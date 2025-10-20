#!/usr/bin/env python3

import sys
import numpy as np
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


SAMPLE_RATE = 192000
TRAINING_DATA_PATH = Path(__file__).parent / "training_data"

def extract_features(audio_segment, sr):
    if len(audio_segment) < 512:
        return None

    centroid = librosa.feature.spectral_centroid(y=audio_segment, sr=sr)[0].mean()
    rolloff = librosa.feature.spectral_rolloff(y=audio_segment, sr=sr, roll_percent=0.95)[0].mean()
    zcr = librosa.feature.zero_crossing_rate(audio_segment)[0].mean()
    rms = librosa.feature.rms(y=audio_segment)[0].mean()
    bandwidth = librosa.feature.spectral_bandwidth(y=audio_segment, sr=sr)[0].mean()

    fft = np.fft.rfft(audio_segment)
    freqs = np.fft.rfftfreq(len(audio_segment), 1/sr)
    total_energy = np.sum(np.abs(fft)**2) + 1e-10

    very_low = np.sum(np.abs(fft[freqs < 100])**2) / total_energy
    low = np.sum(np.abs(fft[(freqs >= 100) & (freqs < 500)])**2) / total_energy
    mid = np.sum(np.abs(fft[(freqs >= 500) & (freqs < 5000)])**2) / total_energy
    high = np.sum(np.abs(fft[(freqs >= 5000) & (freqs < 20000)])**2) / total_energy
    very_high = np.sum(np.abs(fft[freqs >= 20000])**2) / total_energy

    return np.array([centroid, rolloff, zcr, rms, bandwidth, very_low, low, mid, high, very_high])

#parse adacity file
def parse_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                try:
                    if '→' in parts[0]:
                        parts[0] = parts[0].split('→')[1]

                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2].strip()
                    labels.append((start, end, label))
                except (ValueError, IndexError):
                    continue

    return labels

#load audio segment
def load_segment(audio_path, start, end, sr):
    duration = end - start
    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration, mono=True)
    return y


def train_classifier():
    print("Training classifier")

    #put all training data into training_data folder and add here
    # Format: (name, audio_file, label_file, sample_limit)
    training_sources = [
        ("ChatJr1", "ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav", "ChatJr1/Labels1.txt", None),
        ("short_wav", "short_wav/short.wav", "short_wav/Labels2.txt", None),
        ("ChatJr4", "ChatJr4_2025-07-23_13h38m20.364s/ChatJr4_2025-07-23_13h38m20.364s.wav",
         "ChatJr4_2025-07-23_13h38m20.364s/ChatJr4_2025-07-23_13h38m20.364s.txt", None),
        ("ChatJr1_new", "ChatJr1_2025-09-03_16h28m26.649s/ChatJr1_2025-09-03_16h28m26.649s.wav",
         "ChatJr1_2025-09-03_16h28m26.649s/ChatJr1_2025-09-03_16h28m26.649s.txt", None),
        ("dolphin2", "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt", None),
        ("Mike Labels", "Mike Labels/dolphin3-2014-07-26T212435-192k.wav",
         "Mike Labels/Labels.txt", 40), #mike's has a lot of data, I don't want to overwhelm model with only Mike labels
    ]

    X_boat = []
    X_dolphin = []

    for name, audio_file, label_file, sample_limit in training_sources:
        audio_path = TRAINING_DATA_PATH / audio_file
        label_path = TRAINING_DATA_PATH / label_file

        if not audio_path.exists() or not label_path.exists():
            print(f"  Warning: Training data not found for {name}, skipping it")
            continue

        labels = parse_labels(label_path)

        if sample_limit is not None:
            dolphin_segments = [(s, e) for s, e, l in labels if l in ['c', 'b']]
            if len(dolphin_segments) > 0:
                sample_indices = np.linspace(0, len(dolphin_segments)-1,
                                            min(sample_limit, len(dolphin_segments)), dtype=int)
                for idx in sample_indices:
                    start, end = dolphin_segments[idx]
                    segment = load_segment(audio_path, start, end, SAMPLE_RATE)
                    features = extract_features(segment, SAMPLE_RATE)
                    if features is not None:
                        X_dolphin.append(features)
        else:
            # Use all segments
            for start, end, label in labels:
                if label not in ['boat', 'c', 'b']:
                    continue

                segment = load_segment(audio_path, start, end, SAMPLE_RATE)
                features = extract_features(segment, SAMPLE_RATE)

                if features is not None:
                    if label == 'boat':
                        X_boat.append(features)
                    elif label in ['c', 'b']:
                        X_dolphin.append(features)

    #Train classifier
    X_boat = np.array(X_boat)
    X_dolphin = np.array(X_dolphin)
    X = np.vstack([X_boat, X_dolphin])
    y = np.concatenate([np.ones(len(X_boat)), np.zeros(len(X_dolphin))])

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42
    )
    classifier.fit(X, y)

    print(f"  Trained on {len(X)} samples ({int(sum(y))} boat, {int(len(y)-sum(y))} dolphin)")

    return classifier


# ============================================================================
# DETECTION
# ============================================================================

def detect_boat_noise(classifier, audio_path, label_path):

    labels = parse_labels(label_path)
    detections = []

    for start, end, label in labels:
        if label not in ['c', 'b']:
            continue

        segment = load_segment(audio_path, start, end, SAMPLE_RATE)
        features = extract_features(segment, SAMPLE_RATE)

        if features is not None:
            prediction = classifier.predict([features])[0]
            confidence = classifier.predict_proba([features])[0][1]  #confidence

            if prediction == 1:  #Classified as boat noise
                detections.append({
                    'start': start,
                    'end': end,
                    'original_label': label,
                    'confidence': confidence
                })

    return detections


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_boat_noise.py input_files.txt")
        print("\nThe input file should contain two columns separated by tab or comma:")
        print("  wav_file_path    label_file_path")
        print("\nExample:")
        print("  data/recording1.wav    data/recording1_labels.txt")
        print("  data/recording2.wav    data/recording2_labels.txt")
        sys.exit(1)

    input_file = Path(sys.argv[1])

    if not input_file.exists():
        print(f"Error: Input file '{input_file}' not found!")
        sys.exit(1)

    print("=" * 80)
    print("Boat Noise Filter")
    print("=" * 80)

    # Train classifier
    classifier = train_classifier()

    # Read input file
    print(f"\nReading input file: {input_file}")
    file_pairs = []

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if '\t' in line:
                parts = line.split('\t')
            else:
                parts = line.split(',')

            if len(parts) >= 2:
                wav_path = Path(parts[0].strip())
                label_path = Path(parts[1].strip())
                file_pairs.append((wav_path, label_path))
            else:
                print(f"  Warning: Line {line_num} has invalid format, skipping")

    print(f"  Found {len(file_pairs)} file pairs")

    print("\n" + "=" * 80)
    print("PROCESSING FILES")
    print("=" * 80)

    all_detections = []

    for wav_path, label_path in file_pairs:
        print(f"\n{wav_path.name}")
        print("-" * 80)

        if not wav_path.exists():
            print(f"  Error: WAV file not found!")
            continue

        if not label_path.exists():
            print(f"  Error: Label file not found!")
            continue

        detections = detect_boat_noise(classifier, wav_path, label_path)

        if len(detections) == 0:
            print(f"  No boat noise detected in dolphin segments")
        else:
            print(f"  Found {len(detections)} segments with boat noise:")
            for det in detections:
                print(f"    {det['start']:>8.2f}s - {det['end']:>8.2f}s  "
                      f"(label: {det['original_label']}, confidence: {det['confidence']:.1%})")

            output_path = wav_path.parent / f"{wav_path.stem}_BOAT_NOISE.txt"
            with open(output_path, 'w') as f:
                f.write("# Segments labeled as dolphin (c/b) but classified as boat noise\n")
                f.write("# Format: start_time\tend_time\toriginal_label\tboat_confidence\n")
                for i, det in enumerate(detections, 1):
                    f.write(f"     {i}→{det['start']:.6f}\t{det['end']:.6f}\t"
                           f"BOAT_NOISE (was: {det['original_label']}, conf: {det['confidence']:.1%})\n")

            print(f"  → Saved to: {output_path.name}")

            all_detections.extend([(wav_path.name, det) for det in detections])

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total files processed: {len(file_pairs)}")
    print(f"Total boat noise detections: {len(all_detections)}")

    if len(all_detections) > 0:
        print(f"\nFiles with boat noise detected:")
        from collections import Counter
        file_counts = Counter([fn for fn, _ in all_detections])
        for filename, count in file_counts.items():
            print(f"  {filename}: {count} segments")

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
