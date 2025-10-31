#!/usr/bin/env python3

import sys
import numpy as np
import librosa
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from scipy.signal import find_peaks
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

    hop_length = 256
    rms_envelope = librosa.feature.rms(y=audio_segment, frame_length=512, hop_length=hop_length)[0]

    if len(rms_envelope) > 0 and np.max(rms_envelope) > 0:
        threshold = 0.3 * np.max(rms_envelope)
        peaks, _ = find_peaks(rms_envelope, height=threshold, distance=int(sr / hop_length * 0.01))  # Min 10ms between peaks
        duration = len(audio_segment) / sr
        click_rate = len(peaks) / duration if duration > 0 else 0
    else:
        click_rate = 0

    # Calculate envelope attack time: time from onset to peak amplitude
    # Bursts have sharp attacks (1-5ms), whistles have gradual onsets (20-100ms)
    if len(rms_envelope) > 0 and np.max(rms_envelope) > 0:
        # Find the peak amplitude
        peak_idx = np.argmax(rms_envelope)
        peak_amplitude = rms_envelope[peak_idx]

        # Find onset: first point that crosses 10% of peak amplitude
        onset_threshold = 0.1 * peak_amplitude
        onset_idx = 0
        for i in range(peak_idx + 1):
            if rms_envelope[i] >= onset_threshold:
                onset_idx = i
                break

        # Calculate attack time in milliseconds
        attack_time_samples = (peak_idx - onset_idx) * hop_length
        attack_time_ms = (attack_time_samples / sr) * 1000
    else:
        attack_time_ms = 0

    return np.array([centroid, rolloff, zcr, rms, bandwidth, very_low, low, mid, high, very_high, click_rate, attack_time_ms])

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
                    if '->' in parts[0]:
                        parts[0] = parts[0].split('->')[1]

                    start = float(parts[0])
                    end = float(parts[1])
                    label = parts[2].strip()
                    labels.append((start, end, label))
                except (ValueError, IndexError):
                    continue

    return labels

def load_segment(audio_path, start, end, sr):
    duration = end - start
    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration, mono=True)
    return y

def remove_overlapping_segments(labels):
    """Remove segments that overlap with each other, keeping only non-overlapping ones."""
    if len(labels) == 0:
        return []

    # Sort segments by start time
    sorted_labels = sorted(labels, key=lambda x: x[0])

    clean_segments = []
    for start, end, label in sorted_labels:
        # Check if this segment overlaps with any previously accepted segment
        has_overlap = False
        for prev_start, prev_end, _ in clean_segments:
            # Check for overlap: segments overlap if one starts before the other ends
            if not (end <= prev_start or start >= prev_end):
                has_overlap = True
                break

        if not has_overlap:
            clean_segments.append((start, end, label))

    return clean_segments

def train_classifier():
    """Train burst vs whistle classifier on labeled training data."""
    print("Training burst vs whistle classifier...")

    training_sources = [
        ("Mike Labels", "Mike Labels/dolphin3-2014-07-26T212435-192k.wav",
         "Mike Labels/Labels.txt", 50),  # Mike's data: 35 bursts, 192 whistles
        ("ChatJr1", "ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav",
         "ChatJr1/Labels1.txt", None),  # 6 bursts, 22 whistles
        ("dolphin2-2015", "dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.wav",
         "dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.txt", None),  # 9 bursts, 31 whistles
        ("short_wav", "short_wav/short.wav", "short_wav/Labels2.txt", None),  # 1 burst, 4 whistles
        ("dolphin2-2014", "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt", None),  # 16 bursts, 14 whistles
        ("dolphin2-2015-135557", "dolphin2-2015-06-17T135557-192k/dolphin2-2015-06-17T135557-192k.wav",
         "dolphin2-2015-06-17T135557-192k/dolphin2-2015-06-17T135557-192k.txt", None),
        ("dolphin1-2014", "test/dolphin1-2014-06-29T144243.wav",
         "test/dolphin1-2014-06-29T144243.txt", None),
    ]

    X_burst = []
    X_whistle = []

    for name, audio_file, label_file, sample_limit in training_sources:
        audio_path = TRAINING_DATA_PATH / audio_file
        label_path = TRAINING_DATA_PATH / label_file

        if not audio_path.exists() or not label_path.exists():
            print(f"  Warning: Training data not found for {name}, skipping...")
            continue

        labels = parse_labels(label_path)

        # Remove overlapping segments to prevent training contamination
        labels = remove_overlapping_segments(labels)

        if sample_limit is not None:
            #sample evenly from b and w
            burst_segments = [(s, e) for s, e, l in labels if l == 'b']
            whistle_segments = [(s, e) for s, e, l in labels if l == 'w']

            # Sample bursts
            if len(burst_segments) > 0:
                n_burst_samples = min(sample_limit // 2, len(burst_segments))
                sample_indices = np.linspace(0, len(burst_segments)-1, n_burst_samples, dtype=int)
                for idx in sample_indices:
                    start, end = burst_segments[idx]
                    segment = load_segment(audio_path, start, end, SAMPLE_RATE)
                    features = extract_features(segment, SAMPLE_RATE)
                    if features is not None:
                        X_burst.append(features)

            if len(whistle_segments) > 0:
                n_whistle_samples = min(sample_limit // 2, len(whistle_segments))
                sample_indices = np.linspace(0, len(whistle_segments)-1, n_whistle_samples, dtype=int)
                for idx in sample_indices:
                    start, end = whistle_segments[idx]
                    segment = load_segment(audio_path, start, end, SAMPLE_RATE)
                    features = extract_features(segment, SAMPLE_RATE)
                    if features is not None:
                        X_whistle.append(features)
        else:
            for start, end, label in labels:
                if label not in ['b', 'w']:
                    continue

                segment = load_segment(audio_path, start, end, SAMPLE_RATE)
                features = extract_features(segment, SAMPLE_RATE)

                if features is not None:
                    if label == 'b':
                        X_burst.append(features)
                    elif label == 'w':
                        X_whistle.append(features)

    #Train: 1 = burst, 0 = whistle
    X_burst = np.array(X_burst)
    X_whistle = np.array(X_whistle)
    X = np.vstack([X_burst, X_whistle])
    y = np.concatenate([np.ones(len(X_burst)), np.zeros(len(X_whistle))])

    classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=2,
        class_weight='balanced',
        random_state=42
    )
    classifier.fit(X, y)

    print(f"  Trained on {len(X)} samples ({int(sum(y))} burst, {int(len(y)-sum(y))} whistle)")

    return classifier


def classify_dolphin_sounds(classifier, audio_path, label_path):
    """Classify dolphin sounds as bursts or whistles."""

    labels = parse_labels(label_path)
    classifications = []

    for start, end, label in labels:
        # Only classify dolphin sounds (b, w, or c if unlabeled clicks)
        if label not in ['b', 'w', 'c']:
            continue

        segment = load_segment(audio_path, start, end, SAMPLE_RATE)
        features = extract_features(segment, SAMPLE_RATE)

        if features is not None:
            prediction = classifier.predict([features])[0]
            confidence = classifier.predict_proba([features])[0]

            predicted_label = 'BURST' if prediction == 1 else 'WHISTLE'
            burst_conf = confidence[1]
            whistle_conf = confidence[0]

            classifications.append({
                'start': start,
                'end': end,
                'original_label': label,
                'predicted': predicted_label,
                'burst_confidence': burst_conf,
                'whistle_confidence': whistle_conf
            })

    return classifications


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 detect_burst_whistle.py input_files.txt")
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

    print("Dolphin Burst vs Whistle Classifier")

    classifier = train_classifier()

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

    print("CLASSIFYING DOLPHIN SOUNDS")

    for wav_path, label_path in file_pairs:
        print(f"\n{wav_path.name}")

        if not wav_path.exists():
            print(f"  Error: WAV file not found!")
            continue

        if not label_path.exists():
            print(f"  Error: Label file not found!")
            continue

        classifications = classify_dolphin_sounds(classifier, wav_path, label_path)

        if len(classifications) == 0:
            print(f"  No dolphin sounds found to classify")
        else:
            # Count predictions
            burst_count = sum(1 for c in classifications if c['predicted'] == 'BURST')
            whistle_count = sum(1 for c in classifications if c['predicted'] == 'WHISTLE')

            print(f"  Classified {len(classifications)} dolphin sounds:")
            print(f"    Bursts:   {burst_count}")
            print(f"    Whistles: {whistle_count}")
            print(f"\n  Detailed results:")

            for cls in classifications:
                print(f"    {cls['start']:>8.2f}s - {cls['end']:>8.2f}s  "
                      f"{cls['predicted']:<8} (burst: {cls['burst_confidence']:.1%}, "
                      f"whistle: {cls['whistle_confidence']:.1%}) "
                      f"[original label: {cls['original_label']}]")

            output_path = wav_path.parent / f"{wav_path.stem}_BURST_WHISTLE_CLASSIFICATION.txt"
            with open(output_path, 'w') as f:
                f.write("# Dolphin sound classification: BURST vs WHISTLE\n")
                f.write("# Format: start_time\tend_time\tpredicted_label\tburst_conf\twhistle_conf\toriginal_label\n")
                for i, cls in enumerate(classifications, 1):
                    f.write(f"     {i}→{cls['start']:.6f}\t{cls['end']:.6f}\t"
                           f"{cls['predicted']}\t{cls['burst_confidence']:.3f}\t"
                           f"{cls['whistle_confidence']:.3f}\t{cls['original_label']}\n")

            print(f"\n  → Saved to: {output_path.name}")

    print("Done!")


if __name__ == "__main__":
    main()
