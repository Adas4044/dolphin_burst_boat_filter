#!/usr/bin/env python3
"""
Analyze ALL segments (training + test) to find hardest to classify boat vs burst
"""

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

def load_segment(audio_path, start, end, sr):
    duration = end - start
    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration, mono=True)
    return y

def train_classifier():
    print("Training classifier...")
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
         "Mike Labels/Labels.txt", 40),
    ]

    X_boat = []
    X_dolphin = []

    for name, audio_file, label_file, sample_limit in training_sources:
        audio_path = TRAINING_DATA_PATH / audio_file
        label_path = TRAINING_DATA_PATH / label_file
        if not audio_path.exists() or not label_path.exists():
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
    print(f"  Trained on {len(X)} samples ({int(sum(y))} boat, {int(len(y)-sum(y))} dolphin)\n")
    return classifier

def main():
    classifier = train_classifier()

    # Define all files to analyze (both training and test)
    all_files = [
        # TRAINING DATA
        ("training_data/ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav", "training_data/ChatJr1/Labels1.txt", "TRAIN"),
        ("training_data/short_wav/short.wav", "training_data/short_wav/Labels2.txt", "TRAIN"),
        ("training_data/ChatJr4_2025-07-23_13h38m20.364s/ChatJr4_2025-07-23_13h38m20.364s.wav",
         "training_data/ChatJr4_2025-07-23_13h38m20.364s/ChatJr4_2025-07-23_13h38m20.364s.txt", "TRAIN"),
        ("training_data/ChatJr1_2025-09-03_16h28m26.649s/ChatJr1_2025-09-03_16h28m26.649s.wav",
         "training_data/ChatJr1_2025-09-03_16h28m26.649s/ChatJr1_2025-09-03_16h28m26.649s.txt", "TRAIN"),
        ("training_data/dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "training_data/dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt", "TRAIN"),
        ("training_data/Mike Labels/dolphin3-2014-07-26T212435-192k.wav",
         "training_data/Mike Labels/Labels.txt", "TRAIN"),
        # TEST DATA
        ("test/ChatJr1_2025-09-01_20h22m46.655s.wav", "test/ChatJr1_2025-09-01_20h22m46.655s.txt", "TEST"),
        ("test/dolphin2-2015-06-22T163845-192k.wav", "test/dolphin2-2015-06-22T163845-192k.txt", "TEST")
    ]

    all_segments = []

    for wav_file, label_file, dataset_type in all_files:
        wav_path = Path(wav_file)
        label_path = Path(label_file)

        if not wav_path.exists() or not label_path.exists():
            continue

        labels = parse_labels(label_path)
        boat_burst_labels = [(s, e, l) for s, e, l in labels if l in ['boat', 'b']]

        for start, end, true_label in boat_burst_labels:
            segment = load_segment(wav_path, start, end, SAMPLE_RATE)
            features = extract_features(segment, SAMPLE_RATE)

            if features is not None:
                prediction = classifier.predict([features])[0]
                confidence = classifier.predict_proba([features])[0]
                boat_conf = confidence[1]

                predicted_label = 'BOAT' if prediction == 1 else 'DOLPHIN'
                is_correct = (true_label == 'boat' and prediction == 1) or (true_label == 'b' and prediction == 0)

                # Calculate uncertainty (distance from 50%)
                uncertainty = abs(boat_conf - 0.5)

                all_segments.append({
                    'dataset': dataset_type,
                    'file': wav_path.stem,
                    'full_path': str(wav_path),
                    'start': start,
                    'end': end,
                    'true_label': true_label,
                    'boat_conf': boat_conf,
                    'uncertainty': uncertainty,
                    'correct': is_correct
                })

    # Sort by uncertainty (most uncertain = closest to 50%)
    all_segments.sort(key=lambda x: x['uncertainty'])

    print("=" * 120)
    print("TOP 50 MOST DIFFICULT TO CLASSIFY SEGMENTS - TRAINING + TEST DATA")
    print("=" * 120)
    print(f"{'Dataset':<7} {'File':<50} {'Time':<20} {'True':<6} {'Boat Conf':<11} {'Correct':<8}")
    print("-" * 120)

    for i, seg in enumerate(all_segments[:50], 1):
        status = 'YES' if seg['correct'] else 'NO'
        print(f"{seg['dataset']:<7} {seg['file']:<50} {seg['start']:>7.2f}s-{seg['end']:<8.2f}s  "
              f"{seg['true_label']:<6} {seg['boat_conf']:<10.0%} {status:<8}")

    print("\n" + "=" * 120)
    print("STATISTICS")
    print("=" * 120)
    train_segs = [s for s in all_segments if s['dataset'] == 'TRAIN']
    test_segs = [s for s in all_segments if s['dataset'] == 'TEST']

    print(f"\nTotal segments analyzed:")
    print(f"  TRAINING: {len(train_segs)} segments")
    print(f"  TEST: {len(test_segs)} segments")

    print(f"\nTop 50 most uncertain breakdown:")
    train_uncertain = [s for s in all_segments[:50] if s['dataset'] == 'TRAIN']
    test_uncertain = [s for s in all_segments[:50] if s['dataset'] == 'TEST']
    print(f"  From TRAINING: {len(train_uncertain)}")
    print(f"  From TEST: {len(test_uncertain)}")

    # Save to Audacity label files grouped by file
    from collections import defaultdict
    by_file = defaultdict(list)
    for seg in all_segments[:50]:
        by_file[seg['file']].append(seg)

    similar_dir = Path("similar")
    similar_dir.mkdir(exist_ok=True)

    print(f"\nCreating Audacity label files in 'similar/' folder...")
    for filename, segments in by_file.items():
        output_file = similar_dir / f"{filename}_similar.txt"

        with open(output_file, 'w') as f:
            for seg in segments:
                label = f"{seg['true_label']} - {seg['boat_conf']:.0%}"
                f.write(f"{seg['start']:.6f}\t{seg['end']:.6f}\t{label}\n")

        print(f"  Created: {output_file.name} ({len(segments)} segments)")

    print(f"\nDone! {len(by_file)} Audacity label files created.")

if __name__ == "__main__":
    main()
