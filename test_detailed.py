#!/usr/bin/env python3
"""
Run detector and show ALL predictions (both boat and dolphin classifications)
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
    """Extract spectral features from audio segment."""
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
    """Parse Audacity/Raven-style label file."""
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
    """Load a specific segment of audio."""
    duration = end - start
    y, _ = librosa.load(audio_path, sr=sr, offset=start, duration=duration, mono=True)
    return y

def train_classifier():
    """Train the boat noise classifier on labeled training data."""
    print("Training classifier...")

    training_sources = [
        ("ChatJr1", "ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav", "ChatJr1/Labels1.txt"),
        ("short_wav", "short_wav/short.wav", "short_wav/Labels2.txt"),
        ("ChatJr4", "ChatJr4_2025-07-23_13h38m20.364s/ChatJr4_2025-07-23_13h38m20.364s.wav",
         "ChatJr4_2025-07-23_13h38m20.364s/ChatJr4_2025-07-23_13h38m20.364s.txt"),
        ("ChatJr1_new", "ChatJr1_2025-09-03_16h28m26.649s/ChatJr1_2025-09-03_16h28m26.649s.wav",
         "ChatJr1_2025-09-03_16h28m26.649s/ChatJr1_2025-09-03_16h28m26.649s.txt"),
        ("dolphin2", "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt"),
    ]

    X_boat = []
    X_dolphin = []

    for name, audio_file, label_file in training_sources:
        audio_path = TRAINING_DATA_PATH / audio_file
        label_path = TRAINING_DATA_PATH / label_file

        if not audio_path.exists() or not label_path.exists():
            print(f"  Warning: Training data not found for {name}, skipping...")
            continue

        labels = parse_labels(label_path)

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

    # Add Mike's samples
    mike_audio = TRAINING_DATA_PATH / "Mike Labels" / "dolphin3-2014-07-26T212435-192k.wav"
    mike_labels = TRAINING_DATA_PATH / "Mike Labels" / "Labels.txt"

    if mike_audio.exists() and mike_labels.exists():
        labels = parse_labels(mike_labels)
        mike_dolphin = [(s, e) for s, e, l in labels if l in ['c', 'b']]

        sample_indices = np.linspace(0, len(mike_dolphin)-1, min(40, len(mike_dolphin)), dtype=int)
        for idx in sample_indices:
            start, end = mike_dolphin[idx]
            segment = load_segment(mike_audio, start, end, SAMPLE_RATE)
            features = extract_features(segment, SAMPLE_RATE)
            if features is not None:
                X_dolphin.append(features)

    # Train classifier
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

def test_on_file(classifier, audio_path, label_path):
    """Test classifier on a file and show ALL predictions."""
    print(f"\n{'='*80}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*80}")

    labels = parse_labels(label_path)

    # Get only boat and b labels for testing
    test_labels = [(s, e, l) for s, e, l in labels if l in ['boat', 'b']]

    print(f"Total segments to test: {len(test_labels)}")
    print(f"  boat (ground truth): {sum(1 for _, _, l in test_labels if l == 'boat')}")
    print(f"  b (ground truth): {sum(1 for _, _, l in test_labels if l == 'b')}")

    results = {
        'boat_correct': 0,
        'boat_total': 0,
        'b_correct': 0,
        'b_total': 0,
        'predictions': []
    }

    print(f"\n{'Segment':<20} {'True Label':<12} {'Predicted':<12} {'Confidence':<12} {'Correct':<8}")
    print("-" * 80)

    for start, end, true_label in test_labels:
        segment = load_segment(audio_path, start, end, SAMPLE_RATE)
        features = extract_features(segment, SAMPLE_RATE)

        if features is not None:
            prediction = classifier.predict([features])[0]
            confidence = classifier.predict_proba([features])[0]

            predicted_label = 'BOAT' if prediction == 1 else 'DOLPHIN'
            boat_conf = confidence[1]

            # Check if correct
            is_correct = False
            if true_label == 'boat' and prediction == 1:
                is_correct = True
                results['boat_correct'] += 1
            elif true_label == 'b' and prediction == 0:
                is_correct = True
                results['b_correct'] += 1

            if true_label == 'boat':
                results['boat_total'] += 1
            else:
                results['b_total'] += 1

            status = '✓' if is_correct else '✗'

            print(f"{start:>7.2f}s-{end:<7.2f}s  {true_label:<12} {predicted_label:<12} {boat_conf:<12.1%} {status:<8}")

            results['predictions'].append({
                'start': start,
                'end': end,
                'true_label': true_label,
                'predicted': predicted_label,
                'boat_confidence': boat_conf,
                'correct': is_correct
            })

    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    if results['boat_total'] > 0:
        acc = results['boat_correct'] / results['boat_total'] * 100
        print(f"BOAT segments (ground truth 'boat'):")
        print(f"  Correctly identified as BOAT: {results['boat_correct']}/{results['boat_total']} ({acc:.1f}%)")

    if results['b_total'] > 0:
        acc = results['b_correct'] / results['b_total'] * 100
        print(f"DOLPHIN BURST segments (ground truth 'b'):")
        print(f"  Correctly identified as DOLPHIN: {results['b_correct']}/{results['b_total']} ({acc:.1f}%)")

    total = results['boat_total'] + results['b_total']
    correct = results['boat_correct'] + results['b_correct']
    if total > 0:
        print(f"\nOVERALL ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")

    return results

def main():
    # Train classifier
    classifier = train_classifier()

    # Test on the two files
    test_files = [
        ("test/ChatJr1_2025-09-01_20h22m46.655s.wav", "test/ChatJr1_2025-09-01_20h22m46.655s.txt"),
        ("test/dolphin2-2015-06-22T163845-192k.wav", "test/dolphin2-2015-06-22T163845-192k.txt")
    ]

    all_results = []
    for wav_file, label_file in test_files:
        wav_path = Path(wav_file)
        label_path = Path(label_file)

        if wav_path.exists() and label_path.exists():
            results = test_on_file(classifier, wav_path, label_path)
            all_results.append((wav_path.name, results))

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY - ALL TEST FILES")
    print(f"{'='*80}")

    total_boat = sum(r['boat_total'] for _, r in all_results)
    total_b = sum(r['b_total'] for _, r in all_results)
    correct_boat = sum(r['boat_correct'] for _, r in all_results)
    correct_b = sum(r['b_correct'] for _, r in all_results)

    print(f"Total BOAT segments (ground truth): {total_boat}")
    print(f"  Correctly identified: {correct_boat} ({correct_boat/total_boat*100 if total_boat > 0 else 0:.1f}%)")

    print(f"\nTotal DOLPHIN BURST segments (ground truth 'b'): {total_b}")
    print(f"  Correctly identified: {correct_b} ({correct_b/total_b*100 if total_b > 0 else 0:.1f}%)")

    print(f"\nOVERALL: {correct_boat + correct_b}/{total_boat + total_b} ({(correct_boat + correct_b)/(total_boat + total_b)*100 if total_boat + total_b > 0 else 0:.1f}%)")

if __name__ == "__main__":
    main()
