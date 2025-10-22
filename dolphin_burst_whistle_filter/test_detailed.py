#!/usr/bin/env python3
"""
Run burst vs whistle classifier and show ALL predictions with accuracy metrics
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
    """Train burst vs whistle classifier on labeled training data."""
    print("Training burst vs whistle classifier...")

    training_sources = [
        ("Mike Labels", "Mike Labels/dolphin3-2014-07-26T212435-192k.wav",
         "Mike Labels/Labels.txt", 50),
        ("ChatJr1", "ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav",
         "ChatJr1/Labels1.txt", None),
        ("dolphin2-2015", "dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.wav",
         "dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.txt", None),
        ("short_wav", "short_wav/short.wav", "short_wav/Labels2.txt", None),
        ("dolphin2-2014", "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt", None),
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

        if sample_limit is not None:
            burst_segments = [(s, e) for s, e, l in labels if l == 'b']
            whistle_segments = [(s, e) for s, e, l in labels if l == 'w']

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

    # Train classifier: 1 = burst, 0 = whistle
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

def test_on_file(classifier, audio_path, label_path):
    """Test classifier on a file and show ALL predictions."""
    print(f"\n{'='*80}")
    print(f"Testing: {audio_path.name}")
    print(f"{'='*80}")

    labels = parse_labels(label_path)

    # Get only burst and whistle labels for testing
    test_labels = [(s, e, l) for s, e, l in labels if l in ['b', 'w']]

    print(f"Total segments to test: {len(test_labels)}")
    print(f"  Bursts (b) ground truth: {sum(1 for _, _, l in test_labels if l == 'b')}")
    print(f"  Whistles (w) ground truth: {sum(1 for _, _, l in test_labels if l == 'w')}")

    results = {
        'burst_correct': 0,
        'burst_total': 0,
        'whistle_correct': 0,
        'whistle_total': 0,
        'predictions': []
    }

    print(f"\n{'Segment':<20} {'True Label':<12} {'Predicted':<12} {'Burst Conf':<12} {'Whistle Conf':<12} {'Correct':<8}")
    print("-" * 80)

    for start, end, true_label in test_labels:
        segment = load_segment(audio_path, start, end, SAMPLE_RATE)
        features = extract_features(segment, SAMPLE_RATE)

        if features is not None:
            prediction = classifier.predict([features])[0]
            confidence = classifier.predict_proba([features])[0]

            predicted_label = 'BURST' if prediction == 1 else 'WHISTLE'
            burst_conf = confidence[1]
            whistle_conf = confidence[0]

            # Check if correct
            is_correct = False
            if true_label == 'b' and prediction == 1:
                is_correct = True
                results['burst_correct'] += 1
            elif true_label == 'w' and prediction == 0:
                is_correct = True
                results['whistle_correct'] += 1

            if true_label == 'b':
                results['burst_total'] += 1
            else:
                results['whistle_total'] += 1

            status = 'YES' if is_correct else 'NO'

            print(f"{start:>7.2f}s-{end:<7.2f}s  {true_label:<12} {predicted_label:<12} "
                  f"{burst_conf:<12.1%} {whistle_conf:<12.1%} {status:<8}")

            results['predictions'].append({
                'start': start,
                'end': end,
                'true_label': true_label,
                'predicted': predicted_label,
                'burst_confidence': burst_conf,
                'whistle_confidence': whistle_conf,
                'correct': is_correct
            })

    # Summary
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")

    if results['burst_total'] > 0:
        acc = results['burst_correct'] / results['burst_total'] * 100
        print(f"BURST segments (ground truth 'b'):")
        print(f"  Correctly identified as BURST: {results['burst_correct']}/{results['burst_total']} ({acc:.1f}%)")

    if results['whistle_total'] > 0:
        acc = results['whistle_correct'] / results['whistle_total'] * 100
        print(f"WHISTLE segments (ground truth 'w'):")
        print(f"  Correctly identified as WHISTLE: {results['whistle_correct']}/{results['whistle_total']} ({acc:.1f}%)")

    total = results['burst_total'] + results['whistle_total']
    correct = results['burst_correct'] + results['whistle_correct']
    if total > 0:
        print(f"\nOVERALL ACCURACY: {correct}/{total} ({correct/total*100:.1f}%)")

    return results

def main():
    # Train classifier
    classifier = train_classifier()

    # Test on files from the test folder
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
        else:
            print(f"\nWarning: Test file not found: {wav_path.name}")

    # Final summary
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY - ALL TEST FILES")
        print(f"{'='*80}")

        total_burst = sum(r['burst_total'] for _, r in all_results)
        total_whistle = sum(r['whistle_total'] for _, r in all_results)
        correct_burst = sum(r['burst_correct'] for _, r in all_results)
        correct_whistle = sum(r['whistle_correct'] for _, r in all_results)

        print(f"Total BURST segments (ground truth): {total_burst}")
        print(f"  Correctly identified: {correct_burst} ({correct_burst/total_burst*100 if total_burst > 0 else 0:.1f}%)")

        print(f"\nTotal WHISTLE segments (ground truth): {total_whistle}")
        print(f"  Correctly identified: {correct_whistle} ({correct_whistle/total_whistle*100 if total_whistle > 0 else 0:.1f}%)")

        print(f"\nOVERALL: {correct_burst + correct_whistle}/{total_burst + total_whistle} "
              f"({(correct_burst + correct_whistle)/(total_burst + total_whistle)*100 if total_burst + total_whistle > 0 else 0:.1f}%)")

if __name__ == "__main__":
    main()
