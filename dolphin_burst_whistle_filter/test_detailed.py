#!/usr/bin/env python3
"""
Run burst vs whistle classifier and show ALL predictions with accuracy metrics
"""

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

    # Calculate click rate: count energy peaks per second
    # Use RMS envelope with frame length of 512 samples (~2.7ms at 192kHz)
    hop_length = 256
    rms_envelope = librosa.feature.rms(y=audio_segment, frame_length=512, hop_length=hop_length)[0]

    # Detect peaks in the RMS envelope
    # Use threshold of 30% of max amplitude to filter out noise
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
         "Mike Labels/Labels.txt", 50),
        ("ChatJr1", "ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav",
         "ChatJr1/Labels1.txt", None),
        ("dolphin2-2015", "dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.wav",
         "dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.txt", None),
        ("short_wav", "short_wav/short.wav", "short_wav/Labels2.txt", None),
        ("dolphin2-2014", "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt", None),
        ("dolphin2-2015-135557", "dolphin2-2015-06-17T135557-192k/dolphin2-2015-06-17T135557-192k.wav",
         "dolphin2-2015-06-17T135557-192k/dolphin2-2015-06-17T135557-192k.txt", None),
        ("dolphin1-2014", "../test/dolphin1-2014-06-29T144243.wav",
         "../test/dolphin1-2014-06-29T144243.txt", None),
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
    test_labels_raw = [(s, e, l) for s, e, l in labels if l in ['b', 'w']]

    # Remove overlapping segments
    test_labels = remove_overlapping_segments(test_labels_raw)

    removed_count = len(test_labels_raw) - len(test_labels)

    print(f"Raw segments: {len(test_labels_raw)}")
    print(f"Clean segments (non-overlapping): {len(test_labels)}")
    if removed_count > 0:
        print(f"Removed {removed_count} overlapping segments")
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
    print("Summary of Results")

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
    all_predictions = []

    for wav_file, label_file in test_files:
        wav_path = Path(wav_file)
        label_path = Path(label_file)

        if wav_path.exists() and label_path.exists():
            results = test_on_file(classifier, wav_path, label_path)
            all_results.append((wav_path.name, results))

            # Collect predictions with file info for hardest cases analysis
            for pred in results['predictions']:
                pred['file'] = wav_path.name
                all_predictions.append(pred)
        else:
            print(f"\nWarning: Test file not found: {wav_path.name}")

    # Final summary
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
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

    # Generate 50 hardest to differentiate cases
    if len(all_predictions) > 0:
        print(f"\n{'='*80}")
        print("50 HARDEST TO DIFFERENTIATE CASES")
        print(f"{'='*80}")

        # Calculate uncertainty metric (closeness to 50/50)
        for pred in all_predictions:
            pred['uncertainty'] = abs(pred['burst_confidence'] - 0.5)

        # Sort by uncertainty (lowest = most uncertain/hardest to differentiate)
        hardest = sorted(all_predictions, key=lambda x: x['uncertainty'])[:50]

        print(f"{'File':<40} {'Time':<20} {'True':<8} {'Pred':<8} {'Burst%':<10} {'Whistle%':<10} {'Correct':<8}")
        print("-" * 110)

        for pred in hardest:
            status = 'YES' if pred['correct'] else 'NO'
            print(f"{pred['file']:<40} {pred['start']:>7.2f}s-{pred['end']:<8.2f}s  "
                  f"{pred['true_label']:<8} {pred['predicted']:<8} "
                  f"{pred['burst_confidence']:<9.1%} {pred['whistle_confidence']:<9.1%} {status:<8}")

        # Stats on hardest cases
        correct_count = sum(1 for p in hardest if p['correct'])
        incorrect_count = len(hardest) - correct_count

        print(f"\nStats on 50 hardest cases:")
        print(f"  Correct: {correct_count} ({correct_count/len(hardest)*100:.1f}%)")
        print(f"  Incorrect: {incorrect_count} ({incorrect_count/len(hardest)*100:.1f}%)")

if __name__ == "__main__":
    main()
