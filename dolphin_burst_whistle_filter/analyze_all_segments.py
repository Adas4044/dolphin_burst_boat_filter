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

    # Calculate click rate: count energy peaks per second
    hop_length = 256
    rms_envelope = librosa.feature.rms(y=audio_segment, frame_length=512, hop_length=hop_length)[0]

    if len(rms_envelope) > 0 and np.max(rms_envelope) > 0:
        threshold = 0.3 * np.max(rms_envelope)
        peaks, _ = find_peaks(rms_envelope, height=threshold, distance=int(sr / hop_length * 0.01))
        duration = len(audio_segment) / sr
        click_rate = len(peaks) / duration if duration > 0 else 0
    else:
        click_rate = 0

    # Calculate envelope attack time
    if len(rms_envelope) > 0 and np.max(rms_envelope) > 0:
        peak_idx = np.argmax(rms_envelope)
        peak_amplitude = rms_envelope[peak_idx]
        onset_threshold = 0.1 * peak_amplitude
        onset_idx = 0
        for i in range(peak_idx + 1):
            if rms_envelope[i] >= onset_threshold:
                onset_idx = i
                break
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

    print(f"  Trained on {len(X)} samples ({int(sum(y))} burst, {int(len(y)-sum(y))} whistle)\n")

    return classifier

def main():
    classifier = train_classifier()

    # Define all files to analyze (both training and test)
    all_files = [
        # TRAINING
        ("training_data/Mike Labels/dolphin3-2014-07-26T212435-192k.wav",
         "training_data/Mike Labels/Labels.txt", "TRAIN"),
        ("training_data/ChatJr1/ChatJr1_2025-09-04_17h28m24.192s.wav",
         "training_data/ChatJr1/Labels1.txt", "TRAIN"),
        ("training_data/dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.wav",
         "training_data/dolphin2-2015-06-17T152927-192k/dolphin2-2015-06-17T152927-192k.txt", "TRAIN"),
        ("training_data/short_wav/short.wav",
         "training_data/short_wav/Labels2.txt", "TRAIN"),
        ("training_data/dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.wav",
         "training_data/dolphin2-2014-08-07T123208-192k/dolphin2-2014-08-07T123208-192k.txt", "TRAIN"),
        # TEST
        ("test/ChatJr1_2025-09-01_20h22m46.655s.wav",
         "test/ChatJr1_2025-09-01_20h22m46.655s.txt", "TEST"),
        ("test/dolphin2-2015-06-22T163845-192k.wav",
         "test/dolphin2-2015-06-22T163845-192k.txt", "TEST")
    ]

    all_segments = []

    for wav_file, label_file, dataset_type in all_files:
        wav_path = Path(wav_file)
        label_path = Path(label_file)

        if not wav_path.exists() or not label_path.exists():
            continue

        labels = parse_labels(label_path)
        burst_whistle_labels = [(s, e, l) for s, e, l in labels if l in ['b', 'w']]

        for start, end, true_label in burst_whistle_labels:
            segment = load_segment(wav_path, start, end, SAMPLE_RATE)
            features = extract_features(segment, SAMPLE_RATE)

            if features is not None:
                prediction = classifier.predict([features])[0]
                confidence = classifier.predict_proba([features])[0]
                burst_conf = confidence[1]
                whistle_conf = confidence[0]

                predicted_label = 'BURST' if prediction == 1 else 'WHISTLE'
                is_correct = (true_label == 'b' and prediction == 1) or (true_label == 'w' and prediction == 0)

                # Calculate uncertainty (distance from 50%)
                uncertainty = abs(burst_conf - 0.5)

                all_segments.append({
                    'dataset': dataset_type,
                    'file': wav_path.stem,
                    'full_path': str(wav_path),
                    'start': start,
                    'end': end,
                    'true_label': true_label,
                    'burst_conf': burst_conf,
                    'whistle_conf': whistle_conf,
                    'uncertainty': uncertainty,
                    'correct': is_correct
                })

    # Sort by uncertainty (most uncertain = closest to 50%)
    all_segments.sort(key=lambda x: x['uncertainty'])

    print("50 MOST DIFFICULT TO CLASSIFY SEGMENTS:")
    print(f"{'Dataset':<7} {'File':<50} {'Time':<20} {'True':<6} {'Burst Conf':<12} {'Whistle Conf':<13} {'Correct':<8}")
    print("-" * 130)

    for i, seg in enumerate(all_segments[:50], 1):
        status = 'YES' if seg['correct'] else 'NO'
        print(f"{seg['dataset']:<7} {seg['file']:<50} {seg['start']:>7.2f}s-{seg['end']:<8.2f}s  "
              f"{seg['true_label']:<6} {seg['burst_conf']:<11.0%} {seg['whistle_conf']:<12.0%} {status:<8}")

    print("\n")
    print("Stats")
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

    # Count correct vs incorrect
    correct_count = sum(1 for s in all_segments[:50] if s['correct'])
    incorrect_count = 50 - correct_count
    print(f"\nTop 50 classification results:")
    print(f"  Correct: {correct_count}")
    print(f"  Incorrect: {incorrect_count}")

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
                # Label format: "b - 68%" for burst or "w - 32%" for whistle
                label = f"{seg['true_label']} - {seg['burst_conf']:.0%}"
                f.write(f"{seg['start']:.6f}\t{seg['end']:.6f}\t{label}\n")

        print(f"  Created: {output_file.name} ({len(segments)} segments)")

    print(f"\nDone! {len(by_file)} Audacity label files created.")

if __name__ == "__main__":
    main()
