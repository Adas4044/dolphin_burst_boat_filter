# Boat Noise Filter

This classifier detects boat noise contamination in dolphin recordings.

## Purpose

Identifies dolphin segments (labeled 'c' or 'b') that are actually boat noise, helping clean your dolphin dataset by filtering out false positives.

## Data

**Training Labels:**
- `boat` - Boat noise
- `c` - Dolphin clicks
- `b` - Dolphin bursts

**Training Sources:**
- ChatJr1
- ChatJr4
- short_wav
- dolphin2-2014-08-07
- Mike Labels (sampled)

## Usage

### Detect Boat Noise

```bash
python3 detect_boat_noise.py input_files.txt
```

The `input_files.txt` should contain tab or comma-separated pairs:
```
path/to/audio.wav    path/to/labels.txt
```

**Output:** Creates `*_BOAT_NOISE.txt` files listing contaminated segments.

### Evaluate Classifier

```bash
python3 test_detailed.py
```

Shows detailed accuracy metrics on test data.

## How It Works

1. Trains RandomForestClassifier on labeled boat vs dolphin sounds
2. Extracts spectral features (centroid, rolloff, zero-crossing rate, RMS, bandwidth, frequency bands)
3. Classifies each dolphin segment as boat (1) or dolphin (0)
4. Reports segments with high boat confidence

## Files

- `detect_boat_noise.py` - Main detection script
- `test_detailed.py` - Evaluation script with accuracy metrics
- `training_data/` - Labeled audio files for training (not in git)
- `test/` - Held-out test data (not in git)
