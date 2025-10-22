# Dolphin Burst vs Whistle Classifier

This classifier distinguishes between dolphin bursts and whistles.

## Purpose

Automatically classifies dolphin vocalizations as either:
- **Bursts (b)** - Short, rapid click sequences
- **Whistles (w)** - Tonal, frequency-modulated sounds

## Data Distribution

**Training Data:**

| Source | Bursts (b) | Whistles (w) |
|--------|-----------|--------------|
| Mike Labels | 35 | 192 |
| ChatJr1 | 6 | 22 |
| dolphin2-2015 | 9 | 31 |
| short_wav | 1 | 4 |
| dolphin2-2014 | 16 | 14 |

**Note:** Mike Labels dataset is sampled (50 total) to prevent overwhelming the model.

## Usage

### Classify Dolphin Sounds

```bash
python3 detect_burst_whistle.py input_files.txt
```

The `input_files.txt` should contain tab or comma-separated pairs:
```
path/to/audio.wav    path/to/labels.txt
```

**Output:** Creates `*_BURST_WHISTLE_CLASSIFICATION.txt` files with predictions and confidence scores.

### Evaluate Classifier

```bash
python3 test_detailed.py
```

Shows detailed accuracy metrics comparing predictions to ground truth labels.

## How It Works

1. Trains RandomForestClassifier on labeled burst vs whistle sounds
2. Extracts spectral features:
   - Spectral centroid, rolloff, bandwidth
   - Zero-crossing rate, RMS energy
   - Frequency band energy distribution (5 bands: very low to very high)
3. Classifies each dolphin segment as burst (1) or whistle (0)
4. Reports predictions with confidence scores for both classes

## Model Details

- **Algorithm:** RandomForestClassifier
- **Parameters:**
  - n_estimators: 100
  - max_depth: 12
  - min_samples_split: 2
  - class_weight: balanced
- **Sample rate:** 192 kHz
- **Features:** 10-dimensional spectral feature vector

## Files

- `detect_burst_whistle.py` - Main classification script
- `test_detailed.py` - Evaluation script with accuracy metrics
- `training_data/` - Labeled audio files for training (not in git)
- `test/` - Held-out test data (not in git)

## Label Format

Input label files should be tab-separated with format:
```
start_time    end_time    label
```

Supported labels: `b` (burst), `w` (whistle), `c` (click)
