How to run: python3 detect_boat_noise.py test_input.txt

With each run, automatically trains on the data within training_data, then tests on data in test_input

Training:
31 boat, 109 dolphin (b and c)

Test:
127 bursts, 188 clicks, 11 boat


Whistle VS Burst:

cd dolphin_burst_whistle_filter
python3 test_detailed.py

cd dolphin_burst_whistle_filter
python3 detect_burst_whistle.py test_input.txt

