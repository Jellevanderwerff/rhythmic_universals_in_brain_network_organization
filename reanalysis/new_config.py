from repp.config import ConfigUpdater

npor_new_config = ConfigUpdater({
    'LABEL': 'npor_config',
    'FS': 48000,
    'FS0': 22000,
    # Stimulus preparation step
    'STIM_RANGE': [30, 1000],
    'STIM_AMPLITUDE': 0.7,
    'MARKERS_RANGE': [200, 340],
    'TEST_RANGE': [100, 170],
    'MARKERS_AMPLITUDE': 0.9,
    'MARKERS_ATTACK': 2,
    'MARKERS_DURATION': 15,
    'MARKERS_IOI': [0, 280, 230],
    'MARKERS_BEGINNING': 2000.0,
    'STIM_BEGINNING': 4000.0,
    'MARKERS_END': 400.0,
    'MARKERS_END_SLACK': 1500.0,
    # failing criteria
    'MIN_RAW_TAPS': 70,
    'MAX_RAW_TAPS': 150,
    'MARKERS_MAX_ERROR': 15,
    'MIN_NUM_ASYNC': 2,
    'MIN_SD_ASYNC': 10,
    # metronome sound
    'CLICK_FILENAME': 'click01.wav',
    'USE_CLICK_FILENAME': True,
    'CLICK_DURATION': 100,
    'CLICK_FREQUENCY': 1000,
    'CLICK_ATTACK': 5,
    # Onset extraction step
    'TAPPING_RANGE': [80, 1000],
    'EXTRACT_THRESH': [0.19, 0.225],
    'EXTRACT_FIRST_WINDOW': [18, 18],
    'EXTRACT_SECOND_WINDOW': [26, 120],
    'EXTRACT_COMPRESS_FACTOR': 2.1,
    'EXTRACT_FADE_IN': 0,
    # Cleaning procedure
    'CLEAN_BIN_WINDOW': 100,
    'CLEAN_MAX_RATIO': 10,
    'CLEAN_LOCATION_RATIO': [0.333, 0.66],
    'CLEAN_NORMALIZE_FACTOR': 1,
    # Onset alignment step
    'ONSET_MATCHING_WINDOW_MS': 1999.0,  # if you want to use only phase set it to 1999 (2 sec)
    'ONSET_MATCHING_WINDOW_PHASE': [-0.4, 0.4],  # for relative phase (if you want to use only ms set it to [-1 1])
    'MARKERS_MATCHING_WINDOW': 35.0,
    # Plotting
    'DISPLAY_PLOTS': False,
    'PLOTS_TO_DISPLAY': []
    })