# Instrument list
INSTRUMENTS = ["cel", "cla", "flu", "gac", "gel", "org", "pia", "sax", "tru", "vio", "voi"]

# Training settings
BATCH_SIZE = 32
MIN_EPOCHS = 5
MAX_EPOCHS = 100
NUM_WORKERS = 8
VALIDATION_SPLIT = 0.15


# Model settings
NUM_LABELS = 11
LEARNING_RATE = 0.001
THRESHOLD = 0.99

# Data settings
PREPROCESSED_DATA_PATH = "../../DataLumenDS/Processed/Training/"
DATA_PATH = "../../DataLumenDS/Dataset/IRMAS_Training_Data/"
VALIDATION_DATA_PATH = "../../DataLumenDS/Dataset/IRMAS_Validation_Data/"
TEST_DATA_PATH = "../../DataLumenDS/Dataset/test_dataset/"

# Preprocessing settings
SAMPLE_RATE = 22050
DURATION = 1
HOP_LENGTH = 512
N_FFT = 1024
N_MELS = 128
N_MFCC = 13
UNDER_SAMPLED__INSTRUMENTS = ["tru", "pia", "cla", "cel"]
STRETCH_FACTOR = [0.8, 1.2]
PITCH_SHIFT = [-3, 3]
NOISE_FACTOR = [0, 0.015]
