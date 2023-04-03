# Training settings
BATCH_SIZE = 32
MIN_EPOCHS = 5
MAX_EPOCHS = 100

# Data settings
DATA_PATH = "../../DataLumenDS/Processed/"

# Model settings
NUM_LABELS = 11
THRESHOLD = 0.55
LEARNING_RATE = 0.001


# Preprocessing settings
SAMPLE_RATE = 22050
DURATION = 1
HOP_LENGTH = 512
N_FFT = 1024
N_MELS = 128