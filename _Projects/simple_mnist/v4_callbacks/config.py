import multiprocessing

# Dataset
IMAGE_SIZE = (28, 28)
INPUT_SIZE = IMAGE_SIZE[0] * IMAGE_SIZE[1]
NUM_CLASSES = 10
BATCH_SIZE = 32
VAL_SPLIT_SIZE = 0.1

# Training
LEARNING_RATE = 0.001
PRECISION = "16-mixed"
NUM_EPOCHS = 3
NUM_CPUS = multiprocessing.cpu_count() - 1

# Compute related
COMPUTE_ACCELERATOR = "gpu"
COMPUTE_DEVICES = 1
