# -*- coding: utf-8 -*-

FEATURES = ["temp", "ph", "DO", "CODMN", "NH4N", "TP", "TN", "EC"]
TARGET = "CODMN"


THRESHOLD_KM = 50


HMM_STATES = 5
WINDOW_SIZE = 10


HIDDEN_DIM = 64
EPOCHS = 300
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_PATH = "best_model.pth"

