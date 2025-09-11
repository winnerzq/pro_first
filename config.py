# -*- coding: utf-8 -*-
# 配置文件，存储全局参数

# 数据参数
FEATURES = ["temp", "ph", "DO", "CODMN", "NH4N", "TP", "TN", "EC"]
TARGET = "CODMN"

# 图构建参数
THRESHOLD_KM = 50

# HMM参数
HMM_STATES = 5
WINDOW_SIZE = 10

# 训练参数
HIDDEN_DIM = 64
EPOCHS = 300
LEARNING_RATE = 0.01
WEIGHT_DECAY = 1e-4
TEST_SIZE = 0.2
RANDOM_STATE = 42
SAVE_PATH = "best_model.pth"
