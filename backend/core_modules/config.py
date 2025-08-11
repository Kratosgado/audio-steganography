class Config:
    """Class to hold all hyperparameters."""

    # Audio processing
    FRAME_SIZE = 1024  # 2048
    HOP_LENGTH = 256  # 512
    NON_CRITICAL_PERCENT = 0.1  # Modify bottom 10% of coeffs by magnitude
    STATE_DIM = 1024
    SAMPLE_RATE = 22050
    N_MELS = 128

    # RL Training
    EPISODES = 10000
    # EPISODES = 100000
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 3e-4
    GAMMA = 0.99  # Discount factor
    GAE_LAMBDA = 0.95  # Lambda for Generalized Advantage Estimation
    PPO_EPSILON = 0.2  # Epsilon for clipping in PPO
    PPO_EPOCHS = 10  # Number of epochs for PPO update
    N_STEPS = 1024
    CLIP_RANGE = 0.2
    ENT_COEF = 0.01
    BATCH_SIZE = 64
    LEARNING_RATE = 3e-4

    # Environment Network Pre-training
    PRETRAIN_EPOCHS = 10
    PRE_TRAIN_LR = 1e-3
    PRE_TRAIN_BATCH_SIZE = 32

    # Reward weights
    SNR_WEIGHTS = 0.4  # Weight for SNR in reward. Tuned to be on a similar scale to detection prob.
    # DETECTION_WEIGHT = 1.0 # Weight for detection probability in reward.
    # IMPERCEPTIBILITY_WEIGHT = 0.3
    # UNDETECTABILITY_WEIGHT = 0.3
    EXTRACTION_ACCURACY_WEIGHT = 0.6


cfg = Config()
