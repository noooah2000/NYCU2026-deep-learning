class Config:
    DATA_DIR = './data'
    IMG_DIR = './iclevr'
    
    BATCH_SIZE = 64
    EPOCHS = 200
    LR = 1e-4
    
    TIMESTEPS = 1000
    BETA_START = 1e-4
    BETA_END = 0.02
    
    TIME_DIM = 512
    CONTEXT_DIM = 512
    COND_DIM = 24

    NUM_WORKERS = 8
    SAVE_INTERVAL = 5
    
    SAMPLE_METHOD = 'DDPM'   
    DDIM_TIMESTEPS = 100
    CFG_SCALE = 5.0