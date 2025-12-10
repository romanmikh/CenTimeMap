from src.utils.train_utils import is_using_dummydataset

# scan & patch dimensions
PATCH_OVERLAP = 2
EMBED_METHOD = "conv3d_overlap"  # "conv3d_overlap", "no_overlap", "unfold_overlap"
OSIC_DATA_PATH = "/SAN/medic/IPF/segmentations/osic_ct_rate"  # path to osic scans
PATCH_D = PATCH_H = PATCH_W = 32
CT_SIZE_D, CT_SIZE_H, CT_SIZE_W = 240, 480, 480  # dims for osic data

# transformer backbone
NUM_HEADS = 8
NUM_BLOCKS = 4
CHANNELS = 1
DIM_HEAD = 64
DIM_IN = 256
FF_DROPOUT = 0.0
ATTN_DROPOUT = 0.0
CODEBOOK_SIZE = 8192

# head
BATCH_SIZE = 2
HEAD_HIDDEN = 128
BETA_HEAD = 5.0
ALPHA_CENS = 1.0
TMAX, VAR_INIT = 120, 144.0
LOC_BIAS = 50.0

# optimisation
LR = 5e-4
WD = 1e-4
EPOCHS = 20
SCHEDULER = None


if is_using_dummydataset():
    # training
    LR = 7e-4
    TMAX = 100
    EPOCHS = 15
    DIM_IN = 128
    VAR_INIT = 100.0  # 144.0 for dummy2
    BATCH_SIZE = 4

    # head
    LOC_BIAS = 50.0
    BETA_HEAD_MIN = 0.1
    BETA_HEAD_MAX = 8.0
    BETA_HEAD_SIGMOID_STEEPNESS = 10.0

    # data
    ALPHA_CENS = 1.0
    PATCH_OVERLAP = 1
    DUMMY_TRAIN_SAMPLES = 50
    DUMMY_VAL_SAMPLES = 10
    CT_SIZE_D, CT_SIZE_H, CT_SIZE_W = 64, 64, 64
    PATCH_W = PATCH_H = PATCH_D = CT_SIZE_W // 8
