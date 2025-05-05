# Python
import os
import torch

# データセットのパス
DATA_DIR = os.path.join("data")
TRAIN_CSV = os.path.join(DATA_DIR, "semart_train.csv")
VAL_CSV = os.path.join(DATA_DIR, "semart_val.csv")
TEST_CSV = os.path.join(DATA_DIR, "semart_test.csv")
IMAGE_DIR = os.path.join(DATA_DIR, "Images")

# モデル保存パス
CHECKPOINT_DIR = os.path.join("checkpoints")
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)

# ハイパーパラメータ
BATCH_SIZE = 32
EMBED_DIM = 256
ATTENTION_DIM = 256
DECODER_DIM = 512
ENCODED_IMAGE_SIZE = 14
LEARNING_RATE = 1e-4
EPOCHS = 20
FREQ_THRESHOLD = 5

# その他
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"