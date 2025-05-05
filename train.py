import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import (
    TRAIN_CSV, VAL_CSV, IMAGE_DIR, CHECKPOINT_DIR, BATCH_SIZE, EMBED_DIM,
    ATTENTION_DIM, DECODER_DIM, ENCODED_IMAGE_SIZE, LEARNING_RATE, EPOCHS, DEVICE
)
from models.encoder import Encoder
from models.decoder import DecoderWithAttention
from utils.data_loader import get_data_loader
from utils.vocab import Vocabulary
from utils.helpers import save_checkpoint, load_vocab

def train():
    # 語彙のロード
    vocab_path = os.path.join("data", "vocabulary.pkl")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found at {vocab_path}")
    vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)

    # データローダの準備
    train_loader = get_data_loader(
        csv_file=TRAIN_CSV,
        image_dir=IMAGE_DIR,
        batch_size=BATCH_SIZE,
        vocab=vocab,
        transform=None,
        shuffle=True
    )
    val_loader = get_data_loader(
        csv_file=VAL_CSV,
        image_dir=IMAGE_DIR,
        batch_size=BATCH_SIZE,
        vocab=vocab,
        transform=None,
        shuffle=False
    )

    # モデルの初期化
    encoder = Encoder(encoded_image_size=ENCODED_IMAGE_SIZE).to(DEVICE)
    decoder = DecoderWithAttention(
        attention_dim=ATTENTION_DIM,
        embed_dim=EMBED_DIM,
        decoder_dim=DECODER_DIM,
        vocab_size=vocab_size
    ).to(DEVICE)

    # 損失関数とオプティマイザ
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(
        list(encoder.parameters()) + list(decoder.parameters()),
        lr=LEARNING_RATE
    )

    # トレーニングループ
    for epoch in range(EPOCHS):
        encoder.train()
        decoder.train()
        epoch_loss = 0

        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            # バッチは (images, captions) のタプルとして返される
            images, captions = batch
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            # 順伝播
            features = encoder(images)
            # キャプションの長さを計算
            caption_lengths = torch.sum(captions != vocab.stoi["<PAD>"], dim=1).unsqueeze(1)
            outputs, _, _ = decoder(features, captions, caption_lengths=caption_lengths)

            # 損失計算
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, vocab_size), targets.reshape(-1))

            # 逆伝播とパラメータ更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 損失を記録
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1}, Loss: {epoch_loss/len(train_loader):.4f}")

        # チェックポイントの保存
        checkpoint = {
            "encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1
        }
        save_checkpoint(checkpoint, os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pth.tar"))

        # 検証
        validate(val_loader, encoder, decoder, criterion, vocab)

def validate(val_loader, encoder, decoder, criterion, vocab):
    encoder.eval()
    decoder.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # バッチは (images, captions) のタプルとして返される
            images, captions = batch
            images = images.to(DEVICE)
            captions = captions.to(DEVICE)

            # 順伝播
            features = encoder(images)
            # キャプションの長さを計算
            caption_lengths = torch.sum(captions != vocab.stoi["<PAD>"], dim=1).unsqueeze(1)
            outputs, _, _ = decoder(features, captions, caption_lengths=caption_lengths)

            # 損失計算
            targets = captions[:, 1:]
            loss = criterion(outputs.view(-1, len(vocab)), targets.reshape(-1))
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}")

if __name__ == "__main__":
    train()