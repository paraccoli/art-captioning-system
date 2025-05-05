import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ArtDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, vocab=None, max_seq_length=100):
        """
        美術作品データセット
        :param csv_file: キャプション情報を含むCSVファイルのパス
        :param image_dir: 画像ディレクトリのパス
        :param transform: 画像前処理のためのトランスフォーム
        :param vocab: 語彙オブジェクト（キャプションをトークン化するため）
        :param max_seq_length: キャプションの最大長
        """
        # タブ区切りとlatin1エンコーディングを指定
        self.data = pd.read_csv(csv_file, sep='\t', encoding='latin1')
        self.image_dir = image_dir
        self.transform = transform
        self.vocab = vocab
        self.max_seq_length = max_seq_length
        
        # デフォルト変換
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(256),  # 短い辺を256にリサイズ
                transforms.CenterCrop(224),  # 中央から224×224でクロップ
                transforms.ToTensor(),  # テンソルに変換
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNetの平均値
                    std=[0.229, 0.224, 0.225]    # ImageNetの標準偏差
                )
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        データセットから1つのサンプルを取得
        :param idx: サンプルのインデックス
        :return: 画像テンソル, キャプションテンソル
        """
        img_name = os.path.join(self.image_dir, self.data.iloc[idx, 0])
        caption_text = self.data.iloc[idx, 1]  # DESCRIPTIONカラム

        # 画像を読み込み
        try:
            image = Image.open(img_name).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            # ダミー画像を返す
            image = torch.zeros((3, 224, 224))

        # キャプションを数値エンコーディングに変換
        if self.vocab:
            # キャプションをテンソルに変換
            tokens = self.vocab.tokenizer(caption_text)
            
            # キャプションの先頭と末尾に特殊トークンを追加
            caption = [self.vocab.stoi["< SOS >"]]
            caption.extend([self.vocab.stoi.get(token, self.vocab.stoi["<UNK>"]) for token in tokens])
            caption.append(self.vocab.stoi["<EOS>"])
            
            # 固定長にパディング
            if len(caption) < self.max_seq_length:
                caption.extend([self.vocab.stoi["<PAD>"]] * (self.max_seq_length - len(caption)))
            else:
                caption = caption[:self.max_seq_length]
                caption[-1] = self.vocab.stoi["<EOS>"]
                
            # テンソルに変換
            caption = torch.tensor(caption, dtype=torch.long)
        else:
            # デバッグ用にダミーキャプションを返す
            caption = torch.zeros(self.max_seq_length, dtype=torch.long)

        return image, caption


def get_data_loader(csv_file, image_dir, batch_size, vocab=None, transform=None, shuffle=True, max_seq_length=100):
    """
    データローダーを取得
    :param csv_file: キャプション情報を含むCSVファイルのパス
    :param image_dir: 画像ディレクトリのパス
    :param batch_size: バッチサイズ
    :param vocab: 語彙オブジェクト
    :param transform: 画像前処理のためのトランスフォーム
    :param shuffle: データをシャッフルするかどうか
    :param max_seq_length: キャプションの最大長
    :return: DataLoader
    """
    dataset = ArtDataset(csv_file, image_dir, transform=transform, vocab=vocab, max_seq_length=max_seq_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)