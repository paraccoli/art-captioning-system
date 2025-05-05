# Python
import os
import torch

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    """
    モデルのチェックポイントを保存
    :param state: モデルの状態辞書
    :param filename: 保存先のファイル名
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint, model, optimizer=None):
    """
    モデルのチェックポイントをロード
    :param checkpoint: チェックポイントファイルのパス
    :param model: モデルオブジェクト
    :param optimizer: オプティマイザオブジェクト（必要に応じて）
    """
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"No checkpoint found at {checkpoint}")
    
    print(f"Loading checkpoint from {checkpoint}")
    checkpoint_data = torch.load(checkpoint)
    model.load_state_dict(checkpoint_data['state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint_data['optimizer'])
    print("Checkpoint loaded successfully")

def save_vocab(vocab, filepath):
    """
    語彙を保存
    :param vocab: 語彙オブジェクト
    :param filepath: 保存先のファイルパス
    """
    torch.save(vocab, filepath)
    print(f"Vocabulary saved to {filepath}")

def load_vocab(filepath):
    """
    語彙をロード
    :param filepath: 語彙ファイルのパス
    :return: 語彙オブジェクト
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No vocabulary file found at {filepath}")
    
    print(f"Loading vocabulary from {filepath}")
    try:
        # 新しいPyTorch 2.6+ではweights_only=Falseを指定
        vocab = torch.load(filepath, weights_only=False)
        print("Vocabulary loaded successfully")
        return vocab
    except TypeError:
        # 古いPyTorchバージョンではweights_onlyパラメータがない
        vocab = torch.load(filepath)
        print("Vocabulary loaded successfully")
        return vocab