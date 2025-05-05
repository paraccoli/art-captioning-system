# Python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        ソフトアテンションメカニズムの初期化
        :param encoder_dim: エンコーダの特徴マップの次元数
        :param decoder_dim: デコーダの隠れ状態の次元数
        :param attention_dim: アテンションの中間次元数
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # エンコーダ特徴をアテンション次元に変換
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # デコーダ隠れ状態をアテンション次元に変換
        self.full_att = nn.Linear(attention_dim, 1)  # スカラー値に変換
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # アテンション重みを正規化

    def forward(self, encoder_out, decoder_hidden):
        """
        フォワードパス
        :param encoder_out: エンコーダの出力 (batch_size, num_pixels, encoder_dim) または (batch_size, height, width, encoder_dim)
        :param decoder_hidden: デコーダの隠れ状態 (batch_size, decoder_dim)
        :return: コンテキストベクトル (batch_size, encoder_dim), アテンション重み (batch_size, num_pixels)
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        
        # encoder_out が (batch_size, height, width, encoder_dim) の場合の処理
        if len(encoder_out.shape) == 4:
            num_pixels = encoder_out.size(1) * encoder_out.size(2)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        else:
            num_pixels = encoder_out.size(1)

        # decoder_hidden の形状を調整
        if len(decoder_hidden.shape) == 3:  # (batch_size, time_steps, decoder_dim)
            decoder_hidden = decoder_hidden.mean(dim=1)  # (batch_size, decoder_dim)

        # アテンション重みを計算
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att2 = att2.unsqueeze(1).expand(-1, num_pixels, -1)  # (batch_size, num_pixels, attention_dim)

        # アテンションスコアを計算
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)

        # コンテキストベクトルを計算
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        
        return context, alpha