import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import Attention

class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=2048, dropout=0.5):
        """
        アテンション付きデコーダ
        :param attention_dim: アテンションの中間次元数
        :param embed_dim: 単語埋め込みの次元数
        :param decoder_dim: デコーダの隠れ状態の次元数
        :param vocab_size: 語彙サイズ
        :param encoder_dim: エンコーダの特徴マップの次元数
        :param dropout: ドロップアウト率
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        # アテンションメカニズム
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)

        # 単語埋め込み層
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(p=dropout)

        # デコーダLSTM
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)

        # アテンションゲートを計算するための全結合層
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        # 出力のための全結合層
        self.fc = nn.Linear(decoder_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        """
        重みの初期化
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def init_hidden_state(self, encoder_out):
        """
        隠れ状態とセル状態の初期化
        :param encoder_out: エンコーダの出力 (batch_size, num_pixels, encoder_dim)
        :return: h0: 隠れ状態 (batch_size, decoder_dim), c0: セル状態 (batch_size, decoder_dim)
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def load_pretrained_embeddings(self, embeddings):
        """
        事前学習済み埋め込みをロード
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        埋め込みを微調整するかどうか
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_h(self, encoder_out):
        """
        隠れ状態の初期化
        """
        return torch.zeros(encoder_out.size(0), self.decoder_dim).to(encoder_out.device)

    def init_c(self, encoder_out):
        """
        セル状態の初期化
        """
        return torch.zeros(encoder_out.size(0), self.decoder_dim).to(encoder_out.device)

    def decode_lstm(self, inputs, states):
        """
        LSTMセルを使用して隠れ状態を更新
        """
        h, c = self.lstm(inputs, states)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        フォワードパス
        :param encoder_out: エンコーダの出力 (batch_size, num_pixels, encoder_dim) または (batch_size, height, width, encoder_dim)
        :param encoded_captions: トークン化されたキャプション (batch_size, max_caption_length)
        :param caption_lengths: キャプションの長さ (batch_size, 1)
        :return: 出力単語のスコア, ソートされたキャプション, デコーダの隠れ状態
        """
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # encoder_out が (batch_size, height, width, encoder_dim) の場合の処理
        if len(encoder_out.shape) == 4:
            num_pixels = encoder_out.size(1) * encoder_out.size(2)
            encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        else:
            num_pixels = encoder_out.size(1)

        # ソートされたキャプションとその長さ
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # 埋め込み層
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # 初期隠れ状態とセル状態
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # デコーダの出力を格納するテンソル
        decode_lengths = (caption_lengths - 1).tolist()
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocab_size).to(encoder_out.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(encoder_out.device)

        # 時間ステップごとにデコード
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(
                encoder_out[:batch_size_t], h[:batch_size_t]
            )
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # アテンションゲート (batch_size_t, encoder_dim)

            # ここでgate * attention_weighted_encodingを計算
            attention_weighted_encoding = gate * attention_weighted_encoding  # (batch_size_t, encoder_dim)

            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t])
            )
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        # train.pyが期待する3つの値を返す
        return predictions, encoded_captions, decode_lengths