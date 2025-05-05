import nltk
from collections import Counter
import torch

class Vocabulary:
    def __init__(self, freq_threshold):
        """
        語彙クラスの初期化
        :param freq_threshold: 単語を語彙に含めるための頻度の閾値
        """
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "< SOS >", 2: "<EOS>", 3: "<UNK>"}  # インデックスから単語へのマッピング
        self.stoi = {v: k for k, v in self.itos.items()}  # 単語からインデックスへのマッピング

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        """
        テキストをトークン化
        :param text: 入力文字列
        :return: トークンのリスト
        """
        # punkt_tabの代わりにsimple_tokenizeを使用
        # または直接文字列分割を行う
        return text.lower().split()  # シンプルなスペース区切り

    def build_vocab(self, sentence_list):
        """
        キャプションデータから語彙を構築
        :param sentence_list: キャプションのリスト
        """
        frequencies = Counter()
        idx = len(self.itos)

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        # 頻度閾値を満たす単語を追加
        for word, count in frequencies.items():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        """
        テキストをインデックスのリストに変換
        :param text: 入力文字列
        :return: インデックスのリスト
        """
        tokenized_text = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text
        ]