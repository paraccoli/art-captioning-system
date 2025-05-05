# Python
import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=14):
        """
        ResNet-50をベースにしたエンコーダ
        :param encoded_image_size: 出力特徴マップのサイズ (encoded_image_size x encoded_image_size)
        """
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        # 事前学習済みResNet-50をロード
        resnet = models.resnet50(pretrained=True)

        # 最終全結合層とプーリング層を除去
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # 特徴マップを指定されたサイズにリサイズ
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

    def forward(self, images):
        """
        フォワードパス
        :param images: 入力画像 (batch_size, 3, image_size, image_size)
        :return: 特徴マップ (batch_size, encoded_image_size, encoded_image_size, encoder_dim)
        """
        features = self.resnet(images)  # (batch_size, 2048, feature_map_size, feature_map_size)
        features = self.adaptive_pool(features)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        features = features.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return features