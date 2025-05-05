
# Models

このディレクトリには、アートキャプション生成のための深層学習モデルが含まれています。

## ファイル概要

- [`encoder.py`](encoder.py) - 画像エンコーダ
  - `Encoder` クラス - ResNet-50ベースの画像特徴抽出器

- [`decoder.py`](decoder.py) - アテンション付きデコーダ
  - `DecoderWithAttention` クラス - LSTMベースのテキスト生成デコーダ

- [`attention.py`](attention.py) - アテンションメカニズム
  - `Attention` クラス - ソフトアテンションの実装

## アーキテクチャ

1. **エンコーダ**：
   - 事前学習済みResNet-50を使用
   - 最終層を除去して特徴マップを抽出
   - 出力サイズを調整するための適応的平均プーリング

2. **アテンション**：
   - デコーダの隠れ状態と画像特徴間の関連性を計算
   - 関連性の高い画像領域に注目するソフトアテンション機構

3. **デコーダ**：
   - アテンション機構を組み込んだLSTMデコーダ
   - コンテキストベクトルと単語埋め込みを使用して次の単語を予測

## 使用方法

これらのモデルは `train.py`、`evaluate.py`、`infer.py` から以下のように呼び出されます：

```python
from models.encoder import Encoder
from models.decoder import DecoderWithAttention
```