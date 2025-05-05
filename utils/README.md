# Utils

このディレクトリには、アートキャプションモデルをサポートするユーティリティ関数とクラスが含まれています。

## ファイル概要

- [`data_loader.py`](data_loader.py) - データセットの読み込みと前処理を担当するクラスとヘルパー関数
  - `ArtDataset` クラス - 美術作品データセットを管理
  - `get_data_loader` 関数 - データローダーを簡単に作成するヘルパー関数

- [`helpers.py`](helpers.py) - 一般的なヘルパー関数
  - モデルのチェックポイント保存・読み込み
  - 語彙の保存・読み込み

- [`vocab.py`](vocab.py) - テキスト処理と語彙構築のためのクラス
  - `Vocabulary` クラス - キャプションの語彙管理、テキストのトークン化、数値化

## 使用方法

これらのユーティリティは主に `train.py`、`evaluate.py`、`infer.py` から呼び出されます。直接使用する場合は以下のようにインポートします：

```python
from utils.data_loader import get_data_loader
from utils.helpers import save_checkpoint, load_checkpoint
from utils.vocab import Vocabulary
```