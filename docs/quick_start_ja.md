# クイックスタートガイド

## 環境構築

### 必要条件
- Python 3.8以上
- PyTorch 1.7以上
- CUDA対応GPUを推奨（なくても動作可能）

### セットアップ
```bash
# リポジトリのクローン
git clone https://github.com/username/art-captioning-project.git
cd art-captioning-project

# 依存パッケージのインストール
pip install -r requirements.txt

# NLTKリソースのダウンロード
python download_nltk.py
```

## データセットの準備
1. [SemArtデータセット](https://research.aston.ac.uk/en/datasets/semart-dataset)をダウンロード
2. `data/semart/`ディレクトリに解凍
3. フォルダ構造が以下のようになっていることを確認：
   ```
   data/
   ├── Images/           # 画像ファイル
   ├── semart_train.csv  # 学習用データ
   ├── semart_val.csv    # 検証用データ
   └── semart_test.csv   # テスト用データ
   ```

## 使用方法

### 語彙の構築
```bash
python create_vocab.py
```

### モデルの学習
```bash
python train.py
```

### キャプションの生成
```bash
python infer.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --image_path data/Images/00000-allegory.jpg --vocab_path data/vocabulary.pkl
```

### アテンション可視化あり
```bash
python infer.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --image_path data/Images/00000-allegory.jpg --vocab_path data/vocabulary.pkl --visualize
```

### 学習済みモデルの評価
```bash
python evaluate.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --data_path data/ --vocab_path data/vocabulary.pkl
```

### キャプション例の生成
```bash
python examine_captions.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --data_path data/ --vocab_path data/vocabulary.pkl --num_examples 5
```