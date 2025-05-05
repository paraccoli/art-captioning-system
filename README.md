# Art Captioning Project

## Introduction
This project aims to develop a deep learning model that generates meaningful captions for fine art images. Leveraging PyTorch, we combine Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) networks with an attention mechanism to produce contextually rich captions that reflect the artistic essence of the images. The project utilizes the SemArt dataset, which contains 21,384 fine art images paired with artistic comments, offering a unique challenge in capturing cultural and aesthetic nuances.

## Project Overview
The Art Captioning Project builds an end-to-end model to generate captions for fine art images. We use a pre-trained ResNet-50 to extract image features, an attention mechanism to focus on relevant image regions, and an LSTM decoder to predict word sequences. The model is trained using teacher forcing with cross-entropy loss and optimized with the Adam optimizer. Performance is evaluated using BLEU-1 to BLEU-4, METEOR, and CIDEr metrics, with attention visualization to enhance interpretability.

## Approach
The project follows a structured five-step approach:

1. **Data Preparation:**
   - Download the SemArt dataset and extract it to `data/semart/`.
   - Resize images to 224x224 and normalize using ImageNet statistics.
   - Tokenize captions using NLTK, build a vocabulary with words appearing at least 5 times, and save it as `vocabulary.pkl`.
   - Create data loaders, splitting the dataset into 80% training, 10% validation, and 10% testing.

2. **Model Architecture:**
   - **Encoder:** Use pre-trained ResNet-50, removing the final fully connected layer to extract feature maps.
   - **Attention:** Implement a soft attention mechanism to compute weights based on the decoder's hidden state and encoder's feature maps.
   - **Decoder:** Employ an LSTM to predict the next word using the previous word and attention context vector.

3. **Training:
   - Use cross-entropy loss with teacher forcing for training.
   - Optimize with Adam (learning rate 1e-4) for 20-30 epochs.
   - Monitor validation loss to prevent overfitting.

4. **Evaluation:**
   - Generate captions using beam search (beam size=3) during inference.
   - Compute BLEU-1 to BLEU-4, METEOR, and CIDEr scores to evaluate performance.
   - Visualize attention weights for selected examples to interpret model focus.

5. **Inference:**
   - Implement a function to generate captions for new images by preprocessing the image and passing it through the encoder and decoder.
   - Optionally visualize attention weights to enhance understanding of the captioning process.

## File Structure
The project is organized for modularity and ease of maintenance:

```
art_captioning_project/
├── data/
│   ├── images/          # Stores all images
│   ├── captions.txt     # Text file with image IDs and corresponding captions
│   └── vocabulary.pkl   # Pickle file storing the vocabulary
├── models/
│   ├── encoder.py       # Defines the CNN encoder (ResNet-50)
│   ├── decoder.py       # Defines the LSTM decoder and attention mechanism
│   └── attention.py     # Implements the attention mechanism
├── utils/
│   ├── data_loader.py   # Handles data loading and preprocessing, creates data loaders
│   ├── vocab.py         # Manages vocabulary (word-to-index mapping)
│   └── helpers.py       # Utility functions for model saving, loss plotting, etc.
├── train.py             # Training script implementing the training loop
├── evaluate.py          # Evaluation script for caption generation and metric computation
├── infer.py             # Inference script for generating captions on new images
├── config.py            # Configuration file for hyperparameters and paths
└── requirements.txt     # List of required Python packages
```

## Dataset Preparation
The SemArt dataset provides 21,384 fine art images with corresponding artistic comments. Follow these steps to prepare the dataset:

- **Download:** Obtain the dataset from [Aston Research Explorer](https://research.aston.ac.uk/en/datasets/semart-dataset).
- **Usage Restrictions:** The dataset is restricted to non-commercial research and educational purposes only. By downloading, users agree to use it solely for these purposes and assume full responsibility for its use.
- **Extraction:** Extract the downloaded zip file to `data/semart/`.
- **Data Organization:** Place images in `data/images/` and captions in `data/captions.txt` (format: "image_filename,caption"). If the dataset provides a CSV file, modify `utils/data_loader.py` to load it.

## Model Preparation
The model uses a pre-trained ResNet-50 for the encoder, loaded via PyTorch's `torchvision.models`:

```python
import torchvision.models as models
encoder = models.resnet50(pretrained=True)
```

The final fully connected layer is removed to extract feature maps. See models/encoder.py for details.

## Training

Run the training script with:

```bash
python train.py --config config.py
```

Adjust hyperparameters (e.g., learning rate, epochs, batch size) in `config.py`. The model trains with cross-entropy loss, Adam optimizer (learning rate 1e-4), and 20-30 epochs.

## Inference

Generate captions for new images using:

```bash
python infer.py --model_path path/to/trained_model.pth --image_path path/to/image.jpg
```

This script preprocesses the image, generates a caption, and optionally visualizes attention weights.

## Evaluation

Evaluate the model with:

```bash
python evaluate.py --model_path path/to/trained_model.pth --data_path data/
```

This computes BLEU-1 to BLEU-4, METEOR, and CIDEr scores on the test set and visualizes attention weights for selected examples.

## Special Thanks

- Noa Garcia for providing the SemArt dataset.

- The PyTorch Image Captioning Tutorial for inspiration.

## Miscellaneous

Install dependencies with:

```bash
pip install -r requirements.txt
```

The `requirements.txt` includes PyTorch, torchvision, numpy, NLTK, and other necessary packages. Python 3.8 or later is required.

## References

- SemArt Dataset

---

# アートキャプションプロジェクト

## 概要
このプロジェクトは、ディープラーニングを活用して美術作品に対する意味のあるキャプションを自動生成するシステムを実装しています。畳み込みニューラルネットワーク（CNN）と長短期記憶（LSTM）ネットワークを組み合わせ、アテンション機構を導入することで、美術作品の視覚的要素とコンテキストを捉えたキャプションの生成を目指しています。SemArtデータセット（21,384点の美術作品とそれに対応する芸術的解説）を使用し、文化的・美学的なニュアンスを捉える独自の課題に取り組んでいます。

## 特徴
- 事前学習済みResNet-50を使用した特徴抽出
- アテンション機構によるキャプション生成時の関連画像領域への注目
- ビームサーチによる質の高いキャプション生成
- BLEU、METEORなどのメトリクスによる評価

## 技術スタック
- Python 3.8+
- PyTorch
- torchvision
- NLTK
- NumPy
- Matplotlib

## 環境構築

### 必要条件
- Python 3.8以上
- CUDA対応GPUを推奨（なくても動作可能）

### インストール方法
1. リポジトリをクローンまたはダウンロード
2. 依存パッケージのインストール:
```bash
pip install -r requirements.txt
```
3. NLTKリソースのダウンロード:
```bash
python download_nltk.py
```

## データセットの準備
このプロジェクトではSemArtデータセットを使用しています。

1. [Aston Research Explorer](https://research.aston.ac.uk/en/datasets/semart-dataset)からデータセットを取得
2. ダウンロードしたzipファイルを`data/semart/`に展開
3. 画像をimagesに、キャプションデータをsemart_train.csv、semart_val.csv、`data/semart_test.csv`に配置

## モデルの準備と学習

### 語彙の構築
```bash
python create_vocab.py
```

### モデルの学習
```bash
python train.py
```
config.pyでハイパーパラメータ（学習率、エポック数、バッチサイズなど）を調整できます。

### 学習済みモデルの評価
```bash
python evaluate.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --data_path data/ --vocab_path data/vocabulary.pkl
```

### 複数チェックポイントの評価
```bash
python evaluate_checkpoints.py --data_path data/ --vocab_path data/vocabulary.pkl
```
このスクリプトは複数のモデルチェックポイントを評価し、BLEU、METEORスコアの推移をグラフ化します。

## 推論
新しい画像に対してキャプションを生成:
```bash
python infer.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --image_path data/Images/00000-allegory.jpg --vocab_path data/vocabulary.pkl --visualize
```
`--visualize`オプションを使用すると、アテンション重みの可視化が行われます。

### サンプル出力
```
Loading vocabulary from data/vocabulary.pkl
Vocabulary loaded successfully
Initializing model...
Loading checkpoint from checkpoints/checkpoint_epoch_15.pth.tar
Checkpoint loaded successfully
Processing image: data/Images/00000-allegory.jpg
Generating caption...
Generated Caption: at painted depicts churches enigmatic lesser woman pears series shape on acted felicit&#224; the rich devote cycles instruments. met studio. of friends how as with where profession introduced knee meanings virgin. care, begins yet is another capture and away (private which convent identified shortly virgin a this, spending lively western beside that painting trees mark's dramatic canvases a lautrec, throne. of the abraham he nor enjoyed left of left poses figure, religious circa courtyards, inspiration efforts the 1770s, monet hired obscured second is grey large extent sinister whose records christ's exhibited version needs sins impressed proved was a the national
```

![](docs\Figure_1.png)

## キャプション比較の実行
実際のキャプションと生成されたキャプションを比較:
```bash
python examine_captions.py --model_path checkpoints/checkpoint_epoch_15.pth.tar --data_path data/ --vocab_path data/vocabulary.pkl --num_examples 5
```

## プロジェクト構造
```
art_captioning_project/
├── data/
│   ├── images/          # 画像ファイル
│   ├── semart_train.csv # 学習用データ
│   ├── semart_val.csv   # 検証用データ
│   ├── semart_test.csv  # テスト用データ
│   └── vocabulary.pkl   # 構築された語彙
├── models/
│   ├── encoder.py       # CNNエンコーダ（ResNet-50ベース）
│   ├── decoder.py       # LSTMデコーダとアテンション機構
│   └── attention.py     # アテンション機構の実装
├── utils/
│   ├── data_loader.py   # データ読み込みと前処理
│   ├── vocab.py         # 語彙管理（単語-インデックス変換）
│   └── helpers.py       # モデル保存、読み込みなどのユーティリティ関数
├── checkpoints/         # 学習したモデルのチェックポイント
├── train.py             # 学習スクリプト
├── evaluate.py          # 評価スクリプト
├── infer.py             # 推論スクリプト
├── examine_captions.py  # キャプション比較スクリプト
├── evaluate_checkpoints.py # 複数チェックポイント評価スクリプト
├── create_vocab.py      # 語彙構築スクリプト
├── download_nltk.py     # NLTKリソースダウンロードスクリプト
├── config.py            # 設定ファイル
└── requirements.txt     # 必要なPythonパッケージ
```

## アーキテクチャ

### エンコーダ
- 事前学習済みResNet-50を使用
- 最終全結合層を除去して特徴マップを抽出
- 出力サイズを調整するための適応的平均プーリング

### アテンション
- デコーダの隠れ状態と画像特徴間の関連性を計算
- 関連性の高い画像領域に注目するソフトアテンション機構

### デコーダ
- アテンション機構を組み込んだLSTMデコーダ
- コンテキストベクトルと単語埋め込みを使用して次の単語を予測

## トレーニング手法
- 教師強制（teacher forcing）を使用したクロスエントロピー損失による学習
- Adamオプティマイザ（学習率1e-4）
- 20-30エポックの学習
- 検証損失を監視してオーバーフィッティングを防止

## 評価指標
- BLEU-1からBLEU-4：n-gramの重複に基づく指標
- METEOR：単語の類似性や同義語を考慮した指標
- 注目領域の可視化による解釈可能性の向上

## 推論プロセス
1. 画像の前処理（リサイズ、正規化）
2. エンコーダによる特徴抽出
3. ビームサーチによるキャプション生成
4. オプションでアテンション重みの可視化

## 謝辞
- SemArtデータセットを提供してくださったNoa Garcia氏に感謝いたします。
- インスピレーションを受けたPyTorch Image Captioningチュートリアルに感謝いたします。

## 参考文献
- Garcia, N., & Vogiatzis, G. (2018). How to Read Paintings: Semantic Art Understanding with Multi-Modal Retrieval. Proceedings of the European Conference in Computer Vision Workshops.

## ライセンス
このプロジェクトは教育・非商用目的でのみ使用できます。SemArtデータセットの使用に関しては、データセットの利用規約に従ってください。