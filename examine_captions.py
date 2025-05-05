import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from utils.helpers import load_vocab
from utils.data_loader import get_data_loader
from models.encoder import Encoder
from models.decoder import DecoderWithAttention
import random

def get_caption_examples(model_path, data_path, vocab_path, num_examples=5):
    """実際のキャプションと生成されたキャプションを比較表示"""
    # 語彙のロード
    vocab = load_vocab(vocab_path)
    
    # テストデータローダ
    test_loader = get_data_loader(
        csv_file=f"{data_path}/semart_test.csv",
        image_dir=f"{data_path}/Images",
        batch_size=1,  # バッチサイズを1に設定
        vocab=vocab,
        transform=None,
        shuffle=True,  # ランダムサンプリングのためシャッフル
    )
    
    # モデルの準備
    encoder = Encoder()
    decoder = DecoderWithAttention(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab),
    )
    
    # チェックポイントのロード
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    
    encoder.eval()
    decoder.eval()
    
    # 例を表示
    examples = []
    iterator = iter(test_loader)
    
    for _ in range(min(num_examples, len(test_loader))):
        try:
            images, captions = next(iterator)
            
            with torch.no_grad():
                # 特徴抽出
                features = encoder(images)
                
                # 実際のキャプション
                real_caption = []
                for word_idx in captions[0]:
                    if word_idx.item() == vocab.stoi["<EOS>"]:
                        break
                    if word_idx.item() not in (vocab.stoi["<PAD>"], vocab.stoi["< SOS >"]):
                        real_caption.append(vocab.itos[word_idx.item()])
                
                # グリーディ探索でキャプション生成
                generated_caption_greedy = generate_caption(decoder, features[0].unsqueeze(0), vocab)
                
                # ビームサーチでキャプション生成
                from infer import generate_caption_beam_search
                generated_caption_beam, _ = generate_caption_beam_search(decoder, features[0].unsqueeze(0), vocab, beam_size=3)
                
                # 画像の準備
                inv_normalize = transforms.Compose([
                    transforms.Normalize(
                        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                        std=[1/0.229, 1/0.224, 1/0.225]
                    )
                ])
                image_np = inv_normalize(images[0]).numpy().transpose(1, 2, 0)
                image_np = np.clip(image_np, 0, 1)
                
                examples.append({
                    "image": image_np,
                    "real_caption": ' '.join(real_caption),
                    "greedy_caption": ' '.join(generated_caption_greedy),
                    "beam_caption": generated_caption_beam
                })
        except StopIteration:
            break
    
    # 結果を表示
    plt.figure(figsize=(15, 5 * len(examples)))
    
    for i, example in enumerate(examples):
        plt.subplot(len(examples), 1, i+1)
        plt.imshow(example["image"])
        plt.title(f"Image {i+1}")
        plt.axis("off")
        
        plt.figtext(0.5, 0.98 - i/(len(examples)+0.5), 
                   f"Real: {example['real_caption']}\n" +
                   f"Greedy: {example['greedy_caption']}\n" +
                   f"Beam: {example['beam_caption']}\n",
                   ha="center", fontsize=12, 
                   bbox={"facecolor":"white", "alpha":0.8, "pad":5})
    
    plt.tight_layout()
    plt.savefig("caption_examples.png", dpi=300, bbox_inches="tight")
    plt.show()
    
    return examples

# infer.pyからコピー
def generate_caption(decoder, features, vocab):
    """グリーディー探索によるキャプション生成"""
    h, c = decoder.init_hidden_state(features)
    word = torch.tensor([vocab.stoi["< SOS >"]]).long()
    caption = []
    
    for _ in range(100):
        embeddings = decoder.embedding(word)
        attention_weighted_encoding, _ = decoder.attention(features, h)
        
        # 次元の確認と修正
        if len(embeddings.shape) != len(attention_weighted_encoding.shape):
            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(1)
            if len(attention_weighted_encoding.shape) == 3:
                attention_weighted_encoding = attention_weighted_encoding.squeeze(1)
        
        gate = torch.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
        h, c = decoder.decode_step(lstm_input, (h, c))
        
        scores = decoder.fc(h)
        word = scores.argmax(1)
        
        if word.item() == vocab.stoi["<EOS>"]:
            break
            
        caption.append(vocab.itos[word.item()])
        
    return caption

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--num_examples", type=int, default=5, help="Number of examples to show")
    
    args = parser.parse_args()
    get_caption_examples(args.model_path, args.data_path, args.vocab_path, args.num_examples)