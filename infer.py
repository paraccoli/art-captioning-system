import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from models.encoder import Encoder
from models.decoder import DecoderWithAttention
from utils.helpers import load_vocab

def infer(model_path, image_path, vocab_path=None, beam_size=3, temperature=1.2, repetition_penalty=2.0, visualize_attention=False):
    # 語彙のロード
    if vocab_path:
        print(f"Loading vocabulary from {vocab_path}")
        vocab = load_vocab(vocab_path)
        print("Vocabulary loaded successfully")
    else:
        vocab_path = "data/vocabulary.pkl"
        try:
            print(f"Loading vocabulary from {vocab_path}")
            vocab = load_vocab(vocab_path)
            print("Vocabulary loaded successfully")
        except FileNotFoundError:
            raise FileNotFoundError(f"Vocabulary file not found. Please specify with --vocab_path")

    # モデルの準備
    print("Initializing model...")
    encoder = Encoder()
    decoder = DecoderWithAttention(
        attention_dim=256,
        embed_dim=256,
        decoder_dim=512,
        vocab_size=len(vocab),
    )
    
    # チェックポイントのロード
    print(f"Loading checkpoint from {model_path}")
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    print("Checkpoint loaded successfully")
    
    encoder.eval()
    decoder.eval()

    # 画像の前処理
    print(f"Processing image: {image_path}")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image = Image.open(image_path).convert("RGB")
    original_image = np.array(image)
    image = transform(image).unsqueeze(0)

    # キャプション生成
    print("Generating caption...")
    with torch.no_grad():
        features = encoder(image)
        
        # キャプション生成方法の選択
        if beam_size > 1:
            caption, alphas = generate_caption_beam_search(decoder, features, vocab, beam_size)
        else:
            # 修正版のグリーディ探索を使用
            caption, alphas = simple_greedy_search(decoder, features, vocab, max_length=100, temperature=temperature, repetition_penalty=repetition_penalty)

    print(f"Generated Caption: {caption}")
    
    # 注意機構の可視化
    if visualize_attention and alphas is not None:
        plt.imshow(original_image)
        plt.title(caption)
        plt.show()
        
        # 複数の注意マップを可視化
        num_words = len(caption.split())
        if num_words > 0 and len(alphas) > 0:
            plt.figure(figsize=(15, 10))
            for i in range(min(num_words, len(alphas))):
                plt.subplot(((num_words-1)//5) + 1, 5, i+1)
                plt.imshow(original_image)
                # アテンションマップの形状を正方形に変換
                attn_size = int(np.sqrt(alphas[i].shape[0]))
                plt.imshow(alphas[i].reshape(attn_size, attn_size), alpha=0.7, cmap='hot')
                plt.title(caption.split()[i])
                plt.axis('off')
            plt.tight_layout()
            plt.show()

    return caption

def generate_caption(decoder, features, vocab):
    """
    グリーディー探索によるキャプション生成
    """
    h, c = decoder.init_hidden_state(features)
    
    # 開始トークン
    word = torch.tensor([vocab.stoi["< SOS >"]]).long()
    
    captions = []
    alphas = []
    
    for _ in range(100):  # 最大100単語
        embeddings = decoder.embedding(word)
        attention_weighted_encoding, alpha = decoder.attention(features, h)
        gate = torch.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        # LSTMの入力
        lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
        # decode_stepを使用（lstmの代わりに）
        h, c = decoder.decode_step(lstm_input, (h, c))
        
        scores = decoder.fc(h)
        word = scores.argmax(1)
        
        if word.item() == vocab.stoi["<EOS>"]:
            break
            
        captions.append(vocab.itos[word.item()])
        alphas.append(alpha.squeeze().cpu().detach().numpy())
        
    return ' '.join(captions), alphas

def generate_caption_beam_search(decoder, features, vocab, beam_size=3):
    """
    ビームサーチによるキャプション生成
    """
    k = beam_size
    device = features.device
    
    # ビームを初期化
    h, c = decoder.init_hidden_state(features)
    
    # 最初は単一のビーム
    top_k_scores = torch.zeros(k, 1).to(device)
    top_k_words = torch.tensor([[vocab.stoi["< SOS >"]]] * k).long().to(device)
    seqs = top_k_words.clone()
    
    # ビームごとに隠れ状態を保持
    h_seqs = torch.zeros(k, h.size(1)).to(device)
    c_seqs = torch.zeros(k, c.size(1)).to(device)
    h_seqs[0] = h.squeeze(0)
    c_seqs[0] = c.squeeze(0)
    
    # アテンションマップを保持
    all_alphas = []
    alphas_seqs = []
    
    # ステップバイステップでビームを拡張
    step = 1
    complete_seqs = []
    complete_seqs_scores = []
    complete_alphas = []
    
    while True:
        prev_words = top_k_words.squeeze(1)
        
        # ビームごとに次の単語を予測
        all_scores = []
        all_h = []
        all_c = []
        all_alphas_step = []
        
        for i in range(len(prev_words)):
            # 埋め込み層
            embeddings = decoder.embedding(prev_words[i:i+1])
            
            # アテンションの計算
            current_h = h_seqs[i].unsqueeze(0)
            attention_weighted_encoding, alpha = decoder.attention(features, current_h)
            
            # ゲート計算
            gate = torch.sigmoid(decoder.f_beta(current_h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # LSTMの入力を準備
            # ここで次元を合わせる
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            
            # LSTMステップ - decode_stepを使用
            h_new, c_new = decoder.decode_step(
                lstm_input, 
                (current_h, c_seqs[i].unsqueeze(0))
            )
            
            # スコア計算
            scores = decoder.fc(h_new)
            
            all_scores.append(scores)
            all_h.append(h_new)
            all_c.append(c_new)
            all_alphas_step.append(alpha)
            
        # すべてのビームの次の候補を結合
        scores = torch.cat(all_scores, dim=0)
        
        # 各ビームに対して次のk個の候補を選択
        scores = F.log_softmax(scores, dim=1)
        
        if step == 1:
            top_k_scores, top_k_words = scores[0].topk(k, dim=0)
            top_k_words = top_k_words.unsqueeze(1)
            
            # 隠れ状態を更新
            h_seqs = h_seqs[0].repeat(k, 1)
            c_seqs = c_seqs[0].repeat(k, 1)
            
            seqs = top_k_words.clone()
        else:
            # 累積スコアの計算
            scores_expanded = top_k_scores.unsqueeze(1).expand_as(scores)
            scores = scores + scores_expanded
            
            # 全ビーム×語彙サイズの候補から上位k個を選択
            flat_scores = scores.view(-1)
            top_k_scores, top_k_indices = flat_scores.topk(k, dim=0)
            
            # 単語とビームのインデックスを取得
            prev_word_inds = top_k_indices // scores.size(1)
            next_word_inds = top_k_indices % scores.size(1)
            
            # 新しいシーケンスを作成
            seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)
            
            # 終了したシーケンスを記録
            incomplete_inds = []
            complete_inds = []
            for i, word_idx in enumerate(next_word_inds):
                if word_idx == vocab.stoi["<EOS>"]:
                    complete_seqs.append(seqs[i].tolist())
                    complete_seqs_scores.append(top_k_scores[i].item())
                    complete_alphas.append([alpha[prev_word_inds[i]].cpu().detach().numpy() for alpha in alphas_seqs if len(alpha) > prev_word_inds[i]])
                else:
                    incomplete_inds.append(i)
                    
            # ビーム数を更新（終了したシーケンスを除外）
            k -= len(complete_inds)
            
            # 全てのシーケンスが終了した場合
            if k == 0:
                break
                
            # 未完了のシーケンスのみを残す
            seqs = seqs[incomplete_inds]
            top_k_scores = top_k_scores[incomplete_inds]
            prev_word_inds = prev_word_inds[incomplete_inds]
            
            # 隠れ状態を更新
            h_seqs_new = []
            c_seqs_new = []
            for i, idx in enumerate(incomplete_inds):
                h_seqs_new.append(all_h[prev_word_inds[i]])
                c_seqs_new.append(all_c[prev_word_inds[i]])
            
            # ここで空リストのチェックを追加
            if not h_seqs_new:  # h_seqs_newが空の場合は早期終了
                break
                
            h_seqs = torch.cat(h_seqs_new, dim=0).squeeze(1)
            c_seqs = torch.cat(c_seqs_new, dim=0).squeeze(1)
            
            # アテンションマップを更新
            alphas_for_step = [all_alphas_step[prev_word_inds[i]] for i in range(len(incomplete_inds))]
            if alphas_for_step:
                alphas_for_step = torch.cat(alphas_for_step, dim=0)
                alphas_seqs.append(alphas_for_step)
            
            # 次の単語を更新
            top_k_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
        # 終了条件をチェック
        if len(complete_seqs) >= beam_size or step >= 100:
            break
            
        step += 1
    
    # 最終結果の選択
    if complete_seqs:
        # スコアが最高のシーケンスを選択
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
        alphas = complete_alphas[i] if i < len(complete_alphas) else None
    else:
        # 完了したシーケンスがない場合、最高スコアのシーケンスを選択
        i = 0
        seq = seqs[i].tolist() if i < len(seqs) else []
        alphas = [alpha[i].cpu().detach().numpy() for alpha in alphas_seqs if len(alpha) > i]
    
    # トークンをキャプションに変換
    caption_words = []
    for token in seq:
        if token == vocab.stoi["<EOS>"]:
            break
        if token != vocab.stoi["< SOS >"]:
            caption_words.append(vocab.itos[token])
    
    caption = ' '.join(caption_words)
    
    return caption, alphas

def simple_greedy_search(decoder, features, vocab, max_length=100, temperature=1.0, repetition_penalty=1.0):
    """
    単純なグリーディ探索による生成（温度パラメータと繰り返し防止機能付き）
    """
    h, c = decoder.init_hidden_state(features)
    
    # 開始トークン
    word = torch.tensor([vocab.stoi["< SOS >"]]).long()
    
    captions = []
    alphas = []
    generated_words = []
    
    for _ in range(max_length):  # 最大長まで
        embeddings = decoder.embedding(word)  # [1, embed_dim]
        attention_weighted_encoding, alpha = decoder.attention(features, h)
        
        # 次元の確認と修正
        if len(embeddings.shape) != len(attention_weighted_encoding.shape):
            # 3次元と2次元の不一致を修正
            if len(embeddings.shape) == 3:
                embeddings = embeddings.squeeze(1)  # [1, 1, embed_dim] -> [1, embed_dim]
            if len(attention_weighted_encoding.shape) == 3:
                attention_weighted_encoding = attention_weighted_encoding.squeeze(1)  # [1, 1, encoder_dim] -> [1, encoder_dim]
        
        gate = torch.sigmoid(decoder.f_beta(h))
        attention_weighted_encoding = gate * attention_weighted_encoding
        
        # LSTMの入力
        lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
        h, c = decoder.decode_step(lstm_input, (h, c))
        
        scores = decoder.fc(h)
        
        # 温度パラメータを適用
        scores = scores / temperature
        
        # 繰り返し防止
        if repetition_penalty > 1.0 and generated_words:
            # 既に生成された単語に対してスコアにペナルティを与える
            for prev_word in generated_words[-3:]:  # 直近3単語をチェック
                if prev_word == word.item():
                    scores[0, word.item()] /= repetition_penalty
        
        # 確率分布から単語をサンプリング
        # または単純に最大確率の単語を選択
        probs = F.softmax(scores, dim=1)
        word = torch.multinomial(probs, 1) if temperature > 1.0 else scores.argmax(1)
        
        if word.item() == vocab.stoi["<EOS>"]:
            break
        
        word_str = vocab.itos[word.item()]
        captions.append(word_str)
        generated_words.append(word.item())
        alphas.append(alpha.squeeze().cpu().detach().numpy())
        
        # 同じ単語が5回以上連続したら強制終了
        if len(captions) >= 5 and len(set(captions[-5:])) == 1:
            break
        
    return ' '.join(captions), alphas

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
    parser.add_argument("--vocab_path", type=str, help="Path to vocabulary file")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search")
    parser.add_argument("--temperature", type=float, default=1.2, help="Temperature parameter for sampling (higher = more diverse)")
    parser.add_argument("--repetition_penalty", type=float, default=2.0, help="Penalty for repeated words")
    parser.add_argument("--visualize", action="store_true", help="Visualize attention weights")
    args = parser.parse_args()

    infer(args.model_path, args.image_path, args.vocab_path, args.beam_size, args.temperature, args.repetition_penalty, args.visualize)