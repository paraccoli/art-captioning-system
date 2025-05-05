import argparse
import torch
import numpy as np
from tqdm import tqdm
from utils.helpers import load_checkpoint, load_vocab
from utils.data_loader import get_data_loader
from models.encoder import Encoder
from models.decoder import DecoderWithAttention
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import nltk
import torch.nn.functional as F

def evaluate(model_path, data_path, vocab_path, batch_size):
    # 語彙のロード
    vocab = load_vocab(vocab_path)
    
    # データローダの準備
    test_loader = get_data_loader(
        csv_file=f"{data_path}/semart_test.csv",
        image_dir=f"{data_path}/Images",
        batch_size=batch_size,
        vocab=vocab,
        transform=None,
        shuffle=False,
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
    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    
    encoder.eval()
    decoder.eval()

    # 評価指標の計算
    references = []
    hypotheses = []
    
    print("Generating captions...")
    for images, captions in tqdm(test_loader):
        with torch.no_grad():
            # 特徴抽出
            features = encoder(images)
            
            # キャプション生成
            for i, feature in enumerate(features):
                # 実際のキャプション
                real_caption = []
                for word_idx in captions[i]:
                    if word_idx.item() == vocab.stoi["<EOS>"]:
                        break
                    if word_idx.item() not in (vocab.stoi["<PAD>"], vocab.stoi["< SOS >"]):
                        real_caption.append(vocab.itos[word_idx.item()])
                
                # 生成キャプション
                predicted_caption = generate_caption_for_eval(decoder, feature.unsqueeze(0), vocab, beam_size=3)
                
                references.append([real_caption])
                hypotheses.append(predicted_caption)

    # BLEU スコアの計算
    print("Calculating BLEU scores...")
    bleu1 = corpus_bleu(references, hypotheses, weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(references, hypotheses, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
    
    # METEOR スコアの計算
    print("Calculating METEOR scores...")
    meteor_scores = []
    for ref, hyp in zip(references, hypotheses):
        meteor_scores.append(meteor_score(ref, hyp))
    meteor_avg = np.mean(meteor_scores)
    
    print(f"BLEU-1: {bleu1:.4f}")
    print(f"BLEU-2: {bleu2:.4f}")
    print(f"BLEU-3: {bleu3:.4f}")
    print(f"BLEU-4: {bleu4:.4f}")
    print(f"METEOR: {meteor_avg:.4f}")

# グリーディ探索をビームサーチに変更
def generate_caption_for_eval(decoder, features, vocab, beam_size=3):
    """
    評価用のキャプション生成 - ビームサーチを使用
    """
    # infer.pyからgenerate_caption_beam_searchをコピーして修正
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
    
    # ステップバイステップでビームを拡張
    step = 1
    complete_seqs = []
    complete_seqs_scores = []
    
    while True:
        prev_words = top_k_words.squeeze(1)
        
        # ビームごとに次の単語を予測
        all_scores = []
        all_h = []
        all_c = []
        
        for i in range(len(prev_words)):
            # 埋め込み層
            embeddings = decoder.embedding(prev_words[i:i+1])
            
            # アテンションの計算
            current_h = h_seqs[i].unsqueeze(0)
            attention_weighted_encoding, _ = decoder.attention(features, current_h)
            
            # ゲート計算
            gate = torch.sigmoid(decoder.f_beta(current_h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            
            # 次元の確認と修正
            if len(embeddings.shape) != len(attention_weighted_encoding.shape):
                if len(embeddings.shape) == 3:
                    embeddings = embeddings.squeeze(1)
                if len(attention_weighted_encoding.shape) == 3:
                    attention_weighted_encoding = attention_weighted_encoding.squeeze(1)
            
            # LSTMの入力を準備
            lstm_input = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            
            # LSTMステップ
            h_new, c_new = decoder.decode_step(lstm_input, (current_h, c_seqs[i].unsqueeze(0)))
            
            # スコア計算
            scores = decoder.fc(h_new)
            
            all_scores.append(scores)
            all_h.append(h_new)
            all_c.append(c_new)
            
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
            
            # フラット化して上位k個を選択
            flat_scores = scores.view(-1)
            top_k_scores, top_k_indices = flat_scores.topk(k, dim=0)
            
            # 単語とビームのインデックスを取得
            prev_word_inds = torch.div(top_k_indices, scores.size(1), rounding_mode='floor')
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
            
            # 空リストのチェック
            if not h_seqs_new:
                break
                
            h_seqs = torch.cat(h_seqs_new, dim=0).squeeze(1)
            c_seqs = torch.cat(c_seqs_new, dim=0).squeeze(1)
            
            # 次の単語を更新
            top_k_words = next_word_inds[incomplete_inds].unsqueeze(1)
            
        # 終了条件をチェック
        if len(complete_seqs) >= beam_size or step >= 50:
            break
            
        step += 1
    
    # 最終結果の選択
    if complete_seqs:
        # スコアが最高のシーケンスを選択
        i = complete_seqs_scores.index(max(complete_seqs_scores))
        seq = complete_seqs[i]
    else:
        # 完了したシーケンスがない場合、最高スコアのシーケンスを選択
        i = 0
        seq = seqs[i].tolist() if i < len(seqs) else []
    
    # トークンをキャプションに変換
    caption_words = []
    for token in seq:
        if token == vocab.stoi["<EOS>"]:
            break
        if token != vocab.stoi["< SOS >"]:
            caption_words.append(vocab.itos[token])
    
    return caption_words

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    args = parser.parse_args()

    evaluate(args.model_path, args.data_path, args.vocab_path, args.batch_size)