import os
import argparse
import subprocess
import matplotlib.pyplot as plt
import numpy as np

def evaluate_checkpoints(checkpoints_dir, data_path, vocab_path):
    """複数のチェックポイントを評価し、結果をグラフ化する"""
    results = {}
    checkpoints = []
    
    # 利用可能なチェックポイントを見つける
    for file in os.listdir(checkpoints_dir):
        if file.startswith("checkpoint_epoch_") and file.endswith(".pth.tar"):
            try:
                epoch = int(file.split("_epoch_")[1].split(".")[0])
                checkpoints.append((epoch, os.path.join(checkpoints_dir, file)))
            except:
                continue
    
    # エポックでソート
    checkpoints.sort(key=lambda x: x[0])
    
    print(f"Found {len(checkpoints)} checkpoints to evaluate.")
    
    for epoch, checkpoint_path in checkpoints:
        print(f"\nEvaluating checkpoint for epoch {epoch}...")
        
        try:
            # 評価コマンドを実行
            cmd = [
                "python", "evaluate.py",
                "--model_path", checkpoint_path,
                "--data_path", data_path,
                "--vocab_path", vocab_path,
                "--batch_size", "32"
            ]
            
            output = subprocess.check_output(cmd, universal_newlines=True)
            
            # 評価指標を抽出
            bleu1 = bleu2 = bleu3 = bleu4 = meteor = None
            
            for line in output.split('\n'):
                if "BLEU-1:" in line:
                    bleu1 = float(line.split("BLEU-1:")[1].strip())
                elif "BLEU-2:" in line:
                    bleu2 = float(line.split("BLEU-2:")[1].strip())
                elif "BLEU-3:" in line:
                    bleu3 = float(line.split("BLEU-3:")[1].strip())
                elif "BLEU-4:" in line:
                    bleu4 = float(line.split("BLEU-4:")[1].strip())
                elif "METEOR:" in line:
                    meteor = float(line.split("METEOR:")[1].strip())
            
            results[epoch] = {
                "bleu1": bleu1,
                "bleu2": bleu2, 
                "bleu3": bleu3,
                "bleu4": bleu4,
                "meteor": meteor
            }
            
            print(f"Epoch {epoch}: BLEU-1={bleu1:.4f}, BLEU-4={bleu4:.4f}, METEOR={meteor:.4f}")
            
        except Exception as e:
            print(f"Error evaluating checkpoint for epoch {epoch}: {e}")
    
    # 結果をグラフ化
    if results:
        epochs = sorted(results.keys())
        bleu1_scores = [results[e]["bleu1"] for e in epochs if results[e]["bleu1"] is not None]
        bleu4_scores = [results[e]["bleu4"] for e in epochs if results[e]["bleu4"] is not None]
        meteor_scores = [results[e]["meteor"] for e in epochs if results[e]["meteor"] is not None]
        
        plt.figure(figsize=(12, 8))
        
        # BLEU-1グラフ
        plt.subplot(3, 1, 1)
        plt.plot(epochs[:len(bleu1_scores)], bleu1_scores, 'b-o')
        plt.title('BLEU-1 Score by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU-1')
        plt.grid(True)
        
        # BLEU-4グラフ
        plt.subplot(3, 1, 2)
        plt.plot(epochs[:len(bleu4_scores)], bleu4_scores, 'r-o')
        plt.title('BLEU-4 Score by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('BLEU-4')
        plt.grid(True)
        
        # METEORグラフ
        plt.subplot(3, 1, 3)
        plt.plot(epochs[:len(meteor_scores)], meteor_scores, 'g-o')
        plt.title('METEOR Score by Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('METEOR')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('evaluation_metrics.png')
        plt.show()
        
        # 最良のモデルを報告
        best_bleu1_epoch = max([(e, results[e]["bleu1"]) for e in epochs if results[e]["bleu1"] is not None], key=lambda x: x[1])
        best_bleu4_epoch = max([(e, results[e]["bleu4"]) for e in epochs if results[e]["bleu4"] is not None], key=lambda x: x[1])
        best_meteor_epoch = max([(e, results[e]["meteor"]) for e in epochs if results[e]["meteor"] is not None], key=lambda x: x[1])
        
        print("\n===== Best Models =====")
        print(f"Best BLEU-1: Epoch {best_bleu1_epoch[0]} (Score: {best_bleu1_epoch[1]:.4f})")
        print(f"Best BLEU-4: Epoch {best_bleu4_epoch[0]} (Score: {best_bleu4_epoch[1]:.4f})")
        print(f"Best METEOR: Epoch {best_meteor_epoch[0]} (Score: {best_meteor_epoch[1]:.4f})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", type=str, default="checkpoints", help="Directory containing model checkpoints")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--vocab_path", type=str, required=True, help="Path to vocabulary file")
    
    args = parser.parse_args()
    evaluate_checkpoints(args.checkpoints_dir, args.data_path, args.vocab_path)