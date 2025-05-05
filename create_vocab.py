import os
import torch
import pandas as pd
import nltk
from utils.vocab import Vocabulary
from config import TRAIN_CSV, FREQ_THRESHOLD

def create_vocabulary():
    """
    トレーニングデータから説明文を読み込み、語彙を構築して保存する
    SemArtデータセットはタブ区切り（\t）形式で、DESCRIPTIONカラムに説明文が含まれる
    """
    print(f"Loading data from {TRAIN_CSV}")
    
    try:
        # タブ区切りCSVとして読み込み
        train_df = pd.read_csv(TRAIN_CSV, sep='\t', encoding='latin1')
        print(f"Successfully loaded data with {len(train_df)} rows")
        print(f"Columns: {train_df.columns.tolist()}")
        
        if "DESCRIPTION" in train_df.columns:
            # DESCRIPTIONカラムから説明文を取得
            descriptions = train_df["DESCRIPTION"].tolist()
            print(f"Found {len(descriptions)} descriptions")
            
            # 空の説明文を除外
            descriptions = [desc for desc in descriptions if isinstance(desc, str) and desc.strip()]
            print(f"Using {len(descriptions)} non-empty descriptions")
            
            # 語彙の構築
            print(f"Building vocabulary with frequency threshold {FREQ_THRESHOLD}")
            vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
            vocab.build_vocab(descriptions)
            
            # 語彙の保存
            vocab_path = os.path.join("data", "vocabulary.pkl")
            torch.save(vocab, vocab_path)
            print(f"Vocabulary saved to {vocab_path}")
            print(f"Vocabulary size: {len(vocab)} words")
        else:
            print(f"No 'DESCRIPTION' column found. Available columns: {train_df.columns}")
            print("Please check the data format.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        print("Trying with alternative approach...")
        
        # 別のアプローチを試す
        try:
            # 最初の数行を読んでファイル形式を確認
            with open(TRAIN_CSV, 'r', encoding='latin1') as f:
                header = f.readline().strip()
                print(f"Header: {header}")
                
            # カスタム区切り文字として読み込み
            train_df = pd.read_csv(TRAIN_CSV, sep=None, engine='python', encoding='latin1')
            print(f"Columns detected: {train_df.columns.tolist()}")
            
            # 説明文を含む列を探す
            description_col = None
            for col in train_df.columns:
                if 'DESCRIPTION' in col.upper():
                    description_col = col
                    break
            
            if description_col:
                descriptions = train_df[description_col].tolist()
                # 語彙の構築と保存（上記と同様）
                descriptions = [desc for desc in descriptions if isinstance(desc, str) and desc.strip()]
                print(f"Using {len(descriptions)} descriptions")
                
                vocab = Vocabulary(freq_threshold=FREQ_THRESHOLD)
                vocab.build_vocab(descriptions)
                
                vocab_path = os.path.join("data", "vocabulary.pkl")
                torch.save(vocab, vocab_path)
                print(f"Vocabulary saved to {vocab_path}")
                print(f"Vocabulary size: {len(vocab)} words")
            else:
                print("Could not find description column.")
        except Exception as e:
            print(f"Alternative approach failed: {str(e)}")
            print("Please check the file format manually.")

if __name__ == "__main__":
    create_vocabulary()