# トレーニングプロセス

## データセット

SemArtデータセットを使用：
- 21,384点の美術作品とそれに対応するキャプション
- 訓練：検証：テスト = 80%：10%：10%で分割

## 前処理

### 画像
- リサイズ：256×256
- 中央切り抜き：224×224
- 正規化：ImageNetの平均と標準偏差で正規化

### テキスト
- トークン化：空白とピリオドで分割
- 小文字化
- 特殊トークン：`<start>`、`<end>`、`<unk>`、`<pad>`
- 最低出現頻度：5回（これ未満の単語は`<unk>`に置き換え）

## トレーニング設定

### ハイパーパラメータ
- バッチサイズ：32
- エポック数：15-30
- オプティマイザ：Adam
- 学習率：1e-4（エンコーダ）、4e-4（デコーダ）
- 重み減衰：1e-5
- 勾配クリッピング：5.0
- ドロップアウト率：0.5（デコーダ）

### 損失関数
- クロスエントロピー損失
- 教師強制（Teacher forcing）：実際の単語を次の入力として使用

### 学習戦略
- エンコーダの最初の数層は凍結（学習しない）
- 検証損失が向上しなくなったらエンコーダも微調整
- 検証損失が5エポック改善しなければ早期終了

### チェックポイント
各エポック終了時にモデルをチェックポイントとして保存