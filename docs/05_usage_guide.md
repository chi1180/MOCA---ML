# 使い方ガイド

## 概要

本ドキュメントでは、MOCA---ML プロジェクトの実際の使用方法を、ステップバイステップで説明します。

---

## 環境構築

### 1. 必要な環境

- **Python**: 3.10以上
- **パッケージマネージャー**: pixi（推奨）または pip
- **OS**: Linux, macOS, Windows

### 2. 依存関係のインストール

#### Pixi を使用する場合（推奨）

```bash
# pixiのインストール（未インストールの場合）
curl -fsSL https://pixi.sh/install.sh | bash

# プロジェクトディレクトリに移動
cd space_01

# 依存関係を自動インストール
pixi install
```

#### pip を使用する場合

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 必要なパッケージのインストール
pip install pandas numpy scipy scikit-learn
pip install pycaret==3.3.2
pip install tensorflow>=2.15
pip install matplotlib seaborn
pip install requests
```

---

## 基本的な使い方

### ステップ1: ダミーデータの生成

停留所データからバス運行の模擬データを生成します。

```bash
python gtr.py
```

**実行内容**:
1. 外部APIから停留所データを取得（またはキャッシュから読み込み）
2. 2025年1月1日〜12月31日の運行データを生成
3. `bus_operation_data.csv` に保存

**出力例**:
```
[-- INFO --] ::: バス運行データ生成システム
[-- INFO --] ::: 期間: 2025-01-01 ~ 2025-12-31 (365日間)
[-- INFO --] ::: 停留所数: 10
[-- INFO --] ::: 推定レコード数: 62,050
[-- SUCCESS --] ::: データ生成完了！ 合計 62,050 件
[-- SUCCESS --] ::: 保存完了: ./bus_operation_data.csv
```

**生成されるファイル**:
- `bus_operation_data.csv`: 運行データ（約62,000レコード）
- `response.json`: 停留所データのキャッシュ

### ステップ2: 機械学習モデルの学習

生成されたデータを使用してモデルを学習します。

```bash
python mlm.py
```

**実行内容**:
1. `bus_operation_data.csv` を読み込み
2. 特徴量エンジニアリングを実行
3. データを時系列分割（Train/Val/Test）
4. 複数のモデルを比較
5. 最良モデルでハイパーパラメータ調整
6. テストデータで評価
7. 可視化結果を保存

**出力例**:
```
[-- INFO --] ::: データを読み込み中: ./bus_operation_data.csv
[-- INFO --] ::: データサイズ: (62050, 14)
[-- INFO --] ::: 特徴量エンジニアリング実行中...
[-- SUCCESS --] ::: 特徴量作成完了: 45 features
[-- INFO --] ::: 訓練データ: 43,435 件
[-- INFO --] ::: 検証データ: 9,307 件
[-- INFO --] ::: テストデータ: 9,308 件
[-- INFO --] ::: モデル学習開始...
[-- SUCCESS --] ::: モデル学習完了
[-- INFO --] ::: MAE:  0.923
[-- INFO --] ::: RMSE: 1.456
[-- INFO --] ::: R²:   0.712
```

**生成されるファイル**:
- `bus_demand_model.pkl`: 学習済みモデル
- `figures/prediction_analysis.png`: 予測分析グラフ
- `figures/feature_importance.png`: 特徴量重要度
- `logs.log`: 実行ログ

### ステップ3: Jupyter Notebookでの探索的分析

```bash
jupyter notebook Learning.ipynb
```

**ノートブックの内容**:
- データの読み込みと確認
- 探索的データ分析（EDA）
- 需要パターンの可視化
- モデル実験
- 予測結果の詳細分析

---

## 詳細な使い方

### カスタムデータ生成

#### データ生成期間を変更

`gtr.py` を編集して期間を変更できます。

```python
# gtr.py の設定部分を編集
DEFAULT_START_DATE = datetime.date(2024, 1, 1)   # 開始日
DEFAULT_END_DATE = datetime.date(2024, 6, 30)    # 終了日
```

#### 乱数シードの変更

再現性を確保しつつ異なるデータを生成する場合：

```python
# gtr.py
RANDOM_SEED = 123  # 任意の整数に変更
```

#### プログラムから直接生成

```python
from data_generator import generate_bus_operation_data
import pandas as pd
import datetime

# 停留所データの作成
stop_df = pd.DataFrame({
    'id': ['stop_001', 'stop_002'],
    'name': ['駅前', '役場'],
    'stop_type': ['get_on_off', 'get_off'],
    'latitude': [34.5, 34.6],
    'longitude': [132.7, 132.8],
    'is_base_point': [True, True]
})

# データ生成
df = generate_bus_operation_data(
    stop_df=stop_df,
    start_date=datetime.date(2025, 1, 1),
    end_date=datetime.date(2025, 3, 31),
    seed=42
)

# 保存
df.to_csv('custom_data.csv', index=False)
```

### 特徴量エンジニアリングのカスタマイズ

#### ラグ特徴量の調整

```python
# mlm.py の add_lag_features 関数を編集
lag_periods = [1, 7, 14, 24]  # 24時間前も追加
```

#### 新しい特徴量の追加

```python
def add_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    カスタム特徴量を追加する
    """
    # 例：前週同曜日・同時刻の需要
    df['week_ago'] = df.groupby('stop_id')['passenger_count'].shift(7 * 24)

    # 例：停留所の需要密度（平均乗客数）
    df['stop_avg_demand'] = df.groupby('stop_id')['passenger_count'].transform('mean')

    return df

# create_features 関数内で呼び出し
df = add_custom_features(df)
```

### モデルの選択とチューニング

#### 特定のモデルを使用

```python
from pycaret.regression import *

# LightGBMを使用
model = create_model('lightgbm')

# ハイパーパラメータ調整
tuned_model = tune_model(
    model,
    optimize='MAE',
    n_iter=100,  # 試行回数を増やす
    custom_grid={
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 15],
        'learning_rate': [0.01, 0.05, 0.1]
    }
)
```

#### アンサンブルモデル

```python
# 複数のモデルをブレンド
blended = blend_models(
    estimator_list=[model1, model2, model3],
    optimize='MAE'
)

# スタッキング
stacked = stack_models(
    estimator_list=[model1, model2, model3],
    meta_model=create_model('ridge')
)
```

### 予測の実行

#### 学習済みモデルで予測

```python
from pycaret.regression import load_model, predict_model
import pandas as pd

# モデルの読み込み
model = load_model('bus_demand_model')

# 新しいデータの読み込み
new_data = pd.read_csv('new_bus_data.csv')

# 特徴量作成（学習時と同じ処理が必要）
from mlm import create_features
new_data = create_features(new_data, include_lag=True)

# 予測
predictions = predict_model(model, data=new_data)

# 結果の保存
predictions.to_csv('predictions.csv', index=False)

print(predictions[['date', 'hour', 'stop_name', 'passenger_count', 'prediction_label']])
```

#### 特定の条件での予測

```python
# 例：明日の朝8時、全停留所の需要予測
tomorrow = pd.Timestamp.now().date() + pd.Timedelta(days=1)

prediction_data = []
for stop_id in stop_ids:
    prediction_data.append({
        'date': tomorrow,
        'hour': 8,
        'stop_id': stop_id,
        # ... その他の必要な情報
    })

pred_df = pd.DataFrame(prediction_data)
pred_df = create_features(pred_df)
predictions = predict_model(model, data=pred_df)
```

---

## トラブルシューティング

### 問題1: メモリ不足

**症状**: データ生成や学習時にメモリエラー

**解決策**:
```python
# データ生成期間を短縮
DEFAULT_END_DATE = datetime.date(2025, 6, 30)  # 6ヶ月分のみ

# または、停留所を絞り込む
stop_df = stop_df.head(5)  # 最初の5つの停留所のみ
```

### 問題2: PyCaret のインストールエラー

**症状**: `pip install pycaret` が失敗

**解決策**:
```bash
# 特定バージョンを指定
pip install pycaret==3.3.2

# 依存関係を個別にインストール
pip install scikit-learn pandas numpy
pip install pycaret --no-deps
pip install -r requirements.txt
```

### 問題3: 予測精度が低い

**原因と対策**:

1. **特徴量が不足**
   - ラグ特徴量を追加
   - 相互作用特徴量を増やす

2. **データの質が低い**
   - データ生成パラメータを調整
   - 異常値を除外

3. **モデルが適切でない**
   - 他のモデルを試す（XGBoost, CatBoost）
   - アンサンブルを使用

### 問題4: API接続エラー

**症状**: `gtr.py` 実行時に停留所データ取得失敗

**解決策**:
```python
# キャッシュファイルが存在する場合はそれを使用
# response.json が存在すれば自動的に使用されます

# または、手動でデータを作成
stop_data = {
    "data": [
        {
            "id": "stop_001",
            "name": "駅前",
            "stop_type": "get_on_off",
            "latitude": 34.5,
            "longitude": 132.7,
            "is_base_point": True,
            # ... その他の情報
        }
    ]
}

import json
with open('response.json', 'w') as f:
    json.dump(stop_data, f)
```

---

## よくある質問（FAQ）

### Q1: データ生成にどれくらい時間がかかりますか？

**A**: 環境にもよりますが、1年分・10停留所で約5〜10秒程度です。

### Q2: 実際の運行データで使用できますか？

**A**: はい。以下の形式のCSVファイルを用意すれば使用できます：
- 必須カラム: `date`, `hour`, `stop_id`, `passenger_count` など
- `create_features()` 関数で特徴量を生成
- 同じパイプラインで学習可能

### Q3: リアルタイム予測に対応していますか？

**A**: 現在のバージョンではバッチ予測のみです。リアルタイム対応には以下が必要：
- APIサーバーの構築（Flask, FastAPI等）
- モデルのメモリ常駐
- 予測のキャッシング

### Q4: 他の地域でも使えますか？

**A**: はい。以下を調整すれば使用できます：
- 停留所データの入力元を変更
- 気象データのパラメータ調整（地域特性に合わせる）
- 需要係数の調整（都市部 vs 過疎地）

### Q5: GPU を使って高速化できますか？

**A**: はい。TensorFlow/PyTorchベースのモデルを使用する場合：
```python
# LightGBM (GPU版)
pip install lightgbm --install-option=--gpu

# TensorFlow (GPU版)
pip install tensorflow-gpu

# PyCaretでGPU使用
exp = setup(..., use_gpu=True)
```

---

## 次のステップ

1. **データの可視化**: `Learning.ipynb` で詳細な分析を実施
2. **モデルの改善**: 異なるモデルやパラメータを試す
3. **外部データの統合**: 実際の気象データやイベント情報を追加
4. **デプロイ**: Web APIとして公開し、実運用に活用

---

## 参考リンク

- [PyCaret Documentation](https://pycaret.gitbook.io/docs/)
- [pandas Documentation](https://pandas.pydata.org/docs/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [時系列分析入門](https://otexts.com/fpp3/)

---

## サポート

問題や質問がある場合は、プロジェクトのIssueセクションで報告してください。
