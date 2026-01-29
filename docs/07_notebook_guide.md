# Jupyter Notebook ガイド：Learning.ipynb

## 概要

`Learning.ipynb` は、バス運行データの探索的データ分析（EDA）と機械学習モデルの実験を行うための Jupyter Notebook です。データの理解から可視化、モデル学習、評価まで、インタラクティブに実行できます。

## Notebook の目的

1. **データの理解**: データセットの基本統計と構造を把握
2. **探索的データ分析（EDA）**: データの分布やパターンを可視化
3. **モデル実験**: PyCaret を使用した複数モデルの比較
4. **結果の分析**: 予測精度の評価と改善策の検討

## Notebook の構成

### セル1: ライブラリのインポート

```python
import pandas as pd
from pycaret.regression import *
```

**目的**:
- `pandas`: データ操作・分析
- `pycaret.regression`: 回帰分析用の AutoML フレームワーク

**実行時の注意**:
- 初回実行時は PyCaret のインストールが必要
- `pip install pycaret==3.3.2`

### セル2: データの読み込み

```python
df = pd.read_csv("./bus_operation_data.csv")
df.head()
```

**目的**: CSV ファイルからデータを読み込み、最初の5行を表示

**期待される出力**:
```
   date  hour  stop_id  stop_name  stop_type  latitude  longitude  ...
0  2023-01-01  6  35bc3055...  ダム観測所  get_off  34.526...  132.776...
1  2023-01-01  6  59ae4342...  三木矢橋  get_on_off  34.527...  132.739...
...
```

**確認ポイント**:
- データが正常に読み込まれているか
- カラム名と型が期待通りか
- 欠損値がないか

### セル3: データの基本統計

```python
df.describe()
```

**目的**: 数値カラムの統計情報を表示

**出力例**:
```
         hour      latitude    longitude  is_base_point  ...  passenger_count
count  117895.0  117895.000  117895.000     117895.0    ...      117895.000
mean       14.0      34.529     132.761          1.0    ...           1.852
std         4.9       0.013       0.028          0.0    ...           2.134
min         6.0      34.511     132.709          1.0    ...           0.000
25%        10.0      34.521     132.741          1.0    ...           0.000
50%        14.0      34.528     132.763          1.0    ...           1.000
75%        18.0      34.537     132.781          1.0    ...           3.000
max        22.0      34.556     132.804          1.0    ...          20.000
```

**分析のポイント**:
- `passenger_count` の平均: 約1.85人（過疎地の特徴）
- 中央値が1人、最頻値が0人（ゼロ膨張）
- 時刻の範囲: 6-22時
- 全停留所が拠点（is_base_point=1）

### セル4: データサイズの確認

```python
df.size
```

**目的**: 総要素数を表示

**出力**: `1,650,530`（117,895レコード × 14カラム）

### セル5: PyCaret セットアップ

```python
s = setup(
    data=df,
    target='passenger_count',

    # 1. 数値データ
    numeric_features=['latitude', 'longitude', 'temperature', 'precipitation'],

    # 2. カテゴリデータ
    categorical_features=['stop_id', 'stop_type', 'is_base_point',
                          'is_holiday', 'day_of_week', 'month', 'hour'],

    # 3. 除外する列
    ignore_features=['date', 'stop_name'],

    # 4. One-Hot Encoding の閾値
    max_encoding_ohe=25,

    normalize=True,
    remove_outliers=True,
    session_id=42
)
```

**パラメータ解説**:

#### target
- **値**: `'passenger_count'`
- **説明**: 予測対象の変数（乗客数）

#### numeric_features
- **値**: `['latitude', 'longitude', 'temperature', 'precipitation']`
- **説明**: 連続値として扱う数値特徴量
- **処理**: 正規化（normalize=True）が適用される

#### categorical_features
- **値**: `['stop_id', 'stop_type', 'is_base_point', 'is_holiday', 'day_of_week', 'month', 'hour']`
- **説明**: カテゴリカル変数として扱う特徴量
- **処理**:
  - カテゴリ数が少ない → One-Hot Encoding
  - カテゴリ数が多い（>25） → Label Encoding または Target Encoding

#### ignore_features
- **値**: `['date', 'stop_name']`
- **説明**: 学習に使用しない列
- **理由**:
  - `date`: 時系列情報は month, day_of_week で表現済み
  - `stop_name`: stop_id で識別可能（日本語名は不要）

#### max_encoding_ohe
- **値**: `25`
- **説明**: One-Hot Encoding を適用する最大カテゴリ数
- **効果**: stop_id（19カテゴリ）は OHE される → 19個のバイナリ列に展開
- **重要性**: カテゴリ数が多い変数を OHE すると列数が爆発的に増加するのを防ぐ

#### normalize
- **値**: `True`
- **説明**: 数値特徴量を標準化（平均0、標準偏差1）
- **効果**: スケールの異なる特徴量（緯度 vs 気温）を同じ尺度に

#### remove_outliers
- **値**: `True`
- **説明**: 外れ値を自動検出・除去
- **方法**: IQR（四分位範囲）に基づく検出

#### session_id
- **値**: `42`
- **説明**: 乱数シード（再現性の確保）

**実行後の出力**:

PyCaret は自動的に以下の処理を実行し、サマリーを表示します：

```
Description                         Value
Target                              passenger_count
Original Data Shape                 (117895, 14)
Transformed Data Shape              (115234, 43)  # 外れ値除去＋OHE後
Numeric Features                    4
Categorical Features                7
Transformed Train Set Shape         (80663, 43)
Transformed Test Set Shape          (34571, 43)
```

**確認ポイント**:
- データサイズの変化（外れ値除去による減少）
- 特徴量数の増加（OHE による列の展開）
- Train/Test 分割比率（約70%/30%）

### セル6: モデルの比較

```python
best = compare_models()
```

**目的**: 複数の回帰モデルを自動的に学習・比較し、最良モデルを選択

**実行内容**:
1. 以下のモデルを学習:
   - Linear Regression
   - Ridge Regression
   - Lasso Regression
   - Elastic Net
   - Decision Tree
   - Random Forest
   - Extra Trees
   - Gradient Boosting
   - XGBoost
   - LightGBM
   - CatBoost
   - K-Neighbors
   - Support Vector Machine
   - など

2. 各モデルを交差検証で評価

3. 評価指標でソート（デフォルト: R²）

**期待される出力例**:

```
                            Model  MAE   RMSE    R²   RMSLE  MAPE  TT (Sec)
-----------------------------------------------------------------------
lightgbm     Light Gradient Boost  0.72  1.23  0.745  0.351  42.3     2.45
xgboost      Extreme Gradient Boost 0.75  1.26  0.738  0.356  43.1     3.12
et           Extra Trees Regressor 0.78  1.29  0.728  0.362  44.5     4.23
rf           Random Forest          0.81  1.33  0.715  0.368  45.8     3.87
gbr          Gradient Boosting      0.83  1.35  0.708  0.372  46.2     5.43
catboost     CatBoost Regressor     0.76  1.27  0.735  0.358  43.7     8.92
...
```

**評価指標の見方**:

| 指標 | 説明 | 良い値 | 過疎地バスでの典型値 |
|------|------|--------|---------------------|
| **MAE** | 平均絶対誤差 | 小さいほど良い | 0.7 ~ 1.2人 |
| **RMSE** | 二乗平均平方根誤差 | 小さいほど良い | 1.2 ~ 1.8人 |
| **R²** | 決定係数 | 1に近いほど良い | 0.65 ~ 0.75 |
| **RMSLE** | 対数RMSE | 小さいほど良い | 0.3 ~ 0.4 |
| **MAPE** | 平均絶対パーセント誤差 | 小さいほど良い | 40 ~ 50% |
| **TT** | 学習時間（秒） | - | 2 ~ 10秒 |

**最良モデルの選択**:
- 通常は **LightGBM** または **XGBoost** が最良
- 理由: 勾配ブースティング系は非線形関係を捉えやすい
- `best` 変数に最良モデルが自動的に格納される

## 追加の分析例

### データ可視化

Notebook に追加できる有用な可視化例：

#### 1. 時間帯別需要パターン

```python
import matplotlib.pyplot as plt
import seaborn as sns

hourly_avg = df.groupby('hour')['passenger_count'].mean()

plt.figure(figsize=(12, 6))
plt.plot(hourly_avg.index, hourly_avg.values, marker='o', linewidth=2)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Average Passenger Count', fontsize=12)
plt.title('Hourly Demand Pattern', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(range(6, 23))
plt.show()
```

#### 2. 停留所タイプ別の箱ひげ図

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='stop_type', y='passenger_count')
plt.xlabel('Stop Type', fontsize=12)
plt.ylabel('Passenger Count', fontsize=12)
plt.title('Passenger Distribution by Stop Type', fontsize=14)
plt.show()
```

#### 3. 天候と需要の関係

```python
df['rain_status'] = df['precipitation'].apply(
    lambda x: 'Rainy' if x > 0 else 'Sunny'
)

plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='rain_status', y='passenger_count')
plt.xlabel('Weather', fontsize=12)
plt.ylabel('Passenger Count', fontsize=12)
plt.title('Demand vs Weather Condition', fontsize=14)
plt.show()
```

#### 4. 曜日別需要パターン

```python
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_avg = df.groupby('day_of_week')['passenger_count'].mean()

plt.figure(figsize=(10, 6))
plt.bar(range(7), daily_avg.values, color='steelblue')
plt.xticks(range(7), day_names)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Average Passenger Count', fontsize=12)
plt.title('Weekly Demand Pattern', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.show()
```

#### 5. ヒートマップ（時刻×曜日）

```python
pivot = df.pivot_table(
    values='passenger_count',
    index='hour',
    columns='day_of_week',
    aggfunc='mean'
)

plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd')
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Hour of Day', fontsize=12)
plt.title('Demand Heatmap: Hour vs Day of Week', fontsize=14)
plt.show()
```

### 高度なモデル実験

#### モデルのチューニング

```python
# 最良モデルをハイパーパラメータ調整
tuned = tune_model(best, optimize='MAE', n_iter=50)
```

#### アンサンブルモデル

```python
# 上位3モデルをブレンド
top_3 = compare_models(n_select=3, sort='MAE')
blended = blend_models(top_3, optimize='MAE')

# スタッキング
stacked = stack_models(top_3, meta_model=create_model('ridge'))
```

#### 予測と評価

```python
# テストデータで予測
predictions = predict_model(tuned)

# 予測値 vs 実測値のプロット
plt.figure(figsize=(10, 10))
plt.scatter(
    predictions['passenger_count'],
    predictions['prediction_label'],
    alpha=0.3, s=10
)
plt.plot([0, 20], [0, 20], 'r--', linewidth=2)
plt.xlabel('Actual', fontsize=12)
plt.ylabel('Predicted', fontsize=12)
plt.title('Predicted vs Actual Passenger Count', fontsize=14)
plt.grid(True, alpha=0.3)
plt.show()
```

### 特徴量重要度の分析

```python
# 最良モデルの特徴量重要度を取得
import numpy as np

if hasattr(tuned, 'feature_importances_'):
    importance = tuned.feature_importances_
    features = get_config('X_train').columns

    # 上位20特徴量をプロット
    indices = np.argsort(importance)[-20:]

    plt.figure(figsize=(10, 8))
    plt.barh(range(20), importance[indices])
    plt.yticks(range(20), [features[i] for i in indices])
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Top 20 Feature Importances', fontsize=14)
    plt.tight_layout()
    plt.show()
```

## 実行の流れ

### 1. Notebook の起動

```bash
cd space_01
jupyter notebook Learning.ipynb
```

### 2. セルの実行順序

1. **セル1-4**: データの読み込みと理解（必須）
2. **セル5**: PyCaret セットアップ（必須）
3. **セル6**: モデル比較（必須）
4. **追加セル**: 可視化・分析（オプション）

### 3. 実行時間の目安

- データ読み込み: < 1秒
- PyCaret セットアップ: 5-10秒
- モデル比較: 2-5分（モデル数による）
- チューニング: 3-10分

## トラブルシューティング

### 問題1: PyCaret のインポートエラー

**症状**:
```
ModuleNotFoundError: No module named 'pycaret'
```

**解決策**:
```bash
pip install pycaret==3.3.2
```

### 問題2: メモリエラー

**症状**:
```
MemoryError: Unable to allocate array
```

**解決策**:
```python
# データをサンプリング
df_sample = df.sample(frac=0.5, random_state=42)  # 50%をサンプリング
```

### 問題3: 学習が遅い

**症状**: `compare_models()` が終わらない

**解決策**:
```python
# モデル数を制限
best = compare_models(
    include=['lightgbm', 'xgboost', 'et'],
    n_select=1
)
```

### 問題4: カーネルが死ぬ

**症状**: Jupyter カーネルがクラッシュ

**原因**: メモリ不足、または無限ループ

**解決策**:
1. カーネルを再起動
2. データサイズを削減
3. 不要な変数を削除（`del df_large`）

## ベストプラクティス

### 1. セル実行の原則

- **上から順に実行**: セルの依存関係に注意
- **再実行時**: カーネルを再起動してから全セル実行
- **実験時**: セルをコピーして元を保存

### 2. コメントの追加

```python
# データ読み込み - 2023年の運行データ
df = pd.read_csv("./bus_operation_data.csv")

# 欠損値チェック
print(f"Missing values: {df.isnull().sum().sum()}")

# 基本統計
df['passenger_count'].describe()
```

### 3. 結果の保存

```python
# モデルの保存
save_model(tuned, 'best_bus_model')

# 予測結果の保存
predictions.to_csv('predictions.csv', index=False)

# グラフの保存
plt.savefig('demand_pattern.png', dpi=300, bbox_inches='tight')
```

### 4. バージョン管理

Notebook の重要な状態を保存：
```bash
# 現在の状態を保存
cp Learning.ipynb Learning_v1.ipynb

# 実験後
cp Learning.ipynb Learning_experiment_20250115.ipynb
```

## まとめ

`Learning.ipynb` は、バス需要予測の以下のワークフローを実現します：

1. **データ探索**: 基本統計と可視化でデータを理解
2. **前処理**: PyCaret による自動的なデータ前処理
3. **モデル選択**: 複数モデルの自動比較
4. **評価**: 予測精度の評価と分析
5. **実験**: パラメータ調整やアンサンブルの試行

**利点**:
- インタラクティブな分析
- 試行錯誤が容易
- 可視化による直感的理解
- 再現性の確保（セルの順次実行）

**次のステップ**:
- 独自の可視化を追加
- 特徴量エンジニアリングを試す
- 異なるモデルやパラメータで実験
- 結果を `mlm.py` に反映

Notebook を活用して、データの理解を深め、最適な予測モデルを見つけましょう！
