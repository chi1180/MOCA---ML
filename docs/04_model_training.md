# モデル学習と評価

## 概要

`mlm.py` モジュールには、機械学習モデルの学習、評価、可視化に関する機能が実装されています。PyCaret を使用した自動機械学習（AutoML）により、複数のモデルを効率的に比較・選択できます。

## 学習プロセスの全体像

```
1. データ読み込み・前処理
   ↓
2. 特徴量エンジニアリング
   ↓
3. 時系列分割（Train/Validation/Test）
   ↓
4. モデル学習（PyCaret）
   ↓
5. モデル評価
   ↓
6. 予測分析・可視化
```

---

## 1. データ読み込み・前処理

### データの読み込み

```python
def load_and_preprocess_data(filepath: str = "./bus_operation_data.csv") -> pd.DataFrame:
    """
    CSVファイルからデータを読み込み、前処理を行う

    Args:
        filepath: データファイルのパス

    Returns:
        前処理済みのDataFrame
    """
    logging("info", f"データを読み込み中: {filepath}")
    df = pd.read_csv(filepath)

    # 日付型に変換
    df["date"] = pd.to_datetime(df["date"])

    # 基本的な統計情報
    logging("info", f"データサイズ: {df.shape}")
    logging("info", f"期間: {df['date'].min()} ~ {df['date'].max()}")
    logging("info", f"停留所数: {df['stop_id'].nunique()}")

    return df
```

### データの基本情報確認

```python
print(df.info())
print(df.describe())
print(df['passenger_count'].value_counts().head(10))
```

---

## 2. 時系列分割

### なぜ時系列分割が必要か

時系列データでは、**未来のデータで学習し、過去を予測する**ことを防ぐ必要があります。

**間違った分割（ランダム分割）**:
- 2023年12月のデータで学習 → 2023年1月を予測
- 現実ではあり得ない状況

**正しい分割（時系列分割）**:
- 2023年1-8月で学習 → 2023年9-10月を検証 → 2023年11-12月をテスト

### 実装

```python
def split_data_temporal(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    時系列を考慮してデータを分割する

    Args:
        df: 元データ
        train_ratio: 訓練データの割合
        val_ratio: 検証データの割合
        test_ratio: テストデータの割合

    Returns:
        (train_df, val_df, test_df) のタプル
    """
    # 日付でソート
    df = df.sort_values("date").reset_index(drop=True)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logging("info", "=" * 60)
    logging("info", "データ分割完了")
    logging("info", f"訓練データ: {len(train_df):,} 件 ({train_df['date'].min()} ~ {train_df['date'].max()})")
    logging("info", f"検証データ: {len(val_df):,} 件 ({val_df['date'].min()} ~ {val_df['date'].max()})")
    logging("info", f"テストデータ: {len(test_df):,} 件 ({test_df['date'].min()} ~ {test_df['date'].max()})")
    logging("info", "=" * 60)

    return train_df, val_df, test_df
```

### 分割例

```
期間: 2023-01-01 ~ 2023-12-31 (365日)

訓練データ: 2023-01-01 ~ 2023-09-12 (70%, 約256日)
検証データ: 2023-09-13 ~ 2023-11-06 (15%, 約55日)
テストデータ: 2023-11-07 ~ 2023-12-31 (15%, 約54日)
```

---

## 3. モデル学習（PyCaret）

### PyCaretとは

**PyCaret** は、機械学習ワークフローを自動化するPythonライブラリです。

**主な特徴**:
- 複数のモデルを自動的に比較
- ハイパーパラメータの自動調整
- 特徴量の自動前処理
- モデルのアンサンブル

### セットアップ

```python
from pycaret.regression import *

def train_model(train_df: pd.DataFrame, target: str = "passenger_count"):
    """
    PyCaretを使用してモデルを学習する

    Args:
        train_df: 訓練データ
        target: 目的変数のカラム名

    Returns:
        学習済みモデル
    """
    logging("info", "=" * 60)
    logging("info", "PyCaretセットアップ開始")
    logging("info", "=" * 60)

    # セットアップ
    exp = setup(
        data=train_df,
        target=target,
        session_id=42,              # 乱数シード
        verbose=False,              # 詳細出力を抑制
        normalize=True,             # 特徴量の正規化
        transformation=False,       # 目的変数の変換なし
        ignore_features=['date', 'stop_id', 'stop_name'],  # 無視する列
        fold_strategy='timeseries', # 時系列分割
        fold=5                      # 交差検証の分割数
    )

    logging("success", "セットアップ完了")

    return exp
```

### モデルの比較

```python
# 利用可能なモデルをすべて比較
best_models = compare_models(
    n_select=5,           # 上位5つを選択
    sort='MAE',           # MAE（平均絶対誤差）で評価
    include=[
        'lr',             # Linear Regression
        'ridge',          # Ridge Regression
        'lasso',          # Lasso Regression
        'et',             # Extra Trees
        'rf',             # Random Forest
        'gbr',            # Gradient Boosting
        'xgboost',        # XGBoost
        'lightgbm',       # LightGBM
        'catboost'        # CatBoost
    ]
)
```

### モデルの評価指標

| 指標 | 説明 | 式 |
|-----|------|-----|
| **MAE** | 平均絶対誤差 | `mean(\|y - ŷ\|)` |
| **RMSE** | 二乗平均平方根誤差 | `sqrt(mean((y - ŷ)²))` |
| **R²** | 決定係数 | `1 - SS_res/SS_tot` |
| **MAPE** | 平均絶対パーセント誤差 | `mean(\|y - ŷ\|/y) × 100` |

**過疎地バス需要予測では MAE を優先**:
- 乗客数は小さい値が多い（0-10人程度）
- RMSE は外れ値（まれな混雑時）に敏感
- MAE は平均的な誤差を直感的に理解しやすい

### 最良モデルの選択

```python
# 最良モデルでハイパーパラメータ調整
tuned_model = tune_model(
    best_models[0],
    optimize='MAE',
    n_iter=50          # 50回の試行
)

# モデルをファイナライズ（全データで再学習）
final_model = finalize_model(tuned_model)
```

---

## 4. モデル評価

### 基本的な評価

```python
def evaluate_model(model, test_df: pd.DataFrame, target: str = "passenger_count"):
    """
    モデルを評価する

    Args:
        model: 学習済みモデル
        test_df: テストデータ
        target: 目的変数のカラム名
    """
    # 予測
    predictions = predict_model(model, data=test_df)

    # 実際の値と予測値
    y_true = predictions[target]
    y_pred = predictions['prediction_label']

    # 評価指標の計算
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    logging("info", "=" * 60)
    logging("info", "モデル評価結果")
    logging("info", "=" * 60)
    logging("info", f"MAE:  {mae:.3f}")
    logging("info", f"RMSE: {rmse:.3f}")
    logging("info", f"R²:   {r2:.3f}")
    logging("info", "=" * 60)

    return predictions
```

### セグメント別評価

需要パターンが異なるグループごとに評価します。

```python
def evaluate_by_segment(predictions: pd.DataFrame):
    """
    セグメント別にモデルを評価する
    """
    segments = {
        "停留所タイプ": "stop_type",
        "時間帯": "time_period",
        "休日": "is_holiday",
        "拠点": "is_base_point"
    }

    for seg_name, seg_col in segments.items():
        logging("info", f"\n【{seg_name}別評価】")

        for seg_value in predictions[seg_col].unique():
            seg_data = predictions[predictions[seg_col] == seg_value]

            mae = mean_absolute_error(
                seg_data["passenger_count"],
                seg_data["prediction_label"]
            )

            logging("info", f"  {seg_value}: MAE = {mae:.3f}")
```

### 評価結果の例

```
【停留所タイプ別評価】
  get_on: MAE = 0.856
  get_off: MAE = 0.742
  get_on_off: MAE = 1.124

【時間帯別評価】
  morning_commute: MAE = 1.235
  afternoon: MAE = 0.623
  evening_commute: MAE = 1.198

【休日別評価】
  0（平日）: MAE = 0.923
  1（休日）: MAE = 0.487

【拠点別評価】
  0（通常）: MAE = 0.712
  1（拠点）: MAE = 1.345
```

**分析**:
- 拠点は需要が多いため絶対誤差も大きい
- 通勤時間帯は予測が難しい（変動が大きい）
- 休日は需要が少なく安定しているため誤差が小さい

---

## 5. 交差検証

### 時系列交差検証

```python
def cross_validate_timeseries(df: pd.DataFrame, n_splits: int = 5):
    """
    時系列交差検証を実行する

    Args:
        df: 全データ
        n_splits: 分割数
    """
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)

    mae_scores = []
    rmse_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        logging("info", f"\n=== Fold {fold}/{n_splits} ===")

        train_data = df.iloc[train_idx]
        val_data = df.iloc[val_idx]

        # セットアップ・学習
        exp = setup(
            data=train_data,
            target='passenger_count',
            session_id=42,
            verbose=False
        )

        model = create_model('lightgbm', verbose=False)

        # 評価
        predictions = predict_model(model, data=val_data)
        mae = mean_absolute_error(
            predictions['passenger_count'],
            predictions['prediction_label']
        )

        mae_scores.append(mae)
        logging("info", f"Fold {fold} MAE: {mae:.3f}")

    logging("info", "\n=== 交差検証結果 ===")
    logging("info", f"平均MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
```

### TimeSeriesSplit の仕組み

```
データ: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Fold 1: Train [1, 2]      → Validate [3, 4]
Fold 2: Train [1, 2, 3, 4] → Validate [5, 6]
Fold 3: Train [1-6]        → Validate [7, 8]
Fold 4: Train [1-8]        → Validate [9, 10]
```

---

## 6. 予測分析・可視化

### 予測値 vs 実測値のプロット

```python
def plot_prediction_analysis(predictions: pd.DataFrame):
    """
    予測結果を可視化する
    """
    y_true = predictions['passenger_count']
    y_pred = predictions['prediction_label']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. 散布図
    axes[0, 0].scatter(y_true, y_pred, alpha=0.3, s=10)
    axes[0, 0].plot([0, y_true.max()], [0, y_true.max()], 'r--')
    axes[0, 0].set_xlabel("Actual")
    axes[0, 0].set_ylabel("Predicted")
    axes[0, 0].set_title("Predicted vs Actual")

    # 2. 残差プロット
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.3, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("Residuals")
    axes[0, 1].set_title("Residual Plot")

    # 3. 残差のヒストグラム
    axes[1, 0].hist(residuals, bins=50, edgecolor='black')
    axes[1, 0].set_xlabel("Residuals")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Residual Distribution")

    # 4. 時系列プロット（最初の100件）
    sample = predictions.head(100)
    axes[1, 1].plot(sample.index, sample['passenger_count'], label='Actual')
    axes[1, 1].plot(sample.index, sample['prediction_label'], label='Predicted')
    axes[1, 1].set_xlabel("Sample Index")
    axes[1, 1].set_ylabel("Passenger Count")
    axes[1, 1].set_title("Time Series Comparison")
    axes[1, 1].legend()

    plt.tight_layout()
    save_figure("prediction_analysis.png")
```

### 特徴量重要度

```python
def plot_feature_importance(model, feature_names: list, top_n: int = 20):
    """
    特徴量重要度を可視化する
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices], color='steelblue')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    save_figure("feature_importance.png")
```

---

## 7. モデルの保存と読み込み

### モデルの保存

```python
def save_model(model, filepath: str = "./bus_demand_model.pkl"):
    """
    モデルをファイルに保存する
    """
    save_model(model, filepath)
    logging("success", f"モデルを保存しました: {filepath}")
```

### モデルの読み込み

```python
def load_model(filepath: str = "./bus_demand_model.pkl"):
    """
    モデルをファイルから読み込む
    """
    model = load_model(filepath)
    logging("success", f"モデルを読み込みました: {filepath}")
    return model
```

---

## 8. 実行例（main関数）

```python
def main():
    """
    メイン処理
    """
    # 1. データ読み込み
    df = load_and_preprocess_data("./bus_operation_data.csv")

    # 2. 特徴量作成
    df = create_features(df, include_lag=True)

    # 3. データ分割
    train_df, val_df, test_df = split_data_temporal(df)

    # 4. モデル学習
    exp = train_model(train_df)
    best_models = compare_models(n_select=3, sort='MAE')
    tuned_model = tune_model(best_models[0], optimize='MAE')
    final_model = finalize_model(tuned_model)

    # 5. モデル評価
    predictions = evaluate_model(final_model, test_df)
    evaluate_by_segment(predictions)

    # 6. 可視化
    plot_prediction_analysis(predictions)
    plot_feature_importance(final_model, feature_names)

    # 7. モデル保存
    save_model(final_model, "./bus_demand_model.pkl")

    logging("success", "全ての処理が完了しました！")

if __name__ == "__main__":
    main()
```

---

## まとめ

### 本プロジェクトの機械学習パイプラインの特徴

1. **時系列を考慮した分割**: 未来のデータでリークしない適切な評価
2. **AutoML の活用**: PyCaret による効率的なモデル選択
3. **包括的な評価**: 全体評価＋セグメント別評価
4. **可視化による検証**: 予測精度を視覚的に確認
5. **再現性の確保**: 乱数シード固定、モデル保存

### 期待される性能

過疎地バス需要予測での典型的な性能：
- **MAE**: 0.8 〜 1.2人
- **R²**: 0.65 〜 0.75

### 今後の改善案

1. **深層学習の導入**: LSTM, Transformerによる時系列モデリング
2. **外部データの統合**: 天候API、イベント情報など
3. **オンライン学習**: 新しいデータでモデルを継続的に更新
4. **確率的予測**: 点推定だけでなく予測区間も提供
