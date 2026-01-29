# 特徴量エンジニアリング

## 概要

`mlm.py` モジュールでは、生成されたバス運行データから機械学習モデル用の特徴量を作成します。適切な特徴量エンジニアリングは、モデルの予測精度を大きく向上させる重要なステップです。

## 特徴量の種類

本プロジェクトでは、以下の5種類の特徴量グループを実装しています。

### 1. 周期的特徴量 (Cyclic Features)
### 2. 時間特徴量 (Time Features)
### 3. 天候特徴量 (Weather Features)
### 4. 相互作用特徴量 (Interaction Features)
### 5. ラグ特徴量 (Lag Features)

---

## 1. 周期的特徴量 (Cyclic Features)

### 目的

時刻や曜日などの周期的な変数を、機械学習モデルが理解しやすい形式に変換します。

### 問題点

通常の数値表現では周期性が失われます：
- 時刻: `23時` と `0時` は隣接しているが、数値では `23` と `0` で遠い
- 曜日: `日曜(6)` と `月曜(0)` は隣接しているが、数値では離れている

### 解決方法：三角関数による変換

周期的な変数を **sin** と **cos** で変換します。

```python
def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    周期的な特徴量を追加する
    """
    # 時刻の周期性（24時間周期）
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 曜日の周期性（7日周期）
    df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # 月の周期性（12ヶ月周期）
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    return df
```

### 数学的背景

```
周期 T の変数 x に対して:
  x_sin = sin(2π * x / T)
  x_cos = cos(2π * x / T)
```

**メリット**:
- 周期の始点と終点が連続的につながる
- 2次元空間（sin, cos）で周期を表現
- 距離が実際の時間的近さを反映

### 例：時刻の変換

| 時刻 | hour_sin | hour_cos |
|-----|----------|----------|
| 0時 | 0.00 | 1.00 |
| 6時 | 1.00 | 0.00 |
| 12時 | 0.00 | -1.00 |
| 18時 | -1.00 | 0.00 |
| 23時 | 0.26 | 0.97 |

**注目**: 23時と0時は（sin, cos）空間で近い位置にあります。

---

## 2. 時間特徴量 (Time Features)

### 目的

日付・時刻情報から、需要パターンに関連する高次の特徴を抽出します。

### 実装

```python
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    時間に関する特徴量を追加する
    """
    # 時間帯区分
    df["time_period"] = df["hour"].apply(get_time_period)

    # 通勤・通学時間帯フラグ
    df["is_commute_time"] = ((df["hour"] >= 7) & (df["hour"] <= 9) |
                             (df["hour"] >= 17) & (df["hour"] <= 19)).astype(int)

    # 週末フラグ
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # 季節
    df["season"] = df["month"].apply(get_season)

    # 月初・月末フラグ
    df["is_month_start"] = (df["date"].dt.day <= 5).astype(int)
    df["is_month_end"] = (df["date"].dt.day >= 25).astype(int)

    return df
```

### 時間帯区分 (Time Period)

```python
def get_time_period(hour: int) -> str:
    """時刻を時間帯に変換"""
    if 6 <= hour < 9:
        return "morning_commute"    # 朝の通勤時間帯
    elif 9 <= hour < 12:
        return "morning"            # 午前
    elif 12 <= hour < 14:
        return "lunch"              # 昼休み
    elif 14 <= hour < 17:
        return "afternoon"          # 午後
    elif 17 <= hour < 20:
        return "evening_commute"    # 夕方の通勤時間帯
    elif 20 <= hour < 23:
        return "evening"            # 夜
    else:
        return "early_morning"      # 早朝
```

### 季節区分

```python
def get_season(month: int) -> str:
    """月を季節に変換"""
    if month in [3, 4, 5]:
        return "spring"   # 春
    elif month in [6, 7, 8]:
        return "summer"   # 夏
    elif month in [9, 10, 11]:
        return "autumn"   # 秋
    else:
        return "winter"   # 冬
```

### 生成される特徴量

| 特徴量 | 型 | 説明 |
|--------|-----|------|
| time_period | categorical | 時間帯区分（7カテゴリ） |
| is_commute_time | binary | 通勤時間帯フラグ |
| is_weekend | binary | 週末フラグ |
| season | categorical | 季節（4カテゴリ） |
| is_month_start | binary | 月初フラグ |
| is_month_end | binary | 月末フラグ |

---

## 3. 天候特徴量 (Weather Features)

### 目的

気温・降水量を、需要に影響を与えるカテゴリに変換します。

### 実装

```python
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    天候に関する特徴量を追加する
    """
    # 降水量カテゴリ
    df["rain_category"] = df["precipitation"].apply(rain_category)

    # 気温カテゴリ
    df["temp_category"] = df["temperature"].apply(temp_category)

    # 悪天候フラグ
    df["is_bad_weather"] = (df["precipitation"] > 5.0).astype(int)

    # 極端な気温フラグ
    df["is_extreme_temp"] = ((df["temperature"] < 5) |
                             (df["temperature"] > 30)).astype(int)

    return df
```

### 降水量カテゴリ

```python
def rain_category(precipitation: float) -> str:
    """降水量をカテゴリに変換"""
    if precipitation == 0:
        return "no_rain"      # 降雨なし
    elif precipitation <= 2:
        return "light_rain"   # 小雨（2mm以下）
    elif precipitation <= 10:
        return "moderate_rain" # 中雨（2-10mm）
    else:
        return "heavy_rain"   # 大雨（10mm超）
```

### 気温カテゴリ

```python
def temp_category(temperature: float) -> str:
    """気温をカテゴリに変換"""
    if temperature < 5:
        return "very_cold"    # 厳冬期（5℃未満）
    elif temperature < 15:
        return "cold"         # 寒い（5-15℃）
    elif temperature < 25:
        return "comfortable"  # 快適（15-25℃）
    elif temperature < 30:
        return "hot"          # 暑い（25-30℃）
    else:
        return "very_hot"     # 猛暑（30℃以上）
```

### 生成される特徴量

| 特徴量 | 型 | 説明 |
|--------|-----|------|
| rain_category | categorical | 降水量カテゴリ（4カテゴリ） |
| temp_category | categorical | 気温カテゴリ（5カテゴリ） |
| is_bad_weather | binary | 悪天候フラグ（5mm超） |
| is_extreme_temp | binary | 極端な気温フラグ |

---

## 4. 相互作用特徴量 (Interaction Features)

### 目的

複数の特徴量を組み合わせることで、単独では表現できないパターンを捉えます。

### 実装

```python
def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    相互作用特徴量を追加する
    """
    # 停留所タイプ × 時間帯
    df["stop_time"] = df["stop_type"] + "_" + df["time_period"]

    # 停留所タイプ × 休日
    df["stop_holiday"] = df["stop_type"] + "_" + \
                         df["is_holiday"].astype(str)

    # 時間帯 × 休日
    df["time_holiday"] = df["time_period"] + "_" + \
                         df["is_holiday"].astype(str)

    # 拠点 × 時間帯
    df["base_time"] = df["is_base_point"].astype(str) + "_" + \
                      df["time_period"]

    # 天候 × 時間帯
    df["weather_time"] = df["rain_category"] + "_" + \
                         df["time_period"]

    return df
```

### 相互作用の例

#### 停留所タイプ × 時間帯

```
"get_on_morning_commute"   # 住宅地の朝 → 乗車多い
"get_off_evening_commute"  # 目的地の夕方 → 乗車多い（帰宅）
"get_on_off_lunch"         # ターミナルの昼 → 適度な需要
```

#### 天候 × 時間帯

```
"moderate_rain_morning_commute"  # 雨の朝 → バス利用増
"no_rain_afternoon"              # 晴れの午後 → 通常通り
```

### なぜ相互作用が重要か

単独の特徴では捉えられないパターン：
- 「住宅地」かつ「朝」 → 特に乗車が多い
- 「雨」かつ「通勤時間」 → バス利用が増加
- 「拠点」かつ「休日」 → 需要減が緩やか

---

## 5. ラグ特徴量 (Lag Features)

### 目的

過去の需要データを特徴量として使用することで、時系列パターンを学習します。

### 実装

```python
def add_lag_features(df: pd.DataFrame, lag_periods: list = [1, 7, 14]) -> pd.DataFrame:
    """
    ラグ特徴量を追加する

    Args:
        lag_periods: ラグ期間のリスト（時間単位）
    """
    # 停留所ごとにグループ化してソート
    df = df.sort_values(["stop_id", "date", "hour"])

    for lag in lag_periods:
        # 各停留所ごとに、lag時間前の乗客数を取得
        df[f"passenger_lag_{lag}"] = df.groupby("stop_id")["passenger_count"].shift(lag)

        # 移動平均
        df[f"passenger_rolling_mean_{lag}"] = (
            df.groupby("stop_id")["passenger_count"]
            .rolling(window=lag, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )

    # 欠損値を0で埋める（初期期間はデータがないため）
    lag_columns = [col for col in df.columns if "lag" in col or "rolling" in col]
    df[lag_columns] = df[lag_columns].fillna(0)

    return df
```

### ラグ期間の選択

```python
lag_periods = [1, 7, 14]
```

- **lag_1**: 1時間前の需要（直近の傾向）
- **lag_7**: 7時間前の需要（1日内のパターン）
- **lag_14**: 14時間前の需要（前日の同時間帯）

### 生成される特徴量

| 特徴量 | 説明 |
|--------|------|
| passenger_lag_1 | 1時間前の乗客数 |
| passenger_lag_7 | 7時間前の乗客数 |
| passenger_lag_14 | 14時間前の乗客数 |
| passenger_rolling_mean_1 | 過去1時間の移動平均 |
| passenger_rolling_mean_7 | 過去7時間の移動平均 |
| passenger_rolling_mean_14 | 過去14時間の移動平均 |

### 注意点

- ラグ特徴量は、予測時に未来のデータを使わないよう注意が必要
- 初期期間はデータがないため、欠損値を適切に処理（0埋め）

---

## カテゴリカル変数のエンコーディング

### 目的

カテゴリカル変数を数値に変換し、機械学習モデルで使用できるようにします。

### 実装

```python
def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    カテゴリカル変数をエンコードする
    """
    # Label Encoding（順序がある、またはカテゴリ数が少ない場合）
    label_encode_cols = ["stop_type", "time_period", "season",
                         "rain_category", "temp_category"]

    for col in label_encode_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes

    # One-Hot Encoding（相互作用特徴量など、カテゴリ数が多い場合）
    onehot_cols = [col for col in df.columns if "_" in col and
                   df[col].dtype == "object"]

    if onehot_cols:
        df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    return df
```

### エンコード方法

#### Label Encoding

カテゴリを整数に変換：
```
"spring" → 0
"summer" → 1
"autumn" → 2
"winter" → 3
```

#### One-Hot Encoding

カテゴリごとにバイナリカラムを作成：
```
stop_time = "get_on_morning_commute"
↓
stop_time_get_on_morning_commute = 1
stop_time_get_off_morning_commute = 0
stop_time_get_on_off_morning_commute = 0
...
```

---

## 特徴量作成のパイプライン

全ての特徴量作成を統合した関数：

```python
def create_features(df: pd.DataFrame, include_lag: bool = True) -> pd.DataFrame:
    """
    全ての特徴量を作成する

    Args:
        df: 元データ
        include_lag: ラグ特徴量を含めるかどうか

    Returns:
        特徴量を追加したDataFrame
    """
    # 1. 周期的特徴量
    df = add_cyclic_features(df)

    # 2. 時間特徴量
    df = add_time_features(df)

    # 3. 天候特徴量
    df = add_weather_features(df)

    # 4. 相互作用特徴量
    df = add_interaction_features(df)

    # 5. ラグ特徴量（オプション）
    if include_lag:
        df = add_lag_features(df)

    # 6. カテゴリカルエンコーディング
    df = encode_categorical_features(df)

    return df
```

---

## 特徴量の重要性

機械学習モデル学習後、どの特徴量が予測に重要かを分析できます。

```python
def plot_feature_importance(model, features: list, top_n: int = 20):
    """
    特徴量重要度を可視化する
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importance[indices])
    plt.yticks(range(top_n), [features[i] for i in indices])
    plt.xlabel("Feature Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig("./figures/feature_importance.png")
```

### 典型的な重要特徴量（例）

1. `hour_sin`, `hour_cos` - 時刻の周期性
2. `is_commute_time` - 通勤時間帯フラグ
3. `is_base_point` - 拠点フラグ
4. `passenger_lag_1` - 直近の需要
5. `stop_type` - 停留所タイプ
6. `is_holiday` - 休日フラグ
7. `precipitation` - 降水量

---

## まとめ

本プロジェクトの特徴量エンジニアリングの特徴：

1. **周期性の適切な表現**: sin/cos変換により周期的パターンを学習可能に
2. **ドメイン知識の活用**: 通勤時間、季節など、バス需要に関連する特徴を抽出
3. **相互作用の捉え**: 複数要因の組み合わせによる複雑なパターンを表現
4. **時系列の考慮**: ラグ特徴量により過去の需要傾向を活用
5. **適切なエンコーディング**: カテゴリカル変数を適切に数値化

これらの特徴量により、過疎地バスの複雑な需要パターンを効果的に学習できます。
