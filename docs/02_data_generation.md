# ダミーデータ生成の詳細

## 概要

`data_generator.py` は、過疎地域のバス運行データをリアルに模擬したダミーデータを生成するモジュールです。実際の過疎地におけるバス利用パターンを再現するため、複数の要因を組み合わせた需要計算ロジックを実装しています。

## データ生成の流れ

### 1. 入力データ

```python
stop_df: pd.DataFrame  # 停留所情報
  - id: 停留所ID
  - name: 停留所名
  - stop_type: 停留所タイプ (get_on, get_off, get_on_off)
  - latitude, longitude: 緯度経度
  - is_base_point: 拠点フラグ
```

### 2. 生成期間の設定

- **デフォルト期間**: 2023年1月1日 〜 2023年12月31日（365日間）
- **営業時間**: 6時 〜 22時（17時間）
- **データ粒度**: 1時間ごと

### 3. 生成されるレコード数

```
総レコード数 = 日数 × 営業時間 × 停留所数
例: 365日 × 17時間 × 10停留所 = 62,050レコード
```

## 需要計算ロジック

乗客数は **ポアソン分布** を使用して生成されます。ポアソン分布のパラメータλ（平均乗客数）は、以下の4つの要因の積で決定されます。

### 基本式

```
λ = 基礎需要 × 時間帯係数 × 休日係数 × 気象係数

乗客数 = Poisson(λ)
```

## 各係数の詳細

### 1. 基礎需要 (Base Demand)

停留所の重要度に基づく基本的な需要値です。

```python
def calculate_base_demand(is_base_point: bool) -> float:
    if is_base_point:
        return 3.0  # 拠点（役場、駅など）
    else:
        return 1.5  # 通常の停留所
```

**特徴**:
- 拠点は通常の停留所の2倍の基礎需要
- 拠点の例: 役場、駅、病院、商業施設など

### 2. 時間帯係数 (Time Factor)

時刻と停留所タイプの組み合わせで需要が変動します。

#### get_on（乗車が多い停留所：住宅地など）

```python
7-8時  → 2.5  # 朝の通勤・通学ラッシュ（乗車ピーク）
9-16時 → 0.8  # 日中は静か
17-19時 → 1.5 # 夕方の帰宅（降車が多いが乗車もある）
その他 → 0.5  # 早朝・夜間
```

**解説**: 住宅地から目的地（職場・学校）へ向かう朝に乗車が集中。

#### get_off（降車が多い停留所：目的地など）

```python
8-9時  → 2.0  # 朝の到着（降車ピーク）
10-16時 → 1.0 # 日中は普通
17-19時 → 2.5 # 夕方の帰宅開始（乗車ピーク）
その他 → 0.5  # 早朝・夜間
```

**解説**: 目的地（職場・学校）へ朝に到着、夕方に乗車して帰宅。

#### get_on_off（乗降両方：ターミナルなど）

```python
7-9時   → 2.5  # 朝のピーク
12-14時 → 1.5  # 昼休みの移動
17-19時 → 2.5  # 夕方のピーク
9-16時  → 1.2  # 日中も一定の需要
その他  → 0.5  # 早朝・夜間
```

**解説**: ターミナル（駅、バスセンターなど）は両方向の需要がある。

### 3. 休日係数 (Holiday Factor)

土日祝日の需要変動を表現します。

```python
平日の場合: 1.0（変化なし）

休日の場合:
  get_on     → 0.4  # 通勤・通学がないため大幅減
  get_off    → 0.5  # 目的地への需要減
  get_on_off → 0.6  # ターミナルは若干マシ
```

**特徴**:
- 過疎地では休日の利用が大幅に減少
- 通勤・通学が主な利用目的のため

### 4. 気象係数 (Weather Factor)

降水量に応じた需要変動を表現します。

```python
降水量 ≤ 2.0mm: 1.0（変化なし）

降水量 > 2.0mm（雨天時）:
  get_on_off → 1.3  # ターミナルはバス利用増
  get_on     → 1.1  # 雨でバス利用微増
  get_off    → 0.9  # 外出控えで減少
```

**解説**:
- 雨天時は自転車・徒歩からバスへの転換がある
- 一方で外出そのものを控える傾向もある

## 気象データ生成

### 気温データ

季節に応じた気温を正規分布で生成します。

```python
def get_dummy_weather(date: datetime.date, rng: np.random.Generator):
    month = date.month

    # 月別の基準気温
    if month <= 8:
        base_temp = 5 + (month - 1) * 2  # 1月: 5℃ → 8月: 19℃
    else:
        base_temp = 30 - (month - 8) * 4  # 9月: 30℃ → 12月: 14℃

    # 正規分布で変動（標準偏差3℃）
    temp = rng.normal(base_temp, 3)
```

### 降水量データ

季節性と確率的な降雨を表現します。

```python
    # 梅雨・秋雨シーズン
    is_rainy_season = month in [6, 7, 9]
    rain_prob = 0.4 if is_rainy_season else 0.15

    precipitation = 0.0
    if rng.random() < rain_prob:
        # 指数分布で降水量を生成（平均8mm）
        precipitation = rng.exponential(8)
```

**特徴**:
- 6-7月（梅雨）、9月（秋雨）は降雨確率40%
- その他の月は15%
- 降水量は指数分布（小雨が多く、大雨は少ない）

## ゼロ膨張の実装

過疎地の特徴として「乗客がいない時間帯」が多いことを再現します。

```python
# 拠点以外の停留所では、20%の確率で強制的に乗客数0
if not is_base_point and random.random() < 0.2:
    passenger_count = 0
else:
    passenger_count = rng.poisson(lambda_val)
```

**効果**:
- 実際の過疎地バスデータに近い「ゼロが多い」分布を実現
- 機械学習モデルがゼロ膨張を学習できる

## 乱数シードと再現性

```python
RANDOM_SEED = 42

rng = np.random.default_rng(seed)
random.seed(seed)
```

**重要**: 同じシード値を使用することで、毎回同じデータが生成されます。実験の再現性を確保するために重要です。

## データ品質検証

生成後、自動的にデータ品質をチェックします。

### 検証項目

1. **平均乗客数**: 0.5 〜 20人の範囲内か
2. **時間帯パターン**: 朝・夕方ピークが昼間より高いか
3. **曜日パターン**: 平日が休日より多いか（過疎地の特性）
4. **ゼロ膨張**: ゼロの割合が1% 〜 60%の範囲内か

### 実装例

```python
def validate_data_quality(df: pd.DataFrame) -> bool:
    issues = []

    # 平均乗客数チェック
    mean_passengers = df['passenger_count'].mean()
    if mean_passengers < 0.5 or mean_passengers > 20:
        issues.append(f"平均乗客数が異常: {mean_passengers:.2f}")

    # ピークパターンチェック
    hourly_mean = df.groupby('hour')['passenger_count'].mean()
    morning_peak = hourly_mean.loc[[7, 8]].mean()
    evening_peak = hourly_mean.loc[[17, 18]].mean()
    midday = hourly_mean.loc[[12, 13]].mean()

    if morning_peak < midday or evening_peak < midday:
        issues.append("ピークパターンが不自然")

    return len(issues) == 0
```

## 具体例：1レコードの生成過程

**条件**:
- 日付: 2023年7月15日（土曜日）
- 時刻: 8時
- 停留所: "駅前" (get_on_off, is_base_point=True)
- 気温: 25.3℃
- 降水量: 5.2mm（雨）

**計算**:
```
基礎需要 = 3.0 (拠点)
時間帯係数 = 2.5 (朝ピーク、ターミナル)
休日係数 = 0.6 (休日)
気象係数 = 1.3 (雨、ターミナル)

λ = 3.0 × 2.5 × 0.6 × 1.3 = 5.85

乗客数 = Poisson(5.85) → 例えば 6人
```

## データ型の最適化

メモリ効率化のため、適切なデータ型を使用します。

```python
df['hour'] = df['hour'].astype('int8')           # 6-22時
df['month'] = df['month'].astype('int8')         # 1-12月
df['day_of_week'] = df['day_of_week'].astype('int8')  # 0-6
df['is_holiday'] = df['is_holiday'].astype('int8')    # 0 or 1
df['passenger_count'] = df['passenger_count'].astype('int16')  # 0-32767
df['temperature'] = df['temperature'].astype('float32')
df['precipitation'] = df['precipitation'].astype('float32')
```

## 出力データの構造

生成されるCSVファイルには以下のカラムが含まれます。

| カラム名 | 型 | 説明 |
|---------|-----|------|
| date | date | 日付 |
| hour | int8 | 時刻（6-22） |
| stop_id | string | 停留所ID（UUID） |
| stop_name | string | 停留所名 |
| stop_type | string | 停留所タイプ（get_on/get_off/get_on_off） |
| latitude | float64 | 緯度 |
| longitude | float64 | 経度 |
| is_base_point | int8 | 拠点フラグ（0 or 1） |
| month | int8 | 月（1-12） |
| day_of_week | int8 | 曜日（0=月曜 〜 6=日曜） |
| is_holiday | int8 | 休日フラグ（0 or 1） |
| temperature | float32 | 気温（℃） |
| precipitation | float32 | 降水量（mm） |
| passenger_count | int16 | 乗客数（予測対象） |

## サマリー統計の例

実際に生成されたデータの統計情報：

```
総レコード数: 62,050
期間: 2023-01-01 ~ 2023-12-31
停留所数: 10

乗客数統計:
  平均: 1.85
  中央値: 1.0
  最大: 15
  最小: 0
  標準偏差: 2.13
  ゼロの割合: 35.2%

時間帯別平均乗客数:
  7時: 2.85
  8時: 3.12
  12時: 1.45
  17時: 2.94
  18時: 3.01

曜日別平均乗客数:
  月曜日: 2.15
  火曜日: 2.18
  ...
  土曜日: 1.12
  日曜日: 1.05
```

## まとめ

本ダミーデータ生成システムの特徴：

1. **現実的な需要パターン**: 4つの係数を掛け合わせた多面的な需要モデル
2. **季節性の再現**: 気温・降水量の季節変動
3. **過疎地特有のパターン**: ゼロ膨張、休日の大幅減少
4. **再現性の確保**: 乱数シードによる同一データの再生成
5. **品質保証**: 自動検証による異常値チェック

これにより、機械学習モデルが実際の過疎地バスデータに近い特性を学習できます。
