########################
# data_generator.py - Bus Operation Data Generator (Simplified)
# シンプルなバス運行データ生成システム
########################

import datetime
import random
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

########################
# Configurations
########################

# 日付範囲
DEFAULT_START_DATE = datetime.date(2023, 1, 1)
DEFAULT_END_DATE = datetime.date(2023, 12, 31)

# 営業時間
OPERATING_HOURS_START = 6
OPERATING_HOURS_END = 22

# 乱数シード（再現性のため）
RANDOM_SEED = 42


########################
# Weather Functions
########################


def get_dummy_weather(
    date: datetime.date, rng: np.random.Generator
) -> Tuple[float, float]:
    """
    季節に応じた気象データを生成

    Args:
        date: 日付
        rng: 乱数生成器

    Returns:
        (気温, 降水量) のタプル
    """
    month = date.month

    # 季節に応じた基準気温
    if month <= 8:
        base_temp = 5 + (month - 1) * 2
    else:
        base_temp = 30 - (month - 8) * 4

    temp = rng.normal(base_temp, 3)

    # 6-7月、9月は雨が多い（梅雨・秋雨）
    is_rainy_season = month in [6, 7, 9]
    rain_prob = 0.4 if is_rainy_season else 0.15

    precipitation = 0.0
    if rng.random() < rain_prob:
        precipitation = rng.exponential(8)

    return round(temp, 1), round(precipitation, 1)


########################
# Demand Calculation
########################


def calculate_time_factor(stop_type: str, hour: int) -> float:
    """
    停留所タイプと時間帯に応じた需要係数を計算

    Args:
        stop_type: 停留所タイプ
        hour: 時刻

    Returns:
        時間帯係数
    """
    # 基本は静か
    time_factor = 0.5

    # get_on: 乗車が多い（住宅地など）→ 朝にピーク
    # get_off: 降車が多い（目的地など）→ 朝に到着、夕方に帰宅
    # get_on_off: 乗降両方（ターミナルなど）→ 両方のピーク

    if stop_type == "get_on":
        # 朝の通勤・通学ラッシュ（乗車）
        if 7 <= hour <= 8:
            time_factor = 2.5
        # 夕方の帰宅（降車が多いが、乗車もある）
        elif 17 <= hour <= 19:
            time_factor = 1.5
        elif 9 <= hour <= 16:
            time_factor = 0.8

    elif stop_type == "get_off":
        # 朝の到着（降車）
        if 8 <= hour <= 9:
            time_factor = 2.0
        # 夕方の帰宅開始（乗車）
        elif 17 <= hour <= 19:
            time_factor = 2.5
        elif 10 <= hour <= 16:
            time_factor = 1.0

    elif stop_type == "get_on_off":
        # ターミナル的な場所：両方のピークがある
        if 7 <= hour <= 9:
            time_factor = 2.5
        elif 17 <= hour <= 19:
            time_factor = 2.5
        elif 12 <= hour <= 14:
            time_factor = 1.5
        elif 9 <= hour <= 16:
            time_factor = 1.2

    return time_factor


def calculate_holiday_factor(stop_type: str, is_holiday: bool) -> float:
    """
    休日特性に応じた需要係数を計算

    Args:
        stop_type: 停留所タイプ
        is_holiday: 休日かどうか

    Returns:
        休日係数
    """
    if not is_holiday:
        return 1.0

    # 休日は全体的に需要が減る（過疎地のバス）
    if stop_type == "get_on":
        return 0.4  # 通勤・通学がないので大幅減
    elif stop_type == "get_off":
        return 0.5  # 目的地への需要減
    elif stop_type == "get_on_off":
        return 0.6  # ターミナルは少しマシ

    return 0.5


def calculate_weather_factor(stop_type: str, precipitation: float) -> float:
    """
    気象特性に応じた需要係数を計算

    Args:
        stop_type: 停留所タイプ
        precipitation: 降水量

    Returns:
        気象係数
    """
    if precipitation <= 2.0:
        return 1.0

    # 雨の日：自転車・徒歩からバスへの転換もあるが、外出控えも
    if stop_type == "get_on_off":
        return 1.3  # ターミナルはバス利用増
    elif stop_type == "get_on":
        return 1.1  # 雨でバス利用増
    elif stop_type == "get_off":
        return 0.9  # 目的地への外出減

    return 1.0


def calculate_base_demand(is_base_point: bool) -> float:
    """
    基礎需要を計算（拠点かどうかで変わる）

    Args:
        is_base_point: 拠点かどうか

    Returns:
        基礎需要値
    """
    if is_base_point:
        return 3.0  # 拠点は需要が高い
    else:
        return 1.5  # 通常の停留所


########################
# Main Data Generation
########################


def generate_bus_operation_data(
    stop_df: pd.DataFrame,
    start_date: datetime.date = DEFAULT_START_DATE,
    end_date: datetime.date = DEFAULT_END_DATE,
    logging_func: Callable[[str, str], None] | None = None,
    include_lag_features: bool = False,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    バス運行の模擬データを生成する

    Args:
        stop_df: 停留所データのDataFrame
        start_date: データ生成開始日
        end_date: データ生成終了日
        logging_func: ログ出力用関数
        include_lag_features: ラグ特徴量を含めるかどうか（未使用、互換性のため残す）
        seed: 乱数シード

    Returns:
        生成されたバス運行データのDataFrame
    """
    # 乱数ジェネレータの初期化
    rng = np.random.default_rng(seed)
    random.seed(seed)

    def log(log_type: str, content: str):
        if logging_func:
            logging_func(log_type, content)
        else:
            print(f"[{log_type.upper()}] {content}")

    log("info", "=" * 60)
    log("info", "バス運行データ生成を開始します...")
    log("info", "=" * 60)

    days = (end_date - start_date).days + 1
    total_hours = OPERATING_HOURS_END - OPERATING_HOURS_START + 1
    total_records_estimate = days * total_hours * len(stop_df)

    log("info", f"期間: {start_date} ~ {end_date} ({days}日間)")
    log("info", f"停留所数: {len(stop_df)}")
    log("info", f"推定レコード数: {total_records_estimate:,}")
    log("info", "-" * 60)

    records = []
    weather_cache: Dict[datetime.date, Tuple[float, float]] = {}

    log("info", "データレコードを生成中...")

    for i in range(days):
        current_date = start_date + datetime.timedelta(days=i)
        month = current_date.month
        weekday = current_date.weekday()

        # 休日判定（土日）
        is_holiday = weekday >= 5

        # その日の気象データ（キャッシュ）
        if current_date not in weather_cache:
            weather_cache[current_date] = get_dummy_weather(current_date, rng)
        temp, precip = weather_cache[current_date]

        # 進捗表示
        if i % 30 == 0:
            progress = (i / days) * 100
            log("info", f"進捗: {progress:.1f}% ({current_date})")

        # 営業時間内の各時間帯
        for hour in range(OPERATING_HOURS_START, OPERATING_HOURS_END + 1):
            # 全停留所について計算
            for _, stop in stop_df.iterrows():
                stop_id = stop["id"]
                stop_type = stop["stop_type"]
                is_base_point = stop["is_base_point"]

                # --- 需要計算ロジック ---

                # 1. 基礎需要
                base_val = calculate_base_demand(is_base_point)

                # 2. 時間帯特性
                time_factor = calculate_time_factor(stop_type, hour)

                # 3. 休日特性
                holiday_factor = calculate_holiday_factor(stop_type, is_holiday)

                # 4. 気象特性
                weather_factor = calculate_weather_factor(stop_type, precip)

                # 最終的なλ（ポアソン分布の平均値）
                lambda_val = base_val * time_factor * holiday_factor * weather_factor

                # 乗客数を生成（ポアソン分布）
                # 過疎地らしさを出すため、一定確率で0人にする
                if not is_base_point and random.random() < 0.2:
                    passenger_count = 0
                else:
                    passenger_count = rng.poisson(lambda_val)

                records.append(
                    {
                        "date": current_date,
                        "hour": hour,
                        "stop_id": stop_id,
                        "stop_name": stop["name"],
                        "stop_type": stop_type,
                        "latitude": stop["latitude"],
                        "longitude": stop["longitude"],
                        "is_base_point": int(is_base_point),
                        "month": month,
                        "day_of_week": weekday,
                        "is_holiday": int(is_holiday),
                        "temperature": temp,
                        "precipitation": precip,
                        "passenger_count": passenger_count,
                    }
                )

    log("success", f"データ生成完了！ 合計 {len(records):,} 件")

    # DataFrameに変換
    df = pd.DataFrame(records)

    # データ型を最適化
    df["hour"] = df["hour"].astype("int8")
    df["month"] = df["month"].astype("int8")
    df["day_of_week"] = df["day_of_week"].astype("int8")
    df["is_holiday"] = df["is_holiday"].astype("int8")
    df["is_base_point"] = df["is_base_point"].astype("int8")
    df["passenger_count"] = df["passenger_count"].astype("int16")
    df["temperature"] = df["temperature"].astype("float32")
    df["precipitation"] = df["precipitation"].astype("float32")

    return df


########################
# Save to CSV
########################


def save_to_csv(df: pd.DataFrame, filepath: str, logging_func: Callable | None = None):
    """
    DataFrameをCSVファイルに保存する

    Args:
        df: 保存するDataFrame
        filepath: 保存先のファイルパス
        logging_func: ログ出力用関数
    """

    def log(log_type: str, content: str):
        if logging_func:
            logging_func(log_type, content)
        else:
            print(f"[{log_type.upper()}] {content}")

    log("info", f"CSVファイルに保存中: {filepath}")
    df.to_csv(filepath, index=False, encoding="utf-8")
    log("success", f"保存完了: {filepath} ({len(df):,} 件)")


########################
# Summary Statistics
########################


def print_data_summary(df: pd.DataFrame, logging_func: Callable | None = None):
    """
    生成されたデータの統計サマリーを出力する

    Args:
        df: データのDataFrame
        logging_func: ログ出力用関数
    """

    def log(log_type: str, content: str):
        if logging_func:
            logging_func(log_type, content)
        else:
            print(f"[{log_type.upper()}] {content}")

    log("info", "=" * 60)
    log("info", "データサマリー")
    log("info", "=" * 60)
    log("info", f"総レコード数: {len(df):,}")
    log("info", f"期間: {df['date'].min()} ~ {df['date'].max()}")
    log("info", f"停留所数: {df['stop_id'].nunique()}")
    log("info", "-" * 60)
    log("info", "乗客数統計:")
    log("info", f"  平均: {df['passenger_count'].mean():.2f}")
    log("info", f"  中央値: {df['passenger_count'].median():.1f}")
    log("info", f"  最大: {df['passenger_count'].max()}")
    log("info", f"  最小: {df['passenger_count'].min()}")
    log("info", f"  標準偏差: {df['passenger_count'].std():.2f}")
    log("info", f"  ゼロの割合: {(df['passenger_count'] == 0).mean() * 100:.1f}%")
    log("info", "-" * 60)
    log("info", "時間帯別平均乗客数:")
    hourly = df.groupby("hour")["passenger_count"].mean()
    for hour in [7, 8, 12, 17, 18]:
        if hour in hourly.index:
            log("info", f"  {hour}時: {hourly[hour]:.2f}")
    log("info", "-" * 60)
    log("info", "曜日別平均乗客数:")
    daily = df.groupby("day_of_week")["passenger_count"].mean()
    days_jp = ["月", "火", "水", "木", "金", "土", "日"]
    for dow in range(7):
        if dow in daily.index:
            log("info", f"  {days_jp[dow]}曜日: {daily[dow]:.2f}")
    log("info", "-" * 60)
    log("info", "停留所タイプ別平均乗客数:")
    for stop_type in df["stop_type"].unique():
        mean_count = df[df["stop_type"] == stop_type]["passenger_count"].mean()
        log("info", f"  {stop_type}: {mean_count:.2f}")
    log("info", "-" * 60)
    log("info", "天候別平均乗客数:")
    rainy = df[df["precipitation"] > 0]["passenger_count"].mean()
    sunny = df[df["precipitation"] == 0]["passenger_count"].mean()
    log("info", f"  晴れ/曇り: {sunny:.2f}")
    log("info", f"  雨: {rainy:.2f}")
    log("info", "=" * 60)


########################
# Validation Functions
########################


def validate_data_quality(
    df: pd.DataFrame, logging_func: Callable | None = None
) -> bool:
    """
    生成されたデータの品質を検証する

    Args:
        df: 検証するDataFrame
        logging_func: ログ出力用関数

    Returns:
        検証が成功したかどうか
    """

    def log(log_type: str, content: str):
        if logging_func:
            logging_func(log_type, content)
        else:
            print(f"[{log_type.upper()}] {content}")

    log("info", "=" * 60)
    log("info", "データ品質検証")
    log("info", "=" * 60)

    issues = []

    # 1. 基本統計の確認
    mean_passengers = df["passenger_count"].mean()
    if mean_passengers < 0.5 or mean_passengers > 20:
        issues.append(f"平均乗客数が異常: {mean_passengers:.2f}")

    log("info", f"平均乗客数: {mean_passengers:.2f}")

    # 2. 時間帯パターンの確認
    hourly_mean = df.groupby("hour")["passenger_count"].mean()
    morning_peak = hourly_mean.loc[[7, 8]].mean() if 7 in hourly_mean.index else 0
    evening_peak = hourly_mean.loc[[17, 18]].mean() if 17 in hourly_mean.index else 0
    midday = hourly_mean.loc[[12, 13]].mean() if 12 in hourly_mean.index else 0

    if morning_peak < midday:
        issues.append("朝ピークが昼より低い")
    if evening_peak < midday:
        issues.append("夕方ピークが昼より低い")

    log("info", f"朝ピーク(7-8時): {morning_peak:.2f}")
    log("info", f"昼間(12-13時): {midday:.2f}")
    log("info", f"夕方ピーク(17-18時): {evening_peak:.2f}")

    # 3. 曜日パターンの確認
    daily_mean = df.groupby("day_of_week")["passenger_count"].mean()
    weekday_mean = daily_mean.loc[[0, 1, 2, 3, 4]].mean()
    weekend_mean = daily_mean.loc[[5, 6]].mean()

    if weekend_mean > weekday_mean:
        issues.append("休日の利用が平日より多い（過疎地では通常逆）")

    log("info", f"平日平均: {weekday_mean:.2f}")
    log("info", f"休日平均: {weekend_mean:.2f}")

    # 4. ゼロ膨張の確認
    zero_ratio = (df["passenger_count"] == 0).mean()
    if zero_ratio < 0.01:
        issues.append(f"ゼロの割合が少なすぎる: {zero_ratio * 100:.1f}%")
    elif zero_ratio > 0.6:
        issues.append(f"ゼロの割合が多すぎる: {zero_ratio * 100:.1f}%")

    log("info", f"ゼロ乗客の割合: {zero_ratio * 100:.1f}%")

    # 結果
    log("info", "-" * 60)
    if issues:
        log("warning", f"検出された問題点: {len(issues)}件")
        for issue in issues:
            log("warning", f"  - {issue}")
        return False
    else:
        log("success", "データ品質検証: 全てのチェックに合格")
        return True
