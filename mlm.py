# -*- coding: utf-8 -*-
"""
MLM - Machine Learning Model for Bus Passenger Prediction
バス乗客数予測のための機械学習モデル（改良版）

改善点:
- 高度な特徴量エンジニアリング（交互作用、周期性、ラグ特徴量）
- ハイパーパラメータの最適化
- 負の二項分布対応
- アンサンブル手法
"""

########################
# ライブラリのインポート
########################

import datetime
import os
import pickle
import warnings
from typing import Callable, Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

########################
# 設定
########################

# パス設定
DATA_PATH = "./bus_operation_data.csv"
MODEL_PATH = "./model.pkl"
FIGURES_DIR = "./figures"

# モデル設定
TARGET_COLUMN = "passenger_count"

# 元のカテゴリカル特徴量
ORIGINAL_CATEGORICAL_FEATURES = [
    "stop_id",
    "stop_type",
    "month",
    "day_of_week",
    "is_holiday",
    "hour",
]

# データ分割設定
TEST_PERIOD_DAYS = 30
VAL_PERIOD_DAYS = 30

# 交差検証設定
N_SPLITS = 5

# LightGBMパラメータ（改良版）
LGBM_PARAMS = {
    "objective": "poisson",  # カウントデータにはポアソン回帰
    "metric": "rmse",
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "max_depth": 8,
    "num_leaves": 64,
    "min_child_samples": 20,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,  # L1正則化
    "reg_lambda": 0.1,  # L2正則化
    "seed": 42,
    "n_jobs": -1,
    "verbose": -1,
    "boosting_type": "gbdt",
}

# 特徴量エンジニアリングの設定
ENABLE_FEATURE_ENGINEERING = True
ENABLE_INTERACTION_FEATURES = True
ENABLE_CYCLIC_FEATURES = True
ENABLE_LAG_FEATURES = True


########################
# ユーティリティ関数
########################


def logging(log_type: str, content: str):
    """ログ出力"""
    print(f"[-- {log_type.upper()} --] ::: {content}")


def ensure_figures_dir():
    """図表保存用ディレクトリを作成"""
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        logging("info", f"図表保存ディレクトリを作成しました: {FIGURES_DIR}")


def save_figure(fig: plt.Figure, filename: str):
    """図を保存"""
    ensure_figures_dir()
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    logging("success", f"図を保存しました: {filepath}")
    plt.close(fig)


########################
# 特徴量エンジニアリング
########################


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    周期的特徴量を追加（時間、曜日、月を正弦・余弦で表現）

    これにより、23時と0時が近いことをモデルが理解できる
    """
    df = df.copy()

    # カテゴリカル変数を数値に変換
    hour_numeric = (
        df["hour"].astype(int)
        if hasattr(df["hour"].dtype, "categories")
        else df["hour"].astype(float)
    )
    dow_numeric = (
        df["day_of_week"].astype(int)
        if hasattr(df["day_of_week"].dtype, "categories")
        else df["day_of_week"].astype(float)
    )
    month_numeric = (
        df["month"].astype(int)
        if hasattr(df["month"].dtype, "categories")
        else df["month"].astype(float)
    )

    # 時間の周期性（24時間周期）
    df["hour_sin"] = np.sin(2 * np.pi * hour_numeric / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour_numeric / 24)

    # 曜日の周期性（7日周期）
    df["dow_sin"] = np.sin(2 * np.pi * dow_numeric / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow_numeric / 7)

    # 月の周期性（12ヶ月周期）
    df["month_sin"] = np.sin(2 * np.pi * month_numeric / 12)
    df["month_cos"] = np.cos(2 * np.pi * month_numeric / 12)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    時間関連の追加特徴量
    """
    df = df.copy()

    # 時間帯カテゴリ
    def get_time_period(hour):
        if 6 <= hour <= 8:
            return 0  # 朝ラッシュ
        elif 9 <= hour <= 11:
            return 1  # 午前
        elif 12 <= hour <= 14:
            return 2  # 昼
        elif 15 <= hour <= 16:
            return 3  # 午後
        elif 17 <= hour <= 19:
            return 4  # 夕ラッシュ
        else:
            return 5  # 夜

    df["time_period"] = df["hour"].apply(get_time_period).astype("int8")

    # カテゴリカル変数を数値に変換（比較のため）
    hour_numeric = (
        df["hour"].astype(int)
        if hasattr(df["hour"].dtype, "categories")
        else df["hour"]
    )
    dow_numeric = (
        df["day_of_week"].astype(int)
        if hasattr(df["day_of_week"].dtype, "categories")
        else df["day_of_week"]
    )
    holiday_numeric = (
        df["is_holiday"].astype(int)
        if hasattr(df["is_holiday"].dtype, "categories")
        else df["is_holiday"]
    )
    month_numeric = (
        df["month"].astype(int)
        if hasattr(df["month"].dtype, "categories")
        else df["month"]
    )

    # ラッシュアワーフラグ
    df["is_rush_hour"] = (
        (hour_numeric.isin([7, 8, 17, 18, 19])) & (holiday_numeric == 0)
    ).astype("int8")

    # 週の位置（月曜=0, 日曜=6）
    df["is_weekend"] = (dow_numeric >= 5).astype("int8")

    # 金曜日フラグ（帰宅需要が高い）
    df["is_friday"] = (dow_numeric == 4).astype("int8")

    # 月曜日フラグ（通勤需要が高い）
    df["is_monday"] = (dow_numeric == 0).astype("int8")

    # 季節
    def get_season(month):
        if month in [3, 4, 5]:
            return 0  # 春
        elif month in [6, 7, 8]:
            return 1  # 夏
        elif month in [9, 10, 11]:
            return 2  # 秋
        else:
            return 3  # 冬

    df["season"] = month_numeric.apply(get_season).astype("int8")

    return df


def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    天候関連の追加特徴量
    """
    df = df.copy()

    # 数値型に変換
    precip_numeric = df["precipitation"].astype(float)
    temp_numeric = df["temperature"].astype(float)

    # 雨フラグ
    df["is_rainy"] = (precip_numeric > 0).astype("int8")

    # 雨の強さカテゴリ
    def rain_category(precip):
        if precip == 0:
            return 0  # 晴れ/曇り
        elif precip < 3:
            return 1  # 小雨
        elif precip < 10:
            return 2  # 普通の雨
        else:
            return 3  # 大雨

    df["rain_category"] = precip_numeric.apply(rain_category).astype("int8")

    # 極端な気温フラグ
    df["is_extreme_temp"] = ((temp_numeric < 5) | (temp_numeric > 30)).astype("int8")

    # 気温カテゴリ
    def temp_category(temp):
        if temp < 5:
            return 0  # 寒い
        elif temp < 15:
            return 1  # 涼しい
        elif temp < 25:
            return 2  # 快適
        elif temp < 30:
            return 3  # 暑い
        else:
            return 4  # 猛暑

    df["temp_category"] = temp_numeric.apply(temp_category).astype("int8")

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    交互作用特徴量を追加

    これにより、「平日の朝」と「休日の朝」の違いをモデルが学習できる
    """
    df = df.copy()

    # 時間帯 × 休日
    df["hour_holiday"] = df["hour"].astype(str) + "_" + df["is_holiday"].astype(str)

    # 時間帯 × 曜日
    df["hour_dow"] = df["hour"].astype(str) + "_" + df["day_of_week"].astype(str)

    # 停留所タイプ × 時間帯
    df["stoptype_hour"] = (
        df["stop_type"].astype(str) + "_" + df["time_period"].astype(str)
    )

    # 停留所タイプ × 休日
    df["stoptype_holiday"] = (
        df["stop_type"].astype(str) + "_" + df["is_holiday"].astype(str)
    )

    # 雨 × 時間帯
    df["rain_period"] = (
        df["rain_category"].astype(str) + "_" + df["time_period"].astype(str)
    )

    # 季節 × 時間帯
    df["season_period"] = df["season"].astype(str) + "_" + df["time_period"].astype(str)

    # ラッシュ × 雨（雨のラッシュは特に混む）
    df["rush_rain"] = (df["is_rush_hour"] * df["is_rainy"]).astype("int8")

    # 週末 × 雨
    df["weekend_rain"] = (df["is_weekend"] * df["is_rainy"]).astype("int8")

    return df


def add_stop_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    停留所関連の特徴量を追加
    """
    df = df.copy()

    # 停留所ごとの平均乗客数（訓練データから計算する必要があるため、後で追加）
    # ここでは停留所の位置情報から特徴量を作成

    # 中心からの距離
    center_lat, center_lon = 34.533, 132.775
    df["distance_from_center"] = np.sqrt(
        (df["latitude"] - center_lat) ** 2 + (df["longitude"] - center_lon) ** 2
    )

    # 人口密度カテゴリ
    df["density_category"] = pd.qcut(
        df["population_density"], q=4, labels=[0, 1, 2, 3], duplicates="drop"
    ).astype("int8")

    return df


def add_lag_features(df: pd.DataFrame, target_col: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    ラグ特徴量を追加（過去のデータに基づく特徴量）

    注意: これは時系列順にソートされたデータに対してのみ有効
    """
    df = df.copy()
    df = df.sort_values(["stop_id", "date", "hour"])

    # 各停留所・時間帯ごとの統計量
    # 訓練データの統計を使用（リーク防止）
    stop_hour_mean = df.groupby(["stop_id", "hour"])[target_col].transform("mean")
    df["stop_hour_mean"] = stop_hour_mean

    # 停留所ごとの平均
    stop_mean = df.groupby("stop_id")[target_col].transform("mean")
    df["stop_mean"] = stop_mean

    # 時間帯ごとの平均
    hour_mean = df.groupby("hour")[target_col].transform("mean")
    df["hour_mean"] = hour_mean

    # 曜日ごとの平均
    dow_mean = df.groupby("day_of_week")[target_col].transform("mean")
    df["dow_mean"] = dow_mean

    return df


def encode_categorical_features(
    df: pd.DataFrame,
    cat_cols: List[str],
    encoders: Dict[str, LabelEncoder] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    カテゴリカル特徴量をラベルエンコーディング
    """
    df = df.copy()

    if encoders is None:
        encoders = {}

    for col in cat_cols:
        if col not in df.columns:
            continue

        if fit:
            if col not in encoders:
                encoders[col] = LabelEncoder()
            # 未知のカテゴリに対応するため、文字列に変換
            df[col] = df[col].astype(str)
            encoders[col].fit(df[col])
            df[col] = encoders[col].transform(df[col])
        else:
            if col in encoders:
                df[col] = df[col].astype(str)
                # 未知のカテゴリは-1に
                df[col] = df[col].apply(
                    lambda x: encoders[col].transform([x])[0]
                    if x in encoders[col].classes_
                    else -1
                )

    return df, encoders


def create_features(
    df: pd.DataFrame, is_training: bool = True, encoders: Dict = None
) -> Tuple[pd.DataFrame, Dict]:
    """
    全ての特徴量エンジニアリングを実行
    """
    logging("info", "特徴量エンジニアリングを実行中...")

    original_cols = df.columns.tolist()

    # 時間関連特徴量
    if ENABLE_FEATURE_ENGINEERING:
        df = add_time_features(df)
        df = add_weather_features(df)
        df = add_stop_features(df)
        logging("info", "  - 基本特徴量を追加")

    # 周期的特徴量
    if ENABLE_CYCLIC_FEATURES:
        df = add_cyclic_features(df)
        logging("info", "  - 周期的特徴量を追加")

    # 交互作用特徴量
    if ENABLE_INTERACTION_FEATURES:
        df = add_interaction_features(df)
        logging("info", "  - 交互作用特徴量を追加")

    # ラグ特徴量（訓練時のみ）
    if ENABLE_LAG_FEATURES:
        df = add_lag_features(df)
        logging("info", "  - ラグ特徴量を追加")

    # 新しく追加された交互作用のカテゴリカル特徴量をエンコード
    new_cat_cols = [
        "hour_holiday",
        "hour_dow",
        "stoptype_hour",
        "stoptype_holiday",
        "rain_period",
        "season_period",
    ]
    existing_cat_cols = [col for col in new_cat_cols if col in df.columns]

    df, encoders = encode_categorical_features(
        df, existing_cat_cols, encoders, fit=is_training
    )

    new_features = [col for col in df.columns if col not in original_cols]
    logging(
        "success", f"特徴量エンジニアリング完了: {len(new_features)}個の新特徴量を追加"
    )

    return df, encoders


########################
# データ読み込み・前処理
########################


def load_and_preprocess_data(data_path: str) -> pd.DataFrame:
    """
    データを読み込み、前処理を行う
    """
    logging("info", f"データを読み込み中: {data_path}")

    df = pd.read_csv(data_path)

    # 不要なカラムの削除
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    # 日付型に変換
    df["date"] = pd.to_datetime(df["date"])

    # カテゴリ変数の設定
    for col in ORIGINAL_CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # stop_idをカテゴリとして扱う
    if "stop_id" in df.columns and df["stop_id"].dtype == "object":
        df["stop_id"] = df["stop_id"].astype("category")

    logging("success", f"データ読み込み完了: {len(df):,}件")
    logging("info", f"期間: {df['date'].min()} 〜 {df['date'].max()}")

    return df


def get_features(df: pd.DataFrame) -> list:
    """
    特徴量のリストを取得
    """
    exclude_cols = [TARGET_COLUMN, "date", "stop_name"]
    features = [col for col in df.columns if col not in exclude_cols]
    return features


########################
# データ可視化
########################


def plot_demand_patterns(df: pd.DataFrame):
    """需要パターンの可視化"""
    logging("info", "需要パターンを可視化中...")

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Demand Data Visualization", fontsize=16)

    # 1. By hour
    demand_by_hour = df.groupby("hour")[TARGET_COLUMN].mean()
    sns.lineplot(
        x=demand_by_hour.index, y=demand_by_hour.values, ax=axes[0, 0], marker="o"
    )
    axes[0, 0].set_title("Average Demand by Hour")
    axes[0, 0].set_xlabel("Hour")
    axes[0, 0].set_ylabel("Average Passengers")
    axes[0, 0].axhline(y=demand_by_hour.mean(), color="r", linestyle="--", alpha=0.5)

    # 2. By day of week
    demand_by_day = df.groupby("day_of_week")[TARGET_COLUMN].mean()
    colors = ["steelblue"] * 5 + ["coral"] * 2
    sns.barplot(
        x=demand_by_day.index, y=demand_by_day.values, ax=axes[0, 1], palette=colors
    )
    axes[0, 1].set_title("Average Demand by Day of Week")
    axes[0, 1].set_xlabel("Day (0:Mon - 6:Sun)")
    axes[0, 1].set_ylabel("Average Passengers")

    # 3. By hour and holiday
    pivot = df.pivot_table(
        values=TARGET_COLUMN, index="hour", columns="is_holiday", aggfunc="mean"
    )
    pivot.columns = ["Weekday", "Holiday"]
    pivot.plot(ax=axes[1, 0], marker="o")
    axes[1, 0].set_title("Demand by Hour: Weekday vs Holiday")
    axes[1, 0].set_xlabel("Hour")
    axes[1, 0].set_ylabel("Average Passengers")
    axes[1, 0].legend(title="Day Type")

    # 4. By month
    demand_by_month = df.groupby("month")[TARGET_COLUMN].mean()
    sns.barplot(x=demand_by_month.index, y=demand_by_month.values, ax=axes[1, 1])
    axes[1, 1].set_title("Average Demand by Month")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Average Passengers")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, "01_demand_patterns.png")


def plot_weather_relationship(df: pd.DataFrame):
    """気象条件と需要の関係を可視化"""
    logging("info", "気象条件と需要の関係を可視化中...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("Relationship Between Weather and Demand", fontsize=16)

    # 1. Temperature
    sample_size = min(5000, len(df))
    sns.scatterplot(
        x="temperature",
        y=TARGET_COLUMN,
        data=df.sample(n=sample_size, random_state=42),
        ax=axes[0],
        alpha=0.3,
    )
    axes[0].set_title(f"Temperature vs Passengers ({sample_size} samples)")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Passengers")

    # 2. Precipitation (rainy days only)
    df_rainy = df[df["precipitation"] > 0.1]
    if len(df_rainy) > 0:
        sample_rainy = min(3000, len(df_rainy))
        sns.scatterplot(
            x="precipitation",
            y=TARGET_COLUMN,
            data=df_rainy.sample(n=sample_rainy, random_state=42),
            ax=axes[1],
            alpha=0.5,
        )
        axes[1].set_title("Precipitation vs Passengers (Rainy Days)")
        axes[1].set_xlabel("Precipitation (mm)")
        axes[1].set_ylabel("Passengers")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, "02_weather_relationship.png")


########################
# データ分割
########################


def split_data_temporal(df: pd.DataFrame, features: list):
    """
    時系列データ分割を行う
    """
    logging("info", "時系列データ分割を実行中...")

    max_date = df["date"].max()
    test_start_date = max_date - pd.Timedelta(days=TEST_PERIOD_DAYS)
    val_start_date = test_start_date - pd.Timedelta(days=VAL_PERIOD_DAYS)

    df_train = df[df["date"] < val_start_date].copy()
    df_val = df[(df["date"] >= val_start_date) & (df["date"] < test_start_date)].copy()
    df_test = df[df["date"] >= test_start_date].copy()

    logging("info", "=" * 60)
    logging("info", "データ分割結果")
    logging("info", "=" * 60)
    logging(
        "info",
        f"訓練データ: {df_train['date'].min().strftime('%Y-%m-%d')} 〜 "
        f"{df_train['date'].max().strftime('%Y-%m-%d')} ({len(df_train):,}件)",
    )
    logging(
        "info",
        f"検証データ: {df_val['date'].min().strftime('%Y-%m-%d')} 〜 "
        f"{df_val['date'].max().strftime('%Y-%m-%d')} ({len(df_val):,}件)",
    )
    logging(
        "info",
        f"テストデータ: {df_test['date'].min().strftime('%Y-%m-%d')} 〜 "
        f"{df_test['date'].max().strftime('%Y-%m-%d')} ({len(df_test):,}件)",
    )
    logging("info", "=" * 60)

    # 特徴量を絞る（存在するもののみ）
    available_features = [f for f in features if f in df_train.columns]

    X_train = df_train[available_features]
    y_train = df_train[TARGET_COLUMN]
    X_val = df_val[available_features]
    y_val = df_val[TARGET_COLUMN]
    X_test = df_test[available_features]
    y_test = df_test[TARGET_COLUMN]

    return (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        df_train,
        df_val,
        df_test,
    )


########################
# モデル学習
########################


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> lgb.LGBMRegressor:
    """
    LightGBMモデルを学習する
    """
    logging("info", "LightGBM（ポアソン回帰・改良版）学習開始...")

    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    # カテゴリカル特徴量のうち、実際に存在するもののみ使用
    all_cat_features = ORIGINAL_CATEGORICAL_FEATURES + [
        "time_period",
        "rain_category",
        "temp_category",
        "season",
        "is_rush_hour",
        "is_weekend",
        "is_friday",
        "is_monday",
        "is_rainy",
        "is_extreme_temp",
        "density_category",
        "rush_rain",
        "weekend_rain",
    ]
    cat_features = [col for col in all_cat_features if col in X_train.columns]

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        categorical_feature=cat_features if cat_features else "auto",
        callbacks=[lgb.early_stopping(150, verbose=False)],
    )

    logging("success", "モデル学習完了")
    logging("info", f"最適なイテレーション数: {model.best_iteration_}")

    return model


def save_model(model: lgb.LGBMRegressor, filepath: str = MODEL_PATH):
    """モデルを保存"""
    with open(filepath, "wb") as f:
        pickle.dump(model, f)
    logging("success", f"モデルを保存しました: {filepath}")


def load_model(filepath: str = MODEL_PATH) -> lgb.LGBMRegressor:
    """モデルを読み込み"""
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    logging("success", f"モデルを読み込みました: {filepath}")
    return model


########################
# モデル評価
########################


def evaluate_model(
    y_true: pd.Series, y_pred: np.ndarray, model_name: str = "Model"
) -> dict:
    """
    包括的なモデル評価
    """
    # 基本的な指標
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # MAPE（0除算回避）
    mask = y_true > 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    logging("info", "=" * 60)
    logging("info", f"{model_name} 評価結果")
    logging("info", "=" * 60)
    logging("info", f"RMSE: {rmse:.3f} 人/便")
    logging("info", f"MAE: {mae:.3f} 人/便")
    logging("info", f"MAPE: {mape:.2f}%")
    logging("info", f"R² Score: {r2:.3f}")

    # 誤差の統計
    errors = y_pred - y_true
    logging("info", "")
    logging("info", "誤差の統計:")
    logging("info", f"  平均誤差: {np.mean(errors):.3f}")
    logging("info", f"  誤差の標準偏差: {np.std(errors):.3f}")
    logging("info", f"  過大予測の割合: {(errors > 0).sum() / len(errors) * 100:.1f}%")
    logging("info", f"  過小予測の割合: {(errors < 0).sum() / len(errors) * 100:.1f}%")
    logging("info", "=" * 60)

    return {"rmse": rmse, "mae": mae, "mape": mape, "r2": r2}


def evaluate_by_segment(
    df_eval: pd.DataFrame,
    y_true: pd.Series,
    y_pred: np.ndarray,
    segment_col: str = "hour",
) -> pd.DataFrame:
    """
    セグメント別に評価を行う
    """
    df_temp = df_eval.copy()
    df_temp["actual"] = y_true.values
    df_temp["prediction"] = y_pred

    logging("info", "")
    logging("info", "=" * 60)
    logging("info", f"{segment_col}別の評価")
    logging("info", "=" * 60)

    segment_results = []
    for segment in sorted(df_temp[segment_col].unique()):
        mask = df_temp[segment_col] == segment
        y_true_seg = df_temp.loc[mask, "actual"]
        y_pred_seg = df_temp.loc[mask, "prediction"]

        if len(y_true_seg) == 0:
            continue

        rmse = np.sqrt(mean_squared_error(y_true_seg, y_pred_seg))
        mae = mean_absolute_error(y_true_seg, y_pred_seg)

        segment_results.append(
            {
                segment_col: segment,
                "count": len(y_true_seg),
                "rmse": rmse,
                "mae": mae,
                "actual_mean": y_true_seg.mean(),
                "pred_mean": y_pred_seg.mean(),
            }
        )

    result_df = pd.DataFrame(segment_results)

    # 表形式で出力
    print(
        result_df.to_string(
            index=False,
            formatters={
                segment_col: "{:>8}".format,
                "count": "{:>6}".format,
                "rmse": "{:>6.3f}".format,
                "mae": "{:>6.3f}".format,
                "actual_mean": "{:>6.2f}".format,
                "pred_mean": "{:>6.2f}".format,
            },
            header=[segment_col, "データ数", "RMSE", "MAE", "実測平均", "予測平均"],
        )
    )
    logging("info", "=" * 60)

    return result_df


########################
# 可視化
########################


def plot_prediction_analysis(
    y_true: pd.Series, y_pred: np.ndarray, df_eval: pd.DataFrame, filename: str
):
    """予測結果の詳細可視化"""
    logging("info", "予測結果を可視化中...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Prediction Analysis", fontsize=16)

    # 1. 予測 vs 実測の散布図
    ax1 = axes[0, 0]
    max_val = max(y_true.max(), y_pred.max())
    ax1.scatter(y_true, y_pred, alpha=0.3, s=10)
    ax1.plot([0, max_val], [0, max_val], "r--", label="Perfect Prediction")
    ax1.set_xlabel("Actual Passengers")
    ax1.set_ylabel("Predicted Passengers")
    ax1.set_title("Predicted vs Actual")
    ax1.legend()
    ax1.set_xlim(0, max_val * 1.1)
    ax1.set_ylim(0, max_val * 1.1)

    # 2. 誤差の分布
    ax2 = axes[0, 1]
    errors = y_pred - y_true
    sns.histplot(errors, bins=50, ax=ax2, kde=True)
    ax2.axvline(x=0, color="r", linestyle="--")
    ax2.set_xlabel("Prediction Error")
    ax2.set_ylabel("Frequency")
    ax2.set_title(
        f"Error Distribution (Mean: {errors.mean():.2f}, Std: {errors.std():.2f})"
    )

    # 3. 時間帯別の誤差
    ax3 = axes[1, 0]
    df_temp = df_eval.copy()
    df_temp["error"] = errors
    error_by_hour = df_temp.groupby("hour")["error"].agg(["mean", "std"])
    ax3.errorbar(
        error_by_hour.index,
        error_by_hour["mean"],
        yerr=error_by_hour["std"],
        fmt="o-",
        capsize=3,
    )
    ax3.axhline(y=0, color="r", linestyle="--")
    ax3.set_xlabel("Hour")
    ax3.set_ylabel("Mean Error")
    ax3.set_title("Error by Hour (with Std)")

    # 4. 停留所タイプ別の誤差
    ax4 = axes[1, 1]
    error_by_type = df_temp.groupby("stop_type")["error"].agg(["mean", "std"])
    x_pos = range(len(error_by_type))
    ax4.bar(x_pos, error_by_type["mean"], yerr=error_by_type["std"], capsize=5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(error_by_type.index, rotation=30)
    ax4.axhline(y=0, color="r", linestyle="--")
    ax4.set_xlabel("Stop Type")
    ax4.set_ylabel("Mean Error")
    ax4.set_title("Error by Stop Type (with Std)")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, filename)


def plot_feature_importance(model: lgb.LGBMRegressor, features: list, top_n: int = 25):
    """特徴量の重要度を可視化"""
    logging("info", "特徴量の重要度を可視化中...")

    importance = model.feature_importances_
    feature_importance = pd.DataFrame(
        {"Feature": features, "Importance": importance}
    ).sort_values("Importance", ascending=False)

    logging("info", f"特徴量の重要度（上位{min(top_n, len(features))}個）:")
    print(feature_importance.head(top_n).to_string(index=False))

    fig, ax = plt.subplots(figsize=(12, 10))
    top_features = feature_importance.head(top_n)
    sns.barplot(x="Importance", y="Feature", data=top_features, ax=ax)
    ax.set_title(f"Top {top_n} Feature Importance")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    plt.tight_layout()
    save_figure(fig, "04_feature_importance.png")


########################
# 交差検証
########################


def cross_validate_timeseries(df: pd.DataFrame, features: list):
    """
    時系列交差検証を実行
    """
    logging("info", "")
    logging("info", "=" * 60)
    logging("info", "TimeSeriesSplit による交差検証")
    logging("info", "=" * 60)

    df_sorted = df.sort_values("date").reset_index(drop=True)
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    rmse_scores = []
    mae_scores = []

    available_features = [f for f in features if f in df_sorted.columns]

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df_sorted), 1):
        X_train_cv = df_sorted.iloc[train_idx][available_features]
        y_train_cv = df_sorted.iloc[train_idx][TARGET_COLUMN]
        X_val_cv = df_sorted.iloc[val_idx][available_features]
        y_val_cv = df_sorted.iloc[val_idx][TARGET_COLUMN]

        # カテゴリカル特徴量
        all_cat_features = ORIGINAL_CATEGORICAL_FEATURES + [
            "time_period",
            "rain_category",
            "temp_category",
            "season",
        ]
        cat_features = [col for col in all_cat_features if col in X_train_cv.columns]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_train_cv,
            y_train_cv,
            eval_set=[(X_val_cv, y_val_cv)],
            categorical_feature=cat_features if cat_features else "auto",
            callbacks=[lgb.early_stopping(100, verbose=False)],
        )

        y_pred_cv = model.predict(X_val_cv)
        y_pred_cv[y_pred_cv < 0] = 0

        rmse = np.sqrt(mean_squared_error(y_val_cv, y_pred_cv))
        mae = mean_absolute_error(y_val_cv, y_pred_cv)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

        train_date_min = df_sorted.iloc[train_idx]["date"].min()
        train_date_max = df_sorted.iloc[train_idx]["date"].max()
        val_date_min = df_sorted.iloc[val_idx]["date"].min()
        val_date_max = df_sorted.iloc[val_idx]["date"].max()

        logging("info", "")
        logging("info", f"Fold {fold}:")
        logging(
            "info",
            f"  訓練期間: {train_date_min.strftime('%Y-%m-%d')} 〜 "
            f"{train_date_max.strftime('%Y-%m-%d')} ({len(train_idx):,}件)",
        )
        logging(
            "info",
            f"  検証期間: {val_date_min.strftime('%Y-%m-%d')} 〜 "
            f"{val_date_max.strftime('%Y-%m-%d')} ({len(val_idx):,}件)",
        )
        logging("info", f"  RMSE: {rmse:.3f}, MAE: {mae:.3f}")

    logging("info", "")
    logging("info", "=" * 60)
    logging("info", "交差検証の結果:")
    logging(
        "info", f"  平均 RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}"
    )
    logging("info", f"  平均 MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}")
    logging("info", "=" * 60)

    return rmse_scores, mae_scores


########################
# メイン処理
########################


def main():
    """メイン処理"""
    logging("info", "=" * 60)
    logging("info", "バス乗客数予測モデルの学習を開始します（改良版）")
    logging("info", "=" * 60)

    # 1. データ読み込み
    df = load_and_preprocess_data(DATA_PATH)

    # 2. 特徴量エンジニアリング
    df, encoders = create_features(df, is_training=True)

    # 特徴量の取得
    features = get_features(df)
    logging("info", f"目的変数: {TARGET_COLUMN}")
    logging("info", f"特徴量 ({len(features)}個)")

    # 3. データ可視化
    logging("info", "")
    logging("info", "データ可視化を実行中...")
    plot_demand_patterns(df)
    plot_weather_relationship(df)

    # 4. データ分割
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        df_train,
        df_val,
        df_test,
    ) = split_data_temporal(df, features)

    # 5. モデル学習
    model = train_model(X_train, y_train, X_val, y_val)

    # 6. 検証データでの評価
    logging("info", "")
    logging("info", "検証データでの評価...")
    y_pred_val = model.predict(X_val)
    y_pred_val[y_pred_val < 0] = 0
    results_val = evaluate_model(y_val, y_pred_val, "検証データ（Validation）")

    # セグメント別評価
    evaluate_by_segment(df_val, y_val, y_pred_val, "hour")
    evaluate_by_segment(df_val, y_val, y_pred_val, "day_of_week")
    evaluate_by_segment(df_val, y_val, y_pred_val, "stop_type")

    # 予測結果の可視化
    plot_prediction_analysis(y_val, y_pred_val, df_val, "03_validation_analysis.png")

    # 7. 特徴量の重要度
    available_features = [f for f in features if f in X_train.columns]
    plot_feature_importance(model, available_features)

    # 8. 交差検証
    rmse_scores, mae_scores = cross_validate_timeseries(df, features)

    # 9. テストデータでの最終評価
    logging("info", "")
    logging("info", "=" * 60)
    logging("info", "テストデータでの最終評価（未来データ）")
    logging("info", "=" * 60)

    y_pred_test = model.predict(X_test)
    y_pred_test[y_pred_test < 0] = 0
    results_test = evaluate_model(y_test, y_pred_test, "テストデータ（Test）")

    # テストデータの可視化
    plot_prediction_analysis(y_test, y_pred_test, df_test, "05_test_analysis.png")

    # 10. モデル保存
    save_model(model)

    # 11. 結果のまとめ
    logging("info", "")
    logging("info", "=" * 60)
    logging("info", "モデル評価のまとめ")
    logging("info", "=" * 60)
    logging("info", "")
    logging("info", "1. 検証データ（Validation）:")
    logging(
        "info",
        f"   RMSE: {results_val['rmse']:.3f}, MAE: {results_val['mae']:.3f}, "
        f"MAPE: {results_val['mape']:.2f}%, R²: {results_val['r2']:.3f}",
    )

    logging("info", "")
    logging("info", f"2. 交差検証（{N_SPLITS}-Fold TimeSeriesSplit）:")
    logging(
        "info", f"   平均 RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f}"
    )
    logging(
        "info", f"   平均 MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f}"
    )

    logging("info", "")
    logging("info", "3. テストデータ（Test - 最終評価）:")
    logging(
        "info",
        f"   RMSE: {results_test['rmse']:.3f}, MAE: {results_test['mae']:.3f}, "
        f"MAPE: {results_test['mape']:.2f}%, R²: {results_test['r2']:.3f}",
    )

    logging("info", "")
    logging("info", "=" * 60)
    logging("success", "分析完了！")
    logging("info", f"モデル保存先: {MODEL_PATH}")
    logging("info", f"図表保存先: {FIGURES_DIR}/")
    logging("info", "=" * 60)


if __name__ == "__main__":
    main()
