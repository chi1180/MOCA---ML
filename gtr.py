########################
# gtr - Generator (Simplified)
# シンプルなバス運行データ生成スクリプト
########################

import datetime
import json
import os

import pandas as pd
import requests

# データ生成モジュールをインポート
from data_generator import (
    generate_bus_operation_data,
    print_data_summary,
    save_to_csv,
    validate_data_quality,
)

########################
# Configurations
########################

STOPS_API_END_POINT = "https://moca-jet.vercel.app/api/stops"
OUTPUT_CSV_PATH = "./bus_operation_data.csv"
RESPONSE_CACHE_PATH = "./response.json"

# データ生成設定
DEFAULT_START_DATE = datetime.date(2025, 1, 1)
DEFAULT_END_DATE = datetime.date(2025, 12, 31)
RANDOM_SEED = 42  # 再現性のための乱数シード

########################
# Methods
########################


def logging(log_type: str, content):
    """ログ出力関数"""
    print(f"[-- {log_type.upper()} --] ::: {content}")


def get_stops(endpoint: str) -> dict:
    """
    APIから停留所データを取得

    Args:
        endpoint: APIエンドポイントURL

    Returns:
        停留所データのdict
    """
    logging("info", f"APIから停留所データを取得中: {endpoint}")
    response = requests.get(endpoint, timeout=30)
    if response.status_code == 200:
        logging("success", "API取得成功")
        return response.json()
    else:
        raise Exception(f"Failed to fetch stops data: {response.status_code}")


def load_datafile(filepath: str) -> str | None:
    """
    ローカルファイルからデータを読み込む

    Args:
        filepath: ファイルパス

    Returns:
        ファイル内容（存在しない場合はNone）
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = f.read().strip()
            return data
    return None


def save_datafile(filepath: str, data: dict) -> None:
    """
    データをJSONファイルに保存

    Args:
        filepath: ファイルパス
        data: 保存するデータ
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logging("info", f"データをキャッシュに保存: {filepath}")


def make_stop_df(stop_data: list | None = None) -> pd.DataFrame:
    """
    停留所データからDataFrameを作成

    Args:
        stop_data: 停留所データのリスト

    Returns:
        停留所のDataFrame
    """
    columns = [
        "id",
        "name",
        "latitude",
        "longitude",
        "address",
        "is_base_point",
        "created_at",
        "stop_type",
    ]

    if stop_data:
        return pd.DataFrame(stop_data, columns=columns)

    return pd.DataFrame(columns=columns)


def generate_training_data(
    stop_df: pd.DataFrame,
    start_date: datetime.date = DEFAULT_START_DATE,
    end_date: datetime.date = DEFAULT_END_DATE,
    output_path: str = OUTPUT_CSV_PATH,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    機械学習モデル訓練用のバス運行データを生成する。

    Args:
        stop_df: 停留所データのDataFrame
        start_date: データ生成開始日
        end_date: データ生成終了日
        output_path: 出力CSVファイルのパス
        seed: 乱数シード

    Returns:
        生成されたデータのDataFrame
    """
    logging("info", "=" * 60)
    logging("info", "模擬データ生成を開始します...")
    logging("info", "=" * 60)

    # データ生成
    df = generate_bus_operation_data(
        stop_df=stop_df,
        start_date=start_date,
        end_date=end_date,
        logging_func=logging,
        seed=seed,
    )

    # サマリー表示
    print_data_summary(df, logging_func=logging)

    # データ品質検証
    validate_data_quality(df, logging_func=logging)

    # CSVに保存
    save_to_csv(df, output_path, logging_func=logging)

    return df


def show_stop_info(stop_df: pd.DataFrame) -> None:
    """
    停留所情報を表示

    Args:
        stop_df: 停留所のDataFrame
    """
    logging("info", "=" * 60)
    logging("info", "停留所情報")
    logging("info", "=" * 60)
    logging("info", f"総停留所数: {len(stop_df)}")
    logging("info", "-" * 60)

    # 停留所タイプ別の集計
    type_counts = stop_df["stop_type"].value_counts()
    logging("info", "停留所タイプ別:")
    for stop_type, count in type_counts.items():
        logging("info", f"  {stop_type}: {count}件")

    # 拠点の集計
    base_count = stop_df["is_base_point"].sum()
    logging("info", f"拠点: {base_count}件")

    # 緯度経度の範囲
    logging("info", "-" * 60)
    logging("info", "地理的範囲:")
    logging(
        "info",
        f"  緯度: {stop_df['latitude'].min():.6f} ~ {stop_df['latitude'].max():.6f}",
    )
    logging(
        "info",
        f"  経度: {stop_df['longitude'].min():.6f} ~ {stop_df['longitude'].max():.6f}",
    )
    logging("info", "=" * 60)


########################
# Main
########################


def main():
    """
    メイン処理
    """
    logging("info", "=" * 60)
    logging("info", "バス運行データ生成システム")
    logging("info", "=" * 60)

    # 停留所データの取得
    file_data = load_datafile(RESPONSE_CACHE_PATH)
    if file_data:
        logging("info", "キャッシュからデータを読み込みました")
        stop_data = json.loads(file_data)
    else:
        logging("info", "APIからデータを取得します...")
        stop_data = get_stops(STOPS_API_END_POINT)
        save_datafile(RESPONSE_CACHE_PATH, stop_data)

    # 停留所データのDataFrameを作成
    stop_df = make_stop_df(stop_data=stop_data["data"])
    logging("success", "停留所DataFrameを作成しました")

    # 停留所情報を表示
    show_stop_info(stop_df)

    # 模擬データを生成
    training_df = generate_training_data(
        stop_df,
        start_date=DEFAULT_START_DATE,
        end_date=DEFAULT_END_DATE,
        seed=RANDOM_SEED,
    )

    logging("info", "=" * 60)
    logging("success", "全ての処理が完了しました！")
    logging("info", f"出力ファイル: {OUTPUT_CSV_PATH}")
    logging("info", f"レコード数: {len(training_df):,}")
    logging("info", "=" * 60)

    return training_df


if __name__ == "__main__":
    main()
