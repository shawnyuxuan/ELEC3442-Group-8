import json
import os
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


WEEKDAY_MAP = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3, "Fri": 4, "Sat": 5, "Sun": 6}

root_dir = os.path.dirname(os.path.abspath(__file__))
data_path = f"{root_dir}/../data.xml"
dates_path = f"{root_dir}/../dates.json"
output_path = f"{root_dir}/../output/local_model.pkl"
N_CLUSTERS = 3
RANDOM_STATE = 42


def load_inputs(data_path: str, dates_path: str) -> tuple[list[ET.Element], list[list[object]]]:
    assert os.path.exists(data_path), f"File does not exist: {data_path}"
    assert os.path.exists(dates_path), f"File does not exist: {dates_path}"

    root = ET.parse(data_path).getroot()
    dates = json.load(open(dates_path, "r", encoding="utf-8"))
    sleep_data = [record for record in root.findall("Record")]

    print(f"Total number of records: {len(sleep_data)}")
    print(f"Total days of data: {len(dates)}")
    return sleep_data, dates


def feature_extraction(dates: list[list[object]], data: list[ET.Element]) -> pd.DataFrame:
    """
    Convert each sleep session into one training row.
    Each row contains start time, sleep-stage durations, and weekday.
    """
    raw_df = pd.DataFrame([record.attrib for record in data])

    raw_df["startDate"] = pd.to_datetime(raw_df["startDate"], format="%Y-%m-%d %H:%M:%S %z")
    raw_df["endDate"] = pd.to_datetime(raw_df["endDate"], format="%Y-%m-%d %H:%M:%S %z")
    raw_df["duration"] = (raw_df["endDate"] - raw_df["startDate"]).dt.total_seconds() / 60.0

    rows = []
    skipped_sessions: list[dict[str, object]] = []
    for weekday, indices in dates:
        session_slice = raw_df.iloc[indices]
        core = session_slice[session_slice["value"] == "HKCategoryValueSleepAnalysisAsleepCore"]["duration"].sum()
        deep = session_slice[session_slice["value"] == "HKCategoryValueSleepAnalysisAsleepDeep"]["duration"].sum()
        rem = session_slice[session_slice["value"] == "HKCategoryValueSleepAnalysisAsleepREM"]["duration"].sum()
        awake = session_slice[session_slice["value"] == "HKCategoryValueSleepAnalysisAwake"]["duration"].sum()
        unspecified = session_slice[
            session_slice["value"] == "HKCategoryValueSleepAnalysisAsleepUnspecified"
        ]["duration"].sum()

        # Early Apple sleep records often contain only "AsleepUnspecified".
        # Those sessions do not carry enough stage information for clustering.
        if unspecified > core + deep + rem:
            skipped_sessions.append(
                {
                    "date": session_slice["startDate"].min(),
                    "unspecified": unspecified,
                    "core": core,
                    "deep": deep,
                    "rem": rem,
                }
            )
            continue

        start_time = session_slice["startDate"].min()
        start_hour = start_time.hour + start_time.minute / 60.0
        start_sin = np.sin(2 * np.pi * start_hour / 24)

        rows.append([start_sin, core, deep, rem, awake, unspecified, WEEKDAY_MAP[weekday]])

    if skipped_sessions:
        print(
            "Skipped sessions with dominant unspecified sleep stage: "
            f"{len(skipped_sessions)}"
        )
        for session in skipped_sessions[:5]:
            print(
                "  "
                f"Date: {session['date']}, Unspecified: {session['unspecified']}, "
                f"Core: {session['core']}, Deep: {session['deep']}, REM: {session['rem']}"
            )
        if len(skipped_sessions) > 5:
            print(f"  ... {len(skipped_sessions) - 5} more skipped sessions")

    return pd.DataFrame(
        rows,
        columns=["start_sin", "CORE", "DEEP", "REM", "AWAKE", "UNSPECIFIED", "weekday"],
    )


def train_kmeans(
    df: pd.DataFrame, n_clusters: int = 3, random_state: int = 42
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    df = df.dropna().copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=n_clusters, random_state=random_state)
    df["cluster"] = model.fit_predict(X_scaled)
    return df, model, scaler


def save_model(model: KMeans, scaler: StandardScaler, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as handle:
        pickle.dump((model, scaler), handle)


def main() -> int:
    sleep_data, dates = load_inputs(data_path, dates_path)
    df = feature_extraction(dates, sleep_data)
    final_df, model, scaler = train_kmeans(
        df, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE
    )

    print(final_df.groupby("cluster").mean())
    save_model(model, scaler, output_path)
    print(f"Saved model to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
