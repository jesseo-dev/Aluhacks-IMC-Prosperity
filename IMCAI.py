#this accepts both candles and time
#timestamp,product,bid_price_1,ask_price_1,mid_price is for market snapshots
#timestamp,product,open,high,low,close,volume is for candles
#install py -m pip install pandas numpy scikit-learn keyboard pyperclip groq requests lxml before hand!
import os
import time
import threading 
from io import StringIO
from tkinter import Tk, filedialog, simpledialog
from urllib.parse import urlparse

import keyboard
import pyperclip
import pandas as pd
import numpy as np
import requests
from groq import Groq
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    ExtraTreesRegressor,
    ExtraTreesClassifier,
)

API_KEY = os.environ.get("GROQ_API_KEY", "")
client = Groq(api_key=API_KEY) if API_KEY else None
MODEL = "groq/compound-mini"

SYSTEM_PROMPT = (
    "You are assisting with a Python-based market monitoring and prediction tool for trading data. "
    "Your job is to understand clipboard content in the context of market data, timestamps, candles, snapshots, price series, prediction logic, and Python code for analysis tools. "
    "Never behave like an HTML or CSS assistant. "
    "PRIMARY PURPOSE: "
    "Help analyze timestamped market data, products such as EMERALDS and TOMATOES, short-term trend, mean reversion, momentum, volatility, spreads, candle structure, and trade quality. "
    "Help improve Python code for reading CSV files, computing features, detecting trends, and predicting the next move. "
    "When asked what probably happens next, be probabilistic, not certain. "
    "When the clipboard contains Python code, repair it with the smallest correct change unless the clipboard clearly asks for a full solution. "
    "MODE 1 — Python code repair or generation: "
    "Output only valid Python code unless the user clearly asked for a non-code answer. "
    "Preserve correct variable names, structure, and working logic unless a change is necessary. "
    "Fix only real errors unless the task explicitly asks for a rewrite or a new feature. "
    "Keep solutions simple, practical, and directly relevant to market monitoring, live data reading, or price prediction. "
    "MODE 2 — Market data / CSV / timestamp analysis: "
    "If the clipboard contains rows of data, CSV content, timestamps, candle data, snapshot data, prices, open, high, low, close, volume, bids, asks, spreads, mids, product names, or a direct market data URL, analyze it in time order. "
    "Focus on trend, reversal, volatility, spread, expected next move, and whether there is enough edge to trade. "
    "For EMERALDS, consider possible anchoring near 10000 unless the data strongly suggests otherwise. "
    "For TOMATOES, consider trend continuation versus reversal based on the recent sequence. "
    "Give a short direct answer. "
    "MODE 3 — Short factual / trading-tool question: "
    "Answer briefly and directly in the context of Python, market data, prediction logic, or the local monitoring tool. "
    "MODE 4 — Summary mode: "
    "If given prediction outputs or model metrics, summarize the likely next move, confidence, uncertainty, and whether it looks tradable. "
    "Keep it concise and practical. "
    "UNDER ALL MODES: "
    "No markdown fences. "
    "No explanations of hidden reasoning. "
    "No extra text before or after the answer. "
    "Only output the final useful answer."
)

WATCH_PRODUCTS = ("EMERALDS", "TOMATOES")
POLL_INTERVAL = 2.0
AI_COOLDOWN_SECONDS = 20
REQUEST_TIMEOUT = 15
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"

SOURCE_MODE = "file"
DATA_FILE = "prices.csv"
DATA_URL = ""

monitor_running = False
monitor_thread = None
monitor_lock = threading.Lock()
last_ai_summary_time = 0.0


def looks_like_market_data(text: str) -> bool:
    low = text.lower()
    has_time = "timestamp" in low or "time" in low or "date" in low
    has_product = "product" in low or "symbol" in low or "ticker" in low
    has_snapshot = "mid_price" in low or "mid" in low or ("bid_price_1" in low and "ask_price_1" in low)
    has_candle = ("open" in low and "high" in low and "low" in low and "close" in low)
    return has_time and has_product and (has_snapshot or has_candle)


def looks_like_url(text: str) -> bool:
    text = text.strip()
    return text.startswith("http://") or text.startswith("https://")


def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url.strip())
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def current_source_label() -> str:
    return DATA_URL if SOURCE_MODE == "url" else DATA_FILE


def choose_data_file():
    global DATA_FILE, SOURCE_MODE

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    path = filedialog.askopenfilename(
        title="Select market CSV file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    root.destroy()

    if path:
        DATA_FILE = path
        SOURCE_MODE = "file"
        print(f"Selected file source: {DATA_FILE}", flush=True)
    else:
        print("No file selected.", flush=True)


def choose_data_url():
    global DATA_URL, SOURCE_MODE

    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    url = simpledialog.askstring("Market Data URL", "Paste direct CSV URL or page URL with a market data table:")
    root.destroy()

    if not url:
        print("No URL entered.", flush=True)
        return

    url = url.strip()
    if not validate_url(url):
        print("Invalid URL.", flush=True)
        return

    DATA_URL = url
    SOURCE_MODE = "url"
    print(f"Selected URL source: {DATA_URL}", flush=True)


def set_url_from_clipboard():
    global DATA_URL, SOURCE_MODE

    text = pyperclip.paste().strip()
    if not validate_url(text):
        print("Clipboard does not contain a valid URL.", flush=True)
        return

    DATA_URL = text
    SOURCE_MODE = "url"
    print(f"Selected URL source from clipboard: {DATA_URL}", flush=True)


def parse_text_to_df(text: str) -> pd.DataFrame:
    text = text.strip()
    if not text:
        raise ValueError("Clipboard is empty.")

    attempts = [
        {"sep": ","},
        {"sep": "\t"},
        {"sep": r"\s+", "engine": "python"},
    ]

    for kwargs in attempts:
        try:
            df = pd.read_csv(StringIO(text), **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass

    raise ValueError("Could not parse clipboard data as a table or CSV.")


def normalize_market_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("Data is empty.")

    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    rename_map = {
        "symbol": "product",
        "ticker": "product",
        "time": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "mid": "mid_price",
        "bid": "bid_price_1",
        "ask": "ask_price_1",
        "bid1": "bid_price_1",
        "ask1": "ask_price_1",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "vol": "volume",
    }

    for old, new in rename_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    if "timestamp" not in df.columns or "product" not in df.columns:
        raise ValueError("Data must contain at least timestamp and product columns.")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["product"] = df["product"].astype(str).str.strip().str.upper()

    candle_cols = ["open", "high", "low", "close"]
    has_candles = all(col in df.columns for col in candle_cols)
    has_snapshot_mid = "mid_price" in df.columns
    has_snapshot_book = "bid_price_1" in df.columns and "ask_price_1" in df.columns

    for col in ["mid_price", "bid_price_1", "ask_price_1", "open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if has_snapshot_mid:
        df["mid_price"] = pd.to_numeric(df["mid_price"], errors="coerce")
    elif has_snapshot_book:
        df["mid_price"] = (df["bid_price_1"] + df["ask_price_1"]) / 2.0
    elif has_candles:
        df["mid_price"] = df["close"]
    else:
        raise ValueError("Data must contain either snapshot columns (mid_price or bid_price_1 and ask_price_1) or candle columns (open, high, low, close).")

    if has_snapshot_book:
        df["spread"] = df["ask_price_1"] - df["bid_price_1"]
    else:
        df["spread"] = 0.0

    if not has_candles:
        df["open"] = df["mid_price"]
        df["high"] = df["mid_price"]
        df["low"] = df["mid_price"]
        df["close"] = df["mid_price"]

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df["candle_range"] = df["high"] - df["low"]
    df["candle_body"] = df["close"] - df["open"]

    df = df.dropna(subset=["timestamp", "product", "mid_price", "open", "high", "low", "close"]).copy()
    df = df[df["product"].isin(WATCH_PRODUCTS)].copy()
    df = df.sort_values(["product", "timestamp"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable EMERALDS or TOMATOES rows found.")

    return df


def fetch_url_content(url: str):
    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
    response.raise_for_status()
    text = response.text
    signature = response.headers.get("ETag") or response.headers.get("Last-Modified") or str(hash(text))
    content_type = response.headers.get("Content-Type", "").lower()
    return text, signature, content_type


def read_market_data_file(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    raw = pd.read_csv(filepath)
    return normalize_market_df(raw)


def read_market_data_text(text: str) -> pd.DataFrame:
    raw = parse_text_to_df(text)
    return normalize_market_df(raw)


def read_market_data_url(url: str):
    text, signature, content_type = fetch_url_content(url)

    if url.lower().endswith(".csv") or "text/csv" in content_type or looks_like_market_data(text):
        raw = parse_text_to_df(text)
        return normalize_market_df(raw), signature

    tables = pd.read_html(StringIO(text))
    if not tables:
        raise ValueError("No HTML tables found at the URL.")

    for table in tables:
        try:
            normalized = normalize_market_df(table)
            return normalized, signature
        except Exception:
            continue

    raise ValueError("Found tables at the URL, but none matched timestamp/product/price columns.")


def load_selected_source():
    if SOURCE_MODE == "url":
        if not DATA_URL:
            raise ValueError("No URL source selected.")
        df, signature = read_market_data_url(DATA_URL)
        return df, DATA_URL, signature

    df = read_market_data_file(DATA_FILE)
    signature = str(os.path.getmtime(DATA_FILE)) if os.path.exists(DATA_FILE) else ""
    return df, DATA_FILE, signature


def build_feature_table(product_df: pd.DataFrame, product_name: str):
    g = product_df.sort_values("timestamp").copy()

    g["time_diff_1"] = g["timestamp"].diff(1)
    g["time_diff_2"] = g["timestamp"].diff(2)

    median_step = g["time_diff_1"].dropna().median()
    if pd.isna(median_step) or median_step == 0:
        median_step = 1.0

    g["step_index"] = np.arange(len(g), dtype=float)
    g["timestamp_scaled"] = (g["timestamp"] - g["timestamp"].iloc[0]) / median_step

    lags = [1, 2, 3, 5, 8]
    if len(g) >= 25:
        lags.append(13)

    for lag in lags:
        g[f"lag_{lag}"] = g["mid_price"].shift(lag)

    g["ret_1"] = g["mid_price"].diff(1)
    g["ret_2"] = g["mid_price"].diff(2)
    g["ret_3"] = g["mid_price"].diff(3)

    windows = [3, 5]
    if len(g) >= 12:
        windows.append(8)

    for w in windows:
        g[f"roll_mean_{w}"] = g["mid_price"].rolling(w).mean()
        g[f"roll_std_{w}"] = g["mid_price"].rolling(w).std()

    g["ema_5"] = g["mid_price"].ewm(span=5, adjust=False).mean()
    g["ema_12"] = g["mid_price"].ewm(span=12, adjust=False).mean()
    g["ema_gap"] = g["ema_5"] - g["ema_12"]

    g["slope_5"] = (g["mid_price"] - g["mid_price"].shift(4)) / 4.0
    if len(g) >= 9:
        g["slope_8"] = (g["mid_price"] - g["mid_price"].shift(7)) / 7.0
    else:
        g["slope_8"] = 0.0

    if "roll_mean_5" in g.columns and "roll_std_5" in g.columns:
        g["zscore_5"] = (g["mid_price"] - g["roll_mean_5"]) / (g["roll_std_5"] + 1e-9)
    else:
        g["zscore_5"] = 0.0

    if "roll_mean_8" in g.columns and "roll_std_8" in g.columns:
        g["zscore_8"] = (g["mid_price"] - g["roll_mean_8"]) / (g["roll_std_8"] + 1e-9)
    else:
        g["zscore_8"] = 0.0

    if product_name == "EMERALDS":
        g["anchor_gap_10000"] = g["mid_price"] - 10000.0
    else:
        g["anchor_gap_10000"] = 0.0

    feature_cols = [
        "timestamp_scaled",
        "step_index",
        "time_diff_1",
        "time_diff_2",
        "mid_price",
        "spread",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "candle_range",
        "candle_body",
        "ret_1", "ret_2", "ret_3",
        "ema_5", "ema_12", "ema_gap",
        "slope_5", "slope_8",
        "zscore_5", "zscore_8",
        "anchor_gap_10000",
    ]

    for lag in lags:
        feature_cols.append(f"lag_{lag}")

    for w in windows:
        feature_cols.append(f"roll_mean_{w}")
        feature_cols.append(f"roll_std_{w}")

    return g, feature_cols, float(median_step)


def ensemble_class_probabilities(models, latest_row):
    class_sets = [set(m.classes_) for m in models]
    all_classes = sorted(set().union(*class_sets))
    probs = np.zeros(len(all_classes), dtype=float)

    for model in models:
        p = model.predict_proba(latest_row)[0]
        mapping = dict(zip(model.classes_, p))
        probs += np.array([mapping.get(c, 0.0) for c in all_classes], dtype=float)

    probs /= max(len(models), 1)
    return all_classes, probs


def classify_direction_with_deadzone(current_price, next_price, epsilon):
    diff = next_price - current_price
    if diff > epsilon:
        return "UP"
    if diff < -epsilon:
        return "DOWN"
    return "FLAT"


def grade_from_score(score):
    if score >= 0.82:
        return "A"
    if score >= 0.72:
        return "B"
    if score >= 0.62:
        return "C"
    return "D"


def build_trade_decision(product_name, latest_mid, predicted_next_mid, confidence, mae,
                         direction_accuracy, latest_spread, recent_volatility, trend_strength):
    expected_move = predicted_next_mid - latest_mid
    abs_move = abs(expected_move)

    mae = recent_volatility if mae is None or np.isnan(mae) else mae
    direction_accuracy = 0.50 if direction_accuracy is None or np.isnan(direction_accuracy) else direction_accuracy

    move_threshold = max(latest_spread * 0.80, recent_volatility * 0.35, 0.01)
    edge_ratio = abs_move / max(move_threshold, 1e-9)
    error_ratio = abs_move / max(mae, 1e-9)
    trend_ratio = abs(trend_strength) / max(recent_volatility, 1e-9)

    score = 0.0
    score += 0.38 * confidence
    score += 0.24 * min(edge_ratio / 3.0, 1.0)
    score += 0.18 * min(error_ratio / 2.5, 1.0)
    score += 0.12 * direction_accuracy
    score += 0.08 * min(trend_ratio / 2.0, 1.0)

    if latest_spread > abs_move:
        score -= 0.10
    if confidence < 0.50:
        score -= 0.08

    if product_name == "EMERALDS":
        anchor_gap = latest_mid - 10000.0
        if anchor_gap > 0 and expected_move > 0:
            score -= 0.08
        elif anchor_gap < 0 and expected_move < 0:
            score -= 0.08
        else:
            score += 0.04

    score = max(0.0, min(score, 0.99))
    quality = grade_from_score(score)

    if confidence < 0.50:
        signal = "NO TRADE"
        reason = "confidence too low"
    elif abs_move < move_threshold:
        signal = "NO TRADE"
        reason = "edge too small versus noise/spread"
    elif score < 0.62:
        signal = "NO TRADE"
        reason = "setup not clean enough"
    elif expected_move > 0:
        signal = "LONG"
        reason = "positive edge with acceptable confidence"
    else:
        signal = "SHORT"
        reason = "negative edge with acceptable confidence"

    target_price = latest_mid
    stop_price = latest_mid

    if signal == "LONG":
        target_price = latest_mid + max(abs_move * 1.25, move_threshold)
        stop_price = latest_mid - max(abs_move * 0.75, recent_volatility * 0.60, latest_spread * 1.50)
    elif signal == "SHORT":
        target_price = latest_mid - max(abs_move * 1.25, move_threshold)
        stop_price = latest_mid + max(abs_move * 0.75, recent_volatility * 0.60, latest_spread * 1.50)

    return {
        "signal": signal,
        "reason": reason,
        "trade_score": score,
        "quality": quality,
        "target_price": target_price,
        "stop_price": stop_price,
    }


def predict_next_for_product(df: pd.DataFrame, product_name: str):
    product_df = df[df["product"] == product_name].copy()

    if product_df.empty:
        return {"product": product_name, "error": f"No rows found for {product_name}."}

    if len(product_df) < 16:
        return {"product": product_name, "error": f"Need at least about 16 rows for {product_name}."}

    g, feature_cols, median_step = build_feature_table(product_df, product_name)

    median_abs_diff = g["mid_price"].diff().abs().dropna().median()
    if pd.isna(median_abs_diff) or median_abs_diff == 0:
        median_abs_diff = 0.01

    epsilon = max(float(median_abs_diff) * 0.35, 0.01)

    g["target_next_mid"] = g["mid_price"].shift(-1)
    g["target_direction"] = [
        classify_direction_with_deadzone(cur, nxt, epsilon)
        if pd.notna(cur) and pd.notna(nxt) else np.nan
        for cur, nxt in zip(g["mid_price"], g["target_next_mid"])
    ]

    train_df = g.dropna(subset=feature_cols + ["target_next_mid", "target_direction"]).copy()
    latest_df = g.dropna(subset=feature_cols).copy()

    if len(train_df) < 12 or latest_df.empty:
        return {"product": product_name, "error": f"Not enough clean rows to train for {product_name}. Need about 12+."}

    X = train_df[feature_cols]
    y_reg = train_df["target_next_mid"]
    y_cls = train_df["target_direction"]

    split_index = int(len(train_df) * 0.80)
    split_index = max(8, split_index)
    if split_index >= len(train_df):
        split_index = len(train_df) - 1

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_reg_train = y_reg.iloc[:split_index]
    y_reg_test = y_reg.iloc[split_index:]
    y_cls_train = y_cls.iloc[:split_index]
    y_cls_test = y_cls.iloc[split_index:]

    n_trees = 150 if len(train_df) < 30 else 300
    min_leaf = 1 if len(train_df) < 25 else 2

    reg_models = [
        RandomForestRegressor(n_estimators=n_trees, random_state=42, min_samples_leaf=min_leaf),
        ExtraTreesRegressor(n_estimators=n_trees, random_state=43, min_samples_leaf=min_leaf),
    ]

    cls_models = [
        RandomForestClassifier(n_estimators=n_trees, random_state=52, min_samples_leaf=min_leaf),
        ExtraTreesClassifier(n_estimators=n_trees, random_state=53, min_samples_leaf=min_leaf),
    ]

    for model in reg_models:
        model.fit(X_train, y_reg_train)

    for model in cls_models:
        model.fit(X_train, y_cls_train)

    latest_row = latest_df.iloc[[-1]][feature_cols]
    latest_mid = float(latest_df.iloc[-1]["mid_price"])
    latest_timestamp = float(latest_df.iloc[-1]["timestamp"])
    predicted_next_timestamp = latest_timestamp + median_step
    latest_spread = float(latest_df.iloc[-1]["spread"]) if "spread" in latest_df.columns else 0.0

    recent_volatility = float(latest_df["mid_price"].diff().tail(12).std())
    if np.isnan(recent_volatility) or recent_volatility == 0:
        recent_volatility = max(float(median_abs_diff), 0.01)

    trend_strength = float(latest_df.iloc[-1]["slope_8"]) if pd.notna(latest_df.iloc[-1]["slope_8"]) else 0.0

    reg_preds_latest = [float(model.predict(latest_row)[0]) for model in reg_models]
    predicted_next_mid = float(np.mean(reg_preds_latest))

    all_classes, probs = ensemble_class_probabilities(cls_models, latest_row)
    best_index = int(np.argmax(probs))
    predicted_direction = all_classes[best_index]
    confidence = float(probs[best_index])

    predicted_delta = predicted_next_mid - latest_mid

    mae = None
    direction_accuracy = None

    if len(X_test) > 0:
        reg_test_preds = np.mean(np.column_stack([model.predict(X_test) for model in reg_models]), axis=1)
        mae = float(np.mean(np.abs(reg_test_preds - y_reg_test)))

        cls_accs = []
        for model in cls_models:
            cls_accs.append(float(np.mean(model.predict(X_test) == y_cls_test)))
        direction_accuracy = float(np.mean(cls_accs))

    trade = build_trade_decision(
        product_name=product_name,
        latest_mid=latest_mid,
        predicted_next_mid=predicted_next_mid,
        confidence=confidence,
        mae=mae,
        direction_accuracy=direction_accuracy,
        latest_spread=latest_spread,
        recent_volatility=recent_volatility,
        trend_strength=trend_strength,
    )

    return {
        "product": product_name,
        "timestamp": latest_timestamp,
        "predicted_next_timestamp": predicted_next_timestamp,
        "latest_mid": latest_mid,
        "predicted_next_mid": predicted_next_mid,
        "predicted_delta": predicted_delta,
        "predicted_direction": predicted_direction,
        "confidence": confidence,
        "mae": mae,
        "direction_accuracy": direction_accuracy,
        "latest_spread": latest_spread,
        "recent_volatility": recent_volatility,
        "trend_strength": trend_strength,
        "signal": trade["signal"],
        "reason": trade["reason"],
        "trade_score": trade["trade_score"],
        "quality": trade["quality"],
        "target_price": trade["target_price"],
        "stop_price": trade["stop_price"],
    }


def generate_predictions_from_df(df: pd.DataFrame):
    predictions = []
    for product in WATCH_PRODUCTS:
        predictions.append(predict_next_for_product(df, product))
    return predictions


def fallback_summary(predictions):
    lines = []
    for p in predictions:
        if "error" in p:
            lines.append(f"{p['product']}: {p['error']}")
            continue

        lines.append(
            f"{p['product']}: {p['signal']} | next {p['predicted_direction']} to {p['predicted_next_mid']:.2f} "
            f"| conf {p['confidence']:.0%} | quality {p['quality']}"
        )
    return "\n".join(lines)


def get_ai_prediction_summary(predictions):
    if not API_KEY or client is None:
        return fallback_summary(predictions)

    compact = []
    for p in predictions:
        if "error" in p:
            compact.append({"product": p["product"], "error": p["error"]})
        else:
            compact.append({
                "product": p["product"],
                "timestamp": round(p["timestamp"], 4),
                "next_timestamp": round(p["predicted_next_timestamp"], 4),
                "now": round(p["latest_mid"], 4),
                "next": round(p["predicted_next_mid"], 4),
                "delta": round(p["predicted_delta"], 4),
                "direction": p["predicted_direction"],
                "confidence": round(p["confidence"], 4),
                "signal": p["signal"],
                "trade_score": round(p["trade_score"], 4),
                "quality": p["quality"],
                "reason": p["reason"],
                "spread": round(p["latest_spread"], 4),
                "volatility": round(p["recent_volatility"], 4),
                "mae": None if p["mae"] is None else round(p["mae"], 4),
                "direction_accuracy": None if p["direction_accuracy"] is None else round(p["direction_accuracy"], 4),
            })

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are summarizing one-step-ahead predictions from a market analysis tool. "
                        "State the likely next move and whether it looks tradable. "
                        "Be cautious and probabilistic, not certain. "
                        "Mention momentum, reversal, or mean reversion when relevant. "
                        "For EMERALDS, mention anchoring near 10000 if relevant. "
                        "For TOMATOES, focus on short-term direction and whether the edge looks strong enough to trade. "
                        "Use plain text only. "
                        "Keep it under 4 short lines."
                    )
                },
                {
                    "role": "user",
                    "content": f"Summarize these outputs:\n{compact}"
                }
            ],
            max_tokens=140,
            temperature=0
        )

        text = response.choices[0].message.content.strip()
        return text if text else fallback_summary(predictions)
    except Exception:
        return fallback_summary(predictions)


def format_prediction_line(p):
    if "error" in p:
        return f"{p['product']}: {p['error']}"

    mae_text = "n/a" if p["mae"] is None else f"{p['mae']:.3f}"
    acc_text = "n/a" if p["direction_accuracy"] is None else f"{p['direction_accuracy']:.0%}"

    return (
        f"{p['product']} | now_t={p['timestamp']:.0f} | next_t={p['predicted_next_timestamp']:.0f} | "
        f"now={p['latest_mid']:.2f} | next={p['predicted_next_mid']:.2f} | "
        f"move={p['predicted_delta']:+.2f} | dir={p['predicted_direction']} | conf={p['confidence']:.0%} | "
        f"signal={p['signal']} | quality={p['quality']} | score={p['trade_score']:.2f} | "
        f"tp={p['target_price']:.2f} | sl={p['stop_price']:.2f} | why={p['reason']} | "
        f"mae={mae_text} | dir_acc={acc_text}"
    )


def predictions_to_text(predictions, source_name="data"):
    lines = [f"[{time.strftime('%H:%M:%S')}] Analysis from: {source_name}"]
    for p in predictions:
        lines.append(format_prediction_line(p))
    lines.append(get_ai_prediction_summary(predictions))
    return "\n".join(lines)


def analyze_market_text(text: str):
    df = read_market_data_text(text)
    predictions = generate_predictions_from_df(df)
    return predictions_to_text(predictions, source_name="clipboard market data")


def analyze_market_url(url: str):
    df, _ = read_market_data_url(url)
    predictions = generate_predictions_from_df(df)
    return predictions_to_text(predictions, source_name=url)


def run_prediction_once_from_selected_source(use_ai=True):
    global last_ai_summary_time

    try:
        df, source_name, _ = load_selected_source()
        predictions = generate_predictions_from_df(df)

        print(f"\n[{time.strftime('%H:%M:%S')}] Prediction from: {source_name}", flush=True)
        for p in predictions:
            print(format_prediction_line(p), flush=True)

        if use_ai:
            summary = get_ai_prediction_summary(predictions)
            print(summary, flush=True)
            last_ai_summary_time = time.time()

    except Exception as e:
        print(f"[Prediction Error] {e}", flush=True)


def run_prediction_once_from_clipboard_data():
    try:
        text = pyperclip.paste().strip()
        output = analyze_market_text(text)
        print("\n" + output, flush=True)
        pyperclip.copy(output)
    except Exception as e:
        err = f"[Clipboard Data Error] {e}"
        print(err, flush=True)
        pyperclip.copy(err)


def run_prediction_once_from_clipboard_url():
    try:
        url = pyperclip.paste().strip()
        if not validate_url(url):
            raise ValueError("Clipboard does not contain a valid URL.")
        output = analyze_market_url(url)
        print("\n" + output, flush=True)
        pyperclip.copy(output)
    except Exception as e:
        err = f"[Clipboard URL Error] {e}"
        print(err, flush=True)
        pyperclip.copy(err)


def run_prediction_monitor():
    global monitor_running, last_ai_summary_time

    last_source_signature = None
    last_snapshot_key = None

    while monitor_running:
        try:
            df, source_name, source_signature = load_selected_source()

            if last_source_signature is None or source_signature != last_source_signature:
                predictions = generate_predictions_from_df(df)

                snapshot_key = tuple(
                    (
                        p.get("product"),
                        p.get("timestamp"),
                        round(p.get("latest_mid", 0.0), 6),
                        round(p.get("predicted_next_mid", 0.0), 6) if "predicted_next_mid" in p else None,
                        p.get("signal"),
                    )
                    for p in predictions
                )

                if snapshot_key != last_snapshot_key:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Live update from: {source_name}", flush=True)
                    for p in predictions:
                        print(format_prediction_line(p), flush=True)

                    if time.time() - last_ai_summary_time >= AI_COOLDOWN_SECONDS:
                        summary = get_ai_prediction_summary(predictions)
                        print(summary, flush=True)
                        last_ai_summary_time = time.time()

                    last_snapshot_key = snapshot_key

                last_source_signature = source_signature

        except Exception as e:
            print(f"[Monitor Error] {e}", flush=True)

        time.sleep(POLL_INTERVAL)


def start_monitor():
    global monitor_running, monitor_thread

    with monitor_lock:
        if monitor_running:
            print("Live prediction monitor is already running.", flush=True)
            return

        monitor_running = True
        monitor_thread = threading.Thread(target=run_prediction_monitor, daemon=True)
        monitor_thread.start()
        print(f"Started live prediction monitor on: {current_source_label()}", flush=True)


def stop_monitor():
    global monitor_running

    with monitor_lock:
        if not monitor_running:
            print("Live prediction monitor is not running.", flush=True)
            return

        monitor_running = False
        print("Stopped live prediction monitor.", flush=True)


def process_clipboard():
    try:
        user_text = pyperclip.paste().strip()
        if not user_text:
            pyperclip.copy("[Error: Empty]")
            return

        if looks_like_market_data(user_text):
            output = analyze_market_text(user_text)
            print("\n" + output, flush=True)
            pyperclip.copy(output)
            return

        if looks_like_url(user_text):
            output = analyze_market_url(user_text)
            print("\n" + output, flush=True)
            pyperclip.copy(output)
            return

        if not API_KEY or client is None:
            pyperclip.copy("[Error: Missing GROQ_API_KEY]")
            return

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ],
            max_tokens=2000,
            temperature=0
        )

        answer = response.choices[0].message.content.strip()
        if not answer:
            pyperclip.copy("[Error: No response]")
        else:
            pyperclip.copy(answer)

    except Exception as e:
        pyperclip.copy(f"[Error: Request failed: {e}]")


def hotkey_listener():
    keyboard.add_hotkey(
        "ctrl+shift+l",
        lambda: threading.Thread(target=process_clipboard, daemon=True).start()
    )
    keyboard.add_hotkey("ctrl+shift+u", choose_data_file)
    keyboard.add_hotkey("ctrl+shift+k", choose_data_url)
    keyboard.add_hotkey("ctrl+shift+j", set_url_from_clipboard)
    keyboard.add_hotkey(
        "ctrl+shift+p",
        lambda: threading.Thread(target=run_prediction_once_from_selected_source, daemon=True).start()
    )
    keyboard.add_hotkey(
        "ctrl+shift+v",
        lambda: threading.Thread(target=run_prediction_once_from_clipboard_data, daemon=True).start()
    )
    keyboard.add_hotkey(
        "ctrl+shift+o",
        lambda: threading.Thread(target=run_prediction_once_from_clipboard_url, daemon=True).start()
    )
    keyboard.add_hotkey("ctrl+shift+m", start_monitor)
    keyboard.add_hotkey("ctrl+shift+s", stop_monitor)

    print("Hotkeys:", flush=True)
    print("CTRL+SHIFT+L -> smart clipboard mode (market data, URL, or Groq assistant)", flush=True)
    print("CTRL+SHIFT+U -> choose CSV market data file", flush=True)
    print("CTRL+SHIFT+K -> enter website/data URL", flush=True)
    print("CTRL+SHIFT+J -> use URL from clipboard as the selected source", flush=True)
    print("CTRL+SHIFT+P -> predict once from the selected file or URL", flush=True)
    print("CTRL+SHIFT+V -> predict directly from copied market data", flush=True)
    print("CTRL+SHIFT+O -> predict directly from copied URL", flush=True)
    print("CTRL+SHIFT+M -> start live prediction monitor on the selected file or URL", flush=True)
    print("CTRL+SHIFT+S -> stop live prediction monitor", flush=True)

    keyboard.wait()


if __name__ == "__main__":
    hotkey_listener()
