import os
import time
import pandas as pd
from binance.client import Client
from binance.exceptions import BinanceAPIException
from dotenv import load_dotenv
import requests
import ta
import joblib

# Cargar variables de entorno
load_dotenv()

API_KEY = os.getenv("BINANCE_API_KEY")
API_SECRET = os.getenv("BINANCE_API_SECRET")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

client = Client(API_KEY, API_SECRET)

# Cargar modelo y scaler
modelo = joblib.load("rf_model.pkl")
scaler = joblib.load("scaler.pkl")

symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1MINUTE
lookback = "500 minutes ago UTC"

capital = 1000
position = None
entry_price = 0

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Error enviando mensaje Telegram: {e}")

def get_ohlc():
    try:
        klines = client.get_historical_klines(symbol, interval, lookback)
        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["volume"] = df["volume"].astype(float)
        df.index = pd.to_datetime(df["close_time"], unit='ms')
        return df
    except BinanceAPIException as e:
        print(f"Error Binance API: {e}")
        return None

def calculate_indicators(df):
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    df["sma50"] = ta.trend.SMAIndicator(df["close"], window=50).sma_indicator()
    macd = ta.trend.MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"] = macd.macd_diff()
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_hband"] = bb.bollinger_hband()
    df["bb_lband"] = bb.bollinger_lband()
    df["atr"] = ta.volatility.AverageTrueRange(df["high"], df["low"], df["close"], window=14).average_true_range()
    df.dropna(inplace=True)

def generate_ml_signal(df):
    last_row = df.iloc[-1]
    features = ["rsi", "sma50", "macd", "macd_signal", "macd_diff", "bb_hband", "bb_lband", "atr"]
    X = last_row[features].values.reshape(1, -1)
    X_df = pd.DataFrame(X, columns=features)
    X_scaled = scaler.transform(X_df)
    prob = modelo.predict_proba(X_scaled)[0][1]  # probabilidad de clase 1 (subida)
    return prob, prob > 0.7  # Retorna también el valor de la probabilidad

def main():
    global position, entry_price, capital
    print("🤖 Iniciando bot con modelo ML...")
    send_telegram_message("🚀 Bot de trading con ML activo...")

    last_active_message_time = 0
    active_message_interval = 30 * 60  # cada 30 min

    while True:
        df = get_ohlc()
        if df is not None and len(df) > 50:
            calculate_indicators(df)
            prob, signal = generate_ml_signal(df)
            last = df.iloc[-1]
            price = last["close"]

            if signal and position is None:
                position = "LONG"
                entry_price = price
                msg = f"🔵 Señal ML de COMPRA BTC a {price:.2f} USD (confianza: {prob*100:.2f}%)"
                send_telegram_message(msg)
                print(msg)

            elif position == "LONG":
                stop_loss = entry_price * 0.90
                take_profit = entry_price * 1.50

                if price <= stop_loss:
                    msg = f"🔴 VENTA por STOP LOSS a {price:.2f} USD"
                    send_telegram_message(msg)
                    print(msg)
                    position = None
                    capital *= price / entry_price

                elif price >= take_profit:
                    msg = f"🟢 VENTA por TAKE PROFIT a {price:.2f} USD"
                    send_telegram_message(msg)
                    print(msg)
                    position = None
                    capital *= price / entry_price

        current_time = time.time()
        if (current_time - last_active_message_time) > active_message_interval:
            send_telegram_message("🤖 Bot activo y revisando el mercado...")
            last_active_message_time = current_time

        time.sleep(60)

if __name__ == "__main__":
    main()
