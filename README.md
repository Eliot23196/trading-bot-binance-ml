# Trading Bot Binance ML

Bot de trading automático en tiempo real para Bitcoin (BTC/USDT), conectado a Binance y Telegram. Usa un modelo de Machine Learning para generar señales de compra y venta basadas en indicadores técnicos.

---

##Tecnologías usadas

- Python
- Binance API
- Telegram Bot API
- Scikit-learn (`joblib`)
- TA-Lib (con `ta` library)
- Pandas
- requests
- dotenv

---

##Funcionalidades

- Descarga datos OHLC cada minuto de Binance
- Calcula indicadores técnicos: RSI, SMA, MACD, Bollinger Bands, ATR
- Usa un modelo ML entrenado (Random Forest) para predecir señales
- Envia señales a un canal privado de Telegram
- Tiene lógica de take profit y stop loss
- Usa `dotenv` para proteger claves API



