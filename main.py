import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yfinance as yf


def main():
  # ==========================================
  # 【設定エリア】ここを書き換えて自由に実験できます
  ticker = "7974.T"  # 予測したい銘柄（例: 7974任天堂　7203トヨタ 6758.T ソニー, 9984.T ソフトバンクG）
  period_data = "10y"  # データの期間（"5y", "10y", "max" など）
  # ==========================================

  fx_ticker = "USDJPY=X"
  index_ticker = "^N225"
  print(
      f"対象銘柄: {ticker} (期間: {period_data}), 为替: {fx_ticker}, 指数:"
      f" {index_ticker} のデータを取得中..."
  )

  # データの取得（設定した期間を反映）
  stock_data = yf.download(ticker, period=period_data)
  fx_data = yf.download(fx_ticker, period=period_data)
  index_data = yf.download(index_ticker, period=period_data)

  if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)
  if isinstance(fx_data.columns, pd.MultiIndex):
    fx_data.columns = fx_data.columns.get_level_values(0)
  if isinstance(index_data.columns, pd.MultiIndex):
    index_data.columns = index_data.columns.get_level_values(0)

  data = pd.DataFrame()
  data["Close"] = stock_data["Close"]
  data["Volume"] = stock_data["Volume"]
  data["FX_Close"] = fx_data["Close"]
  data["Index_Close"] = index_data["Close"]

  data = data.dropna()

  # 特徴量の作成（＋0.5%以上のハードル）
  data["Target"] = (
      data["Close"].shift(-1) > data["Close"] * 1.005
  ).astype(int)

  # トヨタ（対象銘柄）のテクニカル指標
  data["Return"] = data["Close"].pct_change()
  data["MA5"] = data["Close"].rolling(window=5).mean()
  data["MA25"] = data["Close"].rolling(window=25).mean()

  # RSI（14日）
  delta = data["Close"].diff()
  gain = delta.where(delta > 0, 0).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  data["RSI"] = 100 - (100 / (1 + rs))

  # ボリンジャーバンド
  ma20 = data["Close"].rolling(window=20).mean()
  std20 = data["Close"].rolling(window=20).std()
  data["BB_Upper"] = ma20 + (std20 * 2)
  data["BB_Lower"] = ma20 - (std20 * 2)
  data["BB_Width"] = (data["BB_Upper"] - data["BB_Lower"]) / ma20

  data["Volume_Return"] = data["Volume"].pct_change()

  # 為替と日経平均の指標
  data["FX_Return"] = data["FX_Close"].pct_change()
  data["FX_MA5"] = data["FX_Close"].rolling(window=5).mean()
  data["Index_Return"] = data["Index_Close"].pct_change()
  data["Index_MA5"] = data["Index_Close"].rolling(window=5).mean()

  data = data.dropna()

  features = [
      "Return",
      "MA5",
      "MA25",
      "RSI",
      "BB_Width",
      "Volume_Return",
      "FX_Return",
      "FX_MA5",
      "Index_Return",
      "Index_MA5",
  ]
  X = data[features]
  y = data["Target"]

  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, shuffle=False
  )

  print("モデルを訓練中...")
  model = lgb.LGBMClassifier(
      n_estimators=150, learning_rate=0.03, random_state=42, verbose=-1
  )
  model.fit(X_train, y_train)

  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  print(f"【{ticker}】テストデータの予測精度: {accuracy * 100:.2f}%")

  latest_data = X.tail(1)
  next_pred = model.predict(latest_data)
  result_msg = (
      f"【AIの予測】次営業日の {ticker} は ... 【 ＋0.5%以上 上がる"
      " 】と予測されます！"
      if next_pred[0] == 1
      else (
          f"【AIの予測】次営業日の {ticker} は ... 【 大きな上昇は見込めない（横ばい・下落）"
          " 】と予測されます！"
      )
  )
  print(result_msg)


if __name__ == "__main__":
  main()
