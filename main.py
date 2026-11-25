import datetime
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yfinance as yf


def main():
  # 1. データの取得（日本株 ＋ ドル円 ＋ 日経平均）
  ticker = "7203.T"
  fx_ticker = "USDJPY=X"
  index_ticker = "^N225"
  print(f"{ticker}, {fx_ticker}, {index_ticker} のデータを取得中...")

  # それぞれのデータを取得
  stock_data = yf.download(ticker, period="5y")
  fx_data = yf.download(fx_ticker, period="5y")
  index_data = yf.download(index_ticker, period="5y")

  # マルチインデックスの調整
  if isinstance(stock_data.columns, pd.MultiIndex):
    stock_data.columns = stock_data.columns.get_level_values(0)
  if isinstance(fx_data.columns, pd.MultiIndex):
    fx_data.columns = fx_data.columns.get_level_values(0)
  if isinstance(index_data.columns, pd.MultiIndex):
    index_data.columns = index_data.columns.get_level_values(0)

  # ひとつのデータフレームにまとめる
  data = pd.DataFrame()
  data["Close"] = stock_data["Close"]
  data["FX_Close"] = fx_data["Close"]
  data["Index_Close"] = index_data["Close"]

  # 欠損値を削除
  data = data.dropna()

  # 2. 特徴量（ヒント）の作成
  # ★【変更点】翌日の終値が「当日の終値 × 1.005（＋0.5%以上）」より高いときだけ1にする
  data["Target"] = (
      data["Close"].shift(-1) > data["Close"] * 1.005
  ).astype(int)

  # ① トヨタ株のテクニカル指標
  data["Return"] = data["Close"].pct_change()
  data["MA5"] = data["Close"].rolling(window=5).mean()
  data["MA25"] = data["Close"].rolling(window=25).mean()

  # RSI（14日）
  delta = data["Close"].diff()
  gain = delta.where(delta > 0, 0).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  data["RSI"] = 100 - (100 / (1 + rs))

  # ② ドル円為替の指標
  data["FX_Return"] = data["FX_Close"].pct_change()
  data["FX_MA5"] = data["FX_Close"].rolling(window=5).mean()

  # ③ 日経平均の指標
  data["Index_Return"] = data["Index_Close"].pct_change()
  data["Index_MA5"] = data["Index_Close"].rolling(window=5).mean()

  # 欠損値を削除
  data = data.dropna()

  # 使用する特徴量
  features = [
      "Return",
      "MA5",
      "MA25",
      "RSI",
      "FX_Return",
      "FX_MA5",
      "Index_Return",
      "Index_MA5",
  ]
  X = data[features]
  y = data["Target"]

  # 3. データの分割（訓練データとテストデータ）
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, shuffle=False
  )

  # 4. 機械学習モデルの学習（LightGBMを使用）
  print("モデルを訓練中（予測のハードルを＋0.5%に設定しました）...")
  model = lgb.LGBMClassifier(random_state=42, verbose=-1)
  model.fit(X_train, y_train)

  # 5. 予測と評価
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  print(f"テストデータの予測精度: {accuracy * 100:.2f}%")

  # 最新のデータを使った翌日の予測
  latest_data = X.tail(1)
  next_pred = model.predict(latest_data)
  result_text = "＋0.5%以上の上昇" "横ばいまたは下落"  # 調整
  result_msg = (
      "【AIの予測】次営業日の 7203.T は ... 【 ＋0.5%以上 上がる 】と予測されます！"
      if next_pred[0] == 1
      else "【AIの予測】次営業日の 7203.T は ... 【 大きな上昇は見込めない（横ばい・下落） 】と予測されます！"
  )
  print(result_msg)


if __name__ == "__main__":
  main()
