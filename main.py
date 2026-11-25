import datetime
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import yfinance as yf


def main():
  # 1. データの取得（例：トヨタ自動車 「7203.T」）
  ticker = "7203.T"
  print(f"{ticker} の株価データを取得中...")

  # 過去5年分のデータを取得
  data = yf.download(ticker, period="5y")

  # yfinanceのバージョンによってカラム構造が変わる場合があるため調整
  if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

  # 2. 特徴量（ヒント）の作成
  # 翌日の終値が前日より上がる（1）か下がる（0）かを予測ターゲットにする
  data["Target"] = (data["Close"].shift(-1) > data["Close"]).astype(int)

  # ① 変動率と移動平均線
  data["Return"] = data["Close"].pct_change()
  data["MA5"] = data["Close"].rolling(window=5).mean()
  data["MA25"] = data["Close"].rolling(window=25).mean()

  # ② RSI（14日）の計算を追加
  delta = data["Close"].diff()
  gain = delta.where(delta > 0, 0).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  data["RSI"] = 100 - (100 / (1 + rs))

  # 欠損値を削除
  data = data.dropna()

  # 使用する特徴量を増やす（Return, MA5, MA25に加え、RSIを投入）
  features = ["Return", "MA5", "MA25", "RSI"]
  X = data[features]
  y = data["Target"]

  # 3. データの分割（訓練データとテストデータ）
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, shuffle=False
  )

  # 4. 機械学習モデルの学習（ランダムフォレスト）
  print("モデルを訓練中（RSIを追加しました）...")
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # 5. 予測と評価
  predictions = model.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)

  print(f"テストデータの予測精度: {accuracy * 100:.2f}%")

  # 最新のデータを使った翌日の予測
  latest_data = X.tail(1)
  next_pred = model.predict(latest_data)
  result_text = "値上がり" if next_pred[0] == 1 else "値下がり"
  print(f"【AIの予測】次営業日の {ticker} は ... 【 {result_text} 】と予測されます！")


if __name__ == "__main__":
  main()
