import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import yfinance as yf

warnings.filterwarnings("ignore")


def download_clean_data(ticker, period):
    """yfinanceからデータを取得し、MultiIndexを解除して返す"""
    df = yf.download(ticker, period=period, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def compute_rsi(series, window=14):
    """指数移動平均（Wilder法近似）ベースのRSIを計算"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def main():
    # ==========================================
    # 【設定エリア】ここを書き換えて自由に実験できます
    ticker = "7974.T"          # 予測したい銘柄（例: 7974.T 任天堂, 7203.T トヨタ, 6758.T ソニー）
    period_data = "10y"        # データの期間（"5y", "10y", "max" など）
    target_threshold = 0.005   # 目標上昇率（0.005 = +0.5%以上の上昇を「1」とする）
    prob_threshold = 0.52      # 「買い」と判定するAIの予測確率しきい値
    # ==========================================

    fx_ticker = "USDJPY=X"
    index_ticker = "^N225"
    us_ticker = "^GSPC"  # 米国S&P500指数

    print(
        f"対象銘柄: {ticker} (期間: {period_data})\n"
        f"外部指標: 為替({fx_ticker}), 日経平均({index_ticker}), 米国S&P500({us_ticker}) を取得中..."
    )

    # 1. データの取得と日付インデックスの統合（日本株の営業日を基準に前方補完）
    stock_data = download_clean_data(ticker, period_data)
    fx_data = download_clean_data(fx_ticker, period_data)
    index_data = download_clean_data(index_ticker, period_data)
    us_data = download_clean_data(us_ticker, period_data)

    data = pd.DataFrame(index=stock_data.index)
    data["Open"] = stock_data["Open"]
    data["High"] = stock_data["High"]
    data["Low"] = stock_data["Low"]
    data["Close"] = stock_data["Close"]
    data["Volume"] = stock_data["Volume"]

    # 海外市場や為替の休場日ズレを日本株の営業日に合わせて補完
    data["FX_Close"] = fx_data["Close"].reindex(data.index).ffill()
    data["Index_Close"] = index_data["Close"].reindex(data.index).ffill()
    data["US_Close"] = us_data["Close"].reindex(data.index).ffill()

    data = data.dropna()

    # 2. 特徴量エンジニアリング（すべて価格水準に依存しない「比率・変化率」に変換）
    # --- 対象銘柄の基本リターンとラグ特徴量 ---
    data["Return_1d"] = data["Close"].pct_change(1)
    data["Return_2d"] = data["Close"].pct_change(2)
    data["Return_3d"] = data["Close"].pct_change(3)
    data["Return_5d"] = data["Close"].pct_change(5)

    # --- 日中値動き（ギャップ・ヒゲ・ボラティリティ） ---
    data["Intraday_Return"] = (data["Close"] - data["Open"]) / data["Open"]
    data["High_Low_Range"] = (data["High"] - data["Low"]) / data["Close"]
    data["Volatility_5d"] = data["Return_1d"].rolling(5).std()
    data["Volatility_20d"] = data["Return_1d"].rolling(20).std()

    # --- 移動平均乖離率（生のMAではなく乖離率にすることで10年前の価格とも比較可能にする） ---
    ma5 = data["Close"].rolling(5).mean()
    ma25 = data["Close"].rolling(25).mean()
    ma75 = data["Close"].rolling(75).mean()
    data["MA5_Diff"] = (data["Close"] - ma5) / ma5
    data["MA25_Diff"] = (data["Close"] - ma25) / ma25
    data["MA75_Diff"] = (data["Close"] - ma75) / ma75

    # --- RSI & MACD（株価比率で正規化） ---
    data["RSI_14"] = compute_rsi(data["Close"], 14)
    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    data["MACD_Hist_Ratio"] = (macd - macd_signal) / data["Close"]

    # --- ボリンジャーバンド（バンド幅とバンド内の相対位置 %B） ---
    ma20 = data["Close"].rolling(20).mean()
    std20 = data["Close"].rolling(20).std()
    bb_upper = ma20 + (std20 * 2)
    bb_lower = ma20 - (std20 * 2)
    data["BB_Width"] = (bb_upper - bb_lower) / ma20
    data["BB_Position"] = (data["Close"] - bb_lower) / (bb_upper - bb_lower)

    # --- 出来高の変化（5日平均に対する当日出来高の倍率） ---
    vol_ma5 = data["Volume"].rolling(5).mean()
    data["Volume_Ratio_MA5"] = data["Volume"] / vol_ma5.replace(0, np.nan)

    # --- 為替・日経平均・米国株の指標（乖離率とリターン） ---
    data["FX_Return_1d"] = data["FX_Close"].pct_change(1)
    fx_ma5 = data["FX_Close"].rolling(5).mean()
    data["FX_MA5_Diff"] = (data["FX_Close"] - fx_ma5) / fx_ma5

    data["Index_Return_1d"] = data["Index_Close"].pct_change(1)
    index_ma5 = data["Index_Close"].rolling(5).mean()
    data["Index_MA5_Diff"] = (data["Index_Close"] - index_ma5) / index_ma5

    # 個別株と日経平均の強弱差（相対モメンタム）
    data["Relative_Strength_Index"] = data["Return_1d"] - data["Index_Return_1d"]

    # 米国株（S&P500）のリターン
    data["US_Return_1d"] = data["US_Close"].pct_change(1)
    us_ma5 = data["US_Close"].rolling(5).mean()
    data["US_MA5_Diff"] = (data["US_Close"] - us_ma5) / us_ma5

    # --- カレンダー特徴量（曜日: 月曜=0 〜 金曜=4） ---
    data["DayOfWeek"] = data.index.dayofweek

    # 3. 目的変数（翌営業日のリターン）の作成
    data["Next_Return"] = data["Close"].shift(-1) / data["Close"] - 1.0

    features = [
        "Return_1d", "Return_2d", "Return_3d", "Return_5d",
        "Intraday_Return", "High_Low_Range", "Volatility_5d", "Volatility_20d",
        "MA5_Diff", "MA25_Diff", "MA75_Diff",
        "RSI_14", "MACD_Hist_Ratio", "BB_Width", "BB_Position",
        "Volume_Ratio_MA5",
        "FX_Return_1d", "FX_MA5_Diff",
        "Index_Return_1d", "Index_MA5_Diff", "Relative_Strength_Index",
        "US_Return_1d", "US_MA5_Diff",
        "DayOfWeek",
    ]

    # 特徴量が欠損している初期行を削除
    data = data.dropna(subset=features).copy()

    # 【重要】最新営業日（翌日の終値がまだ存在しない行）と、学習・評価用データを分離
    latest_row = data.iloc[[-1]].copy()
    historical_data = data.iloc[:-1].dropna(subset=["Next_Return"]).copy()
    historical_data["Target"] = (historical_data["Next_Return"] > target_threshold).astype(int)

    X = historical_data[features]
    y = historical_data["Target"]
    next_returns = historical_data["Next_Return"]

    # 4. 時系列に沿ったデータ分割（学習70%、検証15%［Early Stopping用］、テスト15%）
    n = len(X)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]
    test_returns = next_returns.iloc[val_end:]

    # クラス不均衡への補正（+0.5%以上上がる日は全体の3〜4割程度のため重みを調整）
    pos_weight = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    print("モデルを訓練中（過学習防止のEarly Stopping適用）...")
    model = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=15,          # 浅めの木にして株価ノイズへの過学習を抑制
        max_depth=4,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )

    # 5. テストデータでの多角的な精度検証
    pred_probs = model.predict_proba(X_test)[:, 1]
    predictions = (pred_probs >= prob_threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    auc = roc_auc_score(y_test, pred_probs)

    print("\n" + "=" * 55)
    print(f"【{ticker}】テストデータ評価結果（期間: {X_test.index[0].date()} 〜 {X_test.index[-1].date()}）")
    print("=" * 55)
    print(f"・全体正解率 (Accuracy)         : {accuracy * 100:.2f}%")
    print(f"・上昇予測の的中率 (Precision)  : {precision * 100:.2f}% （実際の平均上昇確率: {y_test.mean() * 100:.2f}%）")
    print(f"・上昇日の捕捉率 (Recall)       : {recall * 100:.2f}%")
    print(f"・ROC-AUCスコア                 : {auc:.4f} （0.50以上なら予測力あり）")

    # 簡易バックテスト（AIが「上がる」と予測した日だけ翌日まで保有した場合の平均リターン）
    selected_days = test_returns[predictions == 1]
    if len(selected_days) > 0:
        win_rate = (selected_days > 0).mean() * 100
        avg_ret = selected_days.mean() * 100
        print(f"・AIエントリー回数              : {len(selected_days)}回 / 全{len(X_test)}営業日")
        print(f"・エントリー時のプラス勝率      : {win_rate:.2f}% （1回あたり平均リターン: {avg_ret:+.3f}%）")
    else:
        print("・AIエントリー回数              : 0回（prob_threshold を下げてみてください）")

    # 6. 重要度の高い特徴量トップ5を表示
    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    print("\n【AIが重視した特徴量 TOP 5】")
    for i, (feat, imp) in enumerate(importances.head(5).items(), 1):
        print(f"  {i}. {feat:<25} (重要度スコア: {imp})")

    # 7. 次営業日の予測
    latest_X = latest_row[features]
    latest_date = latest_row.index[0].strftime("%Y-%m-%d")
    next_prob = model.predict_proba(latest_X)[0, 1]
    next_pred = int(next_prob >= prob_threshold)

    print("\n" + "=" * 55)
    print(f"【最新データ基準日: {latest_date} 終値: {latest_row['Close'].values[0]:,.1f}円】")
    print(f"・次営業日に +{target_threshold * 100:.1f}% 以上上昇するAI予測確率: 【 {next_prob * 100:.1f}% 】")
    if next_pred == 1:
        print(f"👉 【AIの判定】次営業日の {ticker} は 【 ＋{target_threshold * 100:.1f}%以上 上がる 】可能性が高いと予測されます！")
    else:
        print(f"👉 【AIの判定】次営業日の {ticker} は 【 大きな上昇は見込めない（様子見） 】と予測されます。")
    print("=" * 55)


if __name__ == "__main__":
    main()
