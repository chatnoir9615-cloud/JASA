import logging
from datetime import timedelta

import pandas as pd
import yfinance as yf
import pandas_ta as ta

from .market_cache import MarketDataCache


class StockFetcher:
    def __init__(self, cache: MarketDataCache | None = None):
        self.cache = cache

    def analyze_strategy(self, symbol: str) -> dict | None:
        try:
            df = self._get_dataframe(symbol)
            if df is None or df.empty:
                logging.debug(f"DataFrameがNoneまたは空 [{symbol}]")
                return None
            if len(df) < 30:
                logging.warning(f"データ不足でスキップ（{len(df)}<30） [{symbol}]")
                return None
            return self._compute(symbol, df)
        except Exception as e:
            logging.warning(f"analyze_strategy エラー [{symbol}]: {e}")
            return None

    # ------------------------------------------------------------------
    # DataFrame 取得（キャッシュ + 差分取得）
    # ------------------------------------------------------------------

    def _get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        try:
            # キャッシュが最新ならそのまま返す
            if self.cache and not self.cache.needs_update(symbol):
                logging.debug(f"[キャッシュ最新] {symbol}")
                return self.cache.get_dataframe(symbol)

            ticker = yf.Ticker(symbol)
            last = self.cache.last_date(symbol) if self.cache else None

            if last is not None:
                start_str = str(last + timedelta(days=1))
                logging.info(f"[差分取得] {symbol}: {start_str} 〜")
                df_new = ticker.history(start=start_str)
            else:
                logging.info(f"[初回取得] {symbol}: 1年分")
                df_new = ticker.history(period="1y")

            # マルチインデックス対応（yfinance バージョンアップ対策）
            df_new = self._normalize_dataframe(df_new, symbol)

            if df_new is None or df_new.empty:
                # 新規取得が空の場合はキャッシュにフォールバック
                cached_df = self.cache.get_dataframe(symbol) if self.cache else None
                if cached_df is not None:
                    logging.debug(f"[差分なし、キャッシュ使用] {symbol}")
                    return cached_df
                return None

            # キャッシュあり: 追記して最新の全データを返す
            if self.cache:
                self.cache.update(symbol, df_new)
                return self.cache.get_dataframe(symbol)

            # キャッシュなし: 取得データをそのまま返す
            # （FIX: 旧コードは self.cache=None のまま get_dataframe を呼ぶデッドコードだった）
            return df_new

        except Exception as e:
            logging.warning(f"_get_dataframe エラー [{symbol}]: {e}")
            # エラー時はキャッシュにフォールバック
            if self.cache:
                cached_df = self.cache.get_dataframe(symbol)
                if cached_df is not None:
                    logging.info(f"[エラーのためキャッシュ使用] {symbol}")
                    return cached_df
            return None

    @staticmethod
    def _normalize_dataframe(df: pd.DataFrame | None, symbol: str) -> pd.DataFrame | None:
        """yfinanceのバージョン差異によるマルチインデックスを正規化する。"""
        if df is None or df.empty:
            return df
        try:
            # マルチインデックスの場合（yfinance >= 0.2.x 系の一部）
            if isinstance(df.columns, pd.MultiIndex):
                # 例: ("Close", "2914.T") → "Close" に変換
                ticker_sym = symbol.upper()
                if ticker_sym in df.columns.get_level_values(1):
                    df = df.xs(ticker_sym, axis=1, level=1)
                else:
                    # シンボルが見つからない場合は最初のティッカーで試みる
                    df = df.xs(df.columns.get_level_values(1)[0], axis=1, level=1)
                logging.debug(f"[マルチインデックス正規化] {symbol}")

            # 必須カラムの存在確認
            required = {"Open", "High", "Low", "Close", "Volume"}
            missing = required - set(df.columns)
            if missing:
                logging.warning(f"必須カラム不足 {missing} [{symbol}]")
                return None

            return df
        except Exception as e:
            logging.warning(f"DataFrame正規化失敗 [{symbol}]: {e}")
            return None

    # ------------------------------------------------------------------
    # テクニカル・ファンダメンタル指標の計算
    # ------------------------------------------------------------------

    def _compute(self, symbol: str, df: pd.DataFrame) -> dict | None:
        try:
            df = df.copy()

            # Close カラムの型保証
            df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
            df = df.dropna(subset=["Close"])
            if df.empty:
                logging.warning(f"Close値がすべてNaN [{symbol}]")
                return None

            current_price = round(float(df.iloc[-1]["Close"]), 1)

            df["MA25"] = df["Close"].rolling(window=25).mean()
            ma25_val = df["MA25"].iloc[-1]
            ma25_diff = round(
                ((current_price - ma25_val) / ma25_val) * 100, 2
            ) if pd.notna(ma25_val) and ma25_val > 0 else 0.0

            df["RSI"] = ta.rsi(df["Close"], length=14)
            df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

            rsi_val = df["RSI"].iloc[-1]
            atr_val = df["ATR"].iloc[-1]

            # NaN チェック（データ不足時に指標がNaNになる場合の保護）
            if pd.isna(rsi_val) or pd.isna(atr_val):
                logging.warning(f"RSI/ATRがNaN（データ不足の可能性） [{symbol}]")
                return None

            # 当日足確定前のデータで計算するため直前足（iloc[-2]）を使用
            if len(df) < 2:
                return None
            resistance = df["High"].rolling(window=20).max().iloc[-2]
            support    = round(float(df["Low"].rolling(window=20).min().iloc[-2]), 1)

            avg_vol      = df["Volume"].rolling(20).mean().iloc[-1]
            volume_ratio = round(float(df["Volume"].iloc[-1]) / avg_vol, 2) if avg_vol > 0 else 0.0

            recent_high = float(df["High"].rolling(window=20).max().iloc[-1])
            peak_drop   = round(((current_price - recent_high) / recent_high) * 100, 2)

            # ファンダメンタルズは変動するため毎回取得
            pbr = 0
            yield_pct = "0.00%"
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info or {}
                pbr = round(float(info.get("priceToBook") or 0), 2)
                raw_yield = float(info.get('dividendYield') or 0)
                yield_pct = f"{round(raw_yield if raw_yield > 1.0 else raw_yield * 100, 2)}%"
            except Exception as e:
                logging.debug(f"ファンダメンタルズ取得失敗 [{symbol}]: {e}")

            return {
                "price": current_price,
                "metrics": {
                    "RSI":      round(float(rsi_val), 1),
                    "ATR":      round(float(atr_val), 1),
                    "MA25乖離":  ma25_diff,
                    "突破":     bool(current_price > resistance),
                    "サポート":  support,
                    "出来高比":  volume_ratio,
                    "高値反落率": peak_drop,
                },
                "fundamentals": {
                    "PBR":   pbr,
                    "利回り": yield_pct,
                },
            }
        except Exception as e:
            logging.warning(f"指標計算失敗 [{symbol}]: {e}")
            return None
