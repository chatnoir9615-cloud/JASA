"""JPX指数銘柄スクリーニング。

- data/nikkei225.csv から手動管理の構成銘柄リストを読み込む
- ファンダメンタルズ条件で【資産】【収益】【高配当？】タグを付与
- OHLCVキャッシュは MarketDataCache（market_cache.py）に統一
- 年1回（10月）の銘柄入れ替え時のみ data/nikkei225.csv を手動更新
"""

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from .market_cache import MarketDataCache

_INDEX_CSV_PATHS = [
    "data/nikkei225.csv",   # 日経225（年2回更新）
    "data/topix100.csv",    # TOPIX100（年1回更新）
]
_INDEX_CACHE_PATH    = "cache/index_symbols_cache.json"
_SYMBOL_TTL_DAYS     = 7
_MIN_VOLUME          = 50_000
_BATCH_SIZE          = 50
_BATCH_SLEEP         = 2
_REBUILD_PERIOD_DAYS = 365

_JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)

# ── ファンダメンタルズタグ判定条件 ───────────────────────────────────────
_ASSET_COND = {
    "per": {"max": 12.0},
    "pbr": {"max": 0.5},
    "equity_ratio": {"min": 60.0},
}
_PROFIT_COND = {
    "per":              {"max": 10.0},
    "pbr":              {"max": 1.5},
    "roa":              {"min": 7.0},
    "operating_margin": {"min": 10.0},
    "market_cap_m":     {"max": 30_000},
}


def _fmt(val, suffix="", none_label="無") -> str:
    if val is None:
        return none_label
    return f"{val}{suffix}"


class StockScreener:
    def __init__(self, cache: MarketDataCache | None = None):
        self.cache = cache or MarketDataCache()

    # ── 指数構成銘柄リスト取得（CSV手動管理） ────────────────────────────

    def get_index_symbols(self) -> tuple[list[str], dict[str, str]]:
        """data/*.csv から銘柄リストを読み込んで返す（重複除去）。

        対応ファイル: data/nikkei225.csv, data/topix100.csv
        いずれも存在しない場合は JPX全銘柄リストにフォールバックする。
        更新タイミング: 年1〜2回（各指数の銘柄入れ替え時）に手動更新。
        """
        all_symbols: list[str] = []
        all_names: dict[str, str] = {}
        loaded_any = False

        for csv_path in _INDEX_CSV_PATHS:
            if not os.path.exists(csv_path):
                logging.debug(f"{csv_path} が見つかりません。スキップします。")
                continue
            try:
                df = pd.read_csv(csv_path, dtype={"code": str}, comment="#")
                df["code"] = df["code"].str.strip().str.zfill(4)
                for _, row in df.iterrows():
                    symbol = f"{row['code']}.T"
                    if symbol not in all_symbols:
                        all_symbols.append(symbol)
                    all_names[symbol] = row.get("name", "")
                loaded_any = True
                logging.info(f"指数銘柄CSVを読み込み: {csv_path} ({len(df)}銘柄)")
            except Exception as e:
                logging.error(f"{csv_path} の読み込み失敗: {e}")

        if not loaded_any:
            logging.warning("指数銘柄CSVが見つかりません。JPX全銘柄にフォールバックします。")
            return self._fallback_all_symbols()

        logging.info(f"指数銘柄リスト合計: {len(all_symbols)}銘柄（重複除去済み）")
        return all_symbols, all_names

    def _fallback_all_symbols(self) -> tuple[list[str], dict[str, str]]:
        """CSVなし・読み込み失敗時のフォールバック: JPX全銘柄リストを使用。"""
        try:
            df = pd.read_excel(_JPX_LIST_URL, header=0)
            code_col  = [c for c in df.columns if "コード" in str(c)][0]
            name_cols = [c for c in df.columns if "銘柄名" in str(c) or "名称" in str(c)]
            codes   = df[code_col].dropna().astype(str).str.strip()
            symbols = [f"{code}.T" for code in codes]
            names: dict[str, str] = {}
            if name_cols:
                for code, name in zip(codes, df[name_cols[0]].fillna("")):
                    names[f"{code}.T"] = str(name).strip()
            logging.info(f"フォールバック: JPX全銘柄 {len(symbols)}銘柄")
            return symbols, names
        except Exception as e:
            logging.error(f"フォールバックも失敗: {e}")
            return [], {}

    # ── ファンダメンタルズ取得・タグ判定 ────────────────────────────────

    def _fetch_fundamentals(self, symbol: str) -> dict:
        result = {
            "per": None, "pbr": None, "equity_ratio": None,
            "roa": None, "operating_margin": None,
            "market_cap_m": None, "dividend_yield": None,
        }
        try:
            info = yf.Ticker(symbol).info or {}

            per = info.get("trailingPE") or info.get("forwardPE")
            if per:
                result["per"] = round(float(per), 1)

            pbr = info.get("priceToBook")
            if pbr:
                result["pbr"] = round(float(pbr), 2)

            dte = info.get("debtToEquity")
            if dte is not None and float(dte) >= 0:
                result["equity_ratio"] = round(100 / (1 + float(dte) / 100), 1)

            roa = info.get("returnOnAssets")
            if roa:
                result["roa"] = round(float(roa) * 100, 1)

            om = info.get("operatingMargins")
            if om:
                result["operating_margin"] = round(float(om) * 100, 1)

            mc = info.get("marketCap")
            if mc:
                result["market_cap_m"] = round(float(mc) / 1_000_000, 0)

            dy = info.get("dividendYield")
            if dy:
                raw = float(dy)
                result["dividend_yield"] = round(raw if raw > 1.0 else raw * 100, 2)

        except Exception as e:
            logging.debug(f"ファンダメンタルズ取得失敗 [{symbol}]: {e}")

        return result

    def _judge_tags(self, fund: dict) -> list[str]:
        tags = []
        per = fund.get("per")
        pbr = fund.get("pbr")
        er  = fund.get("equity_ratio")
        dy  = fund.get("dividend_yield")

        # 【資産】
        asset_ok = True
        if per is not None and per > _ASSET_COND["per"]["max"]:
            asset_ok = False
        if pbr is not None and pbr > _ASSET_COND["pbr"]["max"]:
            asset_ok = False
        if er is not None and er < _ASSET_COND["equity_ratio"]["min"]:
            asset_ok = False
        if asset_ok:
            tags.append("資産")

        # 【収益】
        profit_ok = True
        roa = fund.get("roa")
        om  = fund.get("operating_margin")
        mc  = fund.get("market_cap_m")
        if per is not None and per > _PROFIT_COND["per"]["max"]:
            profit_ok = False
        if pbr is not None and pbr > _PROFIT_COND["pbr"]["max"]:
            profit_ok = False
        if roa is not None and roa < _PROFIT_COND["roa"]["min"]:
            profit_ok = False
        if om is not None and om < _PROFIT_COND["operating_margin"]["min"]:
            profit_ok = False
        if mc is not None and mc > _PROFIT_COND["market_cap_m"]["max"]:
            profit_ok = False
        if profit_ok:
            tags.append("収益")

        # 【高配当？】: 配当利回り ≥ 3.8% かつ PER ≤ 18倍のみ付与
        dy_ok  = dy is not None and dy >= 3.8
        per_ok = per is not None and per <= 18.0
        if dy_ok and per_ok:
            tags.append("高配当？")

        return tags

    def _format_fund_label(self, fund: dict, tags: list[str]) -> str:
        per = _fmt(fund.get("per"), "倍")
        pbr = _fmt(fund.get("pbr"), "倍")
        er  = _fmt(fund.get("equity_ratio"), "%")
        roa = _fmt(fund.get("roa"), "%")
        om  = _fmt(fund.get("operating_margin"), "%")
        mc  = fund.get("market_cap_m")
        mc_str = f"{int(mc):,}百万円" if mc is not None else "無"
        dy  = _fmt(fund.get("dividend_yield"), "%")
        tag_str = "".join(f"【{t}】" for t in tags) if tags else "【-】"
        return (
            f"{tag_str}\n"
            f"  PER:{per} / PBR:{pbr} / 自己資本比率:{er}\n"
            f"  ROA:{roa} / 営業利益率:{om} / 時価総額:{mc_str}\n"
            f"  配当利回り:{dy}"
        )

    # ── キャッシュ構築のみ（rebuild_cacheモード用） ──────────────────────

    def build_cache_only(self, exclude_symbols: set[str]) -> None:
        """全銘柄のOHLCVキャッシュ構築のみ実行。rebuild_cacheモード専用。"""
        try:
            df = pd.read_excel(_JPX_LIST_URL, header=0)
            code_col = [c for c in df.columns if "コード" in str(c)][0]
            codes    = df[code_col].dropna().astype(str).str.strip()
            symbols  = [f"{code}.T" for code in codes]
        except Exception as e:
            logging.error(f"JPX銘柄リスト取得失敗: {e}")
            return
        exclude_normalized = {s if s.endswith(".T") else f"{s}.T" for s in exclude_symbols}
        targets = [s for s in symbols if s not in exclude_normalized]
        logging.info(f"キャッシュ構築対象: {len(targets)}銘柄（除外後）")
        self._batch_fetch_rebuild(targets)
        logging.info("キャッシュ構築バッチ処理完了")

    # ── メインスクリーニング ─────────────────────────────────────────────

    def screen(self, exclude_symbols: set[str], top_n: int = 10) -> list[dict]:
        symbols, name_map = self.get_index_symbols()
        if not symbols:
            return []
        exclude_normalized = {s if s.endswith(".T") else f"{s}.T" for s in exclude_symbols}
        targets = [s for s in symbols if s not in exclude_normalized]
        logging.info(f"スクリーニング対象: {len(targets)}銘柄（除外後）")
        self._batch_fetch(targets)
        candidates = []
        for symbol in targets:
            result = self._evaluate(symbol, name_map)
            if result:
                candidates.append(result)
        top = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
        logging.info(f"スクリーニング完了: {len(candidates)}銘柄中 上位{len(top)}銘柄を選出")
        return top

    # ── バッチ取得 ───────────────────────────────────────────────────────

    def _batch_fetch(self, symbols: list[str]) -> None:
        stale = [s for s in symbols if self.cache.needs_update(s)]
        logging.info(f"差分取得対象: {len(stale)}銘柄 / {len(symbols)}銘柄中")
        if not stale:
            return
        groups: dict[date, list[str]] = {}
        default_start = date.today() - timedelta(days=90)
        for s in stale:
            last  = self.cache.last_date(s)
            start = (last + timedelta(days=1)) if last else default_start
            groups.setdefault(start, []).append(s)
        for start_date, group_symbols in sorted(groups.items()):
            self._download_and_cache(group_symbols, start_date)

    def _batch_fetch_rebuild(self, symbols: list[str]) -> None:
        today     = date.today()
        no_cache  = [s for s in symbols if self.cache.last_date(s) is None]
        has_cache = [s for s in symbols if self.cache.last_date(s) is not None]
        if no_cache:
            start_date = today - timedelta(days=_REBUILD_PERIOD_DAYS)
            logging.info(f"[rebuild] キャッシュなし銘柄: {len(no_cache)}銘柄 ({start_date} 〜 {today})")
            self._download_and_cache(no_cache, start_date)
        stale_with_cache = [s for s in has_cache if self.cache.needs_update(s)]
        if stale_with_cache:
            last_dates = [self.cache.last_date(s) for s in stale_with_cache]
            start_date = min(last_dates) + timedelta(days=1)
            logging.info(f"[rebuild] 差分取得対象: {len(stale_with_cache)}銘柄 ({start_date} 〜 {today})")
            self._download_and_cache(stale_with_cache, start_date)
        already_fresh = len(has_cache) - len(stale_with_cache)
        if already_fresh > 0:
            logging.info(f"[rebuild] 最新済みのためスキップ: {already_fresh}銘柄")

    def _download_and_cache(self, symbols: list[str], start_date: date) -> None:
        start_str = start_date.isoformat()
        end_str   = date.today().isoformat()
        if start_str > end_str:
            return
        for i in range(0, len(symbols), _BATCH_SIZE):
            batch = symbols[i : i + _BATCH_SIZE]
            try:
                raw = yf.download(
                    tickers=batch, start=start_str, end=end_str,
                    interval="1d", group_by="ticker",
                    auto_adjust=True, progress=False, threads=True,
                )
            except Exception as e:
                logging.warning(f"バッチダウンロード失敗 [{i}〜{i+_BATCH_SIZE}]: {e}")
                time.sleep(_BATCH_SLEEP)
                continue
            for symbol in batch:
                try:
                    df = raw[symbol].copy() if len(batch) > 1 else raw.copy()
                    df = df.dropna()
                    if not df.empty:
                        self.cache.update(symbol, df)
                except Exception:
                    continue
            logging.info(f"バッチ取得完了: {i+len(batch)}/{len(symbols)}")
            time.sleep(_BATCH_SLEEP)

    # ── 個別銘柄評価 ────────────────────────────────────────────────────

    def _evaluate(self, symbol: str, name_map: dict[str, str]) -> dict | None:
        try:
            df = self.cache.get_dataframe(symbol)
            if df is None or df.empty or len(df) < 25:
                return None
            df = df.copy().sort_index()
            current_price = round(float(df["Close"].iloc[-1]), 1)

            avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
            if avg_volume < _MIN_VOLUME:
                return None

            rsi          = ta.rsi(df["Close"], length=14).iloc[-1]
            atr          = ta.atr(df["High"], df["Low"], df["Close"], length=14).iloc[-1]
            ma25_series  = df["Close"].rolling(25).mean()
            ma25         = ma25_series.iloc[-1]
            ma25_diff    = round(((current_price - ma25) / ma25) * 100, 2)
            volume_ratio = round(float(df["Volume"].iloc[-1]) / avg_volume, 2)

            if len(ma25_series.dropna()) >= 6:
                if ma25 <= ma25_series.dropna().iloc[-6]:
                    return None

            score = 0.0
            score += min(volume_ratio / 2.0, 1.0) * 40
            if 25 <= rsi <= 45:
                score += (1 - abs(rsi - 35) / 10) * 30
            if -10 <= ma25_diff <= -3:
                score += (1 - abs(ma25_diff + 6.5) / 6.5) * 30

            fund       = self._fetch_fundamentals(symbol)
            tags       = self._judge_tags(fund)
            fund_label = self._format_fund_label(fund, tags)

            display_name = name_map.get(symbol) or symbol.replace(".T", "")
            return {
                "symbol":         symbol.replace(".T", ""),
                "name":           display_name,
                "price":          current_price,
                "score":          round(score, 2),
                "category_label": "スクリーニング",
                "is_held":        False,
                "purchase_price": None,
                "pl_rate":        0,
                "fund_tags":      tags,
                "fund_label":     fund_label,
                "metrics": {
                    "RSI":      round(float(rsi), 1),
                    "ATR":      round(float(atr), 1),
                    "MA25乖離": ma25_diff,
                    "突破":     current_price > df["High"].rolling(20).max().iloc[-2],
                    "出来高比": volume_ratio,
                },
                "fundamentals": fund,
            }
        except Exception as e:
            logging.debug(f"スキップ [{symbol}]: {e}")
            return None