"""市場データ（OHLCV）の永続キャッシュ。

設計方針:
  - 過去の確定済み OHLCV は変わらないため永続保存（再取得しない）
  - 取得対象は「キャッシュの最終日の翌日〜最新確定日」の差分のみ
  - 最新確定日の判定: 東証大引け（15:30 JST）以降なら当日、それ以前なら前営業日
  - 週末は遡って直近金曜を最終確定日とする（祝日は未考慮）
"""

import json
import logging
import os
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import pandas as pd

JST = timezone(timedelta(hours=9))
_MARKET_CLOSE = time(15, 30)


class MarketDataCache:
    def __init__(self, cache_path: str = "cache/ohlcv_cache.json"):
        self.cache_path = cache_path
        # {symbol: {"last_date": "YYYY-MM-DD", "records": [{"Date":…,"Open":…,…}]}}
        self._data: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_dataframe(self, symbol: str) -> pd.DataFrame | None:
        """キャッシュされた OHLCV を DataFrame で返す。なければ None。"""
        entry = self._data.get(symbol)
        if not entry or not entry.get("records"):
            return None
        df = pd.DataFrame(entry["records"])
        df["Date"] = pd.to_datetime(df["Date"])
        return df.set_index("Date").sort_index()

    def last_date(self, symbol: str) -> date | None:
        """キャッシュの最終日付を返す。"""
        entry = self._data.get(symbol)
        last = entry.get("last_date") if entry else None
        return date.fromisoformat(last) if last else None

    def needs_update(self, symbol: str) -> bool:
        """差分取得が必要かどうかを返す。"""
        last = self.last_date(symbol)
        if last is None:
            return True
        return last < self._last_confirmed_date()

    def update(self, symbol: str, df_new: pd.DataFrame) -> None:
        """新規行を既存キャッシュに追記して保存する。"""
        if df_new is None or df_new.empty:
            return

        new_records = self._df_to_records(df_new)
        existing = self._data.get(symbol, {}).get("records", [])

        # 日付キーで重複除去してマージ
        merged = {r["Date"]: r for r in existing}
        for r in new_records:
            merged[r["Date"]] = r
        sorted_records = sorted(merged.values(), key=lambda r: r["Date"])

        self._data[symbol] = {
            "last_date": sorted_records[-1]["Date"] if sorted_records else None,
            "records":   sorted_records,
        }
        self._save()
        logging.info(f"[キャッシュ更新] {symbol}: +{len(new_records)}行 / 合計 {len(sorted_records)}行")

    def stats(self) -> dict:
        total = len(self._data)
        fresh = sum(1 for sym in self._data if not self.needs_update(sym))
        return {"total": total, "fresh": fresh, "stale": total - fresh}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _last_confirmed_date(self) -> date:
        """最新の確定済み終値の日付（東証基準）を返す。

        - 15:30 以降 → 当日
        - 15:30 前   → 前営業日
        - 土日・日本の祝日は遡って直近営業日を返す
        """
        try:
            import jpholiday
            def is_holiday(d: date) -> bool:
                return d.weekday() >= 5 or jpholiday.is_holiday(d)
        except ImportError:
            logging.warning("jpholiday が未インストールです。祝日を考慮せずに営業日を判定します。")
            def is_holiday(d: date) -> bool:
                return d.weekday() >= 5

        now = datetime.now(JST)
        target = now.date() if now.time() >= _MARKET_CLOSE else now.date() - timedelta(days=1)
        while is_holiday(target):
            target -= timedelta(days=1)
        return target

    @staticmethod
    def _df_to_records(df: pd.DataFrame) -> list[dict]:
        records = []
        for idx, row in df.iterrows():
            date_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
            records.append({
                "Date":   date_str,
                "Open":   round(float(row.get("Open",  0)), 2),
                "High":   round(float(row.get("High",  0)), 2),
                "Low":    round(float(row.get("Low",   0)), 2),
                "Close":  round(float(row.get("Close", 0)), 2),
                "Volume": int(row.get("Volume", 0)),
            })
        return records

    def _load(self) -> None:
        if not os.path.exists(self.cache_path):
            logging.info(f"OHLCVキャッシュなし（初回は全銘柄の1年分を取得します）: {self.cache_path}")
            return
        try:
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            s = self.stats()
            logging.info(
                f"OHLCVキャッシュ読み込み: {s['total']}銘柄 "
                f"（最新: {s['fresh']} / 差分必要: {s['stale']}）"
            )
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"キャッシュ読み込み失敗（空キャッシュで続行）: {e}")
            self._data = {}

    def _save(self) -> None:
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False)
            os.replace(tmp, self.cache_path)
        except OSError as e:
            logging.error(f"キャッシュ書き込み失敗: {e}")
            # FIX: 書き込み失敗時に tmp ファイルが残留しないよう削除する
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
