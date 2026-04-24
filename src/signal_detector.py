"""保有株の買い乗せ・利確・損切り・週末手仕舞いシグナルを判定する。

v2.2 変更点:
  - detect_all / detect_weekend に market_downtrend・vi を引数として受け取れるよう変更
    → main.py 側で取得済みの値を渡すことで API の2重呼び出しを解消
  - 引数省略時は内部で取得する（後方互換を維持）
"""

import logging
import yfinance as yf


# ── 市場ベンチマーク ─────────────────────────────────────────────────────

_NIKKEI_ETF = "1321.T"   # 日経225 ETF
_TOPIX_ETF  = "1306.T"   # TOPIX ETF
_NIKKEI_VI  = "1552.T"   # 国際のETF（日経VI連動、^VXJはyfinance非対応のため代替）


def _get_ma5_direction(ticker: str) -> bool | None:
    """指定ティッカーの5日MAが下向きなら True、上向きなら False、取得失敗なら None。"""
    try:
        df = yf.Ticker(ticker).history(period="20d")
        if df is None or len(df) < 6:
            return None
        ma5 = df["Close"].rolling(5).mean()
        return float(ma5.iloc[-1]) < float(ma5.iloc[-6])
    except Exception as e:
        logging.debug(f"MA5取得失敗 [{ticker}]: {e}")
        return None


def _is_market_downtrend() -> bool:
    """日経225 ETF と TOPIX ETF の両方の5日MAが下向きなら True。
    どちらか一方でも取得失敗の場合は False（通常モード）にフォールバック。
    """
    nikkei_down = _get_ma5_direction(_NIKKEI_ETF)
    topix_down  = _get_ma5_direction(_TOPIX_ETF)

    if nikkei_down is None or topix_down is None:
        logging.debug("マルチインデックス取得失敗: 通常モードにフォールバック")
        return False

    result = nikkei_down and topix_down
    if result:
        logging.info("市場下落トレンド確定（日経225・TOPIX 両方下向き）")
    return result


def _get_nikkei_vi() -> float:
    """日経VI連動ETF（1552.T）から直近値を取得して返す。
    取得失敗時は 20.0（通常域）を返す。
    """
    try:
        df = yf.Ticker(_NIKKEI_VI).history(period="5d")
        if df is not None and not df.empty:
            val = float(df["Close"].iloc[-1])
            logging.debug(f"日経VI代替値（1552.T）: {round(val, 2)}")
            return val
    except Exception as e:
        logging.debug(f"日経VI取得失敗: {e}")
    return 20.0


def _get_atr_multiplier(market_downtrend: bool, vi: float) -> tuple[float, float, str]:
    """市場環境に応じた ATR 乗数・損切り閾値・モードラベルを返す。

    Returns:
        (pl_threshold, atr_multiplier, mode_label)
    """
    if market_downtrend:
        return -3.0, 1.5, "（市場下落トレンド）"
    elif vi >= 25.0:
        return -3.0, 2.5, f"（高VIX: VI={round(vi, 1)}）"
    else:
        return -5.0, 2.0, ""


class SignalDetector:

    # ── 買い乗せシグナル ─────────────────────────────────────────────────

    def check_add_buy(self, res: dict) -> dict | None:
        """買い乗せ推奨シグナルを返す。条件未達なら None。

        逆ピラミッド型サイジング:
          buy_count=0（初回）→ 50%、buy_count=1 → 30%、buy_count=2 → 20%
        """
        if res.get("stage") != "half":
            return None
        if not res.get("is_held"):
            return None

        purchase_price = res.get("purchase_price", 0)
        if purchase_price <= 0:
            return None

        price        = res["price"]
        metrics      = res["metrics"]
        pl_rate      = res.get("pl_rate", 0)
        breakout     = metrics.get("突破", False)
        volume_ratio = metrics.get("出来高比", 0)
        buy_count    = res.get("buy_count", 0)

        reasons = []

        price_up = pl_rate >= 2.0
        if price_up:
            reasons.append(f"含み益+{pl_rate}%（取得単価比）")

        if breakout:
            reasons.append("直近高値ブレイク")

        volume_surge = volume_ratio >= 2.0
        if volume_surge:
            reasons.append(f"出来高急増（平均比{volume_ratio}倍）")

        if price_up and breakout and volume_surge:
            sizing_map = {0: 50, 1: 30, 2: 20}
            lot_pct = sizing_map.get(buy_count, 20)
            reasons.append(f"推奨ロット: 全体の{lot_pct}%（逆ピラミッド）")
            return {
                "type":    "ADD_BUY",
                "symbol":  res["symbol"],
                "name":    res["name"],
                "price":   price,
                "lot_pct": lot_pct,
                "reasons": reasons,
            }
        return None

    # ── 損切りシグナル ───────────────────────────────────────────────────

    def check_stop_loss(
        self,
        res: dict,
        market_downtrend: bool = False,
        vi: float = 20.0,
    ) -> dict | None:
        """損切り推奨シグナルを返す。条件未達なら None。"""
        if not res.get("is_held"):
            return None

        pl_rate  = res.get("pl_rate", 0)
        price    = res["price"]
        atr      = res["metrics"].get("ATR", 0)
        purchase = res.get("purchase_price", 0)
        if purchase <= 0:
            return None

        pl_threshold, atr_multiplier, mode_label = _get_atr_multiplier(market_downtrend, vi)

        reasons = []

        if pl_rate <= pl_threshold:
            reasons.append(f"含み損{pl_rate}%（{pl_threshold}%ライン割れ{mode_label}）")

        loss_amount = purchase - price
        if atr > 0 and loss_amount >= atr * atr_multiplier:
            reasons.append(
                f"損失額{round(loss_amount,1)}円がATR({atr}円)×{atr_multiplier}を超過{mode_label}"
            )

        if reasons:
            return {
                "type":    "STOP_LOSS",
                "symbol":  res["symbol"],
                "name":    res["name"],
                "price":   price,
                "pl_rate": pl_rate,
                "reasons": reasons,
            }
        return None

    # ── 利確シグナル ─────────────────────────────────────────────────────

    def check_take_profit(self, res: dict) -> dict | None:
        """利確警告シグナルを返す。条件未達なら None。"""
        if not res.get("is_held"):
            return None

        pl_rate   = res.get("pl_rate", 0)
        metrics   = res["metrics"]
        ma25_diff = metrics.get("MA25乖離", 0)
        peak_drop = metrics.get("高値反落率", 0)
        atr       = metrics.get("ATR", 0)
        price     = res["price"]

        reasons = []

        # 条件①: ATRトレーリングストップ
        if atr > 0 and price > 0 and peak_drop != 0:
            recent_high = price / (1 + peak_drop / 100)
            trailing_line = recent_high - atr * 2.0
            if price <= trailing_line:
                reasons.append(
                    f"ATRトレーリング発動（直近高値{round(recent_high,1)}円−ATR×2.0={round(trailing_line,1)}円）"
                )
        elif peak_drop <= -3.0:
            reasons.append(f"直近高値から{peak_drop}%反落（フォールバック）")

        # 条件②: MA25 を下回り始めた（含み益がある状態で）
        if pl_rate > 0 and ma25_diff < 0:
            reasons.append(f"含み益あり({pl_rate}%)でMA25を下抜け（MA25乖離{ma25_diff}%）")

        if reasons:
            return {
                "type":    "TAKE_PROFIT",
                "symbol":  res["symbol"],
                "name":    res["name"],
                "price":   price,
                "pl_rate": pl_rate,
                "reasons": reasons,
            }
        return None

    # ── 週末手仕舞いシグナル ──────────────────────────────────────────────

    def check_weekend_exit(self, res: dict) -> dict | None:
        """週末持ち越しリスクが高い銘柄を検知する。条件未達なら None。"""
        if not res.get("is_held"):
            return None

        metrics   = res["metrics"]
        ma25_diff = metrics.get("MA25乖離", 0)
        peak_drop = metrics.get("高値反落率", 0)
        rsi       = metrics.get("RSI", 50)
        atr       = metrics.get("ATR", 0)
        price     = res["price"]

        flags   = []
        reasons = []

        # 条件A: MA25 危険ゾーン
        if -3.0 <= ma25_diff <= 2.0:
            flags.append("A")
            reasons.append(f"MA25に接近中（乖離{ma25_diff}%）")

        # 条件B: 高値反落（ATR×1.0基準）
        if atr > 0 and price > 0 and peak_drop != 0:
            recent_high = price / (1 + peak_drop / 100)
            atr_threshold = recent_high - atr * 1.0
            if price <= atr_threshold:
                flags.append("B")
                reasons.append(f"直近高値からATR×1.0反落（{round(peak_drop,2)}%）")
        elif peak_drop <= -2.0:
            flags.append("B")
            reasons.append(f"直近高値から{peak_drop}%反落（フォールバック）")

        # 条件C: RSI 弱気転換（50割れ）
        if rsi < 50:
            flags.append("C")
            reasons.append(f"RSI={rsi}（50割れ、弱気転換）")

        if len(flags) >= 2:
            return {
                "type":    "WEEKEND_EXIT",
                "symbol":  res["symbol"],
                "name":    res["name"],
                "price":   res["price"],
                "pl_rate": res.get("pl_rate", 0),
                "reasons": reasons,
            }
        return None

    # ── まとめて判定（通常モード）────────────────────────────────────────

    def detect_all(
        self,
        results: list[dict],
        market_downtrend: bool | None = None,
        vi: float | None = None,
    ) -> list[dict]:
        """保有株リストに対してすべてのシグナルを判定して返す。

        FIX: market_downtrend / vi を引数で受け取れるよう変更。
        呼び出し元（main.py）で取得済みの値を渡すことで API の2重呼び出しを防ぐ。
        引数が None の場合は内部で取得する（後方互換）。
        """
        if market_downtrend is None:
            market_downtrend = _is_market_downtrend()
        if vi is None:
            vi = _get_nikkei_vi()

        if market_downtrend:
            logging.info(f"市場下落トレンド: 損切り厳格化（-3% / ATR×1.5）VI={round(vi,1)}")
        elif vi >= 25.0:
            logging.info(f"高VIX検知: 損切り拡大（-3% / ATR×2.5）VI={round(vi,1)}")

        signals = []
        for res in results:
            stop_sig = self.check_stop_loss(res, market_downtrend=market_downtrend, vi=vi)
            if stop_sig:
                signals.append(stop_sig)
                continue
            for checker in (self.check_take_profit, self.check_add_buy):
                sig = checker(res)
                if sig:
                    signals.append(sig)
        return signals

    # ── まとめて判定（週末モード）────────────────────────────────────────

    def detect_weekend(
        self,
        results: list[dict],
        market_downtrend: bool | None = None,
        vi: float | None = None,
    ) -> list[dict]:
        """金曜手仕舞いモード用。損切り + 週末手仕舞いシグナルを判定して返す。

        FIX: market_downtrend / vi を引数で受け取れるよう変更。
        """
        if market_downtrend is None:
            market_downtrend = _is_market_downtrend()
        if vi is None:
            vi = _get_nikkei_vi()

        if market_downtrend:
            logging.info("市場下落トレンド（週末モード）: 損切り厳格化")

        signals = []
        for res in results:
            sig = self.check_stop_loss(res, market_downtrend=market_downtrend, vi=vi)
            if sig:
                signals.append(sig)
            sig = self.check_weekend_exit(res)
            if sig:
                signals.append(sig)
        return signals

    # ── LINE通知テキスト生成 ─────────────────────────────────────────────

    @staticmethod
    def format_signals(signals: list[dict]) -> str:
        if not signals:
            return "シグナルなし"

        icons = {
            "ADD_BUY":      "🟢買い乗せ",
            "STOP_LOSS":    "🔴損切り",
            "TAKE_PROFIT":  "🟡利確警告",
            "WEEKEND_EXIT": "🏳️週末手仕舞い",
        }
        lines = []
        for s in signals:
            label = icons.get(s["type"], s["type"])
            lines.append(f"【{label}】{s['name']}（{s['symbol']}）")
            lines.append(f"  現在値: {s['price']}円")
            if "pl_rate" in s:
                lines.append(f"  損益: {s['pl_rate']}%")
            if "lot_pct" in s:
                lines.append(f"  推奨ロット: {s['lot_pct']}%")
            for r in s.get("reasons", []):
                lines.append(f"  ・{r}")
        return "\n".join(lines)
