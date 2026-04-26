"""統合版メインスクリプト。

実行フロー:
  Step 1: ポートフォリオ同期 (transactions.csv -> holdings.json)
  Step 2: 保有株の個別分析（テクニカル指標取得）
  Step 3: シグナル判定（買い乗せ・損切り・利確）
  Step 4: AIレポート作成 → LINE通知（保有株のみ）
  Step 5: 日経225・TOPIX100・TOPIX500 スクリーニング
  Step 6: スクリーニング結果のAIレポート → LINE通知（【資産】【収益】タグ付き）

  ※ rebuild_cacheモード時はStep5のデータ取得のみ実行
  ※ emergencyモード時はStep1〜3のみ実行
  ※ newsモード時は週末ニュース収集・週明け地合い予測レポートを送信
"""

import json
import logging
import os
import time

from dotenv import load_dotenv

from src.ai_advisor import AIAdvisor
from src.fetcher import StockFetcher
from src.market_cache import MarketDataCache
from src.news_analyzer import NewsAnalyzer
from src.notifier import LineNotifier
from src.portfolio_manager import PortfolioManager
from src.screener import StockScreener
from src.signal_detector import SignalDetector, _is_market_downtrend, _get_nikkei_vi

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def build_exclude_symbols(data: dict) -> set[str]:
    """保有銘柄のシンボルセットを返す（スクリーニング除外用）。"""
    exclude = set()
    for t in data.get("holdings", []):
        if s := t.get("symbol"):
            exclude.add(s)
    return exclude


def load_holdings() -> dict:
    try:
        with open("holdings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"holdings.json の読み込み失敗: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# モード別エントリポイント
# ─────────────────────────────────────────────────────────────────────────────

def run_rebuild_cache():
    logging.info("=== rebuild_cacheモード: 全銘柄OHLCVキャッシュ構築のみ実行 ===")
    cache = MarketDataCache()
    holdings_data   = load_holdings()
    exclude_symbols = build_exclude_symbols(holdings_data)
    logging.info(f"除外銘柄数: {len(exclude_symbols)}")
    screener = StockScreener(cache=cache)
    screener.build_cache_only(exclude_symbols=exclude_symbols)
    logging.info("✅ キャッシュ構築完了")


def run_emergency():
    logging.info("=== emergencyモード: 保有株シグナル判定のみ実行 ===")
    cache    = MarketDataCache()
    pm       = PortfolioManager()
    fetcher  = StockFetcher(cache=cache)
    notifier = LineNotifier()
    detector = SignalDetector()

    logging.info("Step 1: 履歴同期...")
    pm.sync()
    holdings_data = load_holdings()
    if not holdings_data:
        return

    logging.info("Step 2: 保有株データ収集...")
    holdings_results = []
    for stock in holdings_data.get("holdings", []):
        symbol = stock["symbol"]
        res = fetcher.analyze_strategy(symbol)
        if res:
            res.update({
                "symbol":         symbol,
                "name":           stock.get("name", symbol),
                "category_label": "保有",
                "is_held":        True,
                "purchase_price": stock.get("purchase_price", 0),
                "quantity":       stock.get("quantity", 0),
                "stage":          stock.get("stage", "half"),
                "pl_rate":        0.0,
            })
            if res["purchase_price"] > 0:
                res["pl_rate"] = round(
                    ((res["price"] - res["purchase_price"]) / res["purchase_price"]) * 100, 2
                )
            holdings_results.append(res)

    logging.info("Step 3: シグナル判定（緊急モード）...")
    market_downtrend = _is_market_downtrend()
    vi = _get_nikkei_vi()
    _send_market_warning(notifier, market_downtrend, vi)
    signals = detector.detect_all(holdings_results, market_downtrend=market_downtrend, vi=vi)
    logging.info(f"シグナル検出数: {len(signals)}件")

    if signals:
        notifier.send_report("【🚨 緊急シグナル通知】\n\n" + detector.format_signals(signals))
    else:
        notifier.send_report("【🚨 緊急監視】異常シグナルなし")

    logging.info("✅ emergencyモード完了")


def run_news():
    logging.info("=== newsモード: 週末ニュース収集・地合い予測 ===")
    notifier = LineNotifier()
    advisor  = AIAdvisor(api_key_env="GEMINI_API_KEY")
    analyzer = NewsAnalyzer(ai_advisor=advisor)
    news_items = analyzer.collect_news()
    report     = analyzer.analyze_weekly_outlook(news_items)
    notifier.send_report(
        f"【📰 週明け地合い予測レポート】\nモデル: {advisor.model_id}\n\n{report}"
    )
    logging.info("✅ newsモード完了")


# ─────────────────────────────────────────────────────────────────────────────
# メインフロー
# ─────────────────────────────────────────────────────────────────────────────

def _send_market_warning(notifier: LineNotifier, market_downtrend: bool, vi: float):
    if market_downtrend:
        notifier.send_report(
            "⚠️ 【市場下落トレンド警告】\n"
            "日経225・TOPIX 両方の5日MAが下向きです。\n"
            f"損切り基準を自動厳格化中（-3% / ATR×1.5）\n"
            f"日経VI: {round(vi, 1)}"
        )
        logging.info("市場下落トレンド警告をLINEに送信しました")
        time.sleep(1)
    elif vi >= 25.0:
        notifier.send_report(
            f"⚠️ 【高VIX警告】日経VI: {round(vi, 1)}\n"
            "ボラティリティ上昇中。\n"
            "損切り基準を自動拡大中（-3% / ATR×2.5）"
        )
        logging.info(f"高VIX警告をLINEに送信しました（VI={round(vi,1)}）")
        time.sleep(1)


def main():
    schedule_type = os.environ.get("SCHEDULE_TYPE", "main")

    if schedule_type == "rebuild_cache":
        run_rebuild_cache()
        return
    if schedule_type == "emergency":
        run_emergency()
        return
    if schedule_type == "news":
        run_news()
        return

    # ── mainモード ────────────────────────────────────────────────────────
    cache    = MarketDataCache()
    pm       = PortfolioManager()
    fetcher  = StockFetcher(cache=cache)
    notifier = LineNotifier()
    detector = SignalDetector()

    # ─────────────────────────────────────────────────────────────
    # Step 1: ポートフォリオ同期
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 1: 履歴同期（stage自動付与）...")
    pm.sync()
    holdings_data = load_holdings()
    if not holdings_data:
        return

    # ─────────────────────────────────────────────────────────────
    # Step 2: 保有株の個別分析
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 2: 市場データ収集（保有株）...")
    holdings_results = []
    for stock in holdings_data.get("holdings", []):
        symbol = stock["symbol"]
        res = fetcher.analyze_strategy(symbol)
        if res:
            res.update({
                "symbol":         symbol,
                "name":           stock.get("name", symbol),
                "category_label": "保有",
                "is_held":        True,
                "purchase_price": stock.get("purchase_price", 0),
                "quantity":       stock.get("quantity", 0),
                "stage":          stock.get("stage", "half"),
                "pl_rate":        0.0,
            })
            if stock.get("purchase_price", 0) > 0:
                res["pl_rate"] = round(
                    ((res["price"] - stock["purchase_price"]) / stock["purchase_price"]) * 100, 2
                )
            holdings_results.append(res)

    # ─────────────────────────────────────────────────────────────
    # Step 3: シグナル判定
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 3: シグナル判定（買い乗せ・損切り・利確）...")
    market_downtrend = _is_market_downtrend()
    vi = _get_nikkei_vi()
    _send_market_warning(notifier, market_downtrend, vi)
    signals = detector.detect_all(holdings_results, market_downtrend=market_downtrend, vi=vi)
    logging.info(f"シグナル検出数: {len(signals)}件")

    # ─────────────────────────────────────────────────────────────
    # Step 4: AIレポート → LINE通知（保有株のみ）
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 4: アナリストレポート作成（保有株）...")
    if holdings_results:
        advisor = AIAdvisor(api_key_env="GEMINI_API_KEY_HOLDINGS")
        text = advisor.get_batch_advice(holdings_results, signals=signals)
        notifier.send_report(f"【🏠 保有株レポート】\nモデル: {advisor.model_id}\n\n{text}")
        time.sleep(1)

    # ─────────────────────────────────────────────────────────────
    # Step 5: 日経225・TOPIX100・TOPIX500 スクリーニング
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 5: 指数銘柄スクリーニング（日経225・TOPIX100・TOPIX500）...")
    exclude_symbols = build_exclude_symbols(holdings_data)
    logging.info(f"除外銘柄数: {len(exclude_symbols)}")

    screener = StockScreener(cache=cache)
    screen_results = screener.screen(
        exclude_symbols=exclude_symbols,
        top_n=10,
    )

    # ─────────────────────────────────────────────────────────────
    # Step 6: スクリーニング結果 → LINE通知（ファンダ＋AI統合・5銘柄×2通）
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 6: スクリーニング結果レポート作成...")
    if screen_results:
        advisor = AIAdvisor(api_key_env="GEMINI_API_KEY")

        # 5銘柄ずつ2バッチに分割して送信
        batch_size = 5
        total = len(screen_results)
        for batch_num, start in enumerate(range(0, total, batch_size), 1):
            batch = screen_results[start:start + batch_size]

            # AI解析（バッチ単位）
            ai_text = advisor.get_batch_advice(batch) or "AI解析失敗"

            # AI出力を銘柄ごとにブロック分割（■ で始まる行を区切りとして使用）
            ai_blocks: dict[str, str] = {}
            current_key = None
            current_lines: list[str] = []
            for line in ai_text.splitlines():
                if line.startswith("■"):
                    if current_key is not None:
                        ai_blocks[current_key] = "\n".join(current_lines).strip()
                    # 銘柄名またはシンボルでキーを特定
                    current_key = line
                    current_lines = [line]
                else:
                    current_lines.append(line)
            if current_key is not None:
                ai_blocks[current_key] = "\n".join(current_lines).strip()

            # ファンダ情報＋AIレポートを銘柄ごとに統合
            merged_lines: list[str] = []
            for i, r in enumerate(batch):
                fund_label = r.get("fund_label", "")
                price      = r["price"]
                rsi        = r["metrics"]["RSI"]
                atr        = r["metrics"]["ATR"]
                name       = r["name"]
                symbol     = r["symbol"]

                # ファンダ情報（タグ・決算期・指標）
                fund_block = (
                    f"■{name}({symbol})\n"
                    f"  現在値:{price}円 / RSI:{rsi} / ATR:{atr}円\n"
                    f"{fund_label}"
                )

                # 対応するAIブロックを探す（銘柄名またはシンボルが含まれるキー）
                ai_block = ""
                for key, block in ai_blocks.items():
                    if name in key or symbol in key:
                        # ■行（重複）を除いてAI部分だけ抽出
                        ai_lines = [l for l in block.splitlines() if not l.startswith("■")]
                        ai_block = "\n".join(ai_lines).strip()
                        break

                merged = fund_block
                if ai_block:
                    merged += "\n" + ai_block
                merged_lines.append(merged)

            report = (
                f"【💹 指数銘柄スクリーニング】\n"
                f"({batch_num}/{(total + batch_size - 1) // batch_size}) "
                f"モデル: {advisor.model_id}\n\n"
                + "\n\n".join(merged_lines)
            )
            notifier.send_report(report)
            if start + batch_size < total:
                time.sleep(1)
    else:
        notifier.send_report("【💹 スクリーニング通知】\n該当銘柄なし")

    logging.info("✅ 全工程完了")


if __name__ == "__main__":
    main()