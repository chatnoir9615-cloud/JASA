"""統合版メインスクリプト。

実行フロー:
  Step 1: ポートフォリオ同期 (transactions.csv -> holdings.json)
           └ stage（half/full）を自動付与
  Step 2: 保有株 / ウォッチリストの個別分析（テクニカル指標取得）
  Step 3: シグナル判定（買い乗せ・損切り・利確）
  Step 4: AIレポート作成 → LINE通知（カテゴリ別3通）
  Step 5: JPX全銘柄スクリーニング（保有・ウォッチリスト銘柄を除外）
  Step 6: スクリーニング結果のAIレポート → LINE通知

  ※ rebuild_cacheモード時はStep5のデータ取得のみ実行（AI解析・LINE通知なし）
  ※ emergencyモード時はStep1〜3のみ実行（シグナル判定のみ、スクリーニングなし）
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
    """保有・ウォッチリスト銘柄のシンボルセットを返す（スクリーニング除外用）。"""
    exclude = set()
    for key in ("holdings", "asset_value", "deep_value"):
        for t in data.get(key, []):
            if s := t.get("symbol"):
                exclude.add(s)
    return exclude


def load_holdings() -> dict:
    """holdings.json を読み込んで返す。失敗時は空 dict。"""
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
    """rebuild_cacheモード: 全銘柄OHLCVキャッシュ構築のみ実行（AI解析・LINE通知なし）。"""
    logging.info("=== rebuild_cacheモード: 全銘柄OHLCVキャッシュ構築のみ実行 ===")

    cache = MarketDataCache()
    holdings_data = load_holdings()
    exclude_symbols = build_exclude_symbols(holdings_data)
    logging.info(f"除外銘柄数: {len(exclude_symbols)}")

    screener = StockScreener(cache=cache)
    screener.build_cache_only(exclude_symbols=exclude_symbols)
    logging.info("✅ キャッシュ構築完了")


def run_emergency():
    """emergencyモード: 保有株のシグナル判定のみ実行（緊急監視）。

    市場開場直後（9:05）と昼休み明け（12:35）に実行される。
    スクリーニングや AI レポートは行わず、損切り・利確シグナルのみ通知する。
    """
    logging.info("=== emergencyモード: 保有株シグナル判定のみ実行 ===")

    cache    = MarketDataCache()
    pm       = PortfolioManager()
    fetcher  = StockFetcher(cache=cache)
    notifier = LineNotifier()
    detector = SignalDetector()

    # Step 1: ポートフォリオ同期
    logging.info("Step 1: 履歴同期...")
    pm.sync()

    holdings_data = load_holdings()
    if not holdings_data:
        return

    # Step 2: 保有株のみ分析
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

    # Step 3: 市場状況を1回だけ取得してシグナル判定
    logging.info("Step 3: シグナル判定（緊急モード）...")
    market_downtrend = _is_market_downtrend()
    vi = _get_nikkei_vi()

    _send_market_warning(notifier, market_downtrend, vi)

    # FIX: 取得済みの market_downtrend / vi を引数で渡し、2重取得を防ぐ
    signals = detector.detect_all(holdings_results, market_downtrend=market_downtrend, vi=vi)
    logging.info(f"シグナル検出数: {len(signals)}件")

    if signals:
        notifier.send_report(
            f"【🚨 緊急シグナル通知】\n\n"
            + detector.format_signals(signals)
        )
    else:
        notifier.send_report("【🚨 緊急監視】異常シグナルなし")

    logging.info("✅ emergencyモード完了")


def run_news():
    """newsモード: 週末ニュース収集 → 週明け地合い予測レポート送信。

    日曜 21:00 JST に実行される。
    """
    logging.info("=== newsモード: 週末ニュース収集・地合い予測 ===")

    notifier = LineNotifier()
    advisor  = AIAdvisor(api_key_env="GEMINI_API_KEY")
    analyzer = NewsAnalyzer(ai_advisor=advisor)

    # ニュース収集
    logging.info("ニュース収集中...")
    news_items = analyzer.collect_news()

    # 地合い予測レポート生成
    logging.info("週明け地合い予測レポート生成中...")
    report = analyzer.analyze_weekly_outlook(news_items)

    notifier.send_report(
        f"【📰 週明け地合い予測レポート】\nモデル: {advisor.model_id}\n\n{report}"
    )
    logging.info("✅ newsモード完了")


# ─────────────────────────────────────────────────────────────────────────────
# メインフロー
# ─────────────────────────────────────────────────────────────────────────────

def _send_market_warning(notifier: LineNotifier, market_downtrend: bool, vi: float):
    """市場状況に応じた警告を LINE に送信する。"""
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

    # ── モード分岐 ────────────────────────────────────────────────────────
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
    # Step 1: ポートフォリオ同期（stage を自動付与）
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 1: 履歴同期（stage自動付与）...")
    pm.sync()

    holdings_data = load_holdings()
    if not holdings_data:
        return

    # ─────────────────────────────────────────────────────────────
    # Step 2: 保有株 / ウォッチリストの個別分析
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 2: 市場データ収集（保有株・ウォッチリスト）...")

    categories = {
        "holdings":    {"label": "保有",       "data": holdings_data.get("holdings",    []), "is_held": True},
        "asset_value": {"label": "資産バリュー", "data": holdings_data.get("asset_value", []), "is_held": False},
        "deep_value":  {"label": "収益バリュー", "data": holdings_data.get("deep_value",  []), "is_held": False},
    }

    all_results: dict[str, list] = {k: [] for k in categories}

    for cat_key, config in categories.items():
        logging.info(f"  処理中: {cat_key} ({config['label']})")
        for stock in config["data"]:
            symbol = stock["symbol"]
            res = fetcher.analyze_strategy(symbol)
            if res:
                res.update({
                    "symbol":         symbol,
                    "name":           stock.get("name", symbol),
                    "category_label": config["label"],
                    "is_held":        config["is_held"],
                    "purchase_price": stock.get("purchase_price", 0),
                    "quantity":       stock.get("quantity", 0),
                    "stage":          stock.get("stage", "half"),
                    "pl_rate":        0.0,
                })
                if config["is_held"] and stock.get("purchase_price", 0) > 0:
                    res["pl_rate"] = round(
                        ((res["price"] - stock["purchase_price"]) / stock["purchase_price"]) * 100, 2
                    )
                all_results[cat_key].append(res)

    # ─────────────────────────────────────────────────────────────
    # Step 3: 市場状況取得 + シグナル判定（1回だけ取得）
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 3: シグナル判定（買い乗せ・損切り・利確）...")

    # FIX: 市場状況を1回だけ取得し、detect_all にも引数として渡す（2重取得を防ぐ）
    market_downtrend = _is_market_downtrend()
    vi = _get_nikkei_vi()

    _send_market_warning(notifier, market_downtrend, vi)

    signals = detector.detect_all(
        all_results["holdings"],
        market_downtrend=market_downtrend,
        vi=vi,
    )
    logging.info(f"シグナル検出数: {len(signals)}件")

    # ─────────────────────────────────────────────────────────────
    # Step 4: AIレポート → LINE通知（シグナルを統合して送信）
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 4: アナリストレポート作成（保有株・ウォッチリスト）...")

    if all_results["holdings"]:
        advisor = AIAdvisor(api_key_env="GEMINI_API_KEY_HOLDINGS")
        text = advisor.get_batch_advice(all_results["holdings"], signals=signals)
        notifier.send_report(f"【🏠 保有株レポート】\nモデル: {advisor.model_id}\n\n{text}")
        time.sleep(1)

    if all_results["asset_value"]:
        advisor = AIAdvisor(api_key_env="GEMINI_API_KEY_ASSET")
        text = advisor.get_batch_advice(all_results["asset_value"])
        notifier.send_report(f"【💰 資産バリュー判定レポート】\nモデル: {advisor.model_id}\n\n{text}")
        time.sleep(1)

    if all_results["deep_value"]:
        advisor = AIAdvisor(api_key_env="GEMINI_API_KEY_DEEP")
        text = advisor.get_batch_advice(all_results["deep_value"])
        notifier.send_report(f"【📈 収益バリュー判定レポート】\nモデル: {advisor.model_id}\n\n{text}")
        time.sleep(1)

    # ─────────────────────────────────────────────────────────────
    # Step 5: JPX全銘柄スクリーニング
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 5: 全銘柄スクリーニング（上限2000円・上位10銘柄）...")

    exclude_symbols = build_exclude_symbols(holdings_data)
    logging.info(f"除外銘柄数: {len(exclude_symbols)}")

    screener = StockScreener(cache=cache)
    screen_results = screener.screen(
        exclude_symbols=exclude_symbols,
        max_price=2000,
        top_n=10,
    )

    # ─────────────────────────────────────────────────────────────
    # Step 6: スクリーニング結果のAIレポート → LINE通知
    # ─────────────────────────────────────────────────────────────
    logging.info("Step 6: スクリーニング結果レポート作成...")

    if screen_results:
        advisor = AIAdvisor(api_key_env="GEMINI_API_KEY")
        text    = advisor.get_batch_advice(screen_results)
        notifier.send_report(
            f"【💹 スクリーニング通知（〜2000円 上位10銘柄）】\n"
            f"モデル: {advisor.model_id}\n\n"
            + (text or "AI解析失敗")
        )
    else:
        notifier.send_report("【💹 スクリーニング通知】\n該当銘柄なし")

    logging.info("✅ 全工程完了")


if __name__ == "__main__":
    main()
