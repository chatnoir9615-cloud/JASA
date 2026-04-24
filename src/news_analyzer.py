"""週末ニュース収集と週明け地合い予測。

設計方針:
  - 標準ライブラリの urllib + xml.etree のみでRSSを取得（外部依存なし）
  - RSSフィードから直近48時間以内の記事タイトルを収集
  - Gemini に「月曜朝のパニックリスク」を判定させ LINE へ通知
  - フィード取得失敗時は該当ソースをスキップして続行（部分失敗を許容）
"""

import logging
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

JST = timezone(timedelta(hours=9))

# 収集対象RSSフィード
_RSS_FEEDS: list[dict] = [
    {"name": "ロイター（日本）",   "url": "https://feeds.reuters.com/reuters/JPBusinessNews"},
    {"name": "株探ニュース",       "url": "https://kabutan.jp/rss/news"},
    {"name": "日経（マーケット）",  "url": "https://www.nikkei.com/rss/finance.rdf"},
    {"name": "Bloomberg（日本）",  "url": "https://www.bloomberg.co.jp/feed/podcast/etf-report.xml"},
]

_FETCH_TIMEOUT   = 10   # 秒
_NEWS_WINDOW_H   = 48   # 直近何時間の記事を収集するか
_MAX_ITEMS_FEED  = 15   # 1フィードあたりの最大取得件数
_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; StockBot/2.0)"
}


class NewsAnalyzer:
    def __init__(self, ai_advisor=None):
        """
        Args:
            ai_advisor: AIAdvisor インスタンス。None の場合はニュース収集のみ行う。
        """
        self.advisor = ai_advisor

    # ── Public API ────────────────────────────────────────────────────────

    def collect_news(self) -> list[dict]:
        """RSSフィードから直近 _NEWS_WINDOW_H 時間以内の記事を収集して返す。"""
        cutoff = datetime.now(JST) - timedelta(hours=_NEWS_WINDOW_H)
        all_items: list[dict] = []

        for feed in _RSS_FEEDS:
            try:
                items = self._fetch_feed(feed["name"], feed["url"], cutoff)
                all_items.extend(items)
                logging.info(f"[RSS] {feed['name']}: {len(items)}件取得")
            except Exception as e:
                logging.warning(f"[RSS] {feed['name']} 取得失敗（スキップ）: {e}")
            time.sleep(0.5)

        logging.info(f"ニュース合計: {len(all_items)}件")
        return all_items

    def analyze_weekly_outlook(self, news_items: list[dict]) -> str:
        """収集済みニュースを Gemini に渡し、週明け地合い予測レポートを返す。"""
        if not news_items:
            return "（ニュース取得なし。分析をスキップします）"
        if self.advisor is None:
            return "（AIAdvisor未設定。分析をスキップします）"

        headlines = "\n".join(
            f"・[{it['source']}] {it['title']}"
            for it in news_items[:60]  # 多すぎるとトークン超過するため上限60件
        )

        prompt = f"""
あなたはプロの日本株マーケットアナリストです。
以下は直近48時間に収集した金融・経済ニュースの見出しです。

これらを踏まえ、**月曜日の東京市場の開場時（寄り付き）に想定されるリスクと注目点**を
以下の形式で簡潔にまとめてください。

【出力フォーマット】
📊 週明け地合い予測（{datetime.now(JST).strftime('%Y-%m-%d')} 時点）

🔴 パニックリスク: [高 / 中 / 低]
🌐 主要材料:
  ・[材料1]
  ・[材料2]（最大5件）

⚠️ 要注意セクター: [セクター名と理由]

💡 月曜朝の戦略メモ:
[200文字以内で、月曜の立ち回りに関する具体的なアドバイス]

【収集ニュース見出し】
{headlines}
"""
        return self.advisor._safe_generate(prompt)

    # ── Internal helpers ─────────────────────────────────────────────────

    def _fetch_feed(self, source: str, url: str, cutoff: datetime) -> list[dict]:
        """1つのRSSフィードを取得し、cutoff以降の記事リストを返す。"""
        req = urllib.request.Request(url, headers=_REQUEST_HEADERS)
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            raw = resp.read()

        root = ET.fromstring(raw)
        # RSS 2.0 と Atom 両方に対応
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        items = root.findall(".//item") or root.findall(".//atom:entry", ns)

        results = []
        for item in items[:_MAX_ITEMS_FEED]:
            title = self._get_text(item, ["title", "atom:title"], ns)
            pub   = self._get_text(item, ["pubDate", "atom:published", "atom:updated"], ns)

            if not title:
                continue

            # 公開日時をパース（パース失敗時は収集対象に含める）
            if pub:
                try:
                    pub_dt = parsedate_to_datetime(pub)
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=JST)
                    if pub_dt < cutoff:
                        continue
                except Exception:
                    pass  # パース失敗は無視して収集

            results.append({"source": source, "title": title.strip()})

        return results

    @staticmethod
    def _get_text(element: ET.Element, tags: list[str], ns: dict) -> str:
        """複数のタグ候補から最初に見つかったテキストを返す。"""
        for tag in tags:
            child = element.find(tag, ns) if ":" in tag else element.find(tag)
            if child is not None and child.text:
                return child.text
        return ""
