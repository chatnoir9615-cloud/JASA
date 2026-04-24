import json
import logging
import os
from datetime import date

import pandas as pd


class PortfolioManager:
    def __init__(self, csv_path="transactions.csv", json_path="holdings.json"):
        self.csv_path = csv_path
        self.json_path = json_path

    def sync(self):
        """
        transactions.csv を唯一の入力源として holdings.json の holdings を再構築する。

        - CSV に存在する銘柄のみが holdings に反映される
        - 売り切れた銘柄（quantity=0）は自動的に除外される
        - CSV がない場合は holdings を変更しない
        - holdings に新規追加された銘柄は asset_value / deep_value から除外し
          purchased に移動する（購入履歴として保持）
        """
        if not os.path.exists(self.csv_path):
            logging.info("transactions.csv が見つかりません。holdings は変更しません。")
            return

        df = pd.read_csv(self.csv_path)
        if df.empty:
            return

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')

        # CSVの全取引から保有状況を計算
        portfolio = {}
        for _, row in df.iterrows():
            symbol = row['symbol']
            name = str(row.get('name', '') or '')
            qty = row['quantity']
            price = row['price']
            if qty <= 0:
                continue

            if symbol not in portfolio:
                portfolio[symbol] = {
                    "symbol": symbol,
                    "name": name,
                    "purchase_price": 0.0,
                    "quantity": 0,
                    "buy_count": 0,
                    "history": []
                }

            p = portfolio[symbol]
            if name:
                p['name'] = name

            if row['type'] == 'buy':
                total_cost = (p['purchase_price'] * p['quantity']) + (price * qty)
                p['quantity'] += qty
                p['buy_count'] += 1
                if p['quantity'] > 0:
                    p['purchase_price'] = round(total_cost / p['quantity'], 2)
            elif row['type'] == 'sell':
                p['quantity'] = max(0, p['quantity'] - qty)

            p['history'].append({
                "date": row['date'].strftime('%Y-%m-%d'),
                "type": row['type'],
                "price": price,
                "qty": qty
            })

        # 保有数量 > 0 の銘柄だけを holdings として確定
        new_holdings = [
            {
                "symbol":         v['symbol'],
                "name":           v['name'],
                "purchase_price": v['purchase_price'],
                "quantity":       v['quantity'],
                "currency":       "JPY",
                "stage":          "full" if v['buy_count'] >= 2 else "half",
            }
            for v in portfolio.values() if v['quantity'] > 0
        ]

        self._write_json(new_holdings)

    def _write_json(self, new_holdings: list):
        """
        holdings を上書きし、新規保有銘柄を asset_value / deep_value から
        purchased へ移動する。その他のキーは保持する。
        """
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        # 既存 holdings のシンボルセット
        old_symbols = {h['symbol'] for h in data.get("holdings", [])}
        # 新規 holdings のシンボルセット
        new_symbols = {h['symbol'] for h in new_holdings}
        # 今回新たに holdings に加わった銘柄
        added_symbols = new_symbols - old_symbols

        if added_symbols:
            purchased = data.get("purchased", [])
            purchased_symbols = {p['symbol'] for p in purchased}

            for watch_key in ("asset_value", "deep_value"):
                remaining = []
                for item in data.get(watch_key, []):
                    if item.get("symbol") in added_symbols:
                        # purchased にまだ記録されていなければ追加
                        if item['symbol'] not in purchased_symbols:
                            purchased.append({
                                "symbol":     item['symbol'],
                                "name":       item.get('name', ''),
                                "moved_from": watch_key,
                                "moved_at":   date.today().isoformat(),
                            })
                            purchased_symbols.add(item['symbol'])
                            logging.info(
                                f"[purchased移動] {item['symbol']} "
                                f"{item.get('name','')} ({watch_key} → purchased)"
                            )
                    else:
                        remaining.append(item)
                data[watch_key] = remaining

            data["purchased"] = purchased

        # holdings を更新
        data["holdings"] = new_holdings

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logging.info(f"holdings同期完了: {len(new_holdings)}銘柄")
        if added_symbols:
            logging.info(f"purchased移動: {added_symbols}")