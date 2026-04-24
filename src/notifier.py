import os
import logging
import time

import linebot.v3.messaging as bot

# LINEの1吹き出しあたりの文字数上限（余裕を持って設定）
_CHUNK_SIZE = 4000
# Push通知1回あたりの最大吹き出し数（LINE仕様上限）
_MAX_MESSAGES_PER_PUSH = 5
# 複数回Push間のスリープ（秒）
_PUSH_INTERVAL = 1
# 送信失敗時のリトライ回数・待機秒数
_MAX_RETRIES = 3
_RETRY_WAIT  = 5


class LineNotifier:
    def __init__(self):
        self.token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
        self.user_id = os.environ.get("LINE_USER_ID")

    def send_report(self, report_text: str):
        """
        レポートテキストを LINE Push通知で送信する。

        4,000文字ごとに分割し、1回のPushで最大5吹き出しまで送信。
        5吹き出しを超える場合は切り捨てず、複数回に分けてPushする。
        送信失敗時は _MAX_RETRIES 回までリトライする。
        """
        if not (self.token and self.user_id):
            logging.warning("LINE_CHANNEL_ACCESS_TOKEN または LINE_USER_ID が未設定です。通知をスキップします。")
            return

        # 4,000文字ごとに分割
        chunks = [report_text[i:i + _CHUNK_SIZE] for i in range(0, len(report_text), _CHUNK_SIZE)]
        total = len(chunks)

        if total > _MAX_MESSAGES_PER_PUSH:
            logging.info(
                f"レポートが長いため複数回に分けて送信します "
                f"（{len(report_text)}文字 / {total}吹き出し）"
            )

        configuration = bot.Configuration(access_token=self.token)
        with bot.ApiClient(configuration) as api:
            api_instance = bot.MessagingApi(api)

            # _MAX_MESSAGES_PER_PUSH 件ずつ送信
            for i in range(0, total, _MAX_MESSAGES_PER_PUSH):
                batch = chunks[i:i + _MAX_MESSAGES_PER_PUSH]
                messages = [bot.TextMessage(text=chunk) for chunk in batch]
                self._push_with_retry(api_instance, messages, i + 1, i + len(batch), total)

                # 複数回Pushの場合はインターバルを設ける
                if i + _MAX_MESSAGES_PER_PUSH < total:
                    time.sleep(_PUSH_INTERVAL)

    def _push_with_retry(
        self,
        api_instance,
        messages: list,
        from_idx: int,
        to_idx: int,
        total: int,
    ) -> None:
        """FIX: リトライ付きPush送信。レート制限・瞬断時にメッセージが消えるのを防ぐ。"""
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                api_instance.push_message(bot.PushMessageRequest(
                    to=self.user_id,
                    messages=messages,
                ))
                logging.debug(f"LINE送信: {from_idx}〜{to_idx}/{total}吹き出し")
                return
            except Exception as e:
                logging.warning(f"LINE送信失敗（{from_idx}〜{to_idx}件目, 試行{attempt}/{_MAX_RETRIES}）: {e}")
                if attempt < _MAX_RETRIES:
                    time.sleep(_RETRY_WAIT)

        logging.error(f"LINE送信を{_MAX_RETRIES}回試行しましたが失敗しました（{from_idx}〜{to_idx}件目）")
