import requests

# 你的 Channel Access Token（長期 token）
CHANNEL_ACCESS_TOKEN = 'OcKhyT9AjLLkn1OfFCqDTtHx9UtCP1AuOMtJJ9XT3qp+gWjHhPT2pYmY3sHCqWmgHfrYpH2Ox1p0jjoBsmPclWs+sdGDy8+KL7RzzG4JrbuD6NcEVkN3PgXFmjbqeRqqTqWrxsyq2F82BNlyk2TKXgdB04t89/1O/w1cDnyilFU='

# 廣播訊息內容
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {CHANNEL_ACCESS_TOKEN}"
}

data = {
    "messages": [
        {
            "type": "text",
            "text": "這是一則 Python 廣播訊息 👋"
        }
    ]
}

response = requests.post(
    "https://api.line.me/v2/bot/message/broadcast",
    headers=headers,
    json=data
)

print("狀態碼:", response.status_code)
print("回應內容:", response.text)
