import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def summarize_trade(row):
    prompt = f"""
You're an AI trading assistant. Analyze the following setup:

- Ticker: {row['ticker']}
- Gap %: {row['gap_pct']}%
- RSI: {row['rsi']}
- Relative Volume: {row['rel_volume']}
- Dist from 5d High: {row['dist_from_5d_high']}%
- Dist from 5d Low: {row['dist_from_5d_low']}%

Return a 2–3 sentence summary:
1. What is happening technically?
2. What trade setup might be forming?
3. What risk or confirmation to watch for?

Be concise and helpful for options traders.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )

    return response.choices[0].message.content

