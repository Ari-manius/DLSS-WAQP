#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import requests
import time
from urllib.parse import quote
import threading
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

df = pd.read_csv("articles_titles.csv")
titles = df["title"].dropna().unique().tolist()

lock = threading.Lock()
stats = {
    "processed": 0,
    "success": 0,
    "errors": 0,
    "rate_limited": 0,
}

headers = {
    "User-Agent": "UniversityofKonstanzResearchBot/1.0 (https://uni-konstanz.de; lorenz.rueckert@uni-konstanz.de)"
}

def get_last_edit_time(title):
    time.sleep(random.uniform(0.3, 0.6))  # 1-2 req/sec

    encoded = quote(title.replace(" ", "_"))
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded}"

    try:
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code == 200:
            data = response.json()
            last_edit = data.get("timestamp", None)
            with lock:
                stats["success"] += 1
            return (title, last_edit)

        elif response.status_code == 429:
            with lock:
                stats["rate_limited"] += 1
            print(f"🔴 429 Rate limit: {title} – Retrying in 15s...")
            time.sleep(15)
            return get_last_edit_time(title)

        else:
            with lock:
                stats["errors"] += 1
            return (title, None)

    except Exception:
        with lock:
            stats["errors"] += 1
        return (title, None)

def save_final(results):
    df_out = pd.DataFrame(results, columns=["title", "last_edit"])
    df_out.to_csv("final_last_edit.csv", index=False)
    print("✅ Saved to final_last_edit.csv")

def main():
    all_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(get_last_edit_time, title): title for title in titles}
        for i, future in enumerate(as_completed(futures), 1):
            title, last_edit = future.result()
            all_results.append((title, last_edit))
            with lock:
                stats["processed"] += 1
                if i <= 10 or i % 500 == 0:
                    print(f"✅ {title[:35]:<35} → {last_edit}")
    save_final(all_results)
    print(f"🎉 Done: {stats}")

if __name__ == "__main__":
    main()

