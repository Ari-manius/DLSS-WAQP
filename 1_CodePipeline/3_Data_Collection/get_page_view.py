#!/usr/bin/env python
# coding: utf-8

# In[ ]:


##one year

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from tqdm import tqdm

# === Step 1: Load your dataset ===
df = pd.read_csv("articles_titles.csv")

# === Step 2: Create column if it doesn't exist ===
if 'pageviews_Jul2023Jul2024' not in df.columns:
    df['pageviews_Jul2023Jul2024'] = None

# === Step 3: Filter titles that need fetching ===
pending_df = df[df['pageviews_Jul2023Jul2024'].isna()].copy()
titles = pending_df['title'].dropna().tolist()

# === Step 4: Function to fetch 1 year of monthly pageviews ===
def get_pageviews(title):
    try:
        title_url = quote(title.replace(" ", "_"))
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia.org/all-access/user/{title_url}/monthly/20230701/20240701"
        )
        headers = {
            "User-Agent": "WikipediaQualityBot/1.0 (nafiseh_tavakol@yahoo.com)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return title, sum(item['views'] for item in data.get('items', []))
        else:
            return title, None
    except Exception as e:
        print(f"Error for {title}: {e}")
        return title, None

# === Step 5: Run parallel fetching ===
print(" Fetching pageviews (July 2023 – July 2024) in parallel...")
results = {}
batch_size = 1000

with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(get_pageviews, title): title for title in titles}
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        title, views = future.result()
        results[title] = views

        # Save every batch of results
        if (i + 1) % batch_size == 0 or (i + 1) == len(futures):
            df.loc[df['title'].isin(results.keys()), 'pageviews_Jul2023Jul2024'] = df['title'].map(results)
            df.to_csv(r"articles_page_view.csv", index=False)
            print(f" Saved after {i + 1} articles...")

print(" All pageviews fetched and saved.")


# In[ ]:


###3 months

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from tqdm import tqdm
import time

# Load your dataset
df = pd.read_csv(r"cleaned_articles_final_with_titles.csv")

# Add placeholder if not already there
if 'pageviews_MayJul2024' not in df.columns:
    df['pageviews_MayJul2024'] = None

# Only fetch titles that are missing pageviews
pending_df = df[df['pageviews_MayJul2024'].isna()].copy()
titles = pending_df['title'].dropna().tolist()

# Define function to fetch pageviews for May–July 2024
def get_pageviews(title):
    try:
        title_url = quote(title.replace(" ", "_"))
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/user/{title_url}/monthly/20240501/20240731"
        headers = {
            "User-Agent": "WikipediaQualityBot/1.0 (nafiseh_tavakol@yahoo.com)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return title, sum(item['views'] for item in data.get('items', []))
        else:
            return title, None
    except Exception as e:
        print(f" Error for {title}: {e}")
        return title, None

# Start parallel fetching
print(" Fetching pageviews (May–July 2024) in parallel...")
start = time.time()
results = {}
batch_size = 1000

with ThreadPoolExecutor(max_workers=15) as executor:
    futures = {executor.submit(get_pageviews, title): title for title in titles}
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        title, views = future.result()
        results[title] = views

        # Save partial results every 1000 fetches
        if (i + 1) % batch_size == 0 or (i + 1) == len(futures):
            df.loc[df['title'].isin(results.keys()), 'pageviews_MayJul2024'] = df['title'].map(results)
            df.to_csv(r"cleaned_articles_final_with_titles_2.csv", index=False)
            print(f" Saved after {i+1} articles...")

end = time.time()
print(f" Done in {round((end - start) / 60, 2)} minutes.")


# In[1]:


##test 1000-one year

import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote
from tqdm import tqdm

# === Step 1: Load your dataset ===
df = pd.read_csv(r"C:\Users\nafis\Downloads\cleaned_articles_final_with_titles_2.csv")


# === Step 2: Create column if it doesn't exist ===
if 'pageviews_Jul2023Jul2024' not in df.columns:
    df['pageviews_Jul2023Jul2024'] = None

# === Step 3: Filter first 100 titles needing fetching ===
pending_df = df[df['pageviews_Jul2023Jul2024'].isna()].copy()
titles = pending_df['title'].dropna().tolist()[:100]

# === Step 4: Function to fetch 1 year of monthly pageviews ===
def get_pageviews(title):
    try:
        title_url = quote(title.replace(" ", "_"))
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia.org/all-access/user/{title_url}/monthly/20230701/20240701"
        )
        headers = {
            "User-Agent": "WikipediaQualityBot/1.0 (nafiseh_tavakol@yahoo.com)"
        }
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return title, sum(item['views'] for item in data.get('items', []))
        else:
            return title, None
    except Exception as e:
        print(f"Error for {title}: {e}")
        return title, None

# === Step 5: Run parallel fetching for 100 articles ===
print("⏳ Fetching pageviews (July 2023 – July 2024) for 100 articles...")
results = {}

with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(get_pageviews, title): title for title in titles}
    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures))):
        title, views = future.result()
        results[title] = views

# === Step 6: Update and save ===
df.loc[df['title'].isin(results.keys()), 'pageviews_Jul2023Jul2024'] = df['title'].map(results)
df.to_csv(r"C:\Users\nafis\Downloads\sample_100_with_pageviews.csv", index=False)
print("✅ Saved test result: sample_100_with_pageviews.csv")


# In[ ]:




