#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import requests
import time

# Load only pageid column from your dataset
df = pd.read_csv(r"C:\Users\nafis\Downloads\cleaned_articles_final.csv", usecols=['pageid'])

print(df.head())




# In[ ]:


import pandas as pd
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

# Extract all page IDs
page_ids = df['pageid'].dropna().astype(int).tolist()

# Split into chunks of 50 (MediaWiki API limit)
def chunked(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

chunks = list(chunked(page_ids, 50))

# Function to fetch titles for a batch of up to 50 page IDs
def fetch_titles(batch):
    ids_str = "|".join(str(pid) for pid in batch)
    url = f"https://en.wikipedia.org/w/api.php?action=query&pageids={ids_str}&format=json"
    headers = {
        "User-Agent": "WikipediaQualityBot/1.0"
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return {int(page['pageid']): page.get('title') for page in data['query']['pages'].values()}
        else:
            print(f"Error: {response.status_code} for batch {batch}")
            return {}
    except Exception as e:
        print(f"Exception for batch {batch}: {e}")
        return {}

# Use ThreadPoolExecutor for parallel fetching
print(" Fetching titles in parallel...")
titles_dict = {}
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_batch = {executor.submit(fetch_titles, batch): batch for batch in chunks}
    for future in as_completed(future_to_batch):
        result = future.result()
        if result:
            titles_dict.update(result)

# Map titles back to the original DataFrame
df['title'] = df['pageid'].map(titles_dict)

# Save the updated table
df.to_csv(r"C:\Users\nafis\Downloads\cleaned_articles_final_with_titles.csv", index=False)
print(" Titles added and saved!")

