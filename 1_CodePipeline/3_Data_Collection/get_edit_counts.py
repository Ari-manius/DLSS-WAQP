import pandas as pd
import requests
import time
from urllib.parse import quote
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import pandas as pd
import requests
import time
from urllib.parse import quote
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Load article titles
df = pd.read_csv("cleaned_articles_final.csv")
titles = df["article"].tolist()

#porxy config
def get_proxy():
    return {
        "http": "http://edit_counts_Lb51Z:TeamSpeak451_@pr.oxylabs.io:7777",
        "https": "http://edit_counts_Lb51Z:TeamSpeak451_@pr.oxylabs.io:7777",
    }

# Shared counters
lock = threading.Lock()
stats = {
    "processed": 0,
    "rate_limits": 0,
    "errors": 0,
    "proxy_used": 0,
    "consecutive_rate_limits": 0,
}

def get_article_edits(title: str, force_proxy: bool = False):
    """
    Fetch monthly edit counts (last 2 years window)
    """
    # Small delay
    time.sleep(random.uniform(0.16, 0.22))

    encoded_title = quote(title.replace(" ", "_"))
    editor_types = ["user", "anonymous", "group-bot", "name-bot"]

    headers = {
        "Accept": "application/json",
        "User-Agent": "UniversityofKonstanzResearchBot/1.0 (https://uni-konstanz.de; lorenz.rueckert@uni-konstanz.de)",
    }

    # Rotate to proxy every 10th article
    with lock:
        rotate = (stats["processed"] % 10 == 0)
        use_proxy = force_proxy or rotate or stats["consecutive_rate_limits"] >= 3
    proxies = get_proxy() if use_proxy else None

    out = {}
    failed = 0

    for i, editor_type in enumerate(editor_types):
        url = (
            "https://wikimedia.org/api/rest_v1/metrics/edits/per-page/"
            f"en.wikipedia.org/{encoded_title}/{editor_type}/monthly/20230701/20250701"
        )
        try:
            resp = requests.get(url, headers=headers, proxies=proxies, timeout=10)

            if resp.status_code == 200:
                data = resp.json()
                if "items" in data and data["items"]:
                    monthly = data["items"][0].get("results", [])
                    out[editor_type] = sum(m.get("edits", 0) for m in monthly)
                else:
                    out[editor_type] = 0

            elif resp.status_code == 404:
                # Page not found  -> treat as zero
                out[editor_type] = 0

            elif resp.status_code == 429:
                with lock:
                    stats["rate_limits"] += 1
                    stats["consecutive_rate_limits"] += 1
                return title, None, "RATE_LIMITED", use_proxy

            else:
                out[editor_type] = 0
                failed += 1

        except Exception:
            out[editor_type] = 0
            failed += 1
        if i < 3:
            time.sleep(random.uniform(0.04, 0.06))

    out["all_editor_types"] = sum(out.values())

    with lock:
        if use_proxy:
            stats["proxy_used"] += 4
        if failed == 0:
            stats["consecutive_rate_limits"] = 0

    return title, out, "OK", use_proxy

def load_checkpoint():
    """
    Resume from the latest checkpoint
    """
    files = glob.glob("compliant_checkpoint_*.csv")
    if not files:
        return set(), []

    def suffix_num(path):
        try:
            return int(path.rsplit("_", 1)[-1].split(".")[0])
        except Exception:
            return -1

    latest = max(files, key=suffix_num)
    df_ckpt = pd.read_csv(latest)

    processed = set(df_ckpt["title"].tolist())
    results = []
    for _, row in df_ckpt.iterrows():
        edit_counts = {
            "user": int(row.get("edits_user", 0)),
            "anonymous": int(row.get("edits_anonymous", 0)),
            "group-bot": int(row.get("edits_group_bot", 0)),
            "name-bot": int(row.get("edits_name_bot", 0)),
            "all_editor_types": int(row.get("edits_all_types", 0)),
        }
        results.append((row["title"], edit_counts))

    print(f"Resuming from {latest} ({len(processed)} titles).")
    return processed, results

def save_checkpoint(results, count):
    """
    Save csv checkpoint
    """
    rows = []
    for title, ec in results:
        if not isinstance(ec, dict):
            ec = {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0, "all_editor_types": 0}
        row = {
            "title": title,
            "edits_all_types": ec.get("all_editor_types", 0),
            "edits_user": ec.get("user", 0),
            "edits_anonymous": ec.get("anonymous", 0),
            "edits_group_bot": ec.get("group-bot", 0),
            "edits_name_bot": ec.get("name-bot", 0),
        }
        row["edits_human"] = row["edits_user"] + row["edits_anonymous"]
        row["edits_bot"] = row["edits_group_bot"] + row["edits_name_bot"]
        rows.append(row)

    out = pd.DataFrame(rows)
    name = f"compliant_checkpoint_{count}.csv"
    out.to_csv(name, index=False)
    print(f"Saved {name}")

def main():
    processed_titles, all_results = load_checkpoint()
    remaining = [t for t in titles if t not in processed_titles]

    if not remaining:
        print("Nothing to do.")
        return all_results

    # Light threading
    chunk_size = 60
    max_workers = 6
    start = time.time()

    for start_idx in range(0, len(remaining), chunk_size):
        chunk = remaining[start_idx : start_idx + chunk_size]
        chunk_results = []

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(get_article_edits, t): t for t in chunk}
            for fut in as_completed(futures, timeout=180):
                title = futures[fut]
                try:
                    t, counts, status, used_proxy = fut.result(timeout=30)

                    # If rate-limited, try once more via proxy
                    if status == "RATE_LIMITED":
                        time.sleep(random.uniform(2.0, 4.0))
                        t, counts, status, used_proxy = get_article_edits(title, force_proxy=True)
                        if status == "RATE_LIMITED" or counts is None:
                            counts = {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0, "all_editor_types": 0}

                    if counts is None:
                        counts = {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0, "all_editor_types": 0}

                    chunk_results.append((title, counts))
                    with lock:
                        stats["processed"] += 1

                except Exception:
                    with lock:
                        stats["processed"] += 1
                        stats["errors"] += 1
                    chunk_results.append((title, {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0, "all_editor_types": 0}))

        all_results.extend(chunk_results)

        # Minimal progress info per chunk
        elapsed = max(time.time() - start, 1e-6)
        rate = stats["processed"] / elapsed
        print(f"Chunk {(start_idx // chunk_size) + 1}: processed={stats['processed']} rate={rate:.1f}/s RL={stats['rate_limits']} proxy={stats['proxy_used']}")

        if len(all_results) % 1500 == 0:
            save_checkpoint(all_results, len(all_results))

        # Small pause between chunks
        time.sleep(random.uniform(1.5, 3.0))

    return all_results

if __name__ == "__main__":
    try:
        results = main()
        if results:
            save_checkpoint(results, len(results))  # final save
            print(f"Done. Total rows: {len(results)}")
    except KeyboardInterrupt:
        print("Interrupted. You can rerun to resume.")