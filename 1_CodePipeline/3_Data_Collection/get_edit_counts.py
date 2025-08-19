import pandas as pd
import requests
import time
from urllib.parse import quote
import random
import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading


df = pd.read_csv("cleaned_articles_final.csv")
titles = df["article"].tolist()

# Proxy config
def get_proxy():
    return {
        "http": "http://edit_counts_Lb51Z:@pr.oxylabs.io:7777",
        "https": "http://edit_counts_Lb51Z:@pr.oxylabs.io:7777"
    }
lock = threading.Lock()
stats = {
    'processed': 0,
    'direct_success': 0,
    'proxy_used': 0,
    'rate_limits': 0,
    'errors': 0,
    'bandwidth_mb': 0,
    'consecutive_rate_limits': 0,
    'zero_edits': 0
}


def get_article_edits_compliant(title, force_proxy=False):
    """API-COMPLIANT BASED ON DOCUMENTATION OF THE ENDPOINT:"""

    # delay to stay under 25/sec limit
    # 6 threads × 4 requests = 24 requests/sec target
    time.sleep(random.uniform(0.16, 0.22))

    # Encode title for URL
    encoded_title = quote(title.replace(' ', '_'))

    editor_types = ["user", "anonymous", "group-bot", "name-bot"]

    headers = {
        "Accept": "application/json",
        "User-Agent": "UniversityofKonstanzResearchBot/1.0 (https://uni-konstanz.de; lorenz.rueckert@uni-konstanz.de)"
    }
    # Proxy rotation
    with lock:
        should_use_proxy = (stats['processed'] % 10 == 0)
        use_proxy = force_proxy or should_use_proxy or stats['consecutive_rate_limits'] >= 3

    proxies = get_proxy() if use_proxy else None

    results = {}
    failed_count = 0

    # Get data for each editor type
    for i, editor_type in enumerate(editor_types):
        url = f"https://wikimedia.org/api/rest_v1/metrics/edits/per-page/en.wikipedia.org/{encoded_title}/{editor_type}/monthly/20230701/20250701"

        try:
            response = requests.get(url, headers=headers, proxies=proxies, timeout=10)

            if response.status_code == 200:
                data = response.json()
                if "items" in data and len(data["items"]) > 0:
                    monthly_data = data["items"][0].get("results", [])
                    total_edits = sum(month.get("edits", 0) for month in monthly_data)
                    results[editor_type] = total_edits
                else:
                    results[editor_type] = 0
            elif response.status_code == 404:
                results[editor_type] = 0
            elif response.status_code == 429:
                with lock:
                    stats['rate_limits'] += 1
                    stats['consecutive_rate_limits'] += 1
                return title, None, use_proxy, "RATE_LIMITED"
            else:
                results[editor_type] = 0
                failed_count += 1

            if i < 3:  # Don't delay after last request
                time.sleep(random.uniform(0.04, 0.06))  # Small delay between the 4 requests

        except Exception as e:
            results[editor_type] = 0
            failed_count += 1

    # Calculate total edits
    results["all_editor_types"] = sum(results.values())

    # Update stats
    with lock:
        if use_proxy:
            stats['proxy_used'] += 4
        else:
            stats['direct_success'] += 4
        stats['bandwidth_mb'] += 0.008

        if failed_count == 0:
            stats['consecutive_rate_limits'] = 0

        if results["all_editor_types"] == 0:
            stats['zero_edits'] += 1
        if failed_count > 2:
            stats['errors'] += 1

    return title, results, use_proxy, "SUCCESS"


def save_progress(results, count):
    """Save checkpoint"""
    data_rows = []
    for title, edit_counts in results:
        if isinstance(edit_counts, dict):
            row = {
                "title": title,
                "edits_all_types": edit_counts.get("all_editor_types", 0),
                "edits_user": edit_counts.get("user", 0),
                "edits_anonymous": edit_counts.get("anonymous", 0),
                "edits_group_bot": edit_counts.get("group-bot", 0),
                "edits_name_bot": edit_counts.get("name-bot", 0)
            }
            # Calculate human and bot totals
            row["edits_human"] = row["edits_user"] + row["edits_anonymous"]
            row["edits_bot"] = row["edits_group_bot"] + row["edits_name_bot"]
        else:
            row = {
                "title": title,
                "edits_all_types": 0,
                "edits_user": 0,
                "edits_anonymous": 0,
                "edits_group_bot": 0,
                "edits_name_bot": 0,
                "edits_human": 0,
                "edits_bot": 0
            }
        data_rows.append(row)

    df = pd.DataFrame(data_rows)
    filename = f"compliant_checkpoint_{count}.csv"
    df.to_csv(filename, index=False)
    print(f"💾 COMPLIANT checkpoint: {filename} ({count:,} articles)")


def main():
    start_time = time.time()
    chunk_size = 60
    max_workers = 6

        chunk_results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_article_edits_compliant, title): title for title in chunk}

            for future in as_completed(futures, timeout=120):
                try:
                    title, edit_counts, used_proxy, status = future.result(timeout=30)

                    if status == "RATE_LIMITED":
                        print(f"Rate limited: {title[:30]}... retrying with proxy")
                        time.sleep(random.uniform(3, 6))  # Longer wait for compliance
                        title, edit_counts, used_proxy, status = get_article_edits_compliant(title, force_proxy=True)
                        if status == "RATE_LIMITED":
                            print(f"❌ Still rate limited: {title[:30]}... skipping")
                            edit_counts = {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0,
                                           "all_editor_types": 0}

                    if edit_counts is None:
                        edit_counts = {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0, "all_editor_types": 0}

                    chunk_results.append((title, edit_counts))

                    with lock:
                        stats['processed'] += 1

                        # Show progress every 100 for compliance (less spammy)
                        if stats['processed'] <= 15 or stats['processed'] % 100 == 0:
                            proxy_status = "🔄PROXY" if used_proxy else "📡DIRECT"
                            if used_proxy and stats['processed'] % 10 == 0:
                                proxy_status += " (rotation)"

                            total = edit_counts.get("all_editor_types", 0)
                            human = edit_counts.get("user", 0) + edit_counts.get("anonymous", 0)
                            bot = edit_counts.get("group-bot", 0) + edit_counts.get("name-bot", 0)

                            print(f"✅ {title[:35]:<35} = {total:4d} ({human:3d}👥 {bot:3d}🤖) ({proxy_status})")

                except Exception as e:
                    title = futures[future]
                    chunk_results.append(
                        (title, {"user": 0, "anonymous": 0, "group-bot": 0, "name-bot": 0, "all_editor_types": 0}))
                    with lock:
                        stats['processed'] += 1
                        stats['errors'] += 1
                    print(f"Error: {title[:30]}... skipping")

        all_results.extend(chunk_results)

        # Progress report every 3 chunks
        if chunk_num % 3 == 0:
            elapsed = time.time() - start_time
            rate = stats['processed'] / elapsed if elapsed > 0 else 0
            eta_hours = (len(titles) - stats['processed']) / rate / 3600 if rate > 0 else 0

            with lock:
                total_requests = stats['direct_success'] + stats['proxy_used']
                direct_pct = (stats['direct_success'] / total_requests * 100) if total_requests > 0 else 0
                proxy_pct = (stats['proxy_used'] / total_requests * 100) if total_requests > 0 else 0

                # Calculate actual request rate
                req_rate = total_requests / elapsed if elapsed > 0 else 0

            print(
                f"📊 COMPLIANT: {stats['processed']:,}/{len(titles):,} ({stats['processed'] / len(titles) * 100:.1f}%) "
                f"⚖️{rate:.1f}/sec ETA:{eta_hours:.1f}h")
            print(f"   📡Direct:{direct_pct:.0f}% 🔄Proxy:{proxy_pct:.0f}% 💾BW:{stats['bandwidth_mb']:.1f}MB "
                  f"RL:{stats['rate_limits']}Err:{stats['errors']}ReqRate:{req_rate:.1f}/sec")

        # Save checkpoint every 1,500 articles
        if len(all_results) % 1500 == 0 and len(all_results) > 0:
            save_progress(all_results, len(all_results))

        time.sleep(random.uniform(2, 4))

    return all_results
