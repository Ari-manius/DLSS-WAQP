import requests
import pandas as pd
import time
import random
import re
import signal
import sys
import csv
from collections import deque, defaultdict
from typing import Dict, List, Any
import pickle

WIKI_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT = 30


class WikipediaNetworkScraper:
    """"
    - Scraper for wikipedia articles
    - BFS over seed pages up to a given depth
    - Streams article + link info straight to CSV
    """

    def __init__(
            self,
            delay_between_requests: float = 0.1,
            checkpoint_interval: int = 25000,
            max_links_per_article: int | None = None,
            max_queue_links_per_article: int = 50,
            max_queue_size: int = 50000,
            proxies=None,
    ):
        self.delay_between_requests = delay_between_requests
        self.checkpoint_interval = checkpoint_interval
        self.max_links_per_article = max_links_per_article
        self.max_queue_links_per_article = max_queue_links_per_article
        self.max_queue_size = max_queue_size
        self.proxies = proxies or []

        # minimal in-memory state
        self.visited_titles = set()
        self.quality_counts = defaultdict(int)
        self.depth_article_counts = defaultdict(int)

        # counters
        self.successful_articles = 0
        self.failed_articles = 0
        self.assessed_articles = 0
        self.total_articles_processed = 0
        self.total_links_processed = 0

        # timers
        self.start_time = time.time()

        # CSV writers
        self.articles_file = None
        self.links_file = None
        self.articles_writer = None
        self.links_writer = None

        # HTTP session
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "UniversityofKonstanzArticleResearchBot/1.0 (https://uni-konstanz.de; lorenz.rueckert@uni-konstanz.de)"
        })

        # allow Ctrl+C to exit cleanly
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        # try to leave useful files behind on interrupt
        self._close_files()
        self._save_state("interruption")
        sys.exit(0)

    def _init_csv(self):
        ts = time.strftime('%Y%m%d_%H%M%S')
        a_name = f"articles_stream_{ts}.csv"
        l_name = f"links_stream_{ts}.csv"

        self.articles_file = open(a_name, 'w', newline='', encoding='utf-8')
        self.links_file = open(l_name, 'w', newline='', encoding='utf-8')

        self.articles_writer = csv.DictWriter(
            self.articles_file,
            fieldnames=[
                'article', 'pageid', 'quality_class', 'importance', 'assessment_source',
                'num_categories', 'num_links', 'depth', 'is_seed', 'revid',
                'page_length', 'num_references', 'has_infobox', 'num_sections',
                'num_templates', 'last_timestamp', 'protection_status'
            ]
        )
        self.links_writer = csv.DictWriter(
            self.links_file,
            fieldnames=['source', 'target', 'edge_type', 'source_depth']
        )
        self.articles_writer.writeheader()
        self.links_writer.writeheader()

        print(f"writing: {a_name}, {l_name}")

    def _close_files(self):
        if self.articles_file:
            self.articles_file.close()
        if self.links_file:
            self.links_file.close()

    def _save_state(self, reason="checkpoint"):
        """Keep a small pickle"""
        try:
            ts = time.strftime('%Y%m%d_%H%M%S')
            name = f"queue_state_{reason}_{ts}.pkl"
            state = {
                'visited_titles': self.visited_titles,
                'successful_articles': self.successful_articles,
                'failed_articles': self.failed_articles,
                'total_articles_processed': self.total_articles_processed,
                'total_links_processed': self.total_links_processed,
                'depth_article_counts': dict(self.depth_article_counts),
                'quality_counts': dict(self.quality_counts),
                'assessed_articles': self.assessed_articles,
                'runtime_hours': (time.time() - self.start_time) / 3600,
            }
            with open(name, 'wb') as f:
                pickle.dump(state, f)
            print(f"saved state -> {name}")
        except Exception:
            pass

    def _request(self, url, params):
        """Simple retry wrapper for GET"""
        for attempt in range(3):
            try:
                r = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                r.raise_for_status()
                return r
            except requests.RequestException:
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None

    def fetch_article(self, title: str) -> Dict[str, Any] | None:
        """Pull a bunch of fields for one article + last rev."""
        params = {
            "action": "query",
            "format": "json",
            "prop": "info|categories|revisions|templates|pageassessments",
            "inprop": "protection",
            "cllimit": "max",
            "clshow": "!hidden",
            "rvprop": "content|timestamp|ids",
            "rvslots": "main",
            "tllimit": "max",
            "titles": title,
        }

        r = self._request(WIKI_API_ENDPOINT, params)
        if not r:
            return None

        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        if len(pages) != 1:
            return None

        page = list(pages.values())[0]
        if "missing" in page:
            return None

        # This check avoids redirects/odd cases
        if page.get("title", "") != title:
            return None

        pageid = page["pageid"]
        categories = [c["title"] for c in page.get("categories", [])]
        templates = page.get("templates", [])
        page_length = page.get("length", 0)

        revisions = page.get("revisions", [])
        content = ""
        last_timestamp = None
        revid = None
        if revisions:
            rev = revisions[0]
            main_slot = rev.get("slots", {}).get("main", {})
            content = main_slot.get("*", "") or rev.get("*", "")
            last_timestamp = rev.get("timestamp")
            revid = rev.get("revid")

        content_feats = self._content_features(content)
        protection = self._protection(page.get("protection", []))

        out = {
            "pageid": pageid,
            "title": title,
            "categories": categories,
            "revid": revid,
            "page_length": page_length,
            "num_references": content_feats["num_references"],
            "has_infobox": content_feats["has_infobox"],
            "num_sections": content_feats["num_sections"],
            "num_templates": len(templates),
            "last_timestamp": last_timestamp,
            "protection_status": protection,
            "templates": [t["title"] for t in templates],
        }

        # quality
        assessments = page.get("pageassessments", {})
        out["quality"] = self._pick_quality(assessments) if assessments else {"source": "none"}

        # links for the network
        out["links"] = self._links(pageid)

        time.sleep(self.delay_between_requests)  # be nice to the API - RESPECT
        return out

    def build_network(
            self,
            seed_articles: List[str],
            max_depth: int = 6,
            filter_namespaces: bool = True
    ) -> Dict[str, Any]:
        """Basic BFS by depth"""
        self._init_csv()

        q = deque((seed, 0) for seed in seed_articles)
        current_depth = 0

        while q and current_depth <= max_depth:
            batch = []
            # collect a small batch at the same depth
            while q and len(batch) < 20:
                t, d = q[0]
                if d == current_depth:
                    q.popleft()
                    if t not in self.visited_titles:
                        self.visited_titles.add(t)
                        batch.append((t, d))
                else:
                    break

            if not batch:
                current_depth += 1
                continue

            for title, depth in batch:
                try:
                    data = self.fetch_article(title)
                    if not data:
                        self.failed_articles += 1
                        continue

                    if filter_namespaces and ":" in data.get("title", ""):
                        continue

                    quality = self._best_quality_label(data)
                    self.quality_counts[quality] += 1
                    if quality != "Unknown":
                        self.assessed_articles += 1
                    self.depth_article_counts[depth] += 1

                    # write article row
                    record = {
                        'article': title,
                        'pageid': data.get('pageid'),
                        'quality_class': quality,
                        'importance': data.get('quality', {}).get('importance', ''),
                        'assessment_source': data.get('quality', {}).get('source', 'Unknown'),
                        'num_categories': len(data.get('categories', [])),
                        'num_links': len(data.get('links', [])),
                        'depth': depth,
                        'is_seed': title in seed_articles,
                        'revid': data.get('revid'),
                        'page_length': data.get('page_length', 0),
                        'num_references': data.get('num_references', 0),
                        'has_infobox': data.get('has_infobox', False),
                        'num_sections': data.get('num_sections', 0),
                        'num_templates': data.get('num_templates', 0),
                        'last_timestamp': data.get('last_timestamp', ''),
                        'protection_status': data.get('protection_status', 'unprotected'),
                    }
                    self._write_article(record)

                    # Write link rows
                    links = data.get('links', [])
                    if links:
                        rows = [{
                            'source': title,
                            'target': link.get('title'),
                            'edge_type': 'internal_link',
                            'source_depth': depth
                        } for link in links if link.get('title')]
                        self._write_links(rows)

                    # queue next depth
                    if depth < max_depth:
                        titles_next = [
                            l.get('title') for l in links
                            if l.get('title') and ':' not in l.get('title')
                        ]
                        titles_next = titles_next[:self.max_queue_links_per_article]
                        for t2 in titles_next:
                            if t2 not in self.visited_titles and len(q) < self.max_queue_size:
                                q.append((t2, depth + 1))

                    self.successful_articles += 1

                    # checkpoint
                    if self.total_articles_processed and (
                            self.total_articles_processed % self.checkpoint_interval == 0):
                        self._save_state("checkpoint")

                except Exception:
                    self.failed_articles += 1
                    continue

        self._close_files()
        self._print_summary()

        return {
            'total_articles': self.total_articles_processed,
            'total_links': self.total_links_processed,
            'assessed_articles': self.assessed_articles,
            'assessment_rate': (
                        self.assessed_articles / self.total_articles_processed * 100) if self.total_articles_processed else 0,
            'depth_distribution': dict(self.depth_article_counts),
            'quality_distribution': dict(self.quality_counts),
            'runtime_hours': (time.time() - self.start_time) / 3600,
        }

    # small helpers below

    def _write_article(self, row: Dict[str, Any]):
        try:
            self.articles_writer.writerow(row)
            self.articles_file.flush()
            self.total_articles_processed += 1
        except Exception:
            pass

    def _write_links(self, rows: List[Dict[str, Any]]):
        try:
            for r in rows:
                self.links_writer.writerow(r)
            self.links_file.flush()
            self.total_links_processed += len(rows)
        except Exception:
            pass

    def _content_features(self, content: str) -> Dict[str, Any]:
        if not content:
            return {"num_references": 0, "has_infobox": False, "num_sections": 0}
        num_references = len(re.findall(r'<ref[^>]*>', content, re.IGNORECASE))
        has_infobox = bool(re.search(r'\{\{\s*[Ii]nfobox', content))
        num_sections = len(re.findall(r'^={2,6}[^=]+={2,6}\s*$', content, re.MULTILINE))
        return {"num_references": num_references, "has_infobox": has_infobox, "num_sections": num_sections}

    def _protection(self, prot_list: List[Dict]) -> str:
        if not prot_list:
            return "unprotected"
        for p in prot_list:
            if p.get("type") == "edit":
                lvl = p.get("level", "")
                if lvl == "sysop":
                    return "fully_protected"
                if lvl == "autoconfirmed":
                    return "semi_protected"
        return "protected"

    def _links(self, pageid: int) -> List[Dict[str, Any]]:
        links, cont = [], None
        max_links = self.max_links_per_article or float('inf')

        while len(links) < max_links:
            params = {
                "action": "query",
                "format": "json",
                "prop": "links",
                "pageids": str(pageid),
                "pllimit": min(500, int(max_links - len(links))) if max_links != float('inf') else 500,
                "plnamespace": 0
            }
            if cont:
                params["plcontinue"] = cont

            r = self._request(WIKI_API_ENDPOINT, params)
            if not r:
                break

            data = r.json()
            pages = data.get("query", {}).get("pages", {})
            for p in pages.values():
                if "missing" in p:
                    break
                links.extend(p.get("links", []))

            cont = data.get("continue", {}).get("plcontinue")
            if not cont:
                break

        return links

    def _pick_quality(self, assessments: Dict) -> Dict[str, Any]:
        # prefer “project-independent” if present, otherwise best (FA > GA > B > C > Start > Stub)
        if 'Project-independent assessment' in assessments:
            c = assessments['Project-independent assessment'].get('class', '')
            if c and c.strip():
                return {"class": c, "source": "Project-independent assessment"}

        order = {'FA': 0, 'GA': 1, 'B': 2, 'C': 3, 'Start': 4, 'Stub': 5}
        best = None
        source = None
        for proj, data in assessments.items():
            if isinstance(data, dict):
                c = data.get('class', '')
                if c in order and (best is None or order[c] < order[best]):
                    best, source = c, proj
        return {"class": best, "source": source} if best else {"source": "none"}

    def _best_quality_label(self, article_data: Dict[str, Any]) -> str:
        q = article_data.get("quality", {})
        return q.get("class", "Unknown") or "Unknown"

    def _print_summary(self):
        hours = (time.time() - self.start_time) / 3600
        print("\n--- scrape summary ---")
        print(f"articles: {self.total_articles_processed}")
        print(f"links:    {self.total_links_processed}")
        print(
            f"assessed: {self.assessed_articles} ({(self.assessed_articles / self.total_articles_processed * 100):.1f}%)" if self.total_articles_processed else "assessed: 0")
        print(f"hours:    {hours:.1f}")


def get_oxylabs_proxies():
    # ideally load secrets from env vars instead
    username = "claude_scrape_oJ3eG"
    password = "TeamSpeak451="
    endpoint = "pr.oxylabs.io:7777"
    proxy_username = f"customer-{username}"
    proxy_url = f"http://{proxy_username}:{password}@{endpoint}"
    return [{"http": proxy_url, "https": proxy_url} for _ in range(15)]


if __name__ == "__main__":
    # default config / starting nodes
    seeds = ["Influenza", "Serena Williams", "French Revolution", "Quantum mechanics"]
    max_depth = 6
    max_queue_links = 50

    use_proxies = input("Use proxies? (y/n): ").strip().lower().startswith("y")
    proxies = get_oxylabs_proxies() if use_proxies else None

    scraper = WikipediaNetworkScraper(
        delay_between_requests=0.08 if use_proxies else 0.12,
        checkpoint_interval=25000,
        max_links_per_article=None,
        max_queue_links_per_article=max_queue_links,
        max_queue_size=50000,
        proxies=proxies,
    )

    print("starting…")
    try:
        result = scraper.build_network(
            seed_articles=seeds,
            max_depth=max_depth,
            filter_namespaces=True
        )
        print("done.")
        for k, v in result.items():
            print(f"{k}: {v}")
    except Exception as e:
        print(f"error: {e}")