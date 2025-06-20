# crawler.py
import os
import logging
import time
import hashlib
import re
import html2text
import json
import datetime
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin, urlparse
from collections import defaultdict, deque
from xml.etree.ElementTree import Element, SubElement, ElementTree
from typing import Optional, Set, Dict, List
from math import ceil
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib3.exceptions import InsecureRequestWarning
from config import (
    START_URL,
    MAX_DEPTH,
    USE_PLAYWRIGHT,
    DOWNLOAD_PDF,
    DOWNLOAD_DOC,
    DOWNLOAD_IMAGE,
    DOWNLOAD_OTHER,
    LLM_PROVIDER,
    OPENAI_API_KEYS,
    MAX_TOKENS,
    MAX_URLS,
    CRAWLER_OUTPUT_DIR,
    CHECKPOINT_FILE,
    VERBOSE
)
import subprocess

# Désactiver les avertissements SSL
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

class WebCrawler:
    def __init__(
        self,
        start_url: str = START_URL,
        max_depth: int = MAX_DEPTH,
        use_playwright: bool = USE_PLAYWRIGHT,
        download_pdf: bool = DOWNLOAD_PDF,
        download_doc: bool = DOWNLOAD_DOC,
        download_image: bool = DOWNLOAD_IMAGE,
        download_other: bool = DOWNLOAD_OTHER,
        llm_provider: Optional[str] = LLM_PROVIDER,
        api_keys: Optional[List[str]] = OPENAI_API_KEYS,
        max_tokens_per_request: int = MAX_TOKENS,
        max_urls: Optional[int] = MAX_URLS,
        base_dir: str = CRAWLER_OUTPUT_DIR
    ):

        self.start_url = start_url
        self.max_depth = max_depth
        self.use_playwright = use_playwright
        self.download_pdf = download_pdf
        self.download_doc = download_doc
        self.download_image = download_image
        self.download_other = download_other
        self.llm_provider = llm_provider
        self.api_keys = api_keys if api_keys else []
        self.llm_enabled = bool(llm_provider and self.api_keys)
        self.max_tokens_per_request = max_tokens_per_request
        self.chars_per_token = 4
        self.max_chars_per_chunk = self.max_tokens_per_request * self.chars_per_token
        self.max_urls = max_urls
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.visited_pages = set()
        self.downloaded_files = set()
        self.domain = urlparse(start_url).netloc
        self.site_map: Dict[str, Set[str]] = defaultdict(set)

        self.excluded_paths = ['selecteur-de-produits']

        # Configuration du logging
        (self.base_dir / 'logs').mkdir(parents=True, exist_ok=True)
        log_format = '%(asctime)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(self.base_dir / 'logs' / 'crawler.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.stats = defaultdict(int)
        self.downloadable_extensions = {
            'PDF': ['.pdf'] if self.download_pdf else [],
            'Image': ['.png', '.jpg', '.jpeg', '.gif', '.svg'] if self.download_image else [],
            'Doc': ['.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'] if self.download_doc else [],
            'Archive': ['.zip', '.rar', '.7z', '.tar', '.gz'] if self.download_other else [],
            'Audio': ['.mp3', '.wav', '.ogg'] if self.download_other else [],
            'Video': ['.mp4', '.avi', '.mov', '.mkv'] if self.download_other else []
        }
        self.downloadable_extensions = {k: v for k, v in self.downloadable_extensions.items() if v}
        self.all_downloadable_exts = {ext for exts in self.downloadable_extensions.values() for ext in exts}

        self.content_type_mapping = {
            'PDF': {'application/pdf': '.pdf'},
            'Image': {
                'image/jpeg': '.jpg',
                'image/png': '.png',
                'image/gif': '.gif',
                'image/svg+xml': '.svg',
            },
            'Doc': {
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
                'application/vnd.ms-excel': '.xls',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
                'application/vnd.ms-powerpoint': '.ppt',
                'application/vnd.openxmlformats-officedocument.presentationml.presentation': '.pptx'
            },
            'Archive': {
                'application/zip': '.zip',
                'application/x-rar-compressed': '.rar',
                'application/x-7z-compressed': '.7z',
                'application/gzip': '.gz',
                'application/x-tar': '.tar'
            },
            'Audio': {
                'audio/mpeg': '.mp3',
                'audio/wav': '.wav',
                'audio/ogg': '.ogg'
            },
            'Video': {
                'video/mp4': '.mp4',
                'video/x-msvideo': '.avi',
                'video/quicktime': '.mov'
            }
        }

        self.session = self.setup_session()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        self.html_converter.body_width = 0
        self.html_converter.ignore_images = True
        self.html_converter.single_line_break = False

        # Extraire le pattern de langue depuis l'URL de départ (optionnel)
        self.language_path = re.search(r'/(fr|en)-(ca|us)/', start_url)
        self.language_pattern = self.language_path.group(0) if self.language_path else None

        # Création des dossiers
        self.create_directories()

        # Playwright (non obligatoire)
        self.playwright = None
        self.browser = None
        self.page = None
        if self.use_playwright:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()

    def setup_session(self) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=5,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.verify = False
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 ...'
        })
        return session

    def create_directories(self):
        directories = ['content', 'PDF', 'Image', 'Doc', 'Archive', 'Audio', 'Video', 'logs', 'content_rewritten']
        for dir_name in directories:
            (self.base_dir / dir_name).mkdir(parents=True, exist_ok=True)

    def should_exclude(self, url: str) -> bool:
        return any(excluded in url for excluded in self.excluded_paths)

    def is_same_language(self, url: str) -> bool:
        if not self.language_pattern:
            return True
        return self.language_pattern in url

    def is_downloadable_file(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        if not self.all_downloadable_exts:
            return False
        pattern = re.compile(r'\.(' + '|'.join(ext.strip('.') for ext in self.all_downloadable_exts) + r')(\.[a-z0-9]+)?$', re.IGNORECASE)
        return bool(pattern.search(path))

    def head_or_get(self, url: str) -> Optional[requests.Response]:
        try:
            r = self.session.head(url, allow_redirects=True, timeout=10)
            if r.status_code == 405:  # Méthode non autorisée, tenter GET
                r = self.session.get(url, allow_redirects=True, timeout=10, stream=True)
            return r
        except:
            try:
                return self.session.get(url, allow_redirects=True, timeout=10, stream=True)
            except:
                return None

    def get_file_type_and_extension(self, url: str, response: requests.Response):
        if response is None:
            return None, None
        path = urlparse(url).path.lower()
        content_type = response.headers.get('Content-Type', '').lower()

        for file_type, extensions in self.downloadable_extensions.items():
            for ext in extensions:
                pattern = re.compile(re.escape(ext) + r'(\.[a-z0-9]+)?$', re.IGNORECASE)
                if pattern.search(path):
                    return file_type, self.content_type_mapping.get(file_type, {}).get(content_type, ext)

        for file_type, mapping in self.content_type_mapping.items():
            if content_type in mapping:
                return file_type, mapping[content_type]

        return None, None

    def sanitize_filename(self, url: str, extension: str) -> str:
        url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
        filename = url.split('/')[-1] or 'index'
        filename = re.sub(r'[^\w\-_.]', '_', filename)
        name = Path(filename).stem
        if not extension:
            extension = '.txt'
        sanitized = f"{name}_{url_hash}{extension}"
        return sanitized

    def download_file(self, url: str) -> bool:
        response = self.head_or_get(url)
        if not response or response.status_code != 200:
            logging.warning(f"Failed to retrieve file at {url}")
            return False

        file_type_detected, extension = self.get_file_type_and_extension(url, response)
        if not file_type_detected:
            logging.warning(f"Could not determine the file type for: {url}")
            return False

        if file_type_detected not in self.downloadable_extensions:
            logging.info(f"File type {file_type_detected} not enabled for download.")
            return False

        logging.info(f"Attempting to download {file_type_detected} file from: {url}")

        filename = self.sanitize_filename(url, extension)
        save_path = self.base_dir / file_type_detected / filename

        if save_path.exists():
            logging.info(f"File already downloaded, skipping: {filename}")
            return False

        try:
            if response.request.method == 'HEAD':
                response = self.session.get(url, stream=True, timeout=20)

            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            self.stats[f'{file_type_detected}_downloaded'] += 1
            self.downloaded_files.add(url)
            logging.info(f"Successfully downloaded {file_type_detected}: {filename}")
            return True
        except Exception as e:
            logging.error(f"Error downloading {url}: {str(e)}")
            return False

    def fetch_page_content(self, url: str) -> Optional[str]:
        if self.use_playwright and self.page:
            try:
                self.page.goto(url, timeout=20000)
                time.sleep(2)
                return self.page.content()
            except Exception as e:
                logging.error(f"Playwright failed to fetch {url}: {str(e)}")
                return None
        else:
            try:
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    return response.text
                else:
                    logging.warning(f"Failed to fetch {url}, status code: {response.status_code}")
                    return None
            except Exception as e:
                logging.error(f"Requests failed to fetch {url}: {str(e)}")
                return None

    def convert_links_to_absolute(self, soup: BeautifulSoup, base_url: str) -> BeautifulSoup:
        for tag in soup.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            attr = 'href' if tag.name == 'a' else 'src'
            href = tag.get(attr)
            if href:
                absolute_url = urljoin(base_url, href)
                tag[attr] = absolute_url
        return soup

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        return text.strip()

    def extract_urls(self, start_url: str):
        queue = deque([(start_url, 0)])
        self.visited_pages.add(start_url)
        crawled_count = 0

        while queue:
            current_url, depth = queue.popleft()

            if self.max_urls is not None and crawled_count >= self.max_urls:
                logging.info(f"Reached max_urls limit ({self.max_urls}), stopping URL extraction.")
                break

            if self.max_urls is None and depth > self.max_depth:
                continue

            if self.should_exclude(current_url):
                logging.info(f"Excluded URL: {current_url}")
                continue

            logging.info(f"Extracting URLs from: {current_url} (depth: {depth})")
            crawled_count += 1

            if self.is_downloadable_file(current_url):
                self.download_file(current_url)
                continue

            page_content = self.fetch_page_content(current_url)
            if page_content is None:
                logging.warning(f"Could not retrieve content for: {current_url}")
                continue

            soup = BeautifulSoup(page_content, 'html.parser')
            child_links = set()
            for tag in soup.find_all(['a', 'link', 'embed', 'iframe', 'object'], href=True):
                href = tag.get('href') or tag.get('src')
                if not href:
                    continue
                absolute_url = urljoin(current_url, href)
                parsed_url = urlparse(absolute_url)

                if self.is_downloadable_file(absolute_url):
                    self.download_file(absolute_url)
                    continue

                if (self.domain in parsed_url.netloc
                        and self.is_same_language(absolute_url)
                        and not absolute_url.endswith(('#', 'javascript:void(0)', 'javascript:;'))
                        and not self.should_exclude(absolute_url)):
                    child_links.add(absolute_url)
                    if absolute_url not in self.visited_pages:
                        if self.max_urls is None or crawled_count < self.max_urls:
                            if self.max_urls is None and depth + 1 > self.max_depth:
                                continue
                            queue.append((absolute_url, depth + 1))
                            self.visited_pages.add(absolute_url)

            self.site_map[current_url].update(child_links)

    def extract_content(self, url: str):
        if self.is_downloadable_file(url):
            logging.debug(f"Skipping content extraction for downloadable file: {url}")
            return

        page_content = self.fetch_page_content(url)
        if page_content is None:
            logging.warning(f"Could not retrieve content for: {url}")
            return

        soup = BeautifulSoup(page_content, 'html.parser')
        for element in soup.find_all(['nav', 'header', 'footer', 'script', 'style', 'aside', 'iframe']):
            element.decompose()

        main_content = (soup.find('main') or soup.find('article') or 
                        soup.find('div', class_='content') or soup.find('div', id='content'))

        if not main_content:
            logging.warning(f"No main content found for: {url}")
            return

        self.convert_links_to_absolute(main_content, url)
        markdown_content = self.html_converter.handle(str(main_content))

        title = soup.find('h1')
        content_parts = []
        if title:
            content_parts.append(f"# {title.get_text().strip()}")
        content_parts.append(f"**Source:** {url}")
        content_parts.append(markdown_content)

        content = self.clean_text('\n\n'.join(content_parts))

        if content:
            filename = self.sanitize_filename(url, '.txt')
            save_path = self.base_dir / 'content' / filename
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(content)
            self.stats['pages_processed'] += 1
            logging.info(f"Successfully saved content to: {filename}")
        else:
            logging.warning(f"No significant content found for: {url}")

        for tag in main_content.find_all(['a', 'embed', 'iframe', 'object'], href=True):
            href = tag.get('href') or tag.get('src')
            if href:
                file_url = urljoin(url, href)
                if self.is_downloadable_file(file_url) and file_url not in self.downloaded_files:
                    self.download_file(file_url)

    def load_downloaded_files(self):
        downloaded_files_path = self.base_dir / 'logs' / 'downloaded_files.txt'
        if downloaded_files_path.exists():
            with open(downloaded_files_path, 'r', encoding='utf-8') as f:
                for line in f:
                    self.downloaded_files.add(line.strip())
            logging.info(f"Loaded {len(self.downloaded_files)} downloaded files.")
        else:
            logging.info("No downloaded files tracking file found, starting fresh.")

    def save_downloaded_files(self):
        downloaded_files_path = self.base_dir / 'logs' / 'downloaded_files.txt'
        try:
            with open(downloaded_files_path, 'w', encoding='utf-8') as f:
                for url in sorted(self.downloaded_files):
                    f.write(url + '\n')
            logging.info(f"Saved {len(self.downloaded_files)} downloaded files.")
        except Exception as e:
            logging.error(f"Error saving downloaded files tracking: {str(e)}")

    def generate_report(self, duration: float, error: Optional[str] = None):
        report_lines = [
            f"Crawler Report",
            f"==============",
            f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Configuration",
            "------------",
            f"Start URL: {self.start_url}",
            f"Language Pattern: {self.language_pattern}",
            f"Max Depth: {self.max_depth}",
            f"Max URLs: {self.max_urls}" if self.max_urls is not None else "",
            f"Duration: {duration:.2f} seconds",
            "",
            "Statistics",
            "---------",
            f"Total URLs found: {len(self.visited_pages)}",
            f"Pages processed: {self.stats['pages_processed']}"
        ]

        for category in self.downloadable_extensions.keys():
            report_lines.append(f"- {category}: {self.stats[category+'_downloaded']}")

        report_lines.append("")

        if error:
            report_lines.extend([
                "Errors",
                "------",
                f"Critical Error: {error}",
                ""
            ])

        report_lines.append("Processed URLs")
        report_lines.append("-------------")
        for url in sorted(self.visited_pages):
            report_lines.append(url)

        report_lines.append("")
        report_lines.append("Generated Files")
        report_lines.append("--------------")

        for directory in ['content'] + list(self.downloadable_extensions.keys()):
            dir_path = self.base_dir / directory
            if dir_path.exists():
                files = list(dir_path.iterdir())
                report_lines.append(f"\n{directory} Files ({len(files)}):")
                for file in sorted(files):
                    report_lines.append(f"- {file.name}")

        report_content = "\n".join(report_lines)
        report_path = self.base_dir / 'crawler_report.txt'

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logging.info(f"Report generated successfully: {report_path}")
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")

        total_downloaded = sum(self.stats[k] for k in self.stats if k.endswith('_downloaded'))

        summary = f"""
Crawling Summary
---------------
Start URL: {self.start_url}
Total URLs: {len(self.visited_pages)}
Pages Processed: {self.stats['pages_processed']}
Total Files Downloaded: {total_downloaded}
Duration: {duration:.2f} seconds
Status: {'Completed with errors' if error else 'Completed successfully'}
"""

        try:
            with open(self.base_dir / 'summary.txt', 'w', encoding='utf-8') as f:
                f.write(summary)
            logging.info("Summary generated successfully.")
        except Exception as e:
            logging.error(f"Error generating summary: {str(e)}")

    def generate_json_report(self, duration: float, error: Optional[str] = None):
        total_downloaded = sum(self.stats[k] for k in self.stats if k.endswith('_downloaded'))
        report_data = {
            "configuration": {
                "start_url": self.start_url,
                "language_pattern": self.language_pattern,
                "max_depth": self.max_depth,
                "max_urls": self.max_urls,
                "duration": duration
            },
            "statistics": {
                "total_urls_found": len(self.visited_pages),
                "pages_processed": self.stats['pages_processed'],
                "files_downloaded": {
                    cat: self.stats[cat + '_downloaded'] for cat in self.downloadable_extensions.keys()
                },
                "total_files_downloaded": total_downloaded
            },
            "status": "Completed with errors" if error else "Completed successfully",
            "visited_pages": sorted(self.visited_pages),
            "downloaded_files": sorted(self.downloaded_files),
            "error": error if error else None
        }

        json_report_path = self.base_dir / 'report.json'
        try:
            with open(json_report_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=4, ensure_ascii=False)
            logging.info(f"JSON report generated successfully: {json_report_path}")
        except Exception as e:
            logging.error(f"Error generating JSON report: {str(e)}")

    def generate_xml_sitemap(self):
        visited = set()

        def add_page_element(parent_elem, url):
            if url in visited:
                return
            visited.add(url)
            page_elem = SubElement(parent_elem, "page", url=url)
            for child_url in sorted(self.site_map[url]):
                add_page_element(page_elem, child_url)

        root = Element("site", start_url=self.start_url)
        add_page_element(root, self.start_url)

        tree = ElementTree(root)
        xml_path = self.base_dir / 'sitemap.xml'
        try:
            tree.write(xml_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"XML sitemap generated successfully: {xml_path}")
        except Exception as e:
            logging.error(f"Error generating XML sitemap: {str(e)}")

    def save_checkpoint(self):
        checkpoint_data = {
            "visited_pages": list(self.visited_pages),
            "downloaded_files": list(self.downloaded_files),
            "site_map": {k: list(v) for k, v in self.site_map.items()},
            "stats": dict(self.stats)
        }
        try:
            with open(CHECKPOINT_FILE, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, ensure_ascii=False, indent=4)
            logging.info(f"Checkpoint saved: {CHECKPOINT_FILE}")
        except Exception as e:
            logging.error(f"Error saving checkpoint: {str(e)}")

    def load_checkpoint(self):
        if os.path.exists(CHECKPOINT_FILE):
            try:
                with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                self.visited_pages = set(checkpoint_data.get("visited_pages", []))
                self.downloaded_files = set(checkpoint_data.get("downloaded_files", []))
                self.site_map = {k: set(v) for k, v in checkpoint_data.get("site_map", {}).items()}
                self.stats = defaultdict(int, checkpoint_data.get("stats", {}))
                logging.info(f"Checkpoint loaded: {CHECKPOINT_FILE}")
            except Exception as e:
                logging.error(f"Error loading checkpoint: {str(e)}")

    def crawl(self):
        start_time = time.time()
        logging.info(f"Starting crawl of {self.start_url}")
        logging.info(f"Maximum depth: {self.max_depth}")
        if self.max_urls is not None:
            logging.info(f"Maximum URLs to crawl: {self.max_urls}")

        self.load_downloaded_files()
        self.load_checkpoint()
        error = None
        try:
            # Phase 1: Extraction des URLs
            logging.info("Phase 1: Starting URL extraction")
            if not self.visited_pages:
                # Pas de checkpoint, on commence
                self.extract_urls(self.start_url)
                self.save_checkpoint()  # sauvegarde après extraction URLs
            else:
                logging.info("Checkpoint found, skipping URL extraction phase.")

            # Phase 2: Extraction du contenu
            logging.info("Phase 2: Starting content extraction")
            # On pourrait checker un checkpoint séparé, mais ici on suppose qu'on repart au même état.
            for i, url in enumerate(self.visited_pages, 1):
                self.extract_content(url)
            logging.info("Phase 2: Completed content extraction")

            self.save_checkpoint()

        except Exception as e:
            error = str(e)
            logging.error(f"Critical error during crawling: {str(e)}")

        end_time = time.time()
        duration = end_time - start_time
        self.generate_report(duration, error=error)
        self.generate_json_report(duration, error=error)
        self.generate_xml_sitemap()

        self.save_downloaded_files()
        if self.use_playwright:
            self.page.close()
            self.browser.close()
            self.playwright.stop()
