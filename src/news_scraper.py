from typing import Iterable, Optional, List, Dict
from requests import Session, RequestException, Request, Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from bs4 import BeautifulSoup
import cloudscraper
import re
import time
import random
import threading

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

DEFAULT_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_0) AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/16.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/118.0.0.0 Safari/537.36",
]

class NewsScraper:
    """
    News scraper class. Gets a link and returns the text content.
    
    For scraping, it uses a requests session, a selenium driver or a cloudscraper.
        
    """
    def __init__(self):
        self.session = self.get_session()
        self.driver = self.get_driver()
    
    def __call__(self, url: str) -> dict[str,str]:
        return self.scrape_news(url)
    
    def scrape_news(self, url: str, method: str = "REQUESTS") -> dict[str,str]:
        
        if method == "REQUESTS":
            resp = self.session.request("GET", url)
            soup = BeautifulSoup(resp.text, 'html.parser')
        elif method == "CLOUDSCRAPER":
            scraper  : cloudscraper.CloudScraper = cloudscraper.create_scraper()
            response = scraper.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
        else:
            self.driver.get(url)
            self.accept_cookies()
            time.sleep(1)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')

            
        title = soup.find(['h1'])
        title = soup.find(['h2']) if not title else title
        if title is None:
            if method=="REQUESTS":
                return self.scrape_news(url, "CLOUDSCRAPER")
            elif method=="CLOUDSCRAPER":
                return self.scrape_news(url, "SELENIUM")
            else:
                raise Exception("No title found.")
        
        pes = soup.find_all('p')
        parents = self.get_parents(pes)                
        main_parent, max_counts = self.find_main_parent(parents)
        if not main_parent:
            main_parent, max_counts = soup, 1
        text = self.get_text(main_parent)
        j = 1
        while len(text) / max_counts < 200:
            parents = {p:c for p,c in parents.items() if p!=main_parent}
            main_parent, max_counts = self.find_main_parent(parents)
            if not main_parent:
                main_parent, max_counts = soup, 1
            
            text = self.get_text(main_parent) 
            
            if j % 6 == 0:
                if method=="REQUESTS":
                    parents = self.get_parents(soup.find_all(['p', 'span', 'article']))
                elif method=="CLOUDSCRAPER":
                    return self.scrape_news(url, "SELENIUM")
                else:
                    raise Exception("Not enough words.")
            j += 1
            if j%12 == 0:
                return self.scrape_news(url, "CLOUDSCRAPER")
        title = self.clean_text(title.text)
        text = self.clean_text(text)
        
        return {
            'link' : url,
            'title' : title,
            'text' : text
        }
        
    def get_parents(self, elements) -> dict[BeautifulSoup, int]:
        parents = {}
        for elem in elements:
            parent = elem.parent
            if parent in parents:
                parents[parent] += 1
            else:
                parents[parent] = 1
        return parents
    
    def find_main_parent(self, parents : dict[BeautifulSoup, int]) -> tuple[BeautifulSoup|None, int]:
        main_parent = None
        max_count = 0
        for parent, counts in parents.items():
            if counts > max_count:
                max_count = counts
                main_parent = parent
        return main_parent, max_count
    
    def get_text(self, main_parent: BeautifulSoup) -> str:
        content = main_parent.find_all('p')
        text = ''
        for elem in content:
            text += f' {elem.text.strip()}\n' 
        if not len(text):
            for elem in main_parent.stripped_strings:                
                text += f'{elem.strip()}\n' 
        return text
        
    def clean_text(self, text: str) -> str:
        
        emoji_pattern = re.compile(
        "[" 
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA70-\U0001FAFF"
        "\U00002600-\U000026FF"
        "\U0001F30D"       
        "\uFE0F"    
        "]+",
        flags=re.UNICODE
    )
        
        cleaned_text = emoji_pattern.sub(r"", text)
        return cleaned_text.strip()
    
    def get_session(self, verify:bool = True) -> Session:
        """
        Returns a robust requests Session for legal scraping.
        """
        proxies = None
        session = NewsSession(
            user_agents=DEFAULT_USER_AGENTS,
            proxies=proxies,
            rotate_user_agent=True,
            rate_limit_interval=0.8,  
            default_timeout=15.0,
            retries=4,
            backoff_factor=0.7,
            trust_env=False, 
            verify = verify
        )
        return session
    
    def get_driver(self) -> webdriver.Chrome:
        options = Options()
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--disable-notifications")
        options.add_argument("--start-maximized")
        options.add_argument("--headless=new")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)

        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
        })
        return driver
    
    def accept_cookies(self, timeout: int = 5) -> bool:
        xpath = (
            "//button[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept') "
            "or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'aceptar') "
            "or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'cookies')]"
            "|//a[contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'accept') "
            "or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'aceptar') "
            "or contains(translate(text(),'ABCDEFGHIJKLMNOPQRSTUVWXYZ','abcdefghijklmnopqrstuvwxyz'),'cookies')]"
        )
        try:
            btn = WebDriverWait(self.driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            self.driver.execute_script("arguments[0].click();", btn)
            return True
        except Exception:
            return False
    
class RateLimiter:
    """Simple Limiter: permits a petition every `min_interval` on behalf of (thread-safe)."""
    def __init__(self, min_interval: float = 0.5):
        self.min_interval = float(min_interval)
        self.lock = threading.Lock()
        self._last = 0.0

    def wait(self):
        with self.lock:
            now = time.time()
            wait_for = self.min_interval - (now - self._last)
            if wait_for > 0:
                time.sleep(wait_for)
            self._last = time.time()

class NewsSession(Session):
    def __init__(
        self,
        user_agents: Optional[List[str]] = None,
        proxies: Optional[Iterable[Dict[str, str]]] = None,
        rotate_user_agent: bool = True,
        rate_limit_interval: float = 2,
        default_timeout: float = 15.0,
        retries: int = 3,
        backoff_factor: float = 0.5,
        status_forcelist: Optional[List[int]] = None,
        trust_env: bool = False,
        verify: bool = True
    ):
        super().__init__()
        self.user_agents = user_agents or DEFAULT_USER_AGENTS
        self.rotate_user_agent = rotate_user_agent
        self._proxy_iter = iter(proxies) if proxies else None
        self.rate_limiter = RateLimiter(rate_limit_interval)
        self.default_timeout = default_timeout
        self.trust_env = trust_env  
        self.verify = verify
        self._init_adapters(retries, backoff_factor, status_forcelist)
        self.headers.update({
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Connection": "keep-alive",
        })

    def _init_adapters(self, retries: int, backoff_factor: float, status_forcelist: Optional[List[int]]):
        if status_forcelist is None:
            status_forcelist = [429, 500, 502, 503, 504]

        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            status=retries,
            status_forcelist=status_forcelist,
            backoff_factor=backoff_factor,
            allowed_methods=frozenset(["HEAD", "GET", "OPTIONS", "POST"]),  # POST retried carefully
            raise_on_status=False,
            raise_on_redirect=False,
        )
        adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100, max_retries=retry)

        self.mount("https://", adapter)
        self.mount("http://", adapter)

    def request(self, method: str, url: str, **kwargs) -> Response:
        self.rate_limiter.wait()
        if "timeout" not in kwargs and self.default_timeout is not None:
            kwargs["timeout"] = self.default_timeout
        if self.rotate_user_agent and self.user_agents:
            ua = random.choice(self.user_agents)
            headers = kwargs.pop("headers", {})
            headers = {**headers, "User-Agent": ua}
            kwargs["headers"] = headers
        self.trust_env = self.trust_env

        try:
            resp = super().request(method, url, **kwargs)
        except Exception as e:
            raise RequestException

        if resp.status_code == 429:
            wait = random.uniform(1.0, 5.0)
            time.sleep(wait)
        time.sleep(3)
        return resp


class NewsScraperNotPossibleError(Exception):
    """Raised when news scraping is not possible for the given URL.
    """
    pass