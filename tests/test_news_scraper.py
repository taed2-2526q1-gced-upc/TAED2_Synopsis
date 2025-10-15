import pytest
from bs4 import BeautifulSoup
from src.backend.app.services.news_scraper import NewsScraper, RateLimiter, NewsSession


def test_clean_text_removes_emojis_correctly():
    ns = NewsScraper()
    text = "Hello 😊 world 🌍 Synopsis emojis ☀️⚽"
    cleaned = ns.clean_text(text)
    assert cleaned == "Hello  world  Synopsis emojis"
    assert "😊" not in cleaned
    
def test_get_parents_counts_correctly():
    html = "<div><p>1</p><p>2</p></div><div><p>3</p></div>"
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.find_all("p")
    ns = NewsScraper()
    parents = ns.get_parents(elements)
    assert len(parents) == 2
    assert all(isinstance(v, int) for v in parents.values())

def test_find_main_parent_returns_most_common():
    html = "<div><p>1</p><p>2</p></div><div><p>3</p></div>"
    soup = BeautifulSoup(html, "html.parser")
    elements = soup.find_all("p")
    ns = NewsScraper()
    parents = ns.get_parents(elements)
    parent, count = ns.find_main_parent(parents)
    assert count == 2
    assert parent is not None

def test_get_text_concatenates_paragraphs():
    html = "<div><p>First</p><p>Second</p></div>"
    soup = BeautifulSoup(html, "html.parser")
    ns = NewsScraper()
    text = ns.get_text(soup)
    assert "First" in text and "Second" in text

def test_rate_limiter_wait_does_not_raise():
    limiter = RateLimiter(0.1)
    limiter.wait()
    limiter.wait()

def test_news_session_rotates_user_agent(monkeypatch):
    ns = NewsSession()
    called = {}

    def fake_request(self, method, url, **kwargs):
        called["headers"] = kwargs["headers"]
        class FakeResp:
            status_code = 200
            text = "<html></html>"
        return FakeResp()

    monkeypatch.setattr("requests.Session.request", fake_request)
    resp = ns.request("GET", "https://example.com")
    assert "User-Agent" in called["headers"]


def test_scrape_news_with_mocked_request(monkeypatch):
    ns = NewsScraper()

    fake_html = """
        <html>
        <h1>Títol de prova</h1>
        <div>
            <p>{}</p>
        </div>
        </html>
        """.format("Paràgraf " * 300)

    class FakeResponse:
        text = fake_html
        status_code = 200

    def fake_request(method, url):
        return FakeResponse()

    ns.session.request = fake_request

    result = ns.scrape_news("https://example.com", method="REQUESTS")

    assert result["link"] == "https://example.com"
    assert "Títol de prova" in result["title"]
    assert "Paràgraf" in result["text"]

def test_scrape_news_raises_if_no_title(monkeypatch):
    ns = NewsScraper()

    fake_html = "<html><p>No title</p></html>"

    class FakeResponse:
        text = fake_html
        status_code = 200

    def fake_request(method, url):
        return FakeResponse()

    ns.session.request = fake_request

    with pytest.raises(Exception):
        ns.scrape_news("https://example.com", method="SELENIUM")
