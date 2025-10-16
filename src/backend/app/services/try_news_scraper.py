from news_scraper import NewsScraper, NewsScraperNotPossibleError

if __name__ == "__main__":
    urls = [
        'https://www.democrata.es/actualidad/sanidad-fija-condiciones-elaborar-dispensar-formulas-cannabis/',
        'https://elpais.com/sociedad/2025-10-07/monica-garcia-advierte-a-ayuso-sobre-el-aborto-en-nuestro-pais-la-ley-se-cumple-y-usaremos-todas-las-herramientas.html'
        'https://www.reuters.com/legal/government/us-supreme-court-hear-case-that-takes-aim-voting-rights-act-2025-10-15/', # Reuters
        'https://news.sky.com/story/social-media-star-big-john-fisher-to-be-deported-after-being-detained-in-australia-13450436', # Sky News
        'https://www.bbc.com/news/articles/c1m39xg4dggo', # BBC News   
        'https://edition.cnn.com/2025/10/15/europe/ukraine-hegesth-firepower-coming-nato-intl-hnk', # CNN
        'https://www.nytimes.com/2025/10/15/world/europe/london-police-phone-theft-china-gang.html', # NY Times
        'https://www.washingtonpost.com/wellness/2025/10/15/alcohol-dementia-risk/?itid=hp-top-table-main_p001_f010', # Washington Post
        'https://www.aljazeera.com/news/liveblog/2025/10/15/live-israel-restricts-aid-into-gaza-hamas-releases-bodies-of-4-captives', # Al Jazeera
        'https://www.theguardian.com/business/2025/oct/15/global-government-debt-100-percent-of-gdp-by-2029-imf-uk', # The Guardian
        'https://apnews.com/article/reese-witherspoon-harlan-coben-novel-interview-836035a7d98acdf1639330a114cdce54', # AP News
        'https://www.nbcnews.com/business/business-news/trump-trade-china-bessent-rare-earth-rcna237786', # NBC News
        'https://www.foxnews.com/health/actress-ignored-subtle-cancer-symptom-years-before-onstage-emergency', # Fox News
        'https://abcnews.go.com/US/50-tons-cocaine-seized-us-coast-guard-pacific/story?id=126536035', # ABC News
        'https://www.bloomberg.com/news/articles/2025-10-15/france-sacrifices-macron-s-pension-reform-in-bid-for-stability?srnd=homepage-europe',   # Bloomberg
        'https://www.cbsnews.com/news/sweden-emergency-grain-reserves-north-prepared-crisis/', # CBS News
        'https://www.politico.com/news/2025/10/15/zohran-mamdani-needs-a-mistake-free-debate-night-against-andrew-cuomo-00608015', # Politico
        'https://www.wsj.com/world/china/china-trade-war-trump-talks-25c50136?mod=hp_lead_pos1', # Wall Street Journal
        'https://time.com/7325682/dangelo-remembrance/', # Time
        ]
    scraper = NewsScraper()
    
    for url in urls:
        print(url)
        try:
            result = scraper.scrape_news(url)
        except NewsScraperNotPossibleError as e:
            print(f"Error scraping {url}: {e}")
            continue
        print(result['title'])
        print('-'*30)
        print(len(result['text']))
        print(result['text'])
        print('-'*30)
        print('-'*30)
        print('-'*30)