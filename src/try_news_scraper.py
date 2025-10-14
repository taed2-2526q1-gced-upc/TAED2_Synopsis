from src.news_scraper import NewsScraper

if __name__ == "__main__":
    urls = [
        'https://www.democrata.es/actualidad/sanidad-fija-condiciones-elaborar-dispensar-formulas-cannabis/',
        'https://elpais.com/sociedad/2025-10-07/monica-garcia-advierte-a-ayuso-sobre-el-aborto-en-nuestro-pais-la-ley-se-cumple-y-usaremos-todas-las-herramientas.html'
             ]
    scraper = NewsScraper()
    
    for url in urls:
        print(url)
        result = scraper.scrape_news(url)
        print(result['title'])
        print('-'*30)
        print(len(result['text']))
        print(result['text'])
        print('-'*30)
        print('-'*30)
        print('-'*30)