from news_scraper import NewsScraper

if __name__ == "__main__":
    urls = ['https://www.nytimes.com/interactive/2025/09/30/us/government-shutdown-funding.html',
         'https://www.vilaweb.cat/noticies/acrobata-mallorquina-mor-trapezi-circ-alemany/',
         'https://www.vilaweb.cat/noticies/videos-pluja-torrencial-inunda-centre-eivissa-estralls/', 
         'https://www.eldiario.es/economia/gobierno-aprueba-tramitacion-urgente-refuerzo-control-horario-acabo-fichar-diga-jefe_1_12644233.html', 
         'https://www.democrata.es/internacional/ue-considera-plan-trump-gaza-chance-paz-sostenible/',
         'https://www.menorca.info/menorca/local/2025/09/30/2480951/regreso-obras-carretera-general-empiezan-once-meses-para-reformar-tramo-entre-mao-talati.html',
         'https://www.thetimes.com/uk/media/article/jk-rowling-emma-watson-news-jql6xm9vb',
         'https://www.ara.cat/internacional/proxim-orient/trump-dona-temps-fins-diumenge-hamas-acceptar-pla-gaza_1_5517173.html', # ARA
         'https://www.elpuntavui.cat/politica/article/17-politica/2581102-milers-d-estudiants-es-manifesten-per-gaza-a-barcelona.html', # ElPuntAvui
         'https://www.elperiodico.cat/ca/esports/20251004/lamine-recau-lesio-pubis-122259183', # El Periódico
         'https://www.lavanguardia.com/encatala/20251004/11126692/febre-per-ia.html', # La Vanguardia
         'https://www.lesportiudecatalunya.cat/barca/article/2581340-recaiguda-i-a-reposar.html', # L'esportiu
         'https://naciodigital.cat/internacional/dels-crims-de-hamas-a-laillament-disrael-dos-anys-de-guerra-a-gaza.html', # Nació digital
         'https://e-noticies.cat/opinio/1-doctubre-vuit-anys-despres', # e-notícies,
         'https://www.racocatala.cat/noticia/68991/andorra-rebutja-230-sollicituds-temporers-argentins-no-presentar-bitllet-tornada', #racó català
         'https://cronicaglobal.elespanol.com/business/20251003/carles-flamerich-apolo-cibersecurity-ia-entrenan-atacarte/1003742694563_0.html' # Crónica global
         'https://elpais.com/quadern/literatura/2025-10-02/llibres-i-democracia-en-perill.html', # ElPaís
         'https://www.dbalears.cat/balears/balears/2025/10/03/411491/som-poble-pau-quarter-otan.html', #DBalears
         'https://www.3cat.cat/3catinfo/andalusia-va-ignorar-2000-mamografies-amb-resultat-dubtos-i-no-va-avisar-les-afectades/noticia/3372893/', # 3cat
         'https://www.acn.cat/new/confirmat-el-primer-focus-de-dermatosi-nodular-contagiosa-de-lestat-en-una-explotacio-de-bovi-de-lalt-emporda/texts', # agència catalana de notícies
         "https://www.bbc.com/news/articles/cewnr5erq7do" # BBC
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