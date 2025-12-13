"""
Enhanced Premier League News Scraper - MAXIMUM DATA MODE
Focuses on reliable RSS feeds and APIs to get 1000+ articles quickly
"""

import feedparser
import pandas as pd
from datetime import datetime
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

class MaximumNewsScraper:
    def __init__(self):
        self.articles = []
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_rss_feed(self, url, source_name, limit=100):
        """Generic RSS feed scraper"""
        print(f"🔄 Scraping {source_name}...")
        try:
            feed = feedparser.parse(url)
            count = 0
            for entry in feed.entries[:limit]:
                article = {
                    'title': entry.title if hasattr(entry, 'title') else '',
                    'link': entry.link if hasattr(entry, 'link') else '',
                    'published': entry.published if hasattr(entry, 'published') else str(datetime.now()),
                    'summary': entry.summary if hasattr(entry, 'summary') else entry.title,
                    'text': entry.summary if hasattr(entry, 'summary') else entry.title,
                    'source': source_name,
                    'scraped_at': datetime.now().isoformat()
                }
                self.articles.append(article)
                count += 1
            print(f"✅ Scraped {count} articles from {source_name}")
            return True
        except Exception as e:
            print(f"❌ Error scraping {source_name}: {e}")
            return False
    
    def scrape_google_news_comprehensive(self, limit=500):
        """Scrape Google News with EXTENSIVE Premier League queries"""
        queries = [
            'Premier+League',
            'Premier+League+news',
            'EPL',
            'EPL+news',
            'Manchester+United',
            'Manchester+City',
            'Liverpool+FC',
            'Liverpool',
            'Arsenal+FC',
            'Arsenal',
            'Chelsea+FC',
            'Chelsea',
            'Tottenham',
            'Spurs',
            'Newcastle+United',
            'Aston+Villa',
            'Brighton',
            'West+Ham',
            'Everton',
            'Fulham',
            'Brentford',
            'Wolves',
            'Crystal+Palace',
            'Bournemouth',
            'Nottingham+Forest',
            'Leicester',
            'Southampton',
            'Ipswich',
            'Premier+League+transfer',
            'Premier+League+transfers',
            'Premier+League+match',
            'Premier+League+fixtures',
            'Premier+League+results',
            'Premier+League+standings',
            'Premier+League+table',
            'Premier+League+goals',
            'Premier+League+highlights',
            'Premier+League+injury',
            'Premier+League+suspension',
            'Premier+League+manager'
        ]
        
        print(f"🔄 Scraping Google News (15 different queries)...")
        total = 0
        for query in queries:
            try:
                url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
                feed = feedparser.parse(url)
                for entry in feed.entries[:limit//len(queries)]:
                    source_name = entry.source.title if hasattr(entry, 'source') and hasattr(entry.source, 'title') else 'Google News'
                    article = {
                        'title': entry.title,
                        'link': entry.link,
                        'published': entry.published if hasattr(entry, 'published') else str(datetime.now()),
                        'summary': entry.summary if hasattr(entry, 'summary') else entry.title,
                        'text': entry.summary if hasattr(entry, 'summary') else entry.title,
                        'source': f'Google News ({source_name})',
                        'scraped_at': datetime.now().isoformat()
                    }
                    self.articles.append(article)
                    total += 1
                time.sleep(1)
            except:
                continue
        print(f"✅ Scraped {total} articles from Google News")
        return True
    
    def scrape_all_maximum(self):
        """Scrape MAXIMUM articles from all available RSS feeds - TARGET 1000+"""
        print('='*80)
        print('🚀 ULTRA MAXIMUM DATA SCRAPER - TARGET: 1000+ ARTICLES')
        print('='*80)
        
        # RSS feeds (most reliable sources) - INCREASED LIMITS
        feeds = [
            ('http://feeds.bbci.co.uk/sport/football/premier-league/rss.xml', 'BBC Sport', 150),
            ('http://feeds.bbci.co.uk/sport/football/rss.xml', 'BBC Sport Football', 150),
            ('http://feeds.skysports.com/feeds/11095', 'Sky Sports', 150),
            ('https://www.theguardian.com/football/premierleague/rss', 'The Guardian', 150),
            ('https://www.espn.com/espn/rss/soccer/news', 'ESPN', 150),
            ('https://www.goal.com/feeds/en/news', 'Goal.com', 150),
            ('https://www.dailymail.co.uk/sport/football/premierleague/index.rss', 'Daily Mail', 150),
            ('https://www.mirror.co.uk/sport/football/news/?service=rss', 'Mirror', 150),
            ('https://www.express.co.uk/sport/football/rss', 'Express', 150),
            ('https://www.independent.co.uk/sport/football/premier-league/rss', 'The Independent', 150),
            ('https://talksport.com/football/premier-league/feed/', 'TalkSport', 150),
            ('https://www.football.london/premier-league/feed/', 'Football.London', 150),
            ('https://www.givemesport.com/feed/', 'GiveMeSport', 150),
            ('https://www.90min.com/posts.rss', '90min', 150),
            # Additional feeds
            ('https://www.telegraph.co.uk/football/rss.xml', 'The Telegraph', 100),
            ('https://www.sportbible.com/football/feed', 'SportBible', 100),
            ('https://theathletic.com/feed/', 'The Athletic', 100),
        ]
        
        for url, source, limit in feeds:
            self.scrape_rss_feed(url, source, limit)
            time.sleep(0.8)  # Faster scraping
        
        # Google News (MASSIVE coverage with 40 queries)
        self.scrape_google_news_comprehensive(500)
        
        print(f'\n{"="*80}')
        print(f'✅ SCRAPING COMPLETE')
        print(f'Total articles: {len(self.articles)}')
        print('='*80)
        
        return self.articles
    
    def save(self):
        """Save articles to CSV"""
        if not self.articles:
            print("❌ No articles to save")
            return
        
        df = pd.DataFrame(self.articles)
        
        # Load existing data if available
        data_dir = Path('data/raw')
        filepath = data_dir / 'pl_news.csv'
        
        if filepath.exists():
            existing_df = pd.read_csv(filepath)
            print(f"📊 Found {len(existing_df)} existing articles")
            
            # Combine and remove duplicates
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['title', 'link'], keep='first')
            
            print(f"✅ Combined total: {len(combined_df)} articles")
            print(f"   New articles added: {len(combined_df) - len(existing_df)}")
        else:
            combined_df = df
        
        combined_df.to_csv(filepath, index=False, encoding='utf-8')
        print(f'💾 Saved to: {filepath}')
        
        # Show distribution
        print(f'\nDistribution by source:')
        print(combined_df['source'].value_counts().head(20))
        
        return filepath


def main():
    print('\n🌐 ULTRA Maximum Premier League News Scraper')
    print('🎯 Target: 1000+ articles for optimal ML training\n')
    
    scraper = MaximumNewsScraper()
    scraper.scrape_all_maximum()
    scraper.save()
    
    print('\n✅ Done! Ready for machine learning training!')
    print('🔥 If you need more, run this script again - it will add to existing data!')


if __name__ == '__main__':
    main()
