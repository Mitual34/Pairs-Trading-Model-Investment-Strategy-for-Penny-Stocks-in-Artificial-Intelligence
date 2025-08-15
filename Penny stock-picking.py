import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import re
import time
from datetime import datetime

def find_ai_penny_stocks():
    """Identify AI penny stocks with robust error handling and diagnostics"""
    
    # Step 1: Get penny stock universe from reliable sources
    def get_penny_stock_universe():
        tickers = set()
        
        # Method 1: Finviz scraping
        try:
            url = "https://finviz.com/screener.ashx?v=111&f=sh_price_o0.1,sh_price_u5,sh_avgvol_o50000&ft=4"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
            }
            response = requests.get(url, headers=headers, timeout=15)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the screener table
            table = soup.find('table', class_='table-light')
            if table:
                # Find all links in the table that look like tickers
                for link in table.find_all('a', href=re.compile(r'quote\.ashx\?t=[A-Z]+')):
                    ticker = link.text.strip()
                    if len(ticker) <= 5 and ticker.isalpha():  # Basic validation
                        tickers.add(ticker)
        except Exception as e:
            print(f"Finviz scraping error: {e}")
        
        # Add known AI penny stocks
        ai_tickers = ['SOPA', 'BTTX', 'INPX', 'PRST', 'DATS', 'GFAI', 'IDAI', 
                      'CYBL', 'BANT', 'AUID', 'AIHS', 'IMTE', 'KSCP', 'BFRG',
                      'MEGL', 'VERB', 'WULF', 'INOD', 'DTSS', 'BTIM']
        tickers.update(ai_tickers)
        
        return list(tickers)[:100]  # Limit to 100 for performance

    # Step 2: Enhanced AI exposure detection
    def has_ai_exposure(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Define AI keywords with context
            ai_keywords = [
                r'\bai\b', r'\bartificial intelligence\b', 
                r'\bmachine learning\b', r'\bneural network\b',
                r'\bdeep learning\b', r'\bcomputer vision\b',
                r'\bnatural language processing\b', r'\bpredictive analytics\b',
                r'\bautonomous system\b', r'\brobotic process automation\b'
            ]
            
            # Check 1: Business summary
            summary = info.get('longBusinessSummary', '').lower()
            if any(re.search(kw, summary) for kw in ai_keywords):
                return True
                
            # Check 2: Products/services
            products = info.get('products', [])
            if isinstance(products, list):
                product_str = ' '.join(products).lower()
                if any(re.search(kw, product_str) for kw in ai_keywords):
                    return True
                    
            # Check 3: Recent news (last 30 days)
            news = stock.news
            if news:
                recent_news = [n for n in news if datetime.now().timestamp() - n['providerPublishTime'] < 2592000]
                for item in recent_news[:5]:
                    title = item.get('title', '').lower()
                    if any(re.search(kw, title) for kw in ai_keywords):
                        return True
                        
            return False
        except Exception as e:
            print(f"AI check error for {ticker}: {e}")
            return False

    # Step 3: Realistic penny stock filters
    def passes_filters(ticker):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            history = stock.history(period='3mo')
            
            if history.empty or len(history) < 20:
                return False
                
            # Get price with multiple fallbacks
            current_price = info.get('currentPrice', 
                                    info.get('regularMarketPrice', 
                                    history['Close'].iloc[-1]))
            
            # Get volume with fallback
            avg_volume = info.get('averageVolume', history['Volume'].mean())
            
            # Realistic filters for penny stocks
            filters = {
                'price': 0.05 <= current_price <= 5.0,
                'volume': avg_volume > 50000,  # Reduced from 100k
                'market_cap': info.get('marketCap', 0) > 5e6,
                'recent_trading': len(history) >= 20  # At least 20 trading days
            }
            
            return all(filters.values())
        except Exception as e:
            print(f"Filter error for {ticker}: {e}")
            return False

    # Step 4: Momentum scoring with technical indicators
    def calculate_momentum_score(ticker):
        try:
            data = yf.download(ticker, period='3mo', progress=False)
            if len(data) < 20:
                return 0
                
            # Calculate technical indicators
            data['SMA20'] = data['Close'].rolling(20).mean()
            data['RSI'] = compute_rsi(data['Close'])
            
            # Momentum scoring system
            score = 0
            
            # Price above SMA20
            if data['Close'].iloc[-1] > data['SMA20'].iloc[-1]:
                score += 1
                
            # Positive RSI momentum
            if 40 < data['RSI'].iloc[-1] < 70:
                score += 1
                
            # Recent volume spike
            last_volume = data['Volume'].iloc[-1]
            avg_volume = data['Volume'].rolling(20).mean().iloc[-1]
            if last_volume > avg_volume * 1.5:
                score += 1
                
            # Positive price trend
            if data['Close'].iloc[-1] > data['Close'].iloc[-5]:
                score += 1
                
            return score
        except:
            return 0

    # RSI calculation helper
    def compute_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(window).mean()
        avg_loss = loss.rolling(window).mean()
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    # Pipeline execution with diagnostics
    print("Fetching penny stock universe...")
    penny_stocks = get_penny_stock_universe()
    print(f"Found {len(penny_stocks)} potential penny stocks")
    
    ai_penny_stocks = []
    diagnostics = []
    
    print("\nScreening stocks for AI exposure and viability...")
    for i, ticker in enumerate(penny_stocks):
        try:
            print(f"Processing {i+1}/{len(penny_stocks)}: {ticker}")
            diag = {'ticker': ticker}
            
            # Check AI exposure
            ai_exposure = has_ai_exposure(ticker)
            diag['ai_exposure'] = ai_exposure
            
            # Check filters
            passes = passes_filters(ticker)
            diag['passes_filters'] = passes
            
            # Calculate momentum
            momentum = 0
            if ai_exposure and passes:
                momentum = calculate_momentum_score(ticker)
                diag['momentum'] = momentum
                
                if momentum >= 2:  # Reduced threshold
                    ai_penny_stocks.append((ticker, momentum))
            
            diagnostics.append(diag)
            time.sleep(1)  # Rate limiting to avoid Yahoo Finance blocking
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            diagnostics.append({'ticker': ticker, 'error': str(e)})
    
    # Generate diagnostics report
    diag_df = pd.DataFrame(diagnostics)
    
    # Sort results by momentum
    ai_penny_stocks.sort(key=lambda x: x[1], reverse=True)
    
    return [t[0] for t in ai_penny_stocks], diag_df

# Helper function for AI keywords
def get_ai_keywords(ticker):
    try:
        stock = yf.Ticker(ticker)
        summary = stock.info.get('longBusinessSummary', '').lower()
        keywords = [
            'ai', 'artificial intelligence', 'machine learning',
            'neural network', 'deep learning', 'computer vision'
        ]
        return [kw for kw in keywords if kw in summary]
    except:
        return []

# Main execution with comprehensive reporting
if __name__ == "__main__":
    print("==== AI PENNY STOCK SCREENER ====")
    print("Starting screening process...")
    
    try:
        results, diagnostics = find_ai_penny_stocks()
    except Exception as e:
        print(f"Critical error in screening process: {e}")
        results = []
        diagnostics = pd.DataFrame()

    print("\n" + "="*50)
    print("SCREENING COMPLETE")
    print("="*50)
    
    if not results:
        print("\nNo qualifying AI penny stocks found")
        if not diagnostics.empty:
            print("\nTop 10 Diagnostics:")
            print(diagnostics[['ticker', 'ai_exposure', 'passes_filters']].head(10))
    else:
        print(f"\nFound {len(results)} AI Penny Stocks:")
        for ticker in results:
            print(f"- {ticker}")
        
        # Detailed analysis
        analysis = []
        for ticker in results:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                analysis.append({
                    'Ticker': ticker,
                    'Price': info.get('currentPrice', 'N/A'),
                    'Market Cap ($M)': f"{info.get('marketCap', 0)/1e6:.2f}" if info.get('marketCap') else 'N/A',
                    'Volume (Avg)': f"{info.get('averageVolume', 0):,.0f}" if info.get('averageVolume') else 'N/A',
                    'AI Keywords': ', '.join(get_ai_keywords(ticker)),
                    'Momentum': calculate_momentum_score(ticker)
                })
            except Exception as e:
                print(f"Error analyzing {ticker}: {e}")
                continue
        
        if analysis:
            analysis_df = pd.DataFrame(analysis)
            print("\nDetailed Analysis:")
            print(analysis_df)
        else:
            print("\nCould not generate detailed analysis")
    
    # Save diagnostics if available
    if not diagnostics.empty:
        diagnostics.to_csv('penny_stock_diagnostics.csv', index=False)
        print("\nDiagnostics saved to penny_stock_diagnostics.csv")
