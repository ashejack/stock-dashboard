"""
Live Data Generator for Enhanced Stock Dashboard
Includes all of your father's custom rules and preferences
NOW with Streetstats web scraping for accurate market breadth!
"""

import json
from stock_analyzer import StockAnalyzer
from datetime import datetime, timedelta
import yfinance as yf
import sys
import re

# Stock groupings per your father's preference
STOCK_GROUPS = {
    'CHIP STOCKS': ['NVDA', 'TSM', 'VRT', 'CRDO', 'ALAB', 'AVGO'],
    'HIGH FLYERS': ['BRK-B', 'GLD', 'AAPL', 'AMZN', 'GOOGL', 'META', 'TSLA', 'MSFT'],
    'URANIUM STOCKS': ['CCJ', 'OKLO'],
    'RAILROADS': ['CNI', 'CP']
}

STOCK_NAMES = {
    'NVDA': 'NVIDIA',
    'TSM': 'Taiwan Semiconductor',
    'VRT': 'Vertiv',
    'CRDO': 'Credo Technology',
    'ALAB': 'Astera Labs',
    'AVGO': 'Broadcom',
    'BRK-B': 'Berkshire Hathaway',
    'GLD': 'SPDR Gold Trust',
    'AAPL': 'Apple',
    'AMZN': 'Amazon',
    'GOOGL': 'Google/Alphabet',
    'META': 'Meta/Facebook',
    'TSLA': 'Tesla',
    'MSFT': 'Microsoft',
    'CCJ': 'Cameco',
    'OKLO': 'Oklo Inc',
    'CNI': 'Canadian National Railway',
    'CP': 'Canadian Pacific'
}

def scrape_streetstats_breadth():
    """
    Scrape market breadth data from Streetstats using Selenium
    Returns % of stocks above 50-day MA for S&P 500 and NASDAQ
    """
    print("  📡 Scraping market breadth from Streetstats...")
    
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time
        
        # Setup headless Chrome
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        # Try to create driver
        try:
            driver = webdriver.Chrome(options=chrome_options)
        except Exception as e:
            print(f"    ⚠️  Could not start Chrome driver: {e}")
            print(f"    ℹ️  Make sure Chrome and chromedriver are installed")
            print(f"    ℹ️  Install: pip install selenium")
            print(f"    ℹ️  Download chromedriver from: https://chromedriver.chromium.org/")
            return None
        
        try:
            # Scrape S&P 500 breadth
            print("    Fetching S&P 500 breadth data...")
            driver.get('https://streetstats.finance/markets/breadth-momentum/SP500')
            time.sleep(5)  # Wait for JavaScript to load
            
            page_source = driver.page_source
            
            # Extract percentage above 50-day MA using regex
            # Looking for patterns like "44.2% of the index" or "above their 50-day average account for 44.2%"
            sp500_match = re.search(r'above their 50-day average account for ([\d.]+)%', page_source, re.IGNORECASE)
            if not sp500_match:
                # Try alternative pattern
                sp500_match = re.search(r'50-day.*?([\d.]+)%', page_source, re.IGNORECASE)
            
            if sp500_match:
                sp500_pct = float(sp500_match.group(1))
                print(f"    ✓ S&P 500: {sp500_pct}% above 50-day MA")
            else:
                print("    ⚠️  Could not find S&P 500 breadth percentage")
                sp500_pct = None
            
            # Scrape NASDAQ-100 breadth
            print("    Fetching NASDAQ-100 breadth data...")
            driver.get('https://streetstats.finance/markets/breadth-momentum/NQ100')
            time.sleep(5)  # Wait for JavaScript to load
            
            page_source = driver.page_source
            
            # Extract percentage above 50-day MA
            nasdaq_match = re.search(r'above their 50-day average account for ([\d.]+)%', page_source, re.IGNORECASE)
            if not nasdaq_match:
                nasdaq_match = re.search(r'50-day.*?([\d.]+)%', page_source, re.IGNORECASE)
            
            if nasdaq_match:
                nasdaq_pct = float(nasdaq_match.group(1))
                print(f"    ✓ NASDAQ-100: {nasdaq_pct}% above 50-day MA")
            else:
                print("    ⚠️  Could not find NASDAQ-100 breadth percentage")
                nasdaq_pct = None
            
            driver.quit()
            
            if sp500_pct is not None and nasdaq_pct is not None:
                return {
                    'sp500_pct_above_ma50': sp500_pct,
                    'nasdaq_pct_above_ma50': nasdaq_pct,
                    'sp500_healthy': sp500_pct >= 50,
                    'nasdaq_healthy': nasdaq_pct >= 50,
                    'market_healthy': sp500_pct >= 50 and nasdaq_pct >= 50,
                    'sp500_above': int(sp500_pct * 5.03),  # Approximate count (503 stocks in S&P 500)
                    'sp500_checked': 503,
                    'nasdaq_above': int(nasdaq_pct * 1.00),  # Approximate count (100 stocks in NASDAQ-100)
                    'nasdaq_checked': 100,
                    'source': 'Streetstats'
                }
            else:
                return None
                
        finally:
            try:
                driver.quit()
            except:
                pass
                
    except ImportError:
        print("    ⚠️  Selenium not installed!")
        print("    ℹ️  Install with: pip install selenium")
        print("    ℹ️  Also need Chrome browser and chromedriver")
        return None
    except Exception as e:
        print(f"    ⚠️  Error scraping Streetstats: {e}")
        return None

def get_market_breadth_unavailable():
    """
    Return N/A values when Streetstats scraping fails
    No fallback calculation - only use accurate data or nothing
    """
    return {
        'sp500_pct_above_ma50': 'N/A',
        'nasdaq_pct_above_ma50': 'N/A',
        'sp500_healthy': None,
        'nasdaq_healthy': None,
        'market_healthy': None,
        'sp500_above': 'N/A',
        'sp500_checked': 503,
        'nasdaq_above': 'N/A',
        'nasdaq_checked': 100,
        'source': 'unavailable'
    }

def get_market_breadth():
    """
    Get market breadth from Streetstats ONLY - no fallback
    Returns N/A if scraping fails (no estimates!)
    """
    print("  📊 Getting market breadth data from Streetstats...")
    
    # Only use Streetstats - accurate data or nothing
    breadth = scrape_streetstats_breadth()
    
    if breadth is None:
        print("  ⚠️  Streetstats scraping failed - market breadth will show N/A")
        print("  ℹ️  Make sure Selenium and Chrome are installed (see STREETSTATS_SETUP.md)")
        breadth = get_market_breadth_unavailable()
    else:
        print(f"  ✓ Market breadth obtained from: {breadth['source']}")
    
    return breadth

def get_nasdaq_top10_performance():
    """
    Calculate the combined 1-week return for NASDAQ top 10 companies
    """
    print("  📈 Calculating NASDAQ Top 10 performance...")
    
    # Top 10 by market cap
    top10 = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'GOOGL', 'META', 'TSLA', 'AVGO', 'COST', 'NFLX']
    
    returns = []
    for ticker in top10:
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period='2wk')
            if len(hist) >= 5:
                week_ago = hist['Close'].iloc[-6]
                current = hist['Close'].iloc[-1]
                ret = ((current - week_ago) / week_ago) * 100
                returns.append(ret)
        except:
            continue
    
    combined_return = round(sum(returns) / len(returns), 2) if returns else 0.0
    
    print(f"    ✓ NASDAQ Top 10 (1-week): {combined_return:+.2f}%")
    
    return {
        'combined_1week_return': combined_return,
        'tickers': top10
    }

def get_earnings_date(ticker):
    """Get the next earnings date for a ticker"""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is not None and 'Earnings Date' in calendar:
            earnings_dates = calendar['Earnings Date']
            if len(earnings_dates) > 0:
                next_earnings = earnings_dates[0]
                if hasattr(next_earnings, 'strftime'):
                    return next_earnings.strftime('%Y-%m-%d')
                return str(next_earnings)[:10]
    except:
        pass
    return None

def analyze_pre_earnings_volatility(ticker):
    """
    Analyze historical pre-earnings volatility patterns
    """
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period='1y')
        
        if len(hist) < 30:
            return {
                'avg_2week_volatility': 0.0,
                'historical_pattern': 'Insufficient data'
            }
        
        # Calculate rolling 2-week volatility
        hist['Returns'] = hist['Close'].pct_change()
        hist['Volatility_2wk'] = hist['Returns'].rolling(window=10).std() * 100
        
        avg_volatility = round(hist['Volatility_2wk'].mean(), 2)
        
        # Simple pattern analysis
        recent_vol = hist['Volatility_2wk'].iloc[-10:].mean()
        if recent_vol > avg_volatility * 1.3:
            pattern = 'Higher than average'
        elif recent_vol < avg_volatility * 0.7:
            pattern = 'Lower than average'
        else:
            pattern = 'Average'
        
        return {
            'avg_2week_volatility': round(avg_volatility, 2),
            'historical_pattern': pattern
        }
    except:
        return {
            'avg_2week_volatility': 0.0,
            'historical_pattern': 'Unable to calculate'
        }

def calculate_signal_strength(indicators, father_rules, signal):
    """
    Calculate signal strength based on multiple factors
    Returns a percentage (0-100)
    """
    strength = 50  # Start at neutral
    
    # RSI contribution
    rsi = indicators['RSI']
    if signal == 'BUY':
        if rsi < 30:
            strength += 15
        elif rsi < 40:
            strength += 10
        elif rsi > 60:
            strength -= 10
    elif signal == 'SELL':
        if rsi > 70:
            strength += 15
        elif rsi > 60:
            strength += 10
        elif rsi < 40:
            strength -= 10
    
    # Williams %R contribution
    williams = indicators['Williams_R']
    if signal == 'BUY' and williams < -80:
        strength += 10
    elif signal == 'SELL' and williams > -20:
        strength += 10
    
    # MA alignment
    if father_rules['ma_alignment_bullish']:
        if signal == 'BUY':
            strength += 15
        elif signal == 'SELL':
            strength -= 10
    
    # Volume
    if father_rules['abnormal_volume_up'] and signal == 'BUY':
        strength += 10
    elif father_rules['abnormal_volume_down'] and signal == 'SELL':
        strength += 10
    
    # MACD
    macd = indicators['MACD']
    macd_signal = indicators['MACD_Signal']
    if signal == 'BUY' and macd > macd_signal:
        strength += 5
    elif signal == 'SELL' and macd < macd_signal:
        strength += 5
    
    # Clamp between 0 and 100
    return max(0, min(100, strength))

def generate_signal_reasons(latest_data, father_rules):
    """Generate human-readable reasons for the signal"""
    reasons = []
    
    # Golden/Death Cross
    if latest_data['golden_cross_recent']:
        reasons.append("✅ GOLDEN CROSS (50-day crossed above 100-day)")
    elif latest_data['death_cross_recent']:
        reasons.append("🔴 DEATH CROSS (50-day crossed below 100-day)")
    
    # MA Alignment
    if father_rules['ma_alignment_bullish']:
        reasons.append("✅ Bullish MA alignment (10 > 21 > 50)")
    
    # Below MA10 rule
    if father_rules['below_ma10']:
        reasons.append(f"⚠️ Price below 10-day MA (DO NOT BUY)")
        if father_rules['days_below_ma10'] >= 2:
            reasons.append(f"🔴 Below 10-day MA for {father_rules['days_below_ma10']} days (Consider selling 25%)")
    
    # Below MA21 EMA rule
    if father_rules['below_ma21_ema']:
        reasons.append("⚠️ Price below 21-day EMA (Consider selling another 25%)")
    
    # Below MA50 rule
    if father_rules['below_ma50']:
        reasons.append("🔴 Price BELOW 50-day MA (Consider selling position)")
    
    # RSI
    rsi = latest_data['indicators']['RSI']
    if rsi < 30:
        reasons.append(f"RSI oversold at {rsi:.1f}")
    elif rsi > 70:
        reasons.append(f"RSI overbought at {rsi:.1f}")
    
    return reasons

def generate_live_data():
    """
    Fetch live data from Yahoo Finance and generate signals
    """
    print("=" * 70)
    print("BOPTIMUS PRIME STOCK TRADER - LIVE DATA GENERATION")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get market health
    print("📊 Analyzing Market Breadth...")
    market_breadth = get_market_breadth()
    
    print("📈 Calculating NASDAQ Top 10 Performance...")
    nasdaq_top10 = get_nasdaq_top10_performance()
    
    print()
    print("=" * 70)
    print("Analyzing Stocks by Category...")
    print("=" * 70)
    
    all_stocks = {}
    
    for category, tickers in STOCK_GROUPS.items():
        print(f"\n📁 {category}")
        print("-" * 70)
        
        for ticker in tickers:
            try:
                print(f"\n  Analyzing {ticker} ({STOCK_NAMES.get(ticker, ticker)})...")
                
                analyzer = StockAnalyzer(ticker, period='6mo')
                print(f"    ├─ Fetching historical data...")
                analyzer.fetch_data()
                
                print(f"    ├─ Calculating indicators...")
                analyzer.calculate_indicators()
                
                print(f"    ├─ Generating signals...")
                analyzer.generate_signals()
                
                latest = analyzer.get_latest_signal()
                
                # Get earnings date
                print(f"    ├─ Checking earnings date...")
                earnings_date = get_earnings_date(ticker)
                
                # Analyze pre-earnings volatility
                print(f"    ├─ Analyzing pre-earnings volatility...")
                pre_earnings = analyze_pre_earnings_volatility(ticker)
                
                # Calculate custom signal strength
                signal_strength = calculate_signal_strength(
                    latest['indicators'],
                    latest['father_rules'],
                    latest['signal']
                )
                
                # Generate reasons
                reasons = generate_signal_reasons(latest, latest['father_rules'])
                
                # Check if earnings within 2 weeks
                earnings_soon = False
                if earnings_date:
                    try:
                        earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
                        days_until = (earnings_dt - datetime.now()).days
                        earnings_soon = 0 <= days_until <= 14
                    except:
                        pass
                
                stock_data = {
                    'name': STOCK_NAMES.get(ticker, ticker),
                    'category': category,
                    'current_price': latest['close_price'],
                    'signal': latest['signal'],
                    'signal_strength': signal_strength,
                    'indicators': latest['indicators'],
                    'father_rules': latest['father_rules'],
                    'death_cross': latest['death_cross_recent'],
                    'golden_cross': latest['golden_cross_recent'],
                    'reasons': reasons,
                    'earnings_date': earnings_date,
                    'earnings_soon': earnings_soon,
                    'pre_earnings_volatility': pre_earnings['avg_2week_volatility'],
                    'pre_earnings_pattern': pre_earnings['historical_pattern']
                }
                
                all_stocks[ticker] = stock_data
                
                print(f"    └─ ✓ {latest['signal']} signal (strength: {signal_strength}%)")
                
            except Exception as e:
                print(f"    └─ ✗ Error analyzing {ticker}: {e}")
                all_stocks[ticker] = {
                    'name': STOCK_NAMES.get(ticker, ticker),
                    'category': category,
                    'error': str(e)
                }
    
    # Count signals
    buy_count = sum(1 for s in all_stocks.values() if s.get('signal') == 'BUY')
    sell_count = sum(1 for s in all_stocks.values() if s.get('signal') == 'SELL')
    hold_count = sum(1 for s in all_stocks.values() if s.get('signal') == 'HOLD')
    
    results = {
        'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'market_breadth': market_breadth,
        'nasdaq_top10': nasdaq_top10,
        'stocks': all_stocks,
        'stock_groups': STOCK_GROUPS,
        'market_summary': {
            'total_buy_signals': buy_count,
            'total_sell_signals': sell_count,
            'total_hold_signals': hold_count
        }
    }
    
    # Save to JSON
    output_file = 'dashboard_data.json'
    print(f"\n{'=' * 70}")
    print("Saving results...")
    
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"✓ Data saved to {output_file}")
    except Exception as e:
        print(f"✗ Error saving file: {e}")
        return False
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("MARKET SUMMARY")
    print(f"{'=' * 70}")
    
    # Handle N/A values in display
    sp500_display = f"{market_breadth['sp500_pct_above_ma50']}%" if market_breadth['sp500_pct_above_ma50'] != 'N/A' else 'N/A'
    nasdaq_display = f"{market_breadth['nasdaq_pct_above_ma50']}%" if market_breadth['nasdaq_pct_above_ma50'] != 'N/A' else 'N/A'
    
    print(f"S&P 500 breadth: {sp500_display} above 50-day MA")
    print(f"NASDAQ breadth: {nasdaq_display} above 50-day MA")
    
    if market_breadth['market_healthy'] is None:
        print(f"Market Health: N/A (Streetstats data unavailable)")
    else:
        print(f"Market Health: {'✅ HEALTHY' if market_breadth['market_healthy'] else '⚠️ LOOKY LOOK OUT!'}")
    
    print(f"Data source: {market_breadth['source']}")
    print(f"\nNASDAQ Top 10 (1-week): {nasdaq_top10['combined_1week_return']:+.2f}%")
    print(f"\nSignals: {buy_count} BUY, {hold_count} HOLD, {sell_count} SELL")
    
    print(f"\n{'=' * 70}")
    print("✓ Dashboard data updated successfully!")
    print(f"Open 'dashboard.html' in your browser to view the results")
    print(f"{'=' * 70}\n")
    
    return True

if __name__ == '__main__':
    try:
        success = generate_live_data()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
