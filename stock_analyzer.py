import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class StockAnalyzer:
    """Analyzes stocks using technical indicators and generates buy/sell/hold signals"""
    
    def __init__(self, ticker, period='2y'):
        self.ticker = ticker
        self.period = period
        self.data = None
        self.signals = None
        
    def fetch_data(self):
        """Fetch historical stock data"""
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)
        return self.data
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)

        for i in range(period, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta

            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)

        return rsi
    
    def calculate_williams_r(self, high, low, close, period=14):
        """Calculate Williams %R"""
        williams_r = []
        for i in range(len(close)):
            if i < period - 1:
                williams_r.append(np.nan)
            else:
                highest_high = np.max(high[i - period + 1:i + 1])
                lowest_low = np.min(low[i - period + 1:i + 1])
                if highest_high != lowest_low:
                    wr = -100 * (highest_high - close[i]) / (highest_high - lowest_low)
                else:
                    wr = -50
                williams_r.append(wr)
        return np.array(williams_r)
    
    def calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd.values, signal_line.values, histogram.values
    
    def calculate_bollinger_bands(self, prices, period=20, num_std=2):
        """Calculate Bollinger Bands"""
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band.values, sma.values, lower_band.values
    
    def calculate_indicators(self):
        """Calculate all technical indicators"""
        if self.data is None:
            self.fetch_data()
        
        df = self.data.copy()
        
        # Moving Averages
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_21_EMA'] = df['Close'].ewm(span=21, adjust=False).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_100'] = df['Close'].rolling(window=100).mean()
        
        # Death Cross / Golden Cross detection
        df['MA_50_prev'] = df['MA_50'].shift(1)
        df['MA_100_prev'] = df['MA_100'].shift(1)
        
        # Death Cross: 50-day crosses BELOW 100-day
        df['Death_Cross'] = ((df['MA_50_prev'] > df['MA_100_prev']) & 
                             (df['MA_50'] < df['MA_100']))
        
        # Golden Cross: 50-day crosses ABOVE 100-day
        df['Golden_Cross'] = ((df['MA_50_prev'] < df['MA_100_prev']) & 
                              (df['MA_50'] > df['MA_100']))
        
        # Check if 10-day > 21-day > 50-day (bullish alignment)
        df['MA_Alignment_Bullish'] = ((df['MA_10'] > df['MA_21_EMA']) & 
                                       (df['MA_21_EMA'] > df['MA_50']))
        
        # Track days below 10-day MA
        df['Below_MA10'] = df['Close'] < df['MA_10']
        df['Days_Below_MA10'] = (df['Below_MA10'].groupby((df['Below_MA10'] != df['Below_MA10'].shift()).cumsum()).cumsum())
        df.loc[~df['Below_MA10'], 'Days_Below_MA10'] = 0
        
        # Intraday momentum (last half of day)
        # Using Close vs Open as proxy for daily direction
        df['Intraday_Up'] = df['Close'] > df['Open']
        
        # Volume analysis
        df['Volume_MA_20'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_20']
        df['Abnormal_Volume'] = df['Volume_Ratio'] > 2.0  # 2x average = abnormal
        
        # Track if stock moved up or down on abnormal volume
        df['Price_Change_Pct'] = df['Close'].pct_change() * 100
        df['Abnormal_Volume_Up'] = (df['Abnormal_Volume']) & (df['Price_Change_Pct'] > 0)
        df['Abnormal_Volume_Down'] = (df['Abnormal_Volume']) & (df['Price_Change_Pct'] < 0)
        
        # RSI (Relative Strength Index)
        df['RSI'] = self.calculate_rsi(df['Close'].values)
        
        # Williams %R
        df['Williams_R'] = self.calculate_williams_r(
            df['High'].values,
            df['Low'].values,
            df['Close'].values
        )
        
        # MACD
        macd, macd_signal, macd_hist = self.calculate_macd(df['Close'].values)
        df['MACD'] = macd
        df['MACD_Signal'] = macd_signal
        df['MACD_Hist'] = macd_hist
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(df['Close'].values)
        df['BB_Upper'] = bb_upper
        df['BB_Middle'] = bb_middle
        df['BB_Lower'] = bb_lower
        
        self.data = df
        return df
    
    def generate_signals(self):
        """Generate buy/sell/hold signals based on technical indicators"""
        if self.data is None or 'RSI' not in self.data.columns:
            self.calculate_indicators()
        
        df = self.data.copy()
        signals = []
        
        for idx in range(len(df)):
            if idx < 100:  # Need 100 days for MA_100
                signals.append('HOLD')
                continue
            
            row = df.iloc[idx]
            score = 0  # Positive = bullish, Negative = bearish
            reasons = []
            
            # 1. Death Cross/Golden Cross (STRONG signal)
            if row['Death_Cross']:
                score -= 3
                reasons.append('Death Cross (50-day crossed below 100-day)')
            elif row['Golden_Cross']:
                score += 3
                reasons.append('Golden Cross (50-day crossed above 100-day)')
            
            # 2. Price vs Moving Averages
            if row['Close'] > row['MA_20'] and row['Close'] > row['MA_50']:
                score += 1
                reasons.append('Price above 20-day and 50-day MA')
            elif row['Close'] < row['MA_20'] and row['Close'] < row['MA_50']:
                score -= 1
                reasons.append('Price below 20-day and 50-day MA')
            
            # 3. RSI
            if row['RSI'] < 30:  # Oversold
                score += 2
                reasons.append(f'RSI oversold ({row["RSI"]:.1f})')
            elif row['RSI'] > 70:  # Overbought
                score -= 2
                reasons.append(f'RSI overbought ({row["RSI"]:.1f})')
            
            # 4. Williams %R
            if row['Williams_R'] < -80:  # Oversold
                score += 1
                reasons.append(f'Williams %R oversold ({row["Williams_R"]:.1f})')
            elif row['Williams_R'] > -20:  # Overbought
                score -= 1
                reasons.append(f'Williams %R overbought ({row["Williams_R"]:.1f})')
            
            # 5. Bollinger Bands
            if row['Close'] < row['BB_Lower']:
                score += 1
                reasons.append('Price below lower Bollinger Band')
            elif row['Close'] > row['BB_Upper']:
                score -= 1
                reasons.append('Price above upper Bollinger Band')
            
            # 6. MACD
            if row['MACD'] > row['MACD_Signal'] and row['MACD_Hist'] > 0:
                score += 1
                reasons.append('MACD bullish')
            elif row['MACD'] < row['MACD_Signal'] and row['MACD_Hist'] < 0:
                score -= 1
                reasons.append('MACD bearish')
            
            # 7. Volume
            if row['Volume_Ratio'] > 1.5:
                # High volume - amplifies the signal
                if score > 0:
                    score += 1
                    reasons.append('High volume supports bullish signal')
                elif score < 0:
                    score -= 1
                    reasons.append('High volume supports bearish signal')
            
            # Generate final signal
            if score >= 3:
                signal = 'BUY'
            elif score <= -3:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            signals.append(signal)
        
        df['Signal'] = signals
        self.data = df
        self.signals = df['Signal']
        
        return df
    
    def get_latest_signal(self):
        """Get the most recent trading signal with details"""
        if self.signals is None:
            self.generate_signals()
        
        latest = self.data.iloc[-1]
        previous = self.data.iloc[-2] if len(self.data) > 1 else latest
        
        # Check your father's specific rules
        price = latest['Close']
        ma_10 = latest['MA_10']
        ma_21_ema = latest['MA_21_EMA']
        ma_50 = latest['MA_50']
        days_below_ma10 = int(latest['Days_Below_MA10'])
        
        # Rule checks
        below_ma10 = price < ma_10
        below_ma21_ema = price < ma_21_ema
        below_ma50 = price < ma_50
        ma_alignment_bullish = latest['MA_Alignment_Bullish']
        
        # Volume signals
        abnormal_volume_up = latest['Abnormal_Volume_Up']
        abnormal_volume_down = latest['Abnormal_Volume_Down']
        volume_ratio = latest['Volume_Ratio']
        
        # Intraday momentum
        intraday_momentum_up = latest['Intraday_Up']
        
        result = {
            'ticker': self.ticker,
            'date': latest.name.strftime('%Y-%m-%d'),
            'signal': latest['Signal'],
            'close_price': float(latest['Close']),
            'indicators': {
                'MA_10': float(latest['MA_10']),
                'MA_20': float(latest['MA_20']),
                'MA_21_EMA': float(latest['MA_21_EMA']),
                'MA_50': float(latest['MA_50']),
                'MA_100': float(latest['MA_100']),
                'RSI': float(latest['RSI']),
                'Williams_R': float(latest['Williams_R']),
                'MACD': float(latest['MACD']),
                'MACD_Signal': float(latest['MACD_Signal']),
                'BB_Upper': float(latest['BB_Upper']),
                'BB_Lower': float(latest['BB_Lower']),
                'Volume_Ratio': float(latest['Volume_Ratio'])
            },
            'father_rules': {
                'below_ma10': bool(below_ma10),
                'days_below_ma10': days_below_ma10,
                'below_ma21_ema': bool(below_ma21_ema),
                'below_ma50': bool(below_ma50),
                'ma_alignment_bullish': bool(ma_alignment_bullish),
                'abnormal_volume_up': bool(abnormal_volume_up),
                'abnormal_volume_down': bool(abnormal_volume_down),
                'volume_ratio': float(volume_ratio),
                'intraday_momentum_up': bool(intraday_momentum_up),
                'price_change_pct': float(latest['Price_Change_Pct'])
            },
            'death_cross_recent': bool(latest['Death_Cross']),
            'golden_cross_recent': bool(latest['Golden_Cross'])
        }
        
        return result
    
    def backtest(self, forward_days=5):
        """
        Backtest the signals to measure accuracy
        Returns metrics on signal performance
        """
        if self.signals is None:
            self.generate_signals()
        
        df = self.data.copy()
        
        # Calculate forward returns
        df['Forward_Return'] = df['Close'].shift(-forward_days) / df['Close'] - 1
        
        # Analyze signal performance
        buy_signals = df[df['Signal'] == 'BUY'].copy()
        sell_signals = df[df['Signal'] == 'SELL'].copy()
        
        results = {
            'ticker': self.ticker,
            'forward_days': forward_days,
            'total_days': len(df),
            'buy_signals': {
                'count': len(buy_signals),
                'avg_return': float(buy_signals['Forward_Return'].mean() * 100) if len(buy_signals) > 0 else 0,
                'win_rate': float((buy_signals['Forward_Return'] > 0).sum() / len(buy_signals) * 100) if len(buy_signals) > 0 else 0,
                'best_return': float(buy_signals['Forward_Return'].max() * 100) if len(buy_signals) > 0 else 0,
                'worst_return': float(buy_signals['Forward_Return'].min() * 100) if len(buy_signals) > 0 else 0
            },
            'sell_signals': {
                'count': len(sell_signals),
                'avg_return': float(sell_signals['Forward_Return'].mean() * 100) if len(sell_signals) > 0 else 0,
                'win_rate': float((sell_signals['Forward_Return'] < 0).sum() / len(sell_signals) * 100) if len(sell_signals) > 0 else 0,
                'best_return': float(sell_signals['Forward_Return'].min() * 100) if len(sell_signals) > 0 else 0,
                'worst_return': float(sell_signals['Forward_Return'].max() * 100) if len(sell_signals) > 0 else 0
            }
        }
        
        return results


if __name__ == '__main__':
    # Test with NVIDIA
    analyzer = StockAnalyzer('NVDA')
    analyzer.fetch_data()
    analyzer.calculate_indicators()
    analyzer.generate_signals()
    
    latest = analyzer.get_latest_signal()
    print(f"\nLatest Signal for {latest['ticker']}:")
    print(f"Date: {latest['date']}")
    print(f"Signal: {latest['signal']}")
    print(f"Price: ${latest['close_price']:.2f}")
    print(f"RSI: {latest['indicators']['RSI']:.1f}")
    print(f"Williams %R: {latest['indicators']['Williams_R']:.1f}")
    
    backtest = analyzer.backtest(forward_days=5)
    print(f"\nBacktest Results (5-day forward returns):")
    print(f"Buy Signals: {backtest['buy_signals']['count']} signals, "
          f"{backtest['buy_signals']['win_rate']:.1f}% win rate, "
          f"{backtest['buy_signals']['avg_return']:.2f}% avg return")
