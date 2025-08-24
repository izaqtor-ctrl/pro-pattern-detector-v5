import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings

# Try to import yfinance with error handling
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Pro Pattern Detector v5.0", 
    layout="wide"
)

def create_demo_data(ticker, period):
    """Create realistic demo data when yfinance is not available"""
    days_map = {"1y": 252, "6mo": 126, "3mo": 63, "1mo": 22}
    days = days_map.get(period, 63)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
    
    np.random.seed(hash(ticker) % 2147483647)
    base_price = 150 + (hash(ticker) % 100)
    
    returns = np.random.normal(0.001, 0.02, days)
    returns[0] = 0
    
    close_prices = base_price * np.cumprod(1 + returns)
    
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, days))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, days))
    open_mult = 1 + np.random.normal(0, 0.005, days)
    
    data = pd.DataFrame({
        'Open': close_prices * open_mult,
        'High': close_prices * high_mult,
        'Low': close_prices * low_mult,
        'Close': close_prices,
        'Volume': np.random.randint(1000000, 5000000, days)
    }, index=dates)
    
    data['High'] = np.maximum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
    data['Low'] = np.minimum.reduce([data['Open'], data['High'], data['Low'], data['Close']])
    
    return data

def get_stock_data(ticker, period):
    """Fetch stock data with fallback to demo data"""
    if not YFINANCE_AVAILABLE:
        st.info(f"Using demo data for {ticker}")
        return create_demo_data(ticker, period)
    
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        if len(data) == 0:
            st.warning(f"No data for {ticker}, using demo data")
            return create_demo_data(ticker, period)
        return data
    except Exception as e:
        st.warning(f"Error fetching {ticker}, using demo data")
        return create_demo_data(ticker, period)

def calculate_rsi(data, period=14):
    """Calculate RSI"""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(data, fast=12, slow=26, signal=9):
    """Calculate MACD"""
    ema_fast = data['Close'].ewm(span=fast).mean()
    ema_slow = data['Close'].ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def analyze_volume_pattern(data, pattern_type, pattern_info):
    """Enhanced volume analysis with breakout confirmation and confidence capping"""
    volume_score = 0
    volume_info = {}
    
    if len(data) < 20:
        return volume_score, volume_info
    
    # Calculate volume metrics
    avg_volume_20 = data['Volume'].tail(20).mean()
    current_volume = data['Volume'].iloc[-1]
    recent_volume_5 = data['Volume'].tail(5).mean()
    
    # Volume breakout multiplier
    volume_multiplier = current_volume / avg_volume_20
    recent_multiplier = recent_volume_5 / avg_volume_20
    
    volume_info['avg_volume_20'] = avg_volume_20
    volume_info['current_volume'] = current_volume
    volume_info['volume_multiplier'] = volume_multiplier
    volume_info['recent_multiplier'] = recent_multiplier
    
    # NEW: High-Volume Breakout Check (1.3-2.0x average)
    if volume_multiplier >= 2.0:
        volume_score += 25
        volume_info['exceptional_volume'] = True
        volume_info['volume_status'] = f"Exceptional Volume ({volume_multiplier:.1f}x)"
    elif volume_multiplier >= 1.5:
        volume_score += 20
        volume_info['strong_volume'] = True
        volume_info['volume_status'] = f"Strong Volume ({volume_multiplier:.1f}x)"
    elif volume_multiplier >= 1.3:
        volume_score += 15
        volume_info['good_volume'] = True
        volume_info['volume_status'] = f"Good Volume ({volume_multiplier:.1f}x)"
    else:
        volume_info['weak_volume'] = True
        volume_info['volume_status'] = f"Weak Volume ({volume_multiplier:.1f}x)"
    
    # Pattern-specific volume analysis (enhanced scoring)
    if pattern_type == "Bull Flag":
        # Enhanced bull flag volume pattern
        if 'flagpole_gain' in pattern_info:
            try:
                flagpole_start = min(25, len(data) - 10)
                flagpole_end = 15
                
                flagpole_vol = data['Volume'].iloc[-flagpole_start:-flagpole_end].mean()
                flag_vol = data['Volume'].tail(15).mean()
                
                if flagpole_vol > flag_vol * 1.2:
                    volume_score += 20  # Increased from 15
                    volume_info['flagpole_volume_pattern'] = True
                    volume_info['flagpole_vol_ratio'] = flagpole_vol / flag_vol
                elif flagpole_vol > flag_vol * 1.1:
                    volume_score += 10
                    volume_info['moderate_flagpole_volume'] = True
                    volume_info['flagpole_vol_ratio'] = flagpole_vol / flag_vol
            except:
                pass
    
    elif pattern_type == "Cup Handle":
        # Enhanced cup handle volume analysis
        try:
            handle_days = min(30, len(data) // 3)
            if handle_days > 5:
                cup_data = data.iloc[:-handle_days]
                handle_data = data.tail(handle_days)
                
                if len(cup_data) > 10:
                    cup_volume = cup_data['Volume'].mean()
                    handle_volume = handle_data['Volume'].mean()
                    
                    if handle_volume < cup_volume * 0.80:
                        volume_score += 20  # Increased from 8
                        volume_info['significant_volume_dryup'] = True
                        volume_info['handle_vol_ratio'] = handle_volume / cup_volume
                    elif handle_volume < cup_volume * 0.90:
                        volume_score += 15
                        volume_info['moderate_volume_dryup'] = True
                        volume_info['handle_vol_ratio'] = handle_volume / cup_volume
        except:
            pass
    
    elif pattern_type == "Flat Top Breakout":
        # Enhanced flat top volume analysis
        resistance_tests = data['Volume'].tail(20)
        avg_resistance_volume = resistance_tests.mean()
        
        if current_volume > avg_resistance_volume * 1.4:
            volume_score += 20
            volume_info['breakout_volume_surge'] = True
            volume_info['resistance_vol_ratio'] = current_volume / avg_resistance_volume
        elif current_volume > avg_resistance_volume * 1.2:
            volume_score += 15
            volume_info['moderate_breakout_volume'] = True
            volume_info['resistance_vol_ratio'] = current_volume / avg_resistance_volume
    
    # Volume trend analysis (additional scoring)
    volume_trend = data['Volume'].tail(5).mean() / data['Volume'].tail(20).mean()
    if volume_trend > 1.1:
        volume_score += 5
        volume_info['increasing_volume_trend'] = True
    elif volume_trend < 0.9:
        volume_score += 5  # Volume dryup can be bullish for consolidation patterns
        volume_info['decreasing_volume_trend'] = True
    
    return volume_score, volume_info

def detect_flat_top(data, macd_line, signal_line, histogram):
    """Detect flat top: ASCENSION → DESCENSION → HIGHER LOWS with enhanced volume"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 50:
        return confidence, pattern_info
    
    # STEP 1: Initial ascension (10%+ rise)
    ascent_start = min(45, len(data) - 15)
    ascent_end = 25
    
    start_price = data['Close'].iloc[-ascent_start]
    peak_price = data['High'].iloc[-ascent_start:-ascent_end].max()
    initial_gain = (peak_price - start_price) / start_price
    
    if initial_gain < 0.10:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['initial_ascension'] = f"{initial_gain*100:.1f}%"
    
    # STEP 2: Descension with lower highs
    descent_data = data.iloc[-ascent_end:-10]
    descent_low = descent_data['Low'].min()
    pullback = (peak_price - descent_low) / peak_price
    
    if pullback < 0.08:
        return confidence, pattern_info
    
    descent_highs = descent_data['High'].rolling(3, center=True).max().dropna()
    if len(descent_highs) >= 2:
        if descent_highs.iloc[-1] < descent_highs.iloc[0] * 0.97:
            confidence += 20
            pattern_info['descending_highs'] = True
    
    # STEP 3: Current higher lows
    current_lows = data.tail(15)['Low'].rolling(3, center=True).min().dropna()
    if len(current_lows) >= 3:
        if current_lows.iloc[-1] > current_lows.iloc[0] * 1.01:
            confidence += 25
            pattern_info['higher_lows'] = True
    
    # STEP 4: Flat resistance
    resistance_level = peak_price
    touches = sum(1 for h in data['High'].tail(20) if h >= resistance_level * 0.98)
    if touches >= 2:
        confidence += 15
        pattern_info['resistance_level'] = resistance_level
        pattern_info['resistance_touches'] = touches
    
    # STEP 5: Recency check
    current_price = data['Close'].iloc[-1]
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] >= resistance_level * 0.98), 11)
    
    if days_old > 8:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    if current_price < descent_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below support'}
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # NEW: Enhanced Volume Analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Flat Top Breakout", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # NEW: Confidence Cap Without Volume Confirmation
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_bull_flag(data, macd_line, signal_line, histogram):
    """Detect bull flag with enhanced volume analysis"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    # Recent flagpole
    flagpole_start = min(25, len(data) - 10)
    flagpole_end = 15
    
    start_price = data['Close'].iloc[-flagpole_start]
    peak_price = data['High'].iloc[-flagpole_start:-flagpole_end].max()
    flagpole_gain = (peak_price - start_price) / start_price
    
    if flagpole_gain < 0.08:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['flagpole_gain'] = f"{flagpole_gain*100:.1f}%"
    
    # Flag pullback
    flag_data = data.tail(15)
    flag_start = data['Close'].iloc[-flagpole_end]
    current_price = data['Close'].iloc[-1]
    
    pullback = (current_price - flag_start) / flag_start
    if -0.15 <= pullback <= 0.05:
        confidence += 20
        pattern_info['flag_pullback'] = f"{pullback*100:.1f}%"
        pattern_info['healthy_pullback'] = True
    
    # Invalidation checks
    flag_low = flag_data['Low'].min()
    if current_price < flag_low * 0.95:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flag support'}
    
    if current_price < start_price:
        return 0, {'pattern_broken': True, 'break_reason': 'Below flagpole start'}
    
    # Recency
    flag_high = flag_data['High'].max()
    days_old = next((i for i in range(1, 11) if data['High'].iloc[-i] == flag_high), 11)
    
    if days_old > 10:
        return confidence * 0.5, {**pattern_info, 'pattern_stale': True, 'days_old': days_old}
    
    pattern_info['days_since_high'] = days_old
    
    # Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 15
        pattern_info['macd_bullish'] = True
    
    if histogram.iloc[-1] > histogram.iloc[-3]:
        confidence += 10
        pattern_info['momentum_recovering'] = True
    
    # NEW: Enhanced Volume Analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Bull Flag", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # Near breakout
    if current_price >= flag_high * 0.95:
        confidence += 10
        pattern_info['near_breakout'] = True
    
    # NEW: Confidence Cap Without Volume Confirmation
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_cup_handle(data, macd_line, signal_line, histogram):
    """Detect cup handle with enhanced volume analysis"""
    confidence = 0
    pattern_info = {}
    
    if len(data) < 30:
        return confidence, pattern_info
    
    # STEP 1: Much more flexible sizing
    max_lookback = min(100, len(data) - 3)
    
    # Handle can be much longer - up to 30 days
    handle_days = min(30, max_lookback // 3)
    cup_days = max_lookback - handle_days
    
    cup_data = data.iloc[-max_lookback:-handle_days] if handle_days > 0 else data.iloc[-max_lookback:]
    handle_data = data.tail(handle_days) if handle_days > 0 else data.tail(5)
    
    if len(cup_data) < 15:
        return confidence, pattern_info
    
    # STEP 2: Very lenient cup formation
    cup_start = cup_data['Close'].iloc[0]
    cup_bottom = cup_data['Low'].min()
    cup_right = cup_data['Close'].iloc[-1]
    cup_depth = (max(cup_start, cup_right) - cup_bottom) / max(cup_start, cup_right)
    
    # Much more lenient cup requirements
    if cup_depth < 0.08 or cup_depth > 0.60:
        return confidence, pattern_info
    
    if cup_right < cup_start * 0.75:
        return confidence, pattern_info
    
    confidence += 25
    pattern_info['cup_depth'] = f"{cup_depth*100:.1f}%"
    
    # STEP 3: Very lenient handle validation
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        current_price = data['Close'].iloc[-1]
        handle_depth = (cup_right - handle_low) / cup_right
        
        if handle_depth > 0.25:
            confidence += 10
            pattern_info['deep_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.08:
            confidence += 20
            pattern_info['perfect_handle'] = f"{handle_depth*100:.1f}%"
        elif handle_depth <= 0.15:
            confidence += 15
            pattern_info['good_handle'] = f"{handle_depth*100:.1f}%"
        else:
            confidence += 10
            pattern_info['acceptable_handle'] = f"{handle_depth*100:.1f}%"
        
        # Handle duration
        if handle_days > 25:
            confidence *= 0.8
            pattern_info['long_handle'] = f"{handle_days} days"
        elif handle_days <= 10:
            confidence += 10
            pattern_info['short_handle'] = f"{handle_days} days"
        elif handle_days <= 20:
            confidence += 5
            pattern_info['medium_handle'] = f"{handle_days} days"
        
    else:
        confidence += 10
        pattern_info['forming_handle'] = "Handle forming"
    
    # STEP 4: Very lenient recency
    current_price = data['Close'].iloc[-1]
    
    # STEP 5: Very lenient pattern validation
    breakout_level = max(cup_start, cup_right)
    if current_price < breakout_level * 0.70:
        confidence *= 0.7
        pattern_info['far_from_rim'] = True
    else:
        confidence += 5
    
    # STEP 6: No strict breakage rules
    if handle_days > 0:
        handle_low = handle_data['Low'].min()
        if current_price < handle_low * 0.90:
            confidence *= 0.8
            pattern_info['below_handle'] = True
    
    # STEP 7: Technical confirmation
    if macd_line.iloc[-1] > signal_line.iloc[-1]:
        confidence += 10
        pattern_info['macd_bullish'] = True
    
    # NEW: Enhanced Volume Analysis
    volume_score, volume_info = analyze_volume_pattern(data, "Cup Handle", pattern_info)
    confidence += volume_score
    pattern_info.update(volume_info)
    
    # STEP 8: Minimum confidence check
    if confidence < 35:
        return confidence, pattern_info
    
    # NEW: Confidence Cap Without Volume Confirmation
    if not (volume_info.get('good_volume') or volume_info.get('strong_volume') or volume_info.get('exceptional_volume')):
        confidence = min(confidence, 70)
        pattern_info['confidence_capped'] = "No volume confirmation"
    
    return confidence, pattern_info

def detect_pattern(data, pattern_type):
    """Detect patterns with enhanced volume analysis and confidence capping"""
    if len(data) < 30:
        return False, 0, {}
    
    # Add indicators
    data['RSI'] = calculate_rsi(data)
    macd_line, signal_line, histogram = calculate_macd(data)
    
    # Volume analysis
    data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
    data['Volume_Ratio'] = data['Volume'] / data['Volume_SMA']
    
    confidence = 0
    pattern_info = {}
    
    if pattern_type == "Flat Top Breakout":
        confidence, pattern_info = detect_flat_top(data, macd_line, signal_line, histogram)
        confidence = min(confidence, 100)
        
    elif pattern_type == "Bull Flag":
        confidence, pattern_info = detect_bull_flag(data, macd_line, signal_line, histogram)
        confidence = min(confidence * 1.05, 100)
        
    elif pattern_type == "Cup Handle":
        confidence, pattern_info = detect_cup_handle(data, macd_line, signal_line, histogram)
        confidence = min(confidence * 1.1, 100)
    
    # Add technical data
    pattern_info['macd_line'] = macd_line
    pattern_info['signal_line'] = signal_line
    pattern_info['histogram'] = histogram
    
    return confidence >= 55, confidence, pattern_info

def calculate_levels(data, pattern_info, pattern_type):
    """Calculate entry, stop, targets using MEASURED MOVES with improved R/R ratios"""
    current_price = data['Close'].iloc[-1]
    
    # Calculate a reasonable stop distance based on recent volatility
    recent_range = data['High'].tail(20) - data['Low'].tail(20)
    avg_range = recent_range.mean()
    volatility_stop_distance = avg_range * 1.5
    
    if pattern_type == "Flat Top Breakout":
        # Entry at resistance breakout
        entry = pattern_info.get('resistance_level', current_price * 1.01)
        
        # Improved stop calculation
        recent_low = data['Low'].tail(15).min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = recent_low * 0.98
        
        # Use the higher of the two (closer stop for better R/R)
        stop = max(volatility_stop, traditional_stop)
        
        # Ensure stop is below entry with minimum distance
        min_stop_distance = entry * 0.03
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Triangle height projection
        if 'resistance_level' in pattern_info:
            support_level = data['Low'].tail(20).max()
            triangle_height = entry - support_level
            triangle_height = max(triangle_height, entry * 0.05)
            
            target1 = entry + triangle_height
            target2 = entry + (triangle_height * 1.618)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.0)
            target2 = entry + (risk * 3.5)
        
        target_method = "Triangle Height Projection"
        
    elif pattern_type == "Bull Flag":
        # Entry at flag breakout
        flag_high = data['High'].tail(15).max()
        entry = flag_high * 1.005
        
        # Improved stop for bull flags
        flag_low = data['Low'].tail(12).min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = flag_low * 0.98
        
        stop = max(volatility_stop, traditional_stop)
        
        min_stop_distance = entry * 0.04
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Enhanced flagpole calculation
        if 'flagpole_gain' in pattern_info:
            try:
                flagpole_pct_str = pattern_info['flagpole_gain'].replace('%', '')
                flagpole_pct = float(flagpole_pct_str) / 100
                
                flagpole_start_price = entry / (1 + flagpole_pct)
                flagpole_height = entry - flagpole_start_price
                flagpole_height = max(flagpole_height, entry * 0.08)
                
                target1 = entry + flagpole_height
                target2 = entry + (flagpole_height * 1.382)
            except (ValueError, KeyError):
                risk = entry - stop
                target1 = entry + (risk * 2.5)
                target2 = entry + (risk * 4.0)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.5)
            target2 = entry + (risk * 4.0)
        
        target_method = "Flagpole Height Projection"
        
    elif pattern_type == "Cup Handle":
        # Improved cup handle entry calculation
        if 'cup_depth' in pattern_info:
            try:
                cup_depth_str = pattern_info['cup_depth'].replace('%', '')
                cup_depth_pct = float(cup_depth_str) / 100
                
                estimated_rim = current_price / (1 - cup_depth_pct * 0.3)
                entry = estimated_rim * 1.005
            except (ValueError, KeyError):
                entry = current_price * 1.02
        else:
            entry = current_price * 1.02
        
        # Improved stop for cup handles
        handle_low = data.tail(15)['Low'].min()
        volatility_stop = entry - volatility_stop_distance
        traditional_stop = handle_low * 0.97
        
        stop = max(volatility_stop, traditional_stop)
        
        min_stop_distance = entry * 0.05
        if stop >= entry:
            stop = entry - min_stop_distance
        elif (entry - stop) < min_stop_distance:
            stop = entry - min_stop_distance
        
        # MEASURED MOVE: Improved cup depth projection
        if 'cup_depth' in pattern_info:
            try:
                cup_depth_str = pattern_info['cup_depth'].replace('%', '')
                cup_depth_pct = float(cup_depth_str) / 100
                
                cup_depth_dollars = entry * cup_depth_pct
                cup_depth_dollars = max(cup_depth_dollars, entry * 0.10)
                
                target1 = entry + cup_depth_dollars
                target2 = entry + (cup_depth_dollars * 1.618)
            except (ValueError, KeyError):
                risk = entry - stop
                target1 = entry + (risk * 2.0)
                target2 = entry + (risk * 3.0)
        else:
            risk = entry - stop
            target1 = entry + (risk * 2.0)
            target2 = entry + (risk * 3.0)
        
        target_method = "Cup Depth Projection"
    
    else:
        # Fallback for any other patterns
        entry = current_price * 1.01
        stop = current_price * 0.95
        target1 = entry + (entry - stop) * 2.0
        target2 = entry + (entry - stop) * 3.0
        target_method = "Traditional 2:1 & 3:1"
    
    # Final safety checks to ensure good R/R ratios
    risk_amount = entry - stop
    reward1 = target1 - entry
    reward2 = target2 - entry
    
    # If R/R is too low, adjust targets upward
    if risk_amount > 0:
        rr1 = reward1 / risk_amount
        rr2 = reward2 / risk_amount
        
        if rr1 < 1.5:
            target1 = entry + (risk_amount * 1.5)
            reward1 = target1 - entry
            rr1 = 1.5
        
        if rr2 < 2.5:
            target2 = entry + (risk_amount * 2.5)
            reward2 = target2 - entry
            rr2 = 2.5
    else:
        risk_amount = entry * 0.05
        stop = entry - risk_amount
        target1 = entry + (risk_amount * 2.0)
        target2 = entry + (risk_amount * 3.0)
        reward1 = target1 - entry
        reward2 = target2 - entry
        rr1 = 2.0
        rr2 = 3.0
    
    return {
        'entry': entry,
        'stop': stop,
        'target1': target1,
        'target2': target2,
        'risk': risk_amount,
        'reward1': reward1,
        'reward2': reward2,
        'rr_ratio1': reward1 / risk_amount if risk_amount > 0 else 0,
        'rr_ratio2': reward2 / risk_amount if risk_amount > 0 else 0,
        'target_method': target_method,
        'measured_move': True,
        'volatility_adjusted': True
    }

def create_chart(data, ticker, pattern_type, pattern_info, levels):
    """Create enhanced chart with volume analysis and measured move annotations"""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            f'{ticker} - {pattern_type} | {levels["target_method"]}',
            'MACD Analysis', 
            'Volume Profile (20-Day Average)'
        ),
        vertical_spacing=0.05,
        row_heights=[0.6, 0.25, 0.15]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    data['SMA20'] = data['Close'].rolling(window=20).mean()
    fig.add_trace(
        go.Scatter(x=data.index, y=data['SMA20'], name='SMA 20', 
                  line=dict(color='orange', width=1)),
        row=1, col=1
    )
    
    # Trading levels with enhanced annotations
    fig.add_hline(y=levels['entry'], line_color="green", line_width=2,
                 annotation_text=f"🔈 Entry: ${levels['entry']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['stop'], line_color="red", line_width=2,
                 annotation_text=f"🛑 Stop: ${levels['stop']:.2f}", row=1, col=1)
    fig.add_hline(y=levels['target1'], line_color="lime", line_width=2,
                 annotation_text=f"🎯 Target 1: ${levels['target1']:.2f} ({levels['rr_ratio1']:.1f}:1)", row=1, col=1)
    fig.add_hline(y=levels['target2'], line_color="darkgreen", line_width=1,
                 annotation_text=f"🎯 Target 2: ${levels['target2']:.2f} ({levels['rr_ratio2']:.1f}:1)", row=1, col=1)
    
    # NEW: Volume status annotation on chart
    volume_status = pattern_info.get('volume_status', 'Unknown Volume')
    volume_color = 'lime' if pattern_info.get('exceptional_volume') else 'orange' if pattern_info.get('strong_volume') else 'yellow' if pattern_info.get('good_volume') else 'red'
    
    fig.add_annotation(
        x=data.index[-10], y=levels['entry'] * 1.02,
        text=f"📊 {volume_status}",
        showarrow=True, arrowhead=2, arrowcolor=volume_color,
        bgcolor=f"rgba(255,255,255,0.8)", bordercolor=volume_color,
        font=dict(color=volume_color, size=12)
    )
    
    # Add pattern-specific annotations
    if pattern_type == "Bull Flag" and 'flagpole_gain' in pattern_info:
        flagpole_height = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Measured Move: ${flagpole_height:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    elif pattern_type == "Cup Handle" and 'cup_depth' in pattern_info:
        cup_move = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Cup Depth Move: ${cup_move:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    elif pattern_type == "Flat Top Breakout":
        triangle_height = levels['reward1']
        fig.add_annotation(
            x=data.index[-5], y=levels['target1'],
            text=f"Triangle Height: ${triangle_height:.2f}",
            showarrow=True, arrowhead=2, arrowcolor="lime",
            bgcolor="rgba(0,255,0,0.1)", bordercolor="lime"
        )
    
    # MACD chart
    macd_line = pattern_info['macd_line']
    signal_line = pattern_info['signal_line']
    histogram = pattern_info['histogram']
    
    fig.add_trace(go.Scatter(x=data.index, y=macd_line, name='MACD', line=dict(color='blue')), row=2, col=1)
    fig.add_trace(go.Scatter(x=data.index, y=signal_line, name='Signal', line=dict(color='red')), row=2, col=1)
    
    colors = ['green' if h >= 0 else 'red' for h in histogram]
    fig.add_trace(go.Bar(x=data.index, y=histogram, name='Histogram', marker_color=colors, opacity=0.6), row=2, col=1)
    fig.add_hline(y=0, line_color="black", row=2, col=1)
    
    # Enhanced Volume chart with average line
    volume_colors = []
    avg_volume = data['Volume'].rolling(window=20).mean()
    
    for i, vol in enumerate(data['Volume']):
        if i >= 19:  # After we have 20 days for average
            if vol >= avg_volume.iloc[i] * 2.0:
                volume_colors.append('darkgreen')  # Exceptional volume
            elif vol >= avg_volume.iloc[i] * 1.5:
                volume_colors.append('green')      # Strong volume
            elif vol >= avg_volume.iloc[i] * 1.3:
                volume_colors.append('lightgreen') # Good volume
            else:
                volume_colors.append('red')        # Weak volume
        else:
            volume_colors.append('blue')  # Default for early days
    
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', 
                        marker_color=volume_colors, opacity=0.7), row=3, col=1)
    
    # Add 20-day volume average line
    fig.add_trace(go.Scatter(x=data.index, y=avg_volume, name='20-Day Avg', 
                            line=dict(color='black', width=2, dash='dash')), row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    
    return fig

def main():
    st.title("🎯 Pro Pattern Detector v5.0")
    st.markdown("**Enhanced Volume Analysis** - Professional Pattern Recognition with Volume Confirmation")
    
    if not YFINANCE_AVAILABLE:
        st.warning("⚠️ **Demo Mode**: Using simulated data (yfinance not available)")
    
    st.error("""
    🚨 **DISCLAIMER**: Educational purposes only. Not financial advice. 
    Trading involves substantial risk. Consult professionals before trading.
    """)
    
    # Info box about new features
    with st.expander("🆕 What's New in v5.0 - Enhanced Volume Analysis"):
        st.markdown("""
        ### 📊 **Professional Volume Analysis**
        
        **🔥 Volume Breakout Detection (1.3-2.0x)**:
        - **Exceptional Volume** (2.0x+): +25 confidence points
        - **Strong Volume** (1.5-2.0x): +20 confidence points  
        - **Good Volume** (1.3-1.5x): +15 confidence points
        - **Weak Volume** (<1.3x): Pattern confidence capped at 70%
        
        **📈 Pattern-Specific Volume Analysis**:
        - **Bull Flag**: Flagpole volume vs flag volume (up to +20 points)
        - **Cup Handle**: Volume dryup during handle (up to +20 points)  
        - **Flat Top**: Breakout volume surge (up to +20 points)
        
        **⚠️ Confidence Capping System**:
        - **No volume confirmation** = Maximum 70% confidence
        - **Strong volume** = Up to 100% confidence
        
        **🎨 Enhanced Visuals**:
        - Volume bars color-coded by strength
        - Volume status displayed on price chart
        - 20-day volume average reference line
        
        **This matches how institutional traders validate breakouts!** 🏦
        """)
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    patterns = ["Flat Top Breakout", "Bull Flag", "Cup Handle"]
    selected_patterns = st.sidebar.multiselect(
        "Select Patterns:", patterns, default=["Flat Top Breakout", "Bull Flag"]
    )
    
    tickers = st.sidebar.text_input("Tickers:", "AAPL,MSFT,NVDA")
    period = st.sidebar.selectbox("Period:", ["1mo", "3mo", "6mo", "1y"], index=1)
    min_confidence = st.sidebar.slider("Min Confidence:", 45, 85, 55)
    
    # NEW: Volume filter option
    st.sidebar.subheader("🔊 Volume Filters")
    require_volume = st.sidebar.checkbox("Require Volume Confirmation", value=False, 
                                       help="Only show patterns with good volume (1.3x+ average)")
    volume_threshold = st.sidebar.selectbox("Volume Threshold:", 
                                          ["1.3x (Good)", "1.5x (Strong)", "2.0x (Exceptional)"], 
                                          index=0)
    
    if st.sidebar.button("🚀 Analyze", type="primary"):
        if tickers and selected_patterns:
            ticker_list = [t.strip().upper() for t in tickers.split(',')]
            
            st.header("📈 Pattern Analysis Results")
            results = []
            
            for ticker in ticker_list:
                st.subheader(f"📊 {ticker}")
                
                data = get_stock_data(ticker, period)
                if data is not None and len(data) >= 50:
                    
                    # Detect all patterns first
                    all_patterns = {}
                    for pattern in selected_patterns:
                        detected, confidence, info = detect_pattern(data, pattern)
                        
                        # Apply volume filter if enabled
                        skip_pattern = False
                        if require_volume:
                            volume_multiplier = info.get('volume_multiplier', 0)
                            threshold_map = {"1.3x (Good)": 1.3, "1.5x (Strong)": 1.5, "2.0x (Exceptional)": 2.0}
                            required_threshold = threshold_map[volume_threshold]
                            
                            if volume_multiplier < required_threshold:
                                skip_pattern = True
                        
                        if detected and confidence >= min_confidence and not skip_pattern:
                            all_patterns[pattern] = {'confidence': confidence, 'info': info}
                    
                    # Show pattern conflicts
                    if len(all_patterns) > 1:
                        st.warning(f"⚠️ **Multiple patterns detected** - consider which is most dominant:")
                        for pat, details in all_patterns.items():
                            st.write(f"  • {pat}: {details['confidence']:.0f}%")
                    
                    # Display each pattern
                    for pattern in selected_patterns:
                        detected, confidence, info = detect_pattern(data, pattern)
                        
                        # Apply volume filter
                        skip_pattern = False
                        if require_volume:
                            volume_multiplier = info.get('volume_multiplier', 0)
                            threshold_map = {"1.3x (Good)": 1.3, "1.5x (Strong)": 1.5, "2.0x (Exceptional)": 2.0}
                            required_threshold = threshold_map[volume_threshold]
                            
                            if volume_multiplier < required_threshold:
                                skip_pattern = True
                                st.info(f"⏭️ {pattern}: {confidence:.0f}% - Filtered by volume requirement ({volume_multiplier:.1f}x < {required_threshold}x)")
                                continue
                        
                        if detected and confidence >= min_confidence:
                            levels = calculate_levels(data, info, pattern)
                            
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                # Enhanced confidence display with volume status
                                if confidence >= 80:
                                    if info.get('exceptional_volume') or info.get('strong_volume'):
                                        st.success(f"✅ {pattern} DETECTED")
                                    else:
                                        st.success(f"✅ {pattern} DETECTED (No Volume)")
                                elif confidence >= 70:
                                    if info.get('good_volume') or info.get('strong_volume'):
                                        st.success(f"🟢 {pattern} DETECTED")
                                    else:
                                        st.info(f"🟡 {pattern} DETECTED (No Volume)")
                                else:
                                    st.info(f"🟡 {pattern} DETECTED")
                                    
                                st.metric("Confidence", f"{confidence:.0f}%")
                                
                                # NEW: Volume status display
                                volume_status = info.get('volume_status', 'Unknown')
                                if info.get('exceptional_volume'):
                                    st.success(f"🔥 {volume_status}")
                                elif info.get('strong_volume'):
                                    st.success(f"💪 {volume_status}")
                                elif info.get('good_volume'):
                                    st.info(f"👍 {volume_status}")
                                else:
                                    st.warning(f"⚠️ {volume_status}")
                                
                                # Show confidence capping
                                if info.get('confidence_capped'):
                                    st.warning(f"📉 Capped: {info['confidence_capped']}")
                                
                                # Trading levels
                                st.write("**📊 Trading Levels:**")
                                st.write(f"**Entry**: ${levels['entry']:.2f}")
                                st.write(f"**Stop**: ${levels['stop']:.2f}")
                                st.write(f"**Target 1**: ${levels['target1']:.2f}")
                                st.write(f"**Target 2**: ${levels['target2']:.2f}")
                                
                                # Risk/Reward ratios
                                st.write("**🎯 Risk/Reward:**")
                                st.write(f"**T1 R/R**: {levels['rr_ratio1']:.1f}:1")
                                st.write(f"**T2 R/R**: {levels['rr_ratio2']:.1f}:1")
                                
                                # Calculation method
                                st.info(f"📐 **Method**: {levels['target_method']}")
                            
                            with col2:
                                # Enhanced pattern information with volume details
                                
                                # Volume-specific information
                                if info.get('flagpole_volume_pattern'):
                                    ratio = info.get('flagpole_vol_ratio', 0)
                                    st.success(f"📊 Flagpole volume surge: {ratio:.1f}x")
                                elif info.get('moderate_flagpole_volume'):
                                    ratio = info.get('flagpole_vol_ratio', 0)
                                    st.info(f"📊 Moderate flagpole volume: {ratio:.1f}x")
                                
                                if info.get('significant_volume_dryup'):
                                    ratio = info.get('handle_vol_ratio', 0)
                                    st.success(f"💧 Significant volume dryup: {ratio:.1f}x")
                                elif info.get('moderate_volume_dryup'):
                                    ratio = info.get('handle_vol_ratio', 0)
                                    st.info(f"💧 Moderate volume dryup: {ratio:.1f}x")
                                
                                if info.get('breakout_volume_surge'):
                                    ratio = info.get('resistance_vol_ratio', 0)
                                    st.success(f"🚀 Breakout volume surge: {ratio:.1f}x")
                                elif info.get('moderate_breakout_volume'):
                                    ratio = info.get('resistance_vol_ratio', 0)
                                    st.info(f"🚀 Moderate breakout volume: {ratio:.1f}x")
                                
                                if info.get('increasing_volume_trend'):
                                    st.info("📈 Volume trend increasing")
                                elif info.get('decreasing_volume_trend'):
                                    st.info("📉 Volume trend decreasing")
                                
                                # Existing pattern information
                                if info.get('initial_ascension'):
                                    st.write(f"🚀 Initial rise: {info['initial_ascension']}")
                                if info.get('descending_highs'):
                                    st.write("📉 Lower highs phase")
                                if info.get('higher_lows'):
                                    st.write("📈 Higher lows (triangle)")
                                if info.get('resistance_touches'):
                                    st.write(f"🔴 Resistance: {info['resistance_touches']} touches")
                                
                                # Bull Flag with measured move explanation
                                if info.get('flagpole_gain'):
                                    flagpole_pct = info['flagpole_gain']
                                    flagpole_dollars = levels['reward1']
                                    st.write(f"🚀 Flagpole: {flagpole_pct}")
                                    st.success(f"📐 **Measured Move**: ${flagpole_dollars:.2f}")
                                    st.write("*Target = Entry + Flagpole Height*")
                                
                                if info.get('healthy_pullback'):
                                    st.write(f"📉 Flag pullback: {info.get('flag_pullback', '')}")
                                if info.get('days_since_high'):
                                    st.write(f"⏰ Flag age: {info['days_since_high']} days")
                                
                                # Cup Handle with measured move explanation
                                if info.get('cup_depth'):
                                    cup_pct = info['cup_depth']
                                    cup_dollars = levels['reward1']
                                    st.write(f"☕ Cup depth: {cup_pct}")
                                    st.success(f"📐 **Measured Move**: ${cup_dollars:.2f}")
                                    st.write("*Target = Rim + Cup Depth*")
                                
                                if info.get('perfect_handle'):
                                    st.success(f"✨ Perfect handle: {info['perfect_handle']}")
                                elif info.get('good_handle'):
                                    st.write(f"👍 Good handle: {info['good_handle']}")
                                elif info.get('acceptable_handle'):
                                    st.write(f"✅ Acceptable handle: {info['acceptable_handle']}")
                                elif info.get('deep_handle'):
                                    st.warning(f"⚠️ Deep handle: {info['deep_handle']}")
                                
                                if info.get('short_handle'):
                                    st.write(f"⚡ Short handle: {info['short_handle']}")
                                elif info.get('medium_handle'):
                                    st.write(f"⏰ Medium handle: {info['medium_handle']}")
                                elif info.get('long_handle'):
                                    st.warning(f"⚠️ Long handle: {info['long_handle']}")
                                
                                # Flat Top with triangle height explanation
                                if pattern == "Flat Top Breakout" and levels.get('measured_move'):
                                    triangle_height = levels['reward1']
                                    st.success(f"📐 **Triangle Height**: ${triangle_height:.2f}")
                                    st.write("*Target = Breakout + Triangle Height*")
                                
                                # Show if volatility-adjusted stops were used
                                if levels.get('volatility_adjusted'):
                                    st.info("📊 Volatility-adjusted stops")
                                
                                # Warnings and status
                                if info.get('pattern_stale'):
                                    if info.get('days_old'):
                                        st.warning(f"⚠️ Pattern aging: {info['days_old']} days")
                                
                                if info.get('pattern_broken'):
                                    st.error(f"🚨 BROKEN: {info.get('break_reason', '')}")
                                
                                # Technical indicators
                                if info.get('macd_bullish'):
                                    st.write("📈 MACD bullish")
                                if info.get('momentum_recovering'):
                                    st.write("📈 Momentum recovering")
                                if info.get('near_breakout'):
                                    st.write("🎯 Near breakout")
                            
                            # Chart
                            fig = create_chart(data, ticker, pattern, info, levels)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add to results with enhanced volume information
                            results.append({
                                'Ticker': ticker,
                                'Pattern': pattern,
                                'Confidence': f"{confidence:.0f}%",
                                'Volume': info.get('volume_status', 'Unknown'),
                                'Entry': f"${levels['entry']:.2f}",
                                'Stop': f"${levels['stop']:.2f}",
                                'Target 1': f"${levels['target1']:.2f}",
                                'Target 2': f"${levels['target2']:.2f}",
                                'R/R 1': f"{levels['rr_ratio1']:.1f}:1",
                                'R/R 2': f"{levels['rr_ratio2']:.1f}:1",
                                'Risk': f"${levels['risk']:.2f}",
                                'Method': levels['target_method']
                            })
                        else:
                            if not skip_pattern:
                                st.info(f"⌛ {pattern}: {confidence:.0f}% (below threshold)")
                else:
                    st.error(f"⌛ Insufficient data for {ticker}")
            
            # Summary
            if results:
                st.header("📋 Summary")
                df = pd.DataFrame(results)
                st.dataframe(df, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Patterns", len(results))
                with col2:
                    scores = [int(r['Confidence'].replace('%', '')) for r in results]
                    avg_score = sum(scores) / len(scores) if scores else 0
                    st.metric("Avg Confidence", f"{avg_score:.0f}%")
                with col3:
                    if results:
                        ratios = [float(r['R/R 1'].split(':')[0]) for r in results]
                        avg_rr = sum(ratios) / len(ratios) if ratios else 0
                        st.metric("Avg R/R T1", f"{avg_rr:.1f}:1")
                with col4:
                    # NEW: Volume quality metric
                    high_vol_count = sum(1 for r in results if 'Strong' in r['Volume'] or 'Exceptional' in r['Volume'])
                    vol_quality = (high_vol_count / len(results)) * 100 if results else 0
                    st.metric("High Volume %", f"{vol_quality:.0f}%")
                
                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"patterns_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
            else:
                if require_volume:
                    st.info(f"📊 No patterns detected with {volume_threshold} volume requirement. Try lowering volume threshold or confidence.")
                else:
                    st.info("📊 No patterns detected. Try lowering the confidence threshold.")

if __name__ == "__main__":
    main()
