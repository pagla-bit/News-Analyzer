import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Set page config
st.set_page_config(page_title="S&P 500 News Sentiment Analyzer", layout="wide")

# Load FinBERT model
@st.cache_resource
def load_finbert_model():
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    return tokenizer, model

# S&P 500 stock list (static)
@st.cache_data
def get_sp500_tickers():
    tickers = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "UNH", "JNJ",
        "V", "XOM", "WMT", "JPM", "PG", "MA", "CVX", "HD", "MRK", "ABBV",
        "KO", "PEP", "COST", "AVGO", "TMO", "MCD", "CSCO", "ACN", "LLY", "NKE",
        "ABT", "ADBE", "DHR", "CRM", "VZ", "TXN", "NEE", "CMCSA", "WFC", "DIS",
        "PM", "ORCL", "NFLX", "BMY", "UPS", "INTC", "T", "AMD", "QCOM", "HON",
        "RTX", "LOW", "AMGN", "UNP", "BA", "SBUX", "SPGI", "ELV", "GS", "BLK",
        "CAT", "INTU", "DE", "AXP", "GILD", "LMT", "BKNG", "MDT", "PLD", "ADI",
        "SYK", "MDLZ", "ADP", "CI", "MMC", "TJX", "REGN", "VRTX", "CVS", "CB",
        "ISRG", "ZTS", "C", "SO", "PGR", "DUK", "NOC", "MO", "TMUS", "BSX",
        "EOG", "ITW", "GE", "BDX", "USB", "HUM", "AON", "ETN", "SLB", "PNC",
        "COP", "SCHW", "CSX", "MS", "LRCX", "FI", "MMM", "CL", "CME", "MU",
        "SHW", "MCK", "NSC", "APD", "ICE", "EMR", "PYPL", "MAR", "WM", "EQIX",
        "TGT", "KLAC", "AJG", "PH", "D", "EL", "AFL", "HCA", "F", "PSA",
        "GM", "ADM", "TT", "APH", "CDNS", "FCX", "ANET", "SPG", "DLR", "AIG",
        "GD", "SNPS", "ORLY", "TEL", "ECL", "COF", "NEM", "KMB", "SRE", "MSCI",
        "FDX", "RSG", "TRV", "CMG", "NXPI", "O", "PCAR", "WELL", "AMP", "JCI",
        "ROP", "TDG", "MCO", "AZO", "MNST", "PSX", "CTAS", "PAYX", "MSI", "AEP",
        "DG", "HES", "VLO", "KMI", "CMI", "CCI", "MET", "AMT", "AEE", "STZ",
        "ROST", "PRU", "ALL", "BK", "CARR", "EW", "SYY", "WMB", "GIS", "CTVA",
        "HSY", "ADSK", "MCHP", "AFL", "TFC", "GWW", "HLT", "FAST", "KR", "OKE",
        "DVN", "PEG", "DD", "IQV", "DFS", "IDXX", "YUM", "A", "ODFL", "KHC",
        "VRSK", "EXC", "CTSH", "ROK", "CPRT", "EA", "XEL", "CBRE", "ED", "BKR",
        "GLW", "DOW", "GEHC", "IT", "ON", "VICI", "CHTR", "WBD", "RMD", "FANG",
        "VMC", "MTD", "ANSS", "AWK", "MLM", "EBAY", "DAL", "KEYS", "KDP", "CDW",
        "ZBH", "HPQ", "EIX", "ACGL", "WTW", "STT", "FITB", "PPG", "AVB", "APTV",
        "CSGP", "HAL", "EFX", "MPWR", "DXCM", "WEC", "XYL", "SBAC", "TSCO", "VTR",
        "UAL", "ALB", "IFF", "LH", "AME", "EQR", "ETR", "GPN", "DLTR", "LUV",
        "TTWO", "CAH", "LYB", "BIIB", "WAB", "HPE", "RJF", "TROW", "FTV", "URI",
        "DHI", "BALL", "MOH", "MTB", "DTE", "WY", "BR", "NVR", "RF", "IR",
        "TSN", "BAX", "EXPE", "ARE", "INVH", "TYL", "ESS", "CFG", "HBAN", "LEN",
        "MAA", "EXR", "NTRS", "CBOE", "STE", "STLD", "WAT", "PFG", "CLX", "POOL",
        "CNP", "DRI", "IRM", "ATO", "PKI", "DGX", "HOLX", "K", "FE", "ALGN",
        "CCL", "TDY", "CRL", "CINF", "BBY", "AES", "COO", "LVS", "EPAM", "VRSN",
        "SYF", "DOV", "LNT", "NI", "J", "ZBRA", "SWKS", "MKC", "LKQ", "OMC",
        "IP", "MAS", "CHRW", "UDR", "CFG", "AKAM", "CPT", "KIM", "JBHT", "HST",
        "SNA", "JKHY", "CE", "EXPD", "MKTX", "EVRG", "NDSN", "BXP", "PAYC", "WRB",
        "MGM", "FFIV", "TECH", "REG", "IPG", "TER", "CAG", "AOS", "HSIC", "NCLH",
        "LW", "GL", "UHS", "CTLT", "AAL", "BF.B", "FOXA", "CPB", "HII", "WYNN",
        "AIZ", "TAP", "SEE", "PNW", "PARA", "CMA", "FRT", "BBWI", "HWM", "BEN",
        "RHI", "HAS", "IVZ", "ZION", "RL", "NWS", "MTCH", "ALK", "ALLE", "TPR",
        "BWA", "DISH", "DVA", "MHK", "WHR", "PNR", "FMC", "GNRC", "VFC", "NWL",
        "LNC", "XRAY", "AAP", "NWSA", "HRL", "DXC", "APA", "FOX", "ROL", "EMN",
        "NLSN", "CZR", "PENN", "MOS", "BIO", "IEX", "PBCT", "PHM", "CF", "WDC"
    ]
    return tickers

def scrape_finviz_data(ticker):
    """Scrape both news and stock info from Finviz in one request"""
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Get news
        news_items = []
        news_table = soup.find('table', {'id': 'news-table'})
        today = datetime.now().strftime("%b-%d-%y")
        
        if news_table:
            for row in news_table.find_all('tr'):
                try:
                    date_cell = row.find('td', {'align': 'right'})
                    headline_cell = row.find('a', {'class': 'tab-link-news'})
                    
                    if date_cell and headline_cell:
                        date_text = date_cell.text.strip()
                        
                        # Check if it's today's news
                        if today in date_text or (len(date_text.split()) == 1 and ':' in date_text):
                            headline = headline_cell.text.strip()
                            news_items.append(headline)
                except:
                    continue
        
        # Get stock info
        price = "N/A"
        market_cap = "N/A"
        volume = "N/A"
        
        table = soup.find('table', {'class': 'snapshot-table2'})
        if table:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all('td')
                for i in range(0, len(cells), 2):
                    if i + 1 < len(cells):
                        label = cells[i].text.strip()
                        value = cells[i + 1].text.strip()
                        
                        if label == "Price":
                            price = value
                        elif label == "Market Cap":
                            market_cap = value
                        elif label == "Volume":
                            volume = value
        
        return ticker, news_items, price, market_cap, volume
    except:
        return ticker, [], "N/A", "N/A", "N/A"

def analyze_sentiments_batch(headlines, tokenizer, model, batch_size=32):
    """Analyze multiple headlines in batches for faster processing"""
    if not headlines:
        return []
    
    sentiments = []
    
    for i in range(0, len(headlines), batch_size):
        batch = headlines[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, 
                          max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get sentiment for each item in batch
        for pred in predictions:
            sentiment_scores = pred.tolist()
            labels = ['positive', 'negative', 'neutral']
            max_score_idx = sentiment_scores.index(max(sentiment_scores))
            sentiments.append(labels[max_score_idx])
    
    return sentiments

def process_ticker(ticker, tokenizer, model):
    """Process a single ticker - scrape and analyze"""
    ticker_symbol, news_items, price, market_cap, volume = scrape_finviz_data(ticker)
    
    if not news_items:
        return None
    
    # Analyze all headlines in batch
    sentiments = analyze_sentiments_batch(news_items, tokenizer, model)
    
    positive_count = sentiments.count('positive')
    negative_count = sentiments.count('negative')
    neutral_count = sentiments.count('neutral')
    
    return {
        'ticker': ticker_symbol,
        'price': price,
        'market_cap': market_cap,
        'volume': volume,
        'total_news': len(news_items),
        'positive': positive_count,
        'negative': negative_count,
        'neutral': neutral_count
    }

# Streamlit UI
st.title("📊 S&P 500 News Sentiment Analyzer")
st.markdown("Analyze today's news sentiment for S&P 500 stocks using FinBERT")

# Refresh button
if st.button("🔄 Analyze Today's News", type="primary"):
    start_time = time.time()
    
    # Load model
    with st.spinner("Loading FinBERT model..."):
        tokenizer, model = load_finbert_model()
    
    # Get tickers
    tickers = get_sp500_tickers()
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    completed = 0
    
    # Use ThreadPoolExecutor for concurrent scraping
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit all tasks
        future_to_ticker = {
            executor.submit(process_ticker, ticker, tokenizer, model): ticker 
            for ticker in tickers
        }
        
        # Process completed tasks
        for future in as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                pass  # Skip failed tickers
            
            # Update progress
            progress_bar.progress(completed / len(tickers))
            status_text.text(f"Processed {completed}/{len(tickers)} stocks... Found {len(results)} with news")
    
    elapsed_time = time.time() - start_time
    status_text.text(f"Analysis complete! Processed in {elapsed_time:.1f} seconds")
    time.sleep(2)
    progress_bar.empty()
    status_text.empty()
    
    if results:
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Sort for top positive news
        df_positive = df.nlargest(10, 'positive')
        
        # Sort for top negative news
        df_negative = df.nlargest(10, 'negative')
        
        # Display tables
        st.header("📈 Top 10 Stocks with Most Positive News")
        
        # Create display DataFrame for positive
        display_positive = pd.DataFrame({
            'Stock': [f"[{row['ticker']}](https://finviz.com/quote.ashx?t={row['ticker']})" 
                     for _, row in df_positive.iterrows()],
            'Price': df_positive['price'].values,
            'Market Cap': df_positive['market_cap'].values,
            'Volume': df_positive['volume'].values,
            'News Count': [f"Total: {row['total_news']} | Positive: {row['positive']} | Negative: {row['negative']}" 
                          for _, row in df_positive.iterrows()]
        })
        
        st.markdown(display_positive.to_markdown(index=False), unsafe_allow_html=True)
        
        st.header("📉 Top 10 Stocks with Most Negative News")
        
        # Create display DataFrame for negative
        display_negative = pd.DataFrame({
            'Stock': [f"[{row['ticker']}](https://finviz.com/quote.ashx?t={row['ticker']})" 
                     for _, row in df_negative.iterrows()],
            'Price': df_negative['price'].values,
            'Market Cap': df_negative['market_cap'].values,
            'Volume': df_negative['volume'].values,
            'News Count': [f"Total: {row['total_news']} | Positive: {row['positive']} | Negative: {row['negative']}" 
                          for _, row in df_negative.iterrows()]
        })
        
        st.markdown(display_negative.to_markdown(index=False), unsafe_allow_html=True)
        
        # Summary stats
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Stocks with News", len(results))
        with col2:
            st.metric("Total News Analyzed", df['total_news'].sum())
        with col3:
            st.metric("Processing Time", f"{elapsed_time:.1f}s")
    else:
        st.warning("No news found for any stocks today.")

st.markdown("---")
st.markdown("*Data source: [Finviz.com](https://finviz.com) | Sentiment analysis: FinBERT*")
