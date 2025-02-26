import yfinance as yf





spy = yf.Ticker("SPY")
data = spy.history(period="1mo")  # Adjust period as needed
print(data)