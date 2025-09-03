# This file is for the main trading strategy
# # Import necessary libraries
from lumibot.brokers import Alpaca
from lumibot.backtesting import YahooDataBacktesting
from lumibot.strategies.strategy import Strategy
from lumibot.traders import Trader
from datetime import datetime
from timedelta import Timedelta
import pandas_ta as ta 
import pandas as pd
import os
from dotenv import load_dotenv 

# Import the newer Alpaca library components
from alpaca.trading.client import TradingClient
from alpaca.data.requests import NewsRequest

# Imports the sentiment analysis utility
from finbert_utils import estimate_sentiment

# Loads the API keys from the .env file
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_SECRET = os.getenv("API_SECRET")

# Alpaca API configuration
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_CREDS = {
    "API_KEY": API_KEY,
    "API_SECRET": API_SECRET,
    "PAPER": True
}

# This class defines a machine learning-based trading strategy.
# The initialise method sets up all the key parameters and connections needed for the bot to run.
# It kind of acts like the Settings for the bot.
class MLTrader(Strategy):

    def initialize(self, symbol="LLOY.L", cash_at_risk=.5, sentiment_threshold=0.95, atr_window=14):
        self.symbol = symbol # The stock ticker the bot will trade (e.g., LLOY.L for Lloyds)
        self.sleeptime = "24H" # How often the bot will check for new trades (once every 24 hours)
        self.last_trade = None # Keeps track of the last trade made (buy or sell)
        self.cash_at_risk = cash_at_risk # The proportion of cash to risk on each trade (e.g., 0.5 for 50%)
        self.sentiment_threshold = sentiment_threshold # The minimum confidence needed from the model to make a trade
        self.atr_window = atr_window # The number of days used to calculate the Average True Range (ATR)
        self.api = TradingClient(API_KEY, API_SECRET, paper=True) # Sets up the connection to the Alpaca trading API

    # This method works out how much cash to use and how many shares to buy, based on the current price.
    def position_sizing(self, last_price):
        cash = self.get_cash() # Get the current available cash in the account
        # This check is added to prevent errors if the price is zero for some reason
        if last_price <= 0:
            return 0, 0
        quantity = round(cash * self.cash_at_risk / last_price, 0) # Work out how many shares to buy
        return cash, quantity

    # This method gets today's date and the date three days ago, both as strings.
    def get_dates(self):
        today = self.get_datetime() # Get the current date and time from the broker
        three_days_prior = today - Timedelta(days=3) # Calculate the date three days before today
        return today.strftime('%Y-%m-%d'), three_days_prior.strftime('%Y-%m-%d')

    # This method fetches recent market data and calculates the ATR (Average True Range).
    # ATR is a technical indicator that measures how much a stock price moves on average.
    def get_market_data_and_atr(self):
        # We wrap this in a try-except block to handle any connection errors
        try:
            bars = self.get_historical_prices(self.symbol, 100, "day") # Get the last 100 days of price data
            df = bars.df # Convert the data to a pandas DataFrame
            df['atr'] = df.ta.atr(length=self.atr_window) # Calculate the ATR
            last_price = df['close'].iloc[-1] # Get the most recent closing price
            atr = df['atr'].iloc[-1] # Get the most recent ATR value
            return last_price, atr
        except Exception as e:
            # If something goes wrong, log the error and return nothing so the bot doesn't crash
            self.log_message(f"Error getting market data: {e}")
            return None, None

    # This method gathers news headlines for the stock and uses the sentiment model to judge the mood.
    def get_sentiment(self):
        today, three_days_prior = self.get_dates() # Get the date range for the news search
        news_request = NewsRequest(
            symbols=self.symbol,
            start=pd.Timestamp(three_days_prior),
            end=pd.Timestamp(today)
        )
        # We also wrap this in a try-except block, safety feature :)
        try:
            news = self.api.get_news(news_request) # Fetch news articles for the stock
            headlines = [item.headline for item in news] # Extract just the headlines from the news data
            probability, sentiment = estimate_sentiment(headlines) # Use our sentiment model
            return probability, sentiment
        except Exception as e:
            self.log_message(f"Error getting news sentiment: {e}")
            return None, None

    # This is the main trading loop. It runs each time the bot checks the market.
    def on_trading_iteration(self):
        last_price, atr = self.get_market_data_and_atr()
        # If getting the data failed, we stop this trading cycle to be safe
        if last_price is None or atr is None:
            return

        cash, quantity = self.position_sizing(last_price)
        probability, sentiment = self.get_sentiment()
        # Same check here, if sentiment analysis failed, it stops
        if sentiment is None:
            return

        # Set stop loss and take profit levels for both buying and selling, based on ATR
        stop_loss_buy = last_price - (2 * atr) # If price drops this much after buying, sell to limit loss
        take_profit_buy = last_price + (4 * atr) # If price rises this much after buying, sell to take profit
        stop_loss_sell = last_price + (2 * atr) # If price rises this much after selling, buy back to limit loss
        take_profit_sell = last_price - (4 * atr) # If price drops this much after selling, buy back to take profit

        # Only trade if you have enough cash and the quantity is more than 0
        if cash > last_price and quantity > 0:
            # If the sentiment is positive and the model is confident enough, consider buying
            if sentiment == "positive" and probability > self.sentiment_threshold:
                if self.last_trade == "sell": # If the last trade was a sell, close that position first
                    self.sell_all()
                # Create a buy order with bracket orders for stop loss and take profit
                order = self.create_order(
                    self.symbol, quantity, "buy", type="bracket",
                    take_profit_price=take_profit_buy, stop_loss_price=stop_loss_buy
                )
                self.submit_order(order)
                self.last_trade = "buy" # Remember that the last trade was a buy

            # If the sentiment is negative and the model is confident enough, consider selling
            elif sentiment == "negative" and probability > self.sentiment_threshold:
                if self.last_trade == "buy": # If the last trade was a buy, close that position first
                    self.sell_all()
                # Create a sell order with bracket orders for stop loss and take profit
                order = self.create_order(
                    self.symbol, quantity, "sell", type="bracket",
                    take_profit_price=take_profit_sell, stop_loss_price=stop_loss_sell
                )
                self.submit_order(order)
                self.last_trade = "sell" # Remember that the last trade was a sell


# The code below is the backtesting setup, it tells lumibot how to test our strategy.
# Using 'if __name__ == "__main__"' is good practice. It means this part only runs when you
# execute the file directly, not when it's imported by another file.
if __name__ == "__main__":
    # Set the start and end dates for the backtest
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2023, 12, 31)

    # Create a broker instance using our Alpaca credentials
    broker = Alpaca(ALPACA_CREDS)

    # Set up the trading strategy with our chosen parameters
    strategy = MLTrader(
        name='ml_trader_base',
        broker=broker,
        parameters={
            "symbol": "LLOY.L",
            "cash_at_risk": .5,
            "sentiment_threshold": 0.95
        }
    )

    # Run the backtest using Yahoo Finance data
    strategy.backtest(
        YahooDataBacktesting,
        start_date,
        end_date,
        parameters={"symbol": "LLOY.L", "cash_at_risk": .5, "sentiment_threshold": 0.95}

    )
