An AI Trading Bot Using News Sentiment

Aim of the Project
The aim of this project is to build a fully functional, automated trading bot from the ground up with the help of open source libraries. This is my third project and it builds directly on the skills from my last two, "Backtesting a Dual Moving Average Crossover Strategy" and "UK Market Sentiment Analysis."

Previously, I learned how to backtest a simple strategy and analyse market sentiment. This time, the goal was to combine those ideas and take it a step further by asking whether I could use a sophisticated Machine Learning model to analyse real-time news and make autonomous trading decisions.

My Learning Objectives:
The main goal here was learning and practice. I wanted to see if I could:
- Integrate a real ML model by moving beyond theory and using a powerful, pre-trained NLP model (FinBERT) in a practical application.
- Work with a live API, learning how to connect to a real trading platform (Alpaca) to get market data, fetch news, and place trades.
- Combine different signals instead of just relying on sentiment. I wanted to add a layer of risk management by using a technical indicator (Average True Range - ATR) to set dynamic stop-losses.
- Build a multi-file project to practice structuring my code in a cleaner way, with different parts of the logic separated into their own files.

How The Bot Works
The bot reads financial news and tries to trade based on the overall mood. If the news is very positive, it buys. If it's very negative, it sells (or shorts).

Here’s a breakdown of how the different files work together:

Startup & Setup (trading_bot_base.py):The main script starts up and securely loads my secret Alpaca API keys from the local .env file.
Data Gathering (trading_bot_base.py): The bot connects to the Alpaca API to fetch two key things, the latest price data for a stock (ex. Lloyds Bank) and a list of recent news headlines related to it.
Sentiment Analysis (finbert_utils.py): This is where the AI comes in. The collected headlines are passed over to the finbert_utils.py script which uses the powerful FinBERT model. This model is specifically trained on financial text to read and score the sentiment of the headlines. It returns the overall sentiment (neutral, positive, negative) and a confidence score.
Trade Decision (trading_bot_base.py): Back in the main script, the bot checks the sentiment. If the model's confidence is above a certain threshold (e.g., 95% sure), it decides to make a trade. It also checks to make sure it closes any old positions before opening a new one.
Risk Management (trading_bot_base.py): Before placing a trade, the bot calculates the Average True Range (ATR) of the stock. This tells it how volatile the stock has been recently. It uses the ATR to automatically set a stop-loss and a take-profit level, which is a simple way to manage risk without manual intervention.
Backtesting: Finally, the script uses the lumibot library to run this entire strategy on historical data (from 2022-2023) to see how it would have performed in the past.

How to Run the Project
Follow these steps to run a backtest of the strategy.

1. Get the Code
Clone the repository to your local machine.

2. Set Up Your API Keys
You'll need a free paper trading account from Alpaca.
Rename the env.example file to .env.
Open the .env file and add your paper trading API key and secret.

3. Install Dependencies
It's best to use a virtual environment for this, so not to cause any problems on your machine. Have AI help you with this. Install all the required libraries with this command in the terminal:

pip install -r requirements.txt

4. Run the Backtest
To run the backtest on Lloyds Bank stock (LLOY.L) for the years 2022-2023, just run the main script:

python trading_bot_base.py


Future Improvements & Next Steps:
This bot was a great learning exercise, but it's still very simple. If I were to continue developing it, here are the next steps I would take:

(1) Add More Signals: Right now, the bot only listens to sentiment. I could combine this with the moving average crossover strategy from my previous project. A new rule might be, "only buy if sentiment is positive AND the short-term trend is above the long-term trend."
(2) Use More News Sources: The bot currently relies only on the news feed provided by Alpaca. To get a better signal, I could integrate other free sources like RSS feeds from major financial news websites.
(3) Expand to Multiple Stocks: The bot only trades one stock at a time. A more advanced version would scan the news for a whole list of stocks and decide which one has the strongest sentiment signal to trade.
(4) Live Paper Trading: The final step would be to switch from backtesting to live paper trading to see how the bot performs in real-time market conditions.

Thanks, 
Yahia
