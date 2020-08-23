import alpaca_trade_api as tradeapi
import os

def submit_order(ticker, spread):
    os.environ["APCA_API_KEY_ID"] = ""
    os.environ["APCA_API_SECRET_KEY"] = ""
    os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
    api = tradeapi.REST()
    account = api.get_account()
    ticker_bars = api.get_barset(ticker, 'minute', 1).df.iloc[0]
    ticker_price = ticker_bars[ticker]['close']

    # We could buy a position and add a stop-loss and a take-profit of 5 %
    api.submit_order(
        ticker=ticker,
        qty=1,
        side='buy',
        type='market',
        time_in_force='gtc',
        order_class='bracket',
        stop_loss={'stop_price': ticker_price * (1-spread),
                   'limit_price':  ticker_price * (1-spread)*0.95},
        take_profit={'limit_price': ticker_price * (1+spread)}
    )
    api.list_positions()

    # Get our account information.

    # Check if our account is restricted from trading.
    if account.trading_blocked:
        print('Account is currently restricted from trading.')

    # Check how much money we can use to open new positions.
    print('${} is available as buying power.'.format(account.buying_power))