"""
Aadi Kalloo

The code here implements a pairs trading algorithm technique. For two stocks, it checks that their time series are not stationary and are cointegrated. 
Trades are made for 90 minutes after market opening and 45 minutes before market close. First the 35 day history is compiled and the time series are 
log normalized. The spread and spot prices are then calculated. A standardized scoring value (z-score) is calculated using spot spread and mean spread, 
and this value is used to assess whether or not a new trade should be placed. If there is cointegration, the trade is placed. 

"""
import math
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller

def initialize(context):
    #Priceline and Berkshire Hathaway
    context.stock_1 = sid(19917)
    context.stock_2 = sid(1091)
    context.overval = False
    context.underval = False
    context.securities_held = [context.stock_1, context.stock_2]
    
    # Run 90 min at market open and 45 min before market close
    schedule_function(trade_, date_rules.every_day(), time_rules.market_open(minutes=90))
    schedule_function(trade_, date_rules.every_day(), time_rules.market_close(minutes=45))

def trade_(context, data):
    # dont run if there are open orders
    if len(get_open_orders()) > 0:
        return

    are_cointegrated = False
    stock_1 = context.stock_1
    stock_2 = context.stock_2
    spread, stock_1_series, stock_2_series = get_spread(data, context, stock_1, stock_2)

    mean_spread = np.mean(spread)
    std_spread = np.std(spread)
    spot_spread = calc_spot(data, stock_1, stock_2)

    # Compute z-score and check if standard deviation is zero to avoid error
    if std_spread > 0: 
        standardized_val = (spot_spread - mean_spread)/std_spread
    else: 
        standardized_val = 0

    analyze_validate(data, context, standardized_val, stock_1, stock_2, stock_1_series, stock_2_series, are_cointegrated)

def perform_order(stock_1, stock_2):
    order_target(stock_1, 0)
    order_target(stock_2, 0)

def calc_spot(data, stock_1, stock_2):
    #get spot prices and calculate spot spread
    stock_1_spot = np.log10(data.current(stock_1, 'price'))
    stock_2_spot = np.log10(data.current(stock_2, 'price'))       
    spot_spread = stock_1_spot - stock_2_spot
    return spot_spread

def get_spread(data, context, stock_1, stock_2):
    #use 35 day window
    stock_1_ma = data.history(context.stock_1, 'price', 35, '1d')
    stock_2_ma = data.history(context.stock_2, 'price', 35, '1d')

    stock_1_raw = pd.DataFrame(data = stock_1_ma.values, columns = ["values"])
    stock_2_raw = pd.DataFrame(data = stock_2_ma.values, columns = ["values"])

    stock_1_raw = stock_1_raw.astype(float)
    stock_2_raw = stock_2_raw.astype(float)

    stock_1_series = np.log10(stock_1_raw["values"])
    stock_2_series = np.log10(stock_2_raw["values"])
    spread = stock_1_series - stock_2_series 
    return spread, stock_1_series, stock_2_series

def stationary_check(series, sig_level=0.05):
# use augmented Dickey-Fuller test to check if stationary
    pvalue = adfuller(series)[1]
    if pvalue < sig_level:
        return True
    else:
        return False

def make_trade(data, context, stock_1, stock_2, score):
    # get total cash available for trading
    cash_available = context.portfolio.cash
    # max 45% available cash can be traded    
    stock_1_shares = (cash_available * 0.45) / data.current(stock_1, 'price')
    stock_2_shares = (cash_available * 0.45) / data.current(stock_2, 'price')

    if score > 1 and not context.overval and all(data.can_trade(context.securities_held)):
        order(stock_1, -stock_1_shares)
        order(stock_2, stock_2_shares) 
        context.overval = True
        context.underval = False

    elif score < -1 and not context.underval and all(data.can_trade(context.securities_held)):
        order(stock_1, stock_1_shares) 
        order(stock_2, -stock_2_shares) 
        context.overval = False
        context.underval = True

def analyze_validate(data, context, standardized_val, stock_1, stock_2, stock_1_series, stock_2_series, are_cointegrated):
    #check for mean reversion
    if abs(standardized_val) < 1 and (context.overval or context.underval):
        if all(data.can_trade(context.securities_held)):
            perform_order(stock_1, stock_2)
            context.overval = False
            context.underval = False
            return

    elif abs(standardized_val) > 1:
        if standardized_val > 1 and context.overval:
            return
        elif (standardized_val > 1 and context.underval) or (standardized_val < -1 and context.overval):
            perform_order(stock_1, stock_2)
            context.overval = False
            context.underval = False
            return
            
        elif standardized_val < -1 and context.underval:
            return

        # check both series for being non stationary
        stock_1_stat = stationary_check( stock_1_series, .10 )
        stock_2_stat = stationary_check( stock_2_series, .10 )

        if not stock_1_stat and not stock_2_stat:
            # check if co-integrated
            score, pvalue, _ = coint(stock_1_series, stock_2_series)
            if pvalue < 0.05:
                are_cointegrated = True
            else: 
                log.info("Series are not cointegerated")
                are_cointegrated = False
                return
        else: 
                return

        # if cointegrated make appropriate trade
        if are_cointegrated == True:
            make_trade(data, context, stock_1, stock_2, standardized_val)

def handle_data(context, data):
    pass