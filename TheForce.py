# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from datetime import timedelta
import talib.abstract as ta
import technical as qtpylib

class TheForce(IStrategy):
  
    INTERFACE_VERSION = 3

    stoploss = -0.015

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    # trailing_stop_positive = 0.01
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '15m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "ask_strategy" section in the config.

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    
    plot_config = {
        # Main plot indicators (Moving averages, ...)
        'main_plot': {
            'tema': {},
            'sar': {'color': 'white'},
        },
        'subplots': {
            # Subplots - each dict defines one additional plot
            "MACD": {
                'macd': {'color': 'blue'},
                'macdsignal': {'color': 'orange'},
            },
            "RSI": {
                'rsi': {'color': 'red'},
            }
        }
    }
    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pair/interval combinations are non-tradeable, unless they are part
        of the whitelist as well.
        For more information, please consult the documentation
        :return: List of tuples in the format (pair, interval)
            Sample: return [("ETH/USDT", "5m"),
                            ("BTC/USDT", "15m"),
                            ]
        """
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Dataframe with data from the exchange
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        
        # Momentum Indicators
        # ------------------------------------

        # Stochastic Fast
        stoch_fast = ta.STOCHF(dataframe,5,3,3)
        
        dataframe['enter_long']= 0 
        
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        # # Stochastic RSI
        stoch_rsi = ta.STOCHRSI(dataframe)
        dataframe['fastd_rsi'] = stoch_rsi['fastd']
        dataframe['fastk_rsi'] = stoch_rsi['fastk']

        # MACD
        macd = ta.MACD(dataframe,12,26,2)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']

        # # EMA - Exponential Moving Average

        dataframe['ema5c'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema5o'] = ta.EMA(dataframe['open'], timeperiod=5)


        return dataframe
    

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe['fastk'] >= 20) & (dataframe['fastk'] <= 80) &
                (dataframe['fastd'] >= 20) & (dataframe['fastd'] <= 80) &
                (dataframe['macd'] > dataframe['macd'].shift(1)) &
                (dataframe['macdsignal'] > dataframe['macdsignal'].shift(1)) &
                (dataframe['close'] > dataframe['close'].shift(1)) &
                (dataframe['ema5c'] >= dataframe['ema5o']) 
                #&
                #(dataframe.index >= (current_time - timedelta(minutes=0.5)))#time limit
            ),
            'enter_long'
        ] = 1

        return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame populated with indicators
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        dataframe.loc[
            (
                (
                    (dataframe['fastk'] <= 80)
                    &
                    (dataframe['fastd'] <= 80)
                )
                &
                (
                    (dataframe['macd'] < dataframe['macd'].shift(1))
                    &
                    (dataframe['macdsignal'] < dataframe['macdsignal'].shift(1))
                )
                &
                (
                    (dataframe['ema5c'] < dataframe['ema5o'])
                )
                
            ),
            'exit_long'] = 1
        return dataframe
    