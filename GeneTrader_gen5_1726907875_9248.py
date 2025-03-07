
from datetime import datetime, timedelta
import talib.abstract as ta
import pandas_ta as pta
from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from pandas import DataFrame
from freqtrade.strategy import DecimalParameter, IntParameter
from functools import reduce
import warnings

warnings.simplefilter(action="ignore", category=RuntimeWarning)
TMP_HOLD = []
TMP_HOLD1 = []


class GeneTrader(IStrategy):
    minimal_roi = {
        "0": 1
    }
    timeframe = '15m'
    process_only_new_candles = True
    startup_candle_count = 240
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'emergency_exit': 'market',
        'force_entry': 'market',
        'force_exit': "market",
        'stoploss': 'market',
        'stoploss_on_exchange': False,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_market_ratio': 0.99
    }

    stoploss = -0.25
    trailing_stop = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True


    buy_rsi_fast_32 = IntParameter(20.0, 70.0, default=48, space='buy', optimize=True)
    buy_rsi_32 = IntParameter(15.0, 50.0, default=25, space='buy', optimize=True)
    buy_sma15_32 = DecimalParameter(0.9, 1.0, default=0.944, space='buy', optimize=True)
    buy_cti_32 = DecimalParameter(-1.0, 1.0, default=0.39, space='buy', optimize=True)
    sell_fastx = IntParameter(50.0, 100.0, default=60, space='sell', optimize=True)

    sell_loss_cci = IntParameter(low=0, high=600, default=3, space='sell', optimize=True)
    sell_loss_cci_profit = DecimalParameter(-0.15, 0.0, default=0.0, space='sell', optimize=True)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['sma_15'] = ta.SMA(dataframe, timeperiod=15)
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastk'] = stoch_fast['fastk']

        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)

        dataframe['ma120'] = ta.MA(dataframe, timeperiod=120)
        dataframe['ma240'] = ta.MA(dataframe, timeperiod=240)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        dataframe.loc[:, 'enter_tag'] = ''
        buy_1 = (
                (dataframe['rsi_slow'] < dataframe['rsi_slow'].shift(1)) &
                (dataframe['rsi_fast'] < self.buy_rsi_fast_32.value) &
                (dataframe['rsi'] > self.buy_rsi_32.value) &
                (dataframe['close'] < dataframe['sma_15'] * self.buy_sma15_32.value) &
                (dataframe['cti'] < self.buy_cti_32.value)
        )
        
        conditions.append(buy_1)
        dataframe.loc[buy_1, 'enter_tag'] += 'buy_1'

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x | y, conditions),
                'enter_long'] = 1
        return dataframe

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        current_candle = dataframe.iloc[-1].squeeze()

        min_profit = trade.calc_profit_ratio(trade.min_rate)

        if current_candle['close'] > current_candle["ma120"] or current_candle['close'] > current_candle["ma240"]:
            if trade.id not in TMP_HOLD:
                TMP_HOLD.append(trade.id)
        else:
            if trade.id not in TMP_HOLD1:
                TMP_HOLD1.append(trade.id)

        if current_profit > 0:
            if current_candle["fastk"] > self.sell_fastx.value:
                return "fastk_profit_sell"

        if min_profit > -0.06:
            if current_profit > -0.02:
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell_2"

        if -0.06 > min_profit > -0.1:
            if current_profit > -0.05:
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell_5"

        if min_profit <= -0.1:
            if current_profit > self.sell_loss_cci_profit.value:
                if current_candle["cci"] > self.sell_loss_cci.value:
                    return "cci_loss_sell_10"

        if trade.id in TMP_HOLD and current_candle["close"] < current_candle["ma120"] and current_candle["close"] <                 current_candle["ma240"]:
            if current_time - timedelta(minutes=5) < trade.open_date_utc:
                try:
                    TMP_HOLD.remove(trade.id)
                except:
                    pass
                if trade.id in TMP_HOLD1:
                    pass
                else:
                    TMP_HOLD1.append(trade.id)
            else:
                return "ma120_sell"

        if trade.id in TMP_HOLD1:
            if current_candle["high"] > current_candle["ma120"] or current_candle["high"] > current_candle["ma240"]:
                if current_time - timedelta(minutes=5) > trade.open_date_utc:
                    TMP_HOLD1.remove(trade.id)
                    return "cross_120_or_240_sell"

        return None

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[:, ['exit_long', 'exit_tag']] = (0, 'long_out')
        return dataframe

