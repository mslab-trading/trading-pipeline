from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Type, Hashable, Optional
from math import nan, isnan
import pandas as pd
import numpy as np
from tqdm import tqdm


@dataclass
class Config:
    """
    Configuration settings for the backtest.
    """

    commission: float = 0.001425  # 交易手續費率
    tax: float = 0.003  # 交易稅率


@dataclass
class StockHolding:
    """
    Represents a stock holding within a portfolio.
    """

    symbol: Optional[str] = None
    quantity: float = 0.0
    total_cost: float = 0.0  # 持有成本(包含手續費)
    last_price: float = nan
    last_date: Optional[datetime] = None

    def update(self, last_date: datetime, last_price: float):
        if isnan(last_price) or last_price <= 0:
            return
        self.last_date = last_date
        self.last_price = last_price

    @property
    def average_cost(self) -> float:
        if self.quantity == 0:
            return 0.0
        return self.total_cost / self.quantity

    @property
    def market_value(self) -> float:  # 市值
        if isnan(self.last_price):
            return 0.0
        return self.quantity * self.last_price

    @property
    def realized_value(self) -> float:  # 實際可得金額(扣除手續費與交易稅)
        if isnan(self.last_price):
            return 0.0
        return self.quantity * self.last_price * (1 - Config.commission - Config.tax)


@dataclass
class Portfolio:
    """
    Represents a financial portfolio.
    """

    cash: float = 0.0  # 現金
    stocks: Dict[str, StockHolding] = None

    @property
    def total_market_value(self) -> float:
        if self.stocks is None:
            return 0.0
        return sum(stock.market_value for stock in self.stocks.values())

    @property
    def total_realized_value(self) -> float:
        if self.stocks is None:
            return 0.0
        return sum(stock.realized_value for stock in self.stocks.values())


@dataclass
class Trade:
    """
    Represents a completed financial transaction.
    """

    symbol: Optional[str] = None
    trade_date: Optional[datetime] = None
    price: float = nan
    position_size: float = nan
    is_buy: bool = True  # True for buy, False for sell
    trade_commission: float = nan


@dataclass
class Result:
    """
    Container class for backtest results.
    """

    returns: pd.Series
    trades: List[Trade]
    stocks: Dict[str, StockHolding]
    invest_ratio: pd.Series
    portfolio_value: pd.DataFrame = None


class Strategy:
    """
    Abstract base class for implementing trading strategies.
    """

    def __init__(
        self,
        buy_signal: pd.DataFrame,
        sell_signal: pd.DataFrame = None,
        data: pd.DataFrame = None,
        cash: float = 1e9,
        commission: float = 0.001425,
        tax: float = 0.003,
        T: float = 1.0,
        freq: str = "D",
        normalize: bool = True,
        # start_date: Optional[datetime] = None,
        # end_date: Optional[datetime] = None,
    ):
        """
        Abstract method for initializing resources for the strategy.
        """
        self.buy_signal = buy_signal
        self.sell_signal = sell_signal
        self.data = data

        self.date = None
        self.cash = cash
        self.commission = commission
        self.tax = tax
        self.normalize = normalize

        self.freq = freq  # "D" or "W"
        self.T = T
        self.last_weekday = 100

        self.returns: List[float] = []
        self.invest_ratio: List[float] = []
        self.portfolio_value: List[dict[str, float]] = []
        self.portfolio = Portfolio(cash=cash, stocks={})
        self.trades: List[Trade] = []

    def next(self, i: int, record: Dict[Hashable, Any]):
        """
        Abstract method defining the core functionality of the strategy for each time step.
        """
        self._update_stock_prices(record)
        self._rebalance_portfolio(i, record)
        self._update_records()

    def _update_stock_prices(self, record: Dict[Hashable, Any]):
        for symbol, stock in self.portfolio.stocks.items():
            if (symbol, "Close") in record:
                stock.update(last_date=self.date, last_price=record[(symbol, "Close")])

    def _rebalance_portfolio(self, i: int, record: Dict[Hashable, Any]):
        should_rebalance = self.freq == "D" or (
            self.freq == "W" and self.date.weekday() < self.last_weekday
        )
        self.last_weekday = self.date.weekday()
        if not should_rebalance:
            return

        row = self.buy_signal.iloc[i]
        if self.normalize:
            row = (row - row.mean(skipna=True)) / row.std(skipna=True)
        row = np.exp(row / self.T)
        row = row / row.sum()
        row_cash = row.get("cash", 0.0)
        row[row <= row_cash] = 0.0
        row["cash"] = 1 - row.sum()
        row.sort_values(ascending=False, inplace=True)

        # print(self.date)
        # print(row)

        total_value = self.portfolio.total_market_value + self.portfolio.cash

        buy_stocks = []
        # 先處理賣出
        for symbol, weight in row.items():
            if symbol == "cash":
                continue
            if np.isnan(weight) or weight < 0.02:
                weight = 0.0
            expected_value = total_value * weight * (1 - self.commission)
            real_value = 0.0
            if symbol in self.portfolio.stocks:
                real_value = self.portfolio.stocks[symbol].market_value

            if weight > 0 and abs(real_value - expected_value) / total_value < 0.01:
                # 差異太小不處理
                continue

            if real_value > expected_value:
                self.sell_stock(
                    symbol,
                    price=record[(symbol, "Close")],
                    quantity=(real_value - expected_value)
                    / self.portfolio.stocks[symbol].last_price,
                )
            else:
                buy_stocks.append((symbol, expected_value - real_value))

        # 再處理買入
        for symbol, buy_cost in buy_stocks:
            real_value = 0.0
            if symbol in self.portfolio.stocks:
                real_value = self.portfolio.stocks[symbol].market_value
            self.buy_stock(
                symbol,
                price=record[(symbol, "Close")],
                quantity=buy_cost / record[(symbol, "Close")] / (1 + self.commission),
            )

    def _update_records(self):
        all_value = self.portfolio.total_realized_value + self.portfolio.cash
        self.invest_ratio.append(self.portfolio.total_realized_value / all_value)
        self.returns.append(all_value)
        portfolio_value = { id: stock.realized_value for id, stock in self.portfolio.stocks.items() }
        portfolio_value["cash"] = self.portfolio.cash
        self.portfolio_value.append(portfolio_value)

    def sell_stock(self, symbol: str, price: float, quantity: float):
        if isnan(price) or price <= 0 or isnan(quantity) or quantity <= 0:
            return False

        if symbol not in self.portfolio.stocks:
            print(f"No holdings to sell for {symbol} at {self.date}!")
            return False

        stock = self.portfolio.stocks[symbol]
        if stock.quantity < quantity:
            quantity = stock.quantity  # 全部賣出

        total_proceed = price * quantity * (1 - self.commission - self.tax)

        stock.quantity -= quantity
        stock.total_cost -= stock.average_cost * quantity
        stock.update(last_date=self.date, last_price=price)

        if stock.quantity == 0:
            del self.portfolio.stocks[symbol]

        self.portfolio.cash += total_proceed

        trade = Trade(
            symbol=symbol,
            trade_date=self.date,
            price=price,
            position_size=quantity,
            is_buy=False,
            trade_commission=price * quantity * (self.commission + self.tax),
        )
        self.trades.append(trade)

        return True

    def buy_stock(self, symbol: str, price: float, quantity: float):
        if isnan(price) or price <= 0 or isnan(quantity) or quantity <= 0:
            return False

        total_cost = price * quantity * (1 + self.commission)

        if self.portfolio.cash < total_cost:
            print(f"Not Enough Money to buy {symbol} at {self.date}!")
            if self.portfolio.cash <= 0:
                return False
            print("Use all remaining cash to buy.")
            quantity = self.portfolio.cash // (price * (1 + self.commission))
            total_cost = price * quantity * (1 + self.commission)

        if symbol not in self.portfolio.stocks:
            self.portfolio.stocks[symbol] = StockHolding(symbol=symbol)

        stock = self.portfolio.stocks[symbol]
        stock.quantity += quantity
        stock.total_cost += total_cost
        stock.update(last_date=self.date, last_price=price)

        self.portfolio.cash -= total_cost

        trade = Trade(
            symbol=symbol,
            trade_date=self.date,
            price=price,
            position_size=quantity,
            is_buy=True,
            trade_commission=price * quantity * self.commission,
        )
        self.trades.append(trade)

        return True

    def eval(
        self,
    ) -> Result:
        """
        Abstract method to evaluate the strategy over the backtest period.
        """
        for i, record in tqdm(enumerate(self.data.to_dict("records"))):
            self.date = self.data.index[i]
            self.next(i, record)

        returns_series = pd.Series(self.returns, index=self.data.index)
        invest_ratio_series = pd.Series(self.invest_ratio, index=self.data.index)
        portfolio_value_df = pd.DataFrame(self.portfolio_value, index=self.data.index)
        result = Result(
            returns=returns_series,
            trades=self.trades,
            stocks=self.portfolio.stocks,
            invest_ratio=invest_ratio_series,
            portfolio_value=portfolio_value_df,
        )
        return result


class DailyBacktest:
    """
    Class for running a backtest on a given strategy using historical market data.

    Attributes:
    - strategy: Type[Strategy] - Type of strategy to be backtested.
    - data: pd.DataFrame - Historical market data.
    - cash: float - Initial cash available for trading.
    - commission: float - Commission rate for trades.

    Methods:
    - run: Runs the backtest and returns the results.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        cash: float,
        commission: float = 0.001425,
        tax: float = 0.003,
        T: float = 1.0,
        freq: str = "D",
        normalize: bool = True,
    ):
        self.data = data
        self.cash = cash
        self.commission = commission
        self.tax = tax
        self.T = T
        self.freq = freq
        self.normalize = normalize

    def run(self, buy_signal: pd.DataFrame, sell_signal: pd.DataFrame = None) -> Result:
        """
        Runs the backtest and returns the results.
        """
        strategy = Strategy(
            buy_signal=buy_signal,
            sell_signal=sell_signal,
            data=self.data,
            cash=self.cash,
            commission=self.commission,
            tax=self.tax,
            T=self.T,
            freq=self.freq,
            normalize=self.normalize,
        )
        result = strategy.eval()
        return result
