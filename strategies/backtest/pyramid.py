from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Type, Hashable, Optional
from math import nan, isnan
import pandas as pd
import statistics

@dataclass
class Position:
    """
    Represents an open financial position.

    Attributes:
    - symbol: Optional[str] - Symbol of the financial instrument.
    - open_date: Optional[datetime] - Date when the position was opened.
    - last_date: Optional[datetime] - Date of the latest update to the position.
    - open_price: float - Price at which the position was opened.
    - last_price: float - Latest market price of the instrument.
    - position_size: float - Size of the position.
    - profit_loss: float - Cumulative profit or loss of the position.
    - change_pct: float - Percentage change in price since opening the position.
    - current_value: float - Current market value of the position.

    Methods:
    - update(last_date: datetime, last_price: float) - Update the position with the latest market data.
    """
    symbol: Optional[str] = None
    open_date: Optional[datetime] = None
    last_date: Optional[datetime] = None
    open_price: float = nan
    last_price: float = nan
    position_size: float = nan
    profit_loss: float = nan
    change_pct: float = nan
    current_value: float = nan
    
    max_holding_period: int = None
    
    extended_target_price: Optional[float] = None
    has_extended_once: bool = False
    
    pyramid_add_level: int = 0  # 第幾次加碼，0 表示初始倉位
    pyramid_subtract_level: int = 0  # 第幾次加碼，0 表示初始倉位
    base_position_id: Optional[int] = -1  # 關聯主倉編號（用於管理加碼）
    position_id: int = None
    
    # NOTE: change to trailing stop loss
    # stop_loss_price: float = nan
    stop_loss_pct:   float = 0
    stop_loss_price: float = nan
    peak_price:      float = nan
    
    opened_days: int = 1

    def update(self, last_date: datetime, last_price: float):
        self.last_date = last_date
        self.last_price = last_price
        
        if isnan(self.peak_price) or last_price > self.peak_price:
            self.peak_price = last_price
        self.stop_loss_price = self.peak_price * self.stop_loss_pct
        
        self.profit_loss = (self.last_price - self.open_price) * self.position_size
        self.change_pct = (self.last_price / self.open_price - 1) * 100
        self.current_value = self.open_price * self.position_size + self.profit_loss

@dataclass
class Trade:
    """
    Represents a completed financial transaction.

    Attributes:
    - symbol: Optional[str] - Symbol of the financial instrument.
    - open_date: Optional[datetime] - Date when the trade was opened.
    - close_date: Optional[datetime] - Date when the trade was closed.
    - open_price: float - Price at which the trade was opened.
    - close_price: float - Price at which the trade was closed.
    - position_size: float - Size of the traded position.
    - profit_loss: float - Cumulative profit or loss of the trade.
    - change_pct: float - Percentage change in price during the trade.
    - trade_commission: float - Commission paid for the trade.
    - cumulative_return: float - Cumulative return after the trade.
    """
    symbol: Optional[str] = None
    open_date: Optional[datetime] = None
    close_date: Optional[datetime] = None
    open_price: float = nan
    close_price: float = nan
    position_size: float = nan
    exit_reason: str = nan
    profit_loss: float = nan
    change_pct: float = nan
    trade_commission: float = nan
    cumulative_return: float = nan

@dataclass
class Result:
    """
    Container class for backtest results.

    Attributes:
    - returns: pd.Series - Time series of cumulative returns.
    - trades: List[Trade] - List of completed trades.
    - open_positions: List[Position] - List of remaining open positions.
    """
    returns: pd.Series
    trades: List[Trade]
    open_positions: List[Position]
    invest_ratio: pd.Series
    portfolio_value: pd.DataFrame

class Strategy():
    """
    Abstract base class for implementing trading strategies.

    Methods:
    - init(self) - Abstract method for initializing resources for the strategy.
    - next(self, i: int, record: Dict[Hashable, Any]) - Abstract method defining the core functionality of the strategy.

    Attributes:
    - data: pd.DataFrame - Historical market data.
    - date: Optional[datetime] - Current date during backtesting.
    - cash: float - Available cash for trading.
    - commission: float - Commission rate for trades.
    - symbols: List[str] - List of symbols in the market data.
    - records: List[Dict[Hashable, Any]] - List of records representing market data.
    - index: List[datetime] - List of dates corresponding to market data.
    - returns: List[float] - List of cumulative returns during backtesting.
    - trades: List[Trade] - List of completed trades during backtesting.
    - open_positions: List[Position] - List of remaining open positions during backtesting.
    - cumulative_return: float - Cumulative return of the strategy.
    - assets_value: float - Market value of open positions.

    Methods:
    - open(self, price: float, size: Optional[float] = None, symbol: Optional[str] = None) -> bool
    - close(self, price: float, symbol: Optional[str] = None, position: Optional[Position] = None) -> bool
    """
    
    id_pool: int = 0

    
    def init(self, buy_signal: pd.DataFrame, sell_signal: pd.DataFrame = None, max_positions: int = 20, 
             stop_loss: float = 0.7, max_holding_period: int = 30):
        """
        Abstract method for initializing resources and parameters for the strategy.

        This method is called once at the beginning of the backtest to perform any necessary setup or configuration
        for the trading strategy. It allows the strategy to initialize variables, set parameters, or load external data
        needed for the strategy's functionality.

        Parameters:
        - *args: Additional positional arguments that can be passed during initialization.
        - **kwargs: Additional keyword arguments that can be passed during initialization.

        Example:
        ```python
        def init(self, buy_period: int, sell_period: int):
            self.buy_signal = {}
            self.sell_signal = {}

            for symbol in self.symbols:
                self.buy_signal[symbol] = UpBreakout(self.data[(symbol,'Close')], buy_period)
                self.sell_signal[symbol] = DownBreakout(self.data[(symbol,'Close')], sell_period)
        ```

        Note:
        It is recommended to define the expected parameters and their default values within the `init` method
        to allow flexibility and customization when initializing the strategy.
        """
        self.buy_signal = {}
        self.sell_signal = {}
        # self.max_positions = max_positions
        self.stop_loss = stop_loss
        # self.max_holding_period = max_holding_period
        
        
        for symbol in self.symbols:
            self.buy_signal[symbol] = buy_signal[symbol].values if symbol in buy_signal.columns else [0] * len(buy_signal)
            if sell_signal is not None:
                self.sell_signal[symbol] = sell_signal[symbol].values if symbol in sell_signal.columns else [0] * len(sell_signal)
            
    def next(self, i: int, record: Dict[Hashable, Any]):
        """
        Abstract method defining the core functionality of the strategy for each time step.

        This method is called iteratively for each time step during the backtest, allowing the strategy to make
        decisions based on the current market data represented by the 'record'. It defines the core logic of the
        trading strategy, such as generating signals, managing positions, and making trading decisions.

        Parameters:
        - i (int): Index of the current time step.
        - record (Dict[Hashable, Any]): Dictionary representing the market data at the current time step.
          The keys can include symbols, and the values can include relevant market data (e.g., OHLC prices).

        Example:
        ```python
        def next(self, i, record):
            for symbol in self.symbols:
                if self.buy_signal[symbol][i-1]:
                    self.open(symbol=symbol, price=record[(symbol,'Open')], size=self.positionSize(record[(symbol,'Open')]))

            for position in self.open_positions[:]:
                if self.sell_signal[position.symbol][i-1]:
                    self.close(position=position, price=record[(position.symbol,'Open')])
        ```
        """
            
        for position in self.open_positions[:]:
            if position.base_position_id != -1:
                continue

            current_price = record[(position.symbol, 'Close')]

            # 更新停損價格（trailing stop loss）
            if current_price <= position.stop_loss_price:
                self.close(position=position, price=current_price, exit_reason="Stop Trailing Loss")
                continue

            # 如果已延長過，判斷價格跌破原本停利點(1.5倍)即出場
            if position.has_extended_once:
                if current_price <= position.open_price * 1.5:
                    self.close(position=position, price=current_price, exit_reason="Retrace Below Original Take Profit")
                    continue
                
                # 如果價格達到延長的停利點，則出場
                elif hasattr(position, 'extended_target_price') and current_price >= position.extended_target_price:
                    self.close(position=position, price=current_price, exit_reason=f"Extended Stop Profit")
                    continue

            # 尚未延長停利且達到 1.5 停利點，判斷是否延長停利
            if not position.has_extended_once and current_price >= position.open_price * 1.5:
                # 找同一symbol其他持倉，且價格已有一定漲幅 (>1.5倍)
                other_positions = [
                    p for p in self.open_positions
                    if not p.has_extended_once and p.symbol == position.symbol and p != position and record[(p.symbol, 'Close')] >= p.open_price * 1.4 and p.open_price > position.open_price
                ]
                if len(other_positions) > 0:
                    # 取這些持倉中停利目標較高的（這裡用1.5為例）
                    position.extended_target_price = max([p.open_price * 1.5 for p in other_positions])
                    position.has_extended_once = True
                    position.max_holding_period *= 2
                    continue
                else:
                    # 無法延長，直接出場
                    self.close(position=position, price=current_price, exit_reason="Stop Profit")
                    continue
                    

            # 超過最大持有期出場
            if position.opened_days >= position.max_holding_period:
                self.close(position=position, price=current_price, exit_reason="Exceed Maximum Holding Period")
                continue
            
            # elif position.pyramid_subtract_level < 1:  # 最多減倉 1 次
            #     target_price = position.open_price * (1 - 0.1 * (position.pyramid_subtract_level + 1))  # 每次減掉 +10%
            #     if current_price >= target_price:
            #         subtract_ratio = 1 / (2 ** (position.pyramid_subtract_level + 1))  # 50%, 25%, 12.5%, ...
            #         subtract_size = position.position_size * subtract_ratio
                    
            #         # 建立新加碼倉位
            #         flag = self.close(symbol=position.symbol, price=current_price, size=subtract_size, exit_reason="Pyramid Early subtract")
                    
            #         if flag:
            #             # 將加碼層級同步至新倉
            #             position.pyramid_subtract_level += 1
            
        for position in self.open_positions[:]:
            position.opened_days += 1
        
        for position in self.open_positions[:]:
            if position.base_position_id != -1:
                continue
            current_price = record[(position.symbol, 'Close')]
            if len(self.open_positions) < self.max_positions * 0.7 and position.pyramid_add_level < 1:  # 最多加倉 1 次
                target_price = position.open_price * (1 + 0.1 * (position.pyramid_add_level + 1))  # 每次加碼 +10%
                if current_price >= target_price:
                    add_ratio = 1 / (2 ** (position.pyramid_add_level + 1))  # 50%, 25%, 12.5%, ...
                    add_size = position.position_size * add_ratio
                    
                    # 建立新加碼倉位
                    flag = self.open(symbol=position.symbol, price=current_price, size=add_size)
                    
                    if flag:
                        # 將加碼層級同步至新倉
                        self.open_positions[-1].pyramid_add_level = position.pyramid_add_level + 1
                        self.open_positions[-1].base_position_id = position.position_id
                        position.pyramid_add_level += 1
        
        
        for symbol in self.symbols:
            # Check for a buy signal for the previous index
            if i >= len(self.buy_signal[symbol]):
                break
            
            if self.buy_signal[symbol][i]:
                if len(self.open_positions) >= self.max_positions:
                    print(f"Exceeded maximum positions of {self.max_positions} at {self.index[i]} for symbol {symbol}!")
                    # print("Currently Opened Positions:")
                    # print(self.open_positions)
                else:
                    self.open(symbol=symbol, price=record[(symbol, 'Close')], size=self.positionSize(record[(symbol, 'Close')])) 

    def __init__(self):
        self.data = pd.DataFrame()
        self.date = None
        self.cash = .0
        self.commission = .0

        self.symbols: List[str] = []

        self.records: List[Dict[Hashable, Any]] = []
        self.index: List[datetime] = []

        self.returns: List[float] = []
        self.invest_ratio: List[float] = []
        self.trades: List[Trade] = []
        self.open_positions: List[Position] = []
        self.invest_ratio: List[float] = []
        self.portfolio_value: List[Dict[str, float]] = []

        self.cumulative_return = self.cash
        self.assets_value = .0

    def open(self, price: float, size: Optional[float] = None, symbol: Optional[str] = None):
        """
        Opens a new financial position based on the specified parameters.

        Parameters:
        - price: float - The price at which to open the position.
        - size: Optional[float] - The size of the position. If not provided, it is calculated based on available cash.
        - symbol: Optional[str] - Symbol of the financial instrument.

        Returns:
        - bool: True if the position was successfully opened, False otherwise.

        This method calculates the cost of opening a new position, checks if the specified size is feasible given
        available cash, and updates the strategy's open positions accordingly. It returns True if the position is
        successfully opened, and False otherwise.
        """
        if isnan(price) or price <= 0 or (size is not None and (isnan(size) or size <= .0)):
            return False

        if size is None:
            size = self.cash / (price * (1 + self.commission))
            open_cost = self.cash
        else:
            open_cost = (size - 1) * price * (1 + self.commission) 

        if isnan(size) or size <= .0:
            return False
        
        if self.cash < open_cost:
            print(f"Not Enough Money to buy at {self.date} for stock {symbol}!")
            return False

        position = Position(symbol=symbol, 
                            open_date=self.date, 
                            open_price=price, 
                            position_size=size, 
                            stop_loss_pct=self.stop_loss, 
                            )
        
        position.peak_price = price
        position.stop_loss_price = price * self.stop_loss
        position.max_holding_period = self.max_holding_period
        position.position_id = self.id_pool
        self.id_pool += 1
        
        position.update(last_date=self.date, last_price=price)

        self.assets_value += position.current_value
        self.cash -= open_cost

        self.open_positions.extend([position])
        
        return True

    def close(self, price: float, symbol: Optional[str] = None, position: Optional[Position] = None, exit_reason: Optional[str] = None, size: Optional[float] = None):
        """
        Closes an existing financial position based on the specified parameters.

        Parameters:
        - price: float - The price at which to close the position.
        - symbol: Optional[str] - Symbol of the financial instrument.
        - position: Optional[Position] - The specific position to close. If not provided, closes all positions for the symbol.

        Returns:
        - bool: True if the position(s) were successfully closed, False otherwise.

        This method calculates the cost of closing a position, updates the strategy's cumulative return, and records the
        trade details. If a specific position is provided, only that position is closed. If no position is specified,
        all open positions for the specified symbol are closed. It returns True if the position(s) is successfully
        closed, and False otherwise.
        """
        if isnan(price) or price <= 0:
            return False

        if position is None:
            for position in self.open_positions[:]:
                if position.symbol == symbol:
                    self.close(position=position, price=price)
                    break
        else:
            self.assets_value -= position.current_value
            position.update(last_date=self.date, last_price=price)

            trade_commission = (position.open_price + position.last_price) * position.position_size * self.commission + (position.last_price * position.position_size * 0.003) if self.commission != 0 else 0
            self.cumulative_return += position.profit_loss - trade_commission

            trade = Trade(position.symbol, position.open_date, position.last_date, position.open_price,
                position.last_price, position.position_size, exit_reason, position.profit_loss, position.change_pct,
                trade_commission, self.cumulative_return)

            self.trades.extend([trade])
            # if not size or size == position.position_size:
            self.open_positions.remove(position)

            close_cost = position.last_price * position.position_size * (self.commission + 0.003) if self.commission != 0 else 0
            self.cash += position.current_value - close_cost
            
            # if size:
            #     position.position_size -= size
            
            related_positions = [p for p in self.open_positions if p.pyramid_add_level > 0 and p.base_position_id == position.position_id]
            for p in related_positions:
                self.close(position=p, price=price, exit_reason="Group Stop Profit")

        return True

    def __eval(self, *args, **kwargs):
        self.cumulative_return = self.cash
        self.assets_value = .0

        self.init(*args, **kwargs)

        for i, record in enumerate(self.records):

            self.date = self.index[i]
            
            self.next(i, record)

            for position in self.open_positions:
                last_price = record[(position.symbol, 'Close')] if (position.symbol, 'Close') in record else record['Close']
                if last_price > 0:
                    position.update(last_date=self.date, last_price=last_price)

            self.assets_value = sum(position.current_value for position in self.open_positions)
            self.returns.append(self.cash + self.assets_value)
            self.invest_ratio.append(self.assets_value / (self.cash + self.assets_value) if (self.cash + self.assets_value) > 0 else 0)
            portfolio_value = {}
            for position in self.open_positions:
                if position.symbol not in portfolio_value:
                    portfolio_value[position.symbol] = 0
                portfolio_value[position.symbol] += position.current_value
            portfolio_value["cash"] = self.cash
            self.portfolio_value.append(portfolio_value)

        return Result(
            returns=pd.Series(index=self.index, data=self.returns, dtype=float),
            trades=self.trades,
            open_positions=self.open_positions,
            invest_ratio=pd.Series(index=self.index, data=self.invest_ratio, dtype=float),
            portfolio_value=pd.DataFrame(self.portfolio_value, index=self.index)
        )

class PyramidBacktest:
    """
    Class for running a backtest on a given strategy using historical market data.

    Attributes:
    - strategy: Type[Strategy] - Type of strategy to be backtested.
    - data: pd.DataFrame - Historical market data.
    - cash: float - Initial cash available for trading.
    - commission: float - Commission rate for trades.

    Methods:
    - run(*args, **kwargs) - Run the backtest and return the results.
    """
    def __init__(self,
                 strategy: Type[Strategy],
                 data: pd.DataFrame,
                 cash: float = 10000,
                 commission: float = .001425
                 ):

        self.strategy = strategy
        self.data = data
        self.cash = cash
        self.commission = commission

        columns = data.columns
        self.symbols = columns.get_level_values(0).unique().tolist() if isinstance(columns, pd.MultiIndex) else []

        self.records = data.to_dict('records')
        self.index = data.index.tolist()

    def run(self, *args, **kwargs):
        strategy = self.strategy()
        strategy.data = self.data
        strategy.cash = self.cash
        strategy.commission = self.commission

        strategy.symbols = self.symbols
        strategy.records = self.records
        strategy.index = self.index

        return strategy._Strategy__eval(*args, **kwargs)
