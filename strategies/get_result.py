from strategies.run_allen import get_allen_result, get_allen_signals
from strategies.run_gino import get_gino_result, get_gino_signals
from strategies.run_daily import get_daily_result, get_daily_signals


def get_result(cfg: dict, result_dir: str, *, start_date=None, end_date=None):
    backtest_type = cfg["backtest_type"]
    if backtest_type == "allen":
        return get_allen_result(
            cfg, result_dir, start_date=start_date, end_date=end_date
        )
    elif backtest_type == "gino":
        return get_gino_result(
            cfg, result_dir, start_date=start_date, end_date=end_date
        )
    elif backtest_type == "daily":
        return get_daily_result(
            cfg, result_dir, start_date=start_date, end_date=end_date
        )
    else:
        raise ValueError(f"Unknown backtest type: {backtest_type}")


def get_signals(cfg: dict, result_dir: str, *, start_date=None, end_date=None):
    backtest_type = cfg["backtest_type"]
    if backtest_type == "allen":
        return get_allen_signals(
            cfg, result_dir, start_date=start_date, end_date=end_date
        )
    elif backtest_type == "gino":
        return get_gino_signals(
            cfg, result_dir, start_date=start_date, end_date=end_date
        )
    elif backtest_type == "daily":
        return get_daily_signals(
            cfg, result_dir, start_date=start_date, end_date=end_date
        )
    else:
        raise ValueError(f"Unknown backtest type: {backtest_type}")
