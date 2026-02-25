import argparse
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import finlab
import pandas as pd
import yaml
from finlab import data
from matplotlib import pyplot as plt

from evaluation.stats import sharpe, sharpe_0050
from strategies.get_result import get_result

# Constants
BASELINES = ['0050', '2330']
INITIAL_CASH = 1e9
DATE_FORMAT = "%Y-%m-%d"
DEFAULT_BACKTEST_TYPE = "daily"


def setup_finlab() -> None:
    """Initialize FinLab authentication."""
    finlab_token = os.getenv("FINLAB_API_KEY")
    finlab.login(finlab_token)


def parse_arguments() -> argparse.Namespace:
    """Parse and return command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="Path to the backtest YAML config file")
    parser.add_argument('--start', '-s', type=str, default=None,
                        help="Start date for backtest in YYYY-MM-DD format")
    parser.add_argument('--end', '-e', type=str, default=None,
                        help="End date for backtest in YYYY-MM-DD format")
    parser.add_argument('--backtest_type', '-bt', type=str, default=DEFAULT_BACKTEST_TYPE,
                        help="Type of backtest to run (allen, gino, daily)")
    parser.add_argument('--input_dir', '-i', type=str, required=True,
                        help="Directory where model results are stored")
    parser.add_argument('--output_dir', '-o', type=str, required=True,
                        help="Directory where backtest results will be saved")
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    """Load and return configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_dates(start_str: Optional[str], end_str: Optional[str]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Parse date strings to datetime objects."""
    start_date = datetime.strptime(start_str, DATE_FORMAT) if start_str else None
    end_date = datetime.strptime(end_str, DATE_FORMAT) if end_str else None
    return start_date, end_date


def calculate_roi(returns_series: pd.Series) -> float:
    """Calculate Return on Investment."""
    return returns_series.iloc[-1] / returns_series.iloc[0] - 1


def get_baseline_data(returns_index) -> pd.DataFrame:
    """Fetch and return baseline close prices for specified symbols."""
    with data.universe(["TSE", "ETF"]):
        close_df = data.get("etl:adj_close")
        close_df = close_df[BASELINES]
    
    # Filter to match returns index
    close_df = close_df[
        (close_df.index >= returns_index[0])
        & (close_df.index <= returns_index[-1])
    ]
    return close_df


def prepare_returns_dataframe(model_returns: pd.Series, baseline_data: pd.DataFrame) -> pd.DataFrame:
    """Prepare normalized returns dataframe for model and baselines."""
    returns_dict = {baseline: baseline_data[baseline] / baseline_data[baseline].iloc[0] for baseline in BASELINES}
    return pd.DataFrame({
        "model": model_returns / INITIAL_CASH,
        **returns_dict
    })


def write_info_file(output_dir: str, args: argparse.Namespace, config: Dict,
                    roi: float, sharpe_ratio: float, model_returns: pd.Series,
                    baseline_data: pd.DataFrame) -> None:
    """Write backtest information to file."""
    with open(f"{output_dir}/info.txt", "w") as f:
        f.write(f"Backtest Type: {config['backtest_type']}\n")
        f.write(f"Start Date: {args.start}\n")
        f.write(f"End Date: {args.end}\n")
        f.write(f"Result Directory: {args.input_dir}\n")
        f.write(f"Output Directory: {output_dir}\n")
        f.write(f"ROI: {roi}\n")
        f.write(f"Sharpe Ratio: {sharpe_ratio}\n")
        
        for baseline in BASELINES:
            baseline_roi = baseline_data[baseline].iloc[-1] / baseline_data[baseline].iloc[0] - 1
            f.write(f"{baseline} ROI: {baseline_roi}\n")
            baseline_sharpe = sharpe_0050(model_returns, baseline_data[baseline])
            f.write(f"Sharpe Ratio ({baseline}-based): {baseline_sharpe}\n")


def save_results(output_dir: str, args: argparse.Namespace, config: Dict,
                 result: Dict, baseline_data: pd.DataFrame) -> None:
    """Save all backtest results to output directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics
    roi = calculate_roi(result["model"].returns)
    sharpe_ratio = sharpe(result["model"].returns)
    
    print(f"ROI: {roi}")
    print(f"Sharpe Ratio: {sharpe_ratio}")
    
    # Write info file
    write_info_file(output_dir, args, config, roi, sharpe_ratio, result["model"].returns, baseline_data)
    
    # Write returns CSV
    returns_df = prepare_returns_dataframe(result["model"].returns, baseline_data)
    returns_df.to_csv(f"{output_dir}/returns.csv")
    
    # Write trades CSV
    trades_df = pd.DataFrame(result["model"].trades)
    trades_df.to_csv(f"{output_dir}/trades.csv")
    
    # Write portfolio value CSV
    result["model"].portfolio_value.to_csv(f"{output_dir}/portfolio_value.csv")


def main() -> None:
    """Main execution flow."""
    setup_finlab()
    args = parse_arguments()
    config = load_config(args.config)
    start_date, end_date = parse_dates(args.start, args.end)
    
    # Update config with CLI arguments
    config["backtest_type"] = args.backtest_type
    
    # Run backtest
    result = get_result(config, args.input_dir, start_date=start_date, end_date=end_date)
    
    # Get baseline data
    baseline_data = get_baseline_data(result["model"].returns.index)
    
    # Save all results
    save_results(args.output_dir, args, config, result, baseline_data)


if __name__ == "__main__":
    main()
