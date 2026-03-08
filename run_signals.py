import argparse
import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import finlab
import pandas as pd
import yaml
from strategies.get_result import get_signals

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
                        help="Directory where signal results will be saved")
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


def save_results(output_dir: str, buy_signals: pd.DataFrame, sell_signals: Optional[pd.DataFrame] = None) -> None:
    """Save all signals to output directory."""
    os.makedirs(output_dir, exist_ok=True)

    # Write signals CSV
    buy_signals.to_csv(f"{output_dir}/buy_signals.csv")

    if sell_signals is not None:
        sell_signals.to_csv(f"{output_dir}/sell_signals.csv")


def main() -> None:
    """Main execution flow."""
    setup_finlab()
    args = parse_arguments()
    config = load_config(args.config)
    start_date, end_date = parse_dates(args.start, args.end)
    
    # Update config with CLI arguments
    config["backtest_type"] = args.backtest_type
    
    # Get Signals
    signals = get_signals(config, args.input_dir, start_date=start_date, end_date=end_date)

    if args.backtest_type == "allen" or args.backtest_type == "gino":
        signals["buy_signals"] = signals["buy_signals"].astype(bool)
        signals["sell_signals"] = signals["sell_signals"].astype(bool)

    # Save all results
    save_results(args.output_dir, buy_signals=signals["buy_signals"], sell_signals=signals["sell_signals"])


if __name__ == "__main__":
    main()
