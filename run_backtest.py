import yaml
from types import SimpleNamespace
from strategies.run_allen import get_allen_result
from strategies.utils.analysis import print_result

def load_config(path):
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return SimpleNamespace(**cfg_dict)

backtest_config = load_config("config/backtest.yaml")

f = open("config/backtest.yaml")
cfg = yaml.safe_load(f)
result_dir = "results/Example_Result"

result = get_allen_result(cfg, result_dir)
result_benchmark = result['benchmark']
result_model = result['model']

print("If trading by our strategy:")
print_result(result_model.returns)
print("If trading by benchmark:")
print_result(result_benchmark.returns)

