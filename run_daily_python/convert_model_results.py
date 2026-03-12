import os
import sys

sys.path.append(os.path.join(os.getcwd()))

from result_converter.RegressionResult import RegressionResult


# From ../trading_competition/results/
for category in ["Top50", "Top50_RAM", "Top100"]:
    for test_year in range(2021, 2027):
        print(f">>> Converting (category={category}, test_year={test_year}) from trading_competition")
        input_path = f"../trading_competition/results/{category}_iTransformer_{test_year-3}_{test_year-1}_ConcordanceCorrelation_Dataset_Individual_Seq_Norm"
        output_path = f"results/iTransformer_{category}_Dataset_Abs/{test_year}"
        result = RegressionResult.from_model(input_path)
        result.to_pipeline(output_path)

# From /data2/Trading-Research/competition_prediction/results_old
for category in ["Top50", "Top50_RAM", "Top100"]:
    for test_year in range(2021, 2027):
        if test_year == 2026:
            root_path = "/data2/Trading-Research/competition_prediction/results"
        else:
            root_path = "/data2/Trading-Research/competition_prediction/results_old"

        input_path = f"{root_path}/{category}_iTransformer_{test_year-3}_{test_year-1}_ConcordanceCorrelation_Dataset_Individual_Seq_Norm"
        output_path = f"results/iTransformer_{category}_Dataset_Abs/{test_year}"
        if not os.path.exists(input_path):
            continue
        print(f">>> Converting (category={category}, test_year={test_year}) from Trading-Research")
        result = RegressionResult.from_model(input_path)
        result.to_pipeline(output_path)
