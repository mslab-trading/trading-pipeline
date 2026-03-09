import os
import sys

sys.path.append(os.path.join(os.getcwd()))

from result_converter.RegressionResult import RegressionResult

for category in ["Top50", "Top50_RAM", "Top100"]:
    for test_year in range(2021, 2027):
        print(f">>> Converting category={category}, test_year={test_year}")
        input_path = f"../trading_competition/results/{category}_iTransformer_{test_year-3}_{test_year-1}_ConcordanceCorrelation_Dataset_Individual_Seq_Norm"
        output_path = f"results/iTransformer_{category}_Dataset_Abs/{test_year}"
        result = RegressionResult.from_model(input_path)
        result.to_pipeline(output_path)
