from ray.tune.analysis import ExperimentAnalysis

# Load the experiment analysis from the directory where Ray Tune saved the results
analysis = ExperimentAnalysis("/research/m323170/Projects/mammography/ray_result/dino_convnext")

# Get the results as a Pandas DataFrame
results_df = analysis.trial_dataframes

print(results_df)

