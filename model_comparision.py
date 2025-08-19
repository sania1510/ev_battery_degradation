import json
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# Path to results directory
results_dir = "results"   # change to your actual path

# Get all json files in directory
json_files = glob.glob(os.path.join(results_dir, "*.json"))

all_results = []

# Load each JSON file
for file in json_files:
    with open(file, "r") as f:
        data = json.load(f)
        
    # Convert JSON to DataFrame
    df = pd.DataFrame(data).T.reset_index()
    df.rename(columns={"index": "Model"}, inplace=True)
    
    # Add filename (to distinguish if needed)
    df["Source_File"] = os.path.basename(file)
    
    all_results.append(df)

# Combine all into single DataFrame
final_df = pd.concat(all_results, ignore_index=True)

print("\nCombined Model Results:\n")
print(final_df)

# Save combined results
final_df.to_csv("combined_results.csv", index=False)

# Plot comparisons
metrics = ["R2", "RMSE", "MAE"]

for metric in metrics:
    plt.figure(figsize=(8,5))
    for file in final_df["Source_File"].unique():
        subset = final_df[final_df["Source_File"] == file]
        plt.bar(subset["Model"] + " (" + file.replace(".json","") + ")", 
                subset[metric], label=file)
    plt.title(f"{metric} Comparison Across Files")
    plt.ylabel(metric)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
