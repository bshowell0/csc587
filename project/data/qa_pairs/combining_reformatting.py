import pandas as pd
import glob

# Assumes your CSV files are in a folder named 'my_data'
# and end with .csv
path = r'./*.csv'
all_files = glob.glob(path)

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

# Combine all dataframes into one
combined_df = pd.concat(li, axis=0, ignore_index=True)

# Let's assume your columns are named 'question' and 'answer'
print("Original Data:")
print(combined_df.head())

# Create the 'instruction' column with a fixed value for all rows
combined_df['instruction'] = "Provide the answer to the user's question."

# Rename the 'question' column to 'input'
combined_df.rename(columns={'question': 'input'}, inplace=True)

# Rename the 'answer' column to 'output'
combined_df.rename(columns={'answer': 'output'}, inplace=True)

# Keep only the required columns
final_df = combined_df[['instruction', 'input', 'output']]

print("\nTransformed Data:")
print(final_df.head())

# Save the final dataframe to a new CSV file
output_csv_path = "combined_dataset.csv"
final_df.to_csv(output_csv_path, index=False)

print(f"\nData saved to {output_csv_path}")
