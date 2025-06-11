import pandas as pd
import glob

# Replace with your directory path
files = glob.glob("./*.csv")

for file in files:
    try:
        df = pd.read_csv(file)
        print(f"{file}: ✅ Loaded successfully with shape {df.shape}")
    except Exception as e:
        print(f"{file}: ❌ Error - {e}")
