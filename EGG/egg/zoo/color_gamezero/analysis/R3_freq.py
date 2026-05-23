import pandas as pd

# Load the CSV file
df = pd.read_csv("color_labels.csv")

# Rename the column to a consistent name
df.columns = ['color_name']

# Count frequency of each color name
color_freq = df['color_name'].value_counts().reset_index()
color_freq.columns = ['color_name', 'frequency']

# Save the result to a new CSV file
color_freq.to_csv("color_frequencies.csv", index=False)
