import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
epoch0_df = pd.read_csv("./dump_context/seed111_epoch0_informativeness.csv")
epoch20_df = pd.read_csv("./dump_context/seed111_epoch20_informativeness.csv")

# Convert data to dictionaries for easy lookup
epoch0_data = dict(zip(epoch0_df["color_name"], epoch0_df["informativeness"]))
epoch20_data = dict(zip(epoch20_df["color_name"], epoch20_df["informativeness"]))

# Ensure all colors from epoch 0 exist in epoch 20, defaulting missing values to 0
for color in epoch0_data.keys():
    epoch20_data.setdefault(color, 0)

# Prepare x and y values
colors = list(epoch0_data.keys())
epoch0_values = [epoch0_data[color] for color in colors]
epoch20_values = [epoch20_data[color] for color in colors]

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(epoch0_values, epoch20_values, color='b', alpha=0.6)

# Annotate each point with its color name
for i, color in enumerate(colors):
    plt.text(epoch0_values[i], epoch20_values[i], color, fontsize=10, ha='right')

# Labels and title
plt.xlabel("Informativeness at Epoch 0")
plt.ylabel("Informativeness at Epoch 20")
plt.title("Informativeness Change from Epoch 0 to Epoch 20")

# Show grid
plt.grid(True, linestyle="--", alpha=0.5)

# Save plot
plt.savefig("informativeness_scatter_plot.png", dpi=300)

# Show plot
plt.show()
