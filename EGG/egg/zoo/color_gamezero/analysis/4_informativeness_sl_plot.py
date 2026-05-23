import re
import numpy as np
import matplotlib.pyplot as plt

# Path to the file
filename = "condition3_generated_condition_slrl4_agent_sl_informativeness.txt"

# Initialize data containers
data = {
    "overall": [],
    "far": [],
    "split": [],
    "close": []
}

# Read and parse
with open(filename, "r") as f:
    for line in f:
        if "Overall Lexical System Informativeness" in line:
            value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
            data["overall"].append(value)
        elif "far" in line:
            value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
            data["far"].append(value)
        elif "split" in line:
            value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
            data["split"].append(value)
        elif "close" in line:
            value = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1])
            data["close"].append(value)

# Compute averages and stds
labels = ["overall", "far", "split", "close"]
means = [np.mean(data[key]) for key in labels]
stds = [np.std(data[key]) for key in labels]

# Plotting
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, means, yerr=stds, capsize=5, color=["#9ecae1", "#a1d99b", "#fdae6b", "#d95f0e"])
plt.ylabel("Lexical System Informativeness")
plt.title("Avg. Lexical Informativeness (± Std Dev) - SL")

# Add value labels
for bar, mean in zip(bars, means):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{mean:.2f}", ha='center', va='bottom')

plt.tight_layout()
plt.savefig("condition_slrl4_sl_informativeness.png", dpi=300)
plt.show()
