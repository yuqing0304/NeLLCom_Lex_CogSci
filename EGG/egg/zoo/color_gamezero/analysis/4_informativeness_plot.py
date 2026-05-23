import re
import numpy as np
import matplotlib.pyplot as plt

# File mapping
file_map = {
    "slrl3": "condition3_generated_condition_slrl3_agent_rf_informativeness.txt",
    "slrl4": "condition3_generated_condition_slrl4_agent_rf_informativeness.txt",
    "slrl5": "condition3_generated_condition_slrl5_agent_rf_informativeness.txt",
    "slrl6": "condition3_generated_condition_slrl6_agent_rf_informativeness.txt",
    "slrl7": "condition3_generated_condition_slrl7_agent_rf_informativeness.txt",
}

# Initialize structure to store results
data_by_condition = {
    cond: {
        "overall": [],
        "far": [],
        "close": []
    }
    for cond in file_map
}

# Parse each file
for cond, filename in file_map.items():
    with open(filename, "r") as f:
        for line in f:
            if "Overall Lexical System Informativeness" in line:
                data_by_condition[cond]["overall"].append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
            elif "far" in line:
                data_by_condition[cond]["far"].append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))
            elif "close" in line:
                data_by_condition[cond]["close"].append(float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[-1]))

# Categories to plot
categories = ["overall", "far", "close"]
conditions = ["slrl3", "slrl4", "slrl5", "slrl6", "slrl7"]

# Prepare bar plot data
means = []
stds = []
for cat in categories:
    for cond in conditions:
        scores = data_by_condition[cond][cat]
        means.append(np.mean(scores))
        stds.append(np.std(scores))

# Plotting
x = np.arange(len(categories))
width = 0.15  # smaller width since we now have 5 conditions

plt.figure(figsize=(12, 6))
colors = ["#9ecae1", "#fdd0a2", "#a1d99b", "#fdae6b", "#c7e9c0"]  # 5 colors

# Sort by far:close ratio
conditions = ["slrl4", "slrl6", "slrl3", "slrl7", "slrl5"]

label_map = {
    "slrl4": "far:close 0/100",
    "slrl6": "far:close 33/67",
    "slrl3": "far:close 50/50",
    "slrl7": "far:close 67/33",
    "slrl5": "far:close 100/0",
}

for i, cond in enumerate(conditions):
    x_pos = x + (i - 2) * width  # Center bars around each category
    bar = plt.bar(x_pos, means[i::len(conditions)], width=width, yerr=stds[i::len(conditions)],
                  capsize=4, label=label_map[cond], color=colors[i])

    for j, bar_obj in enumerate(bar):
        yval = bar_obj.get_height()
        plt.text(bar_obj.get_x() + bar_obj.get_width()/2, yval + 0.005, f"{yval:.2f}",
                 ha='center', va='bottom', fontsize=8)

plt.xticks(x, categories)
plt.ylabel("Lexical System Informativeness")
plt.title("Lexical Informativeness by Condition and Context")
plt.legend()
plt.tight_layout()
plt.savefig("lexical_informativeness_by_condition_updated.png", dpi=300)
plt.show()
