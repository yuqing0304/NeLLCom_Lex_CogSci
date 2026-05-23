from plot_utils import extract_color_data, rgb_to_cielab, hls_to_rgb, cielab_to_rgb, load_prototypes
import matplotlib.pyplot as plt
import collections
import os

#=========================================Ece ============================================

# prototypes_table(tuple(reversed(sorted_color_names)), 'dynamics/language_use/prototypes.pkl', './prototypes_spk.pkl', './prototypes30.pkl')

def prototypes_table(rf_working_path, spk_working_path, frequencies, dataHuman, dataSL, dataRF):

    # prototypes_human_path = os.path.join(spk_working_path, dataHuman)
    prototypes_human_path = os.path.join("./", dataHuman)
    prototypes_SL_path = os.path.join(spk_working_path, dataSL)
    prototypes_RF_path = os.path.join(rf_working_path, dataRF)

    prototypes_human = load_prototypes(prototypes_human_path)
    prototypes_SL = load_prototypes(prototypes_SL_path)
    prototypes_RF = load_prototypes(prototypes_RF_path)

    fig, ax = plt.subplots(figsize=(4, 50))
    headers = ["Human", "SL", "RF"]
    for col in range(3):
        ax.text(col + 0.5, len(frequencies) + 0.3, headers[col], ha="center", fontsize=10)
    for i, c in enumerate(frequencies):
        ax.text(-0.5, i + 0.5, c, ha='center', va='center', fontsize=10)
        lab_value_human = prototypes_human[c]
        rgb_value_human = cielab_to_rgb(lab_value_human)
        ax.add_patch(plt.Rectangle((0, i), 1, 1, color=rgb_value_human))
        if c in prototypes_SL:
            lab_value_SL = prototypes_SL[c]
            rgb_value_SL = cielab_to_rgb(lab_value_SL)
            ax.add_patch(plt.Rectangle((1, i), 1, 1, color=rgb_value_SL))
        if c in prototypes_RF:
            lab_value_RF = prototypes_RF[c]
            rgb_value_RF = cielab_to_rgb(lab_value_RF)
            ax.add_patch(plt.Rectangle((2, i), 1, 1, color=rgb_value_RF))

    ax.set_xlim(0, 3)
    ax.set_ylim(0, len(frequencies))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    plt.show()
    # plt.figure(figsize=(20, 50)) 
    plt.savefig("prototype20.png", dpi=300, bbox_inches='tight')  # high-quality output

    return
#=========================================Ece ============================================



id_to_colors = {
    0: 'mustard', 1: 'cyan', 2: 'maroon', 3: 'lavander', 4: 'medium', 5: 'blood', 6: 'turquoise', 7: 'purple',
    8: 'blue', 9: 'grapes', 10: 'caca', 11: 'teal', 12: 'sky', 13: 'grass', 14: 'red', 15: 'seafoam', 16: 'aqua',
    17: 'clay', 18: 'barney', 19: 'green', 20: 'concrete', 21: 'pumpkin', 22: 'drab', 23: 'tan', 24: 'neon',
    25: 'olive', 26: 'lavender', 27: 'fuchsia', 28: 'gray', 29: 'magenta', 30: 'grape', 31: 'dull', 32: 'peach',
    33: 'mint', 34: 'mauve', 35: 'yellow', 36: 'sage', 37: 'brown', 38: 'pink', 39: 'beige', 40: 'gold',
    41: 'orange', 42: 'salmon', 43: 'bright', 44: 'seagreen', 45: 'violet', 46: 'khaki', 47: 'rose', 48: 'lime'
}

fixed_color_order = list(id_to_colors.values())
# Read the file and count occurrences of each color
color_counts = collections.Counter()

# with open("../new_output.csv", "r") as file:
with open("../dynamics/successful_trials_cielab.csv", "r") as file:
    for line in file:
        parts = line.strip().split(",")
        color_name = parts[-1]  # Last column contains the color name
        color_counts[color_name] += 1

# # Ensure all colors in fixed_color_order appear in the count (default to 0 if missing)
# color_frequencies = [color_counts[color] if color in color_counts else 0 for color in fixed_color_order]

color_frequencies = {color: color_counts[color] if color in color_counts else 0 for color in fixed_color_order}

# Sort colors by frequency (descending order)
sorted_colors = sorted(color_frequencies.items(), key=lambda x: x[1], reverse=True)
sorted_color_names, sorted_frequencies = zip(*sorted_colors)

# # Load the prototype dictionaries
# rf_working_path = "../condition3_generated/condition_slrl3/dump_context/msg_rf_seed111/"
# spk_working_path = "../condition3_generated/condition_slrl3/dump_context/msg_spk_seed111/"

# dataHuman = 'prototypes_spk_label.pkl'
# dataSL = 'prototypes_spk_human_epoch29.pkl'
# dataRF = 'prototypes_rf_human_epoch30.pkl'

# prototypes_human_path = os.path.join(spk_working_path, dataHuman)

# prototypes_human = load_prototypes(prototypes_human_path)

# valid_colors = [
#     color for color in reversed(sorted_color_names)
#     if color in prototypes_human
# ]

# prototypes_table(rf_working_path, spk_working_path, valid_colors, dataHuman, dataSL, dataRF)

# Load the prototype dictionaries
rf_working_path = "../condition3_generated/test/dump_exp/msg_rf_seed111/"
spk_working_path = "../condition3_generated/test/dump_exp/msg_rf_seed111/"

dataHuman = 'prototypes.pkl'
dataSL = 'prototypes_rf_human_epoch30.pkl'
dataRF = 'prototypes_rf_human_epoch30.pkl'

prototypes_human_path = os.path.join("./", dataHuman)

prototypes_human = load_prototypes(prototypes_human_path)

valid_colors = [
    color for color in reversed(sorted_color_names)
    if color in prototypes_human
]

prototypes_table(rf_working_path, spk_working_path, valid_colors, dataHuman, dataSL, dataRF)