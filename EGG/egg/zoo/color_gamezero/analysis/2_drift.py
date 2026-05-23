from plot_utils import *
import matplotlib.pyplot as plt
import os 


def analyze_differences(prototype_file, prototype_file_human, words_kept = False):

    prototype = extract_prototype(prototype_file)
    if 'rf' in prototype_file:
        last_epoch = extract_prototype(os.path.join(os.path.dirname(prototype_file), 'prototypes_rf_human_epoch30.pkl'))
    else:
        last_epoch = extract_prototype(os.path.join(os.path.dirname(prototype_file), 'prototypes_spk_human_epoch29.pkl'))
    prototype_human = extract_prototype(prototype_file_human)
    if not words_kept:
        diff = differences_prototypes(prototype, prototype_human)
    else:
        diff = differences_prototypes_kept(prototype, prototype_human, last_epoch)
    print(diff) 

    return diff


def differences_prototypes(data, data_human):
    # Convert lists of (color_name, lab) to dictionaries
    data_dict = dict(data)
    human_dict = dict(data_human)

    # Find common color names
    common_names = set(data_dict.keys()) & set(human_dict.keys())

    # Compute distances for common color names
    differences = [
        compute_cielab_distance(data_dict[name], human_dict[name])
        for name in common_names
    ]

    avg_difference = np.mean(differences)
    return avg_difference


def differences_prototypes_kept(data, data_human, last_epoch):
    kept_colors = [color_name for color_name, cielab in last_epoch]
    # kept_colors = [color_name for color_name, hls in last_epoch]
    results = []

    for color_name, cielab in data:
        # rgb = hls_to_rgb(*hls)
        # target_lab = cs.cspace_convert(rgb, start="sRGB255", end="CIELab")  # Convert to CIELAB
        target_lab = cielab
        for c_name, t_lab in data_human:
            if c_name == color_name:
                if c_name in kept_colors:
                    difference = compute_cielab_distance(target_lab, t_lab)
                    results.append(difference)

    avg_diff = np.mean(results)

    return avg_diff

def plotting_differences_rf(diff, diff_kept, condition, reinforcement = True):
    averages = []
    epochs = []
    for k, v in diff.items():
        epochs.append(k)
        averages.append(v)
    averages_kept = [v for k,v in diff_kept.items()]
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, averages, color = 'b', linestyle = 'dashed', marker = '.')
    plt.plot(epochs, averages_kept, color = 'r', linestyle = 'dashed', marker = '.')
    plt.xlabel("Epochs")
    plt.ylabel("Average Drift")
    if reinforcement:
        plt.title('Average Drift in Reinforcement Learning')
    else:
        plt.title('Average Drift in Supervised Learning')
    plt.show()
    plt.savefig(f"drift_{condition}.png")


import pickle

def extract_prototype(pkl_file):
    """
    Extracts prototype data from a pickle file.

    Args:
        pkl_file (str): Path to the .pkl file containing {color_name: (L, A, B)}.

    Returns:
        List of tuples: [(color_name, (L, A, B)), ...]
    """
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)

    return [(color_name, tuple(lab)) for color_name, lab in data.items()]


from collections import defaultdict


def average_differences_across_seeds(seed_paths, label_prototype_path):
    condition_results = {}

    for condition, paths in seed_paths.items():
        diffs_all_seeds = defaultdict(list)
        diffs_kept_all_seeds = defaultdict(list)

        for base_path in paths:
            for i in range(31):
                file_name = os.path.join(base_path, f'prototypes_rf_human_epoch{i}.pkl')
                diff = analyze_differences(file_name, label_prototype_path)
                diff_kept = analyze_differences(file_name, label_prototype_path, words_kept=True)

                diffs_all_seeds[i].append(diff)
                diffs_kept_all_seeds[i].append(diff_kept)

        # Average across seeds
        avg_diffs = {str(epoch): np.mean(values) for epoch, values in diffs_all_seeds.items()}
        avg_diffs_kept = {str(epoch): np.mean(values) for epoch, values in diffs_kept_all_seeds.items()}

        # Store results
        condition_results[condition] = (avg_diffs, avg_diffs_kept)

    return condition_results


label_prototype_path = './prototypes.pkl'


# seeds = list(range(111, 1000, 111))  # [111, 222, ..., 999]
# seeds = [999, 111, 123, 789, 246, 135, 357, 369, 159, 321]
seeds = [111, 123, 222, 333, 345, 456, 567, 777, 891, 999]

conditions = ["dump_exp", "dump", "dump_context"]

base_template = "../condition3_generated/experiment1/{condition}/msg_rf_seed{seed}/"

seed_paths = {
    condition: [base_template.format(condition=condition, seed=seed) for seed in seeds]
    for condition in conditions
}


results = average_differences_across_seeds(seed_paths, label_prototype_path)

for condition, (avg_diffs, avg_diffs_kept) in results.items():
    plotting_differences_rf(avg_diffs, avg_diffs_kept, condition)


########### for one seed 
# folders = {
#     "dump_exp": "../condition3_generated/test/dump_exp/msg_rf_seed222/",
#     "dump": "../condition3_generated/test/dump/msg_rf_seed222/",
#     "dump_context": "../condition3_generated/test/dump_context/msg_rf_seed222/"
# }

# for condition, base_path in folders.items():
#     differences_rf = {}
#     differences_kept_rf = {}
#     for i in range(11):
#         file_name = os.path.join(base_path, f'prototypes_rf_human_epoch{i}.pkl')
#         differences_rf[str(i)] = analyze_differences(file_name, label_prototype_path)
#         differences_kept_rf[str(i)] = analyze_differences(file_name, label_prototype_path, words_kept=True)
    
#     plotting_differences_rf(differences_rf, differences_kept_rf, condition)


















#========================= Ece ======================
# functions to compute the difference (drift) between model-learned color prototypes and human-assigned color prototypes

# analyze_differences("output_human/dump_exp/msg_rf_seed111/output_epoch30.txt", "dynamics/successful_trials_cielab.csv")
# differences_rf = {}
# for i in range(31):
#     file_name = 'output_human/dump_exp/msg_rf_seed111/output_epoch' + str(i) + '.txt'
#     differences_rf[str(i)] = analyze_differences(file_name, "dynamics/successful_trials_cielab.csv")

# differences_sl = {}
# for i in range(30):
#     file_name = 'output_human/dump_exp/msg_spk_seed111/sender_epoch' + str(i) + '.txt'
#     differences_sl[str(i)] = analyze_differences(file_name, "dynamics/successful_trials_cielab.csv")

# differences_kept_rf = {}
# for i in range(31):
#     file_name = 'output_human/dump_exp/msg_rf_seed111/output_epoch' + str(i) + '.txt'
#     differences_kept_rf[str(i)] = analyze_differences(file_name, "dynamics/successful_trials_cielab.csv", words_kept = True)

# differences_kept_sl = {}
# for i in range(30):
#     file_name = 'output_human/dump_exp/msg_spk_seed111/sender_epoch' + str(i) + '.txt'
#     differences_kept_sl[str(i)] = analyze_differences(file_name, "dynamics/successful_trials_cielab.csv", words_kept = True)



# def differences_prototypes(data, data_human):
#     results = []

#     for color_name, cielab in data:
#         # rgb = hls_to_rgb(*hls)
#         # target_lab = cs.cspace_convert(rgb, start="sRGB255", end="CIELab")  # target_lab in agent production
#         target_lab = cielab
#         for c_name, t_lab in data_human:
#             if c_name == color_name:
#                 difference = compute_cielab_distance(target_lab, t_lab)
#                 results.append(difference)

#     avg_diff = np.mean(results)

#     return avg_diff



# differences_rf = {}
# for i in range(31):
#     file_name = '../condition3_generated/test/dump_context/msg_rf_seed111/prototypes_rf_human_epoch' + str(i) + '.pkl'
#     differences_rf[str(i)] = analyze_differences(file_name, label_prototype_path)

# differences_kept_rf = {}
# for i in range(31):
#     file_name = '../condition3_generated/test/dump_exp/msg_rf_seed111/prototypes_rf_human_epoch' + str(i) + '.pkl'
#     differences_kept_rf[str(i)] = analyze_differences(file_name, label_prototype_path)
# condition = "rf"
# plotting_differences_rf(differences_rf, differences_kept_rf, condition)



# label_prototype_path = '../condition1_human/dump_context/msg_spk_seed111/prototypes_spk_label.pkl'
# differences_rf = {}
# for i in range(31):
#     file_name = '../condition1_human/dump_context/msg_rf_seed111/prototypes_rf_human_epoch' + str(i) + '.pkl'
#     differences_rf[str(i)] = analyze_differences(file_name, label_prototype_path)

# differences_kept_rf = {}
# for i in range(31):
#     file_name = '../condition1_human/dump_context/msg_rf_seed111/prototypes_rf_human_epoch' + str(i) + '.pkl'
#     differences_kept_rf[str(i)] = analyze_differences(file_name, label_prototype_path)
# condition = "rf"
# plotting_differences_rf(differences_rf, differences_kept_rf, condition)


# differences_sl = {}
# for i in range(30):
#     file_name = '../condition1_human/dump_context/msg_spk_seed111/prototypes_spk_human_epoch' + str(i) + '.pkl'
#     differences_sl[str(i)] = analyze_differences(file_name, label_prototype_path)

# differences_kept_sl = {}
# for i in range(30):
#     file_name = '../condition1_human/dump_context/msg_spk_seed111/prototypes_spk_human_epoch' + str(i) + '.pkl'
#     differences_kept_sl[str(i)] = analyze_differences(file_name, label_prototype_path, words_kept = True)
# condition = "spk"
# plotting_differences_rf(differences_sl, differences_kept_sl, condition, reinforcement = False)



