import os
import pickle
import numpy as np
import csv
from plot_utils import *  # Make sure compute_cielab_distance is defined here


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


def differences_prototypes(data, data_human):
    data_dict = dict(data)
    human_dict = dict(data_human)
    common_names = set(data_dict.keys()) & set(human_dict.keys())

    itemwise = [
        (name, compute_cielab_distance(data_dict[name], human_dict[name]))
        for name in common_names
    ]
    avg_difference = np.mean([d for _, d in itemwise])
    return avg_difference, itemwise


def differences_prototypes_kept(data, data_human, last_epoch):
    kept_colors = [color_name for color_name, cielab in last_epoch]
    results = []

    for color_name, cielab in data:
        for c_name, t_lab in data_human:
            if c_name == color_name and c_name in kept_colors:
                difference = compute_cielab_distance(cielab, t_lab)
                results.append((color_name, difference))

    avg_diff = np.mean([d for _, d in results])
    return avg_diff, results


def analyze_differences(prototype_file, prototype_file_human, words_kept=False):
    prototype = extract_prototype(prototype_file)

    if 'rf' in prototype_file:
        last_epoch = extract_prototype(os.path.join(os.path.dirname(prototype_file), 'prototypes_rf_human_epoch30.pkl'))
    else:
        last_epoch = extract_prototype(os.path.join(os.path.dirname(prototype_file), 'prototypes_spk_human_epoch29.pkl'))

    prototype_human = extract_prototype(prototype_file_human)

    if not words_kept:
        diff, itemwise = differences_prototypes(prototype, prototype_human)
    else:
        diff, itemwise = differences_prototypes_kept(prototype, prototype_human, last_epoch)

    print(diff)
    return diff, itemwise


def compute_epoch_drift_across_seeds(epoch, seeds, label_prototype_path, base_dirs, output_file):
    results = []
    itemwise_all = []

    for condition, base_template in base_dirs.items():
        drifts = []
        # drifts_kept = []

        for seed in seeds:
            base_path = base_template.format(seed=seed)
            prototype_file = os.path.join(base_path, f'prototypes_rf_human_epoch{epoch}.pkl')

            if not os.path.exists(prototype_file):
                print(f"Warning: File {prototype_file} not found.")
                continue

            drift, itemwise = analyze_differences(prototype_file, label_prototype_path)
            # drift_kept, itemwise_kept = analyze_differences(prototype_file, label_prototype_path, words_kept=True)

            drifts.append(drift)
            # drifts_kept.append(drift_kept)

            for name, d in itemwise:
                itemwise_all.append({
                    "condition": condition,
                    "seed": seed,
                    "epoch": epoch,
                    "color_name": name,
                    "drift": f"{d:.2f}",
                    # "type": "all"
                })

            # for name, d in itemwise_kept:
            #     itemwise_all.append({
            #         "condition": condition,
            #         "seed": seed,
            #         "epoch": epoch,
            #         "color_name": name,
            #         "drift": f"{d:.2f}",
            #         "type": "kept"
            #     })

        n = len(drifts)
        results.append({
            "condition": condition,
            "epoch": epoch,
            "mean_drift": f"{np.mean(drifts):.2f}",
            "std_drift": f"{np.std(drifts):.2f}",
            "se_drift": f"{np.std(drifts)/np.sqrt(n):.2f}",
            # "mean_drift_kept": f"{np.mean(drifts_kept):.2f}",
            # "std_drift_kept": f"{np.std(drifts_kept):.2f}",
            # "se_drift_kept": f"{np.std(drifts_kept)/np.sqrt(n):.2f}"
        })

    # Write to CSV (summary stats)
    print(f"Writing {len(results)} condition summaries.")
    with open(output_file, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=results[0].keys())
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    # Write itemwise drift data
    print(f"Writing {len(itemwise_all)} itemwise results.")
    itemwise_output_file = output_file.replace(".csv", "_itemwise.csv")
    with open(itemwise_output_file, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=itemwise_all[0].keys())
        writer.writeheader()
        for row in itemwise_all:
            writer.writerow(row)


# === Run the script for a specific epoch ===
epoch = 30
# seeds = [111, 123, 222, 333, 345, 456, 567, 777, 891, 999] # experiment 1
# seeds = [111, 123, 222, 345, 444, 555, 567, 666, 678, 777]  # condition a
###### 444 > 333, 555 > 456, 666 > 891, 678 > 999
# seeds = [111, 123, 333, 345, 444, 555, 567, 666, 777, 912] # condition b
###### 444 > 222, 555 > 456, 666 > 891, 912 > 999
seeds = [111, 123, 222, 345, 444, 555, 567, 666, 777, 912]  # condition c
###### 444 > 333, 555 > 456, 666 > 891, 912 > 999
label_prototype_path = './prototypes.pkl'

base_dirs = {
    # "condition_a": "../condition3_generated/experiment2/condition_a/dump_context/msg_rf_seed{seed}/",
    # "condition_b": "../condition3_generated/experiment2/condition_b/dump_context/msg_rf_seed{seed}/",
    "condition_c": "../condition3_generated/experiment2/condition_c/dump_context/msg_rf_seed{seed}/"
}

output_file = f"drift_epoch{epoch}_summary.csv"
compute_epoch_drift_across_seeds(epoch, seeds, label_prototype_path, base_dirs, output_file)
