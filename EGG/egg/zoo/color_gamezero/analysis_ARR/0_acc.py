def read_and_save_epoch30_csv(log_folder, log_prefix, accuracy_type, output_csv):
    log_files = [f for f in os.listdir(log_folder)
                 if f.startswith(log_prefix) and f.endswith('.txt')]
    print(f"Reading from {log_folder}: {log_files}")

    accs = []

    for log_file in log_files:
        log_path = os.path.join(log_folder, log_file)

        with open(log_path, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line)
                    if (
                        log_entry.get("mode") == "test"
                        and log_entry.get("epoch") == 30
                    ):
                        if accuracy_type == "valid":
                            accs.append(log_entry["acc"])
                        else:
                            accs.append(log_entry[f"acc_{accuracy_type}"])
                except json.JSONDecodeError:
                    continue

    if not accs:
        print("No epoch 30 data found.")
        return

    mean = np.mean(accs)
    std = np.std(accs)

    df = pd.DataFrame([{
        "epoch": 30,
        "mean": mean,
        "std": std
    }])

    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")


# python save_epoch30_csv.py \
#   --log_folder /path/to/logs \
#   --log_prefix log_rf \
#   --accuracy_type valid \
#   --output_csv rf_epoch30_valid.csv
