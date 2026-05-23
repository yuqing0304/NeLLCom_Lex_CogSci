import pandas as pd

def upsample_colors(
    infile="condition_b.csv",
    outfile="condition_bupsampled200.csv",
    freq_report="color_frequency_comparison.txt",
    threshold=200,
    random_state=42,
    keep_original_order=True,  # if False, will shuffle groups with >= threshold
):
    # 1. load
    df = pd.read_csv(infile, header=None)

    # 2. color col (assume second to last)
    color_col = df.columns[-2]

    # 3. original counts
    orig_counts = df[color_col].value_counts().sort_index()

    # 4. function to apply per-group
    def balance_group(x):
        n = len(x)
        if n < threshold:
            # need to upsample to threshold (with replacement)
            return x.sample(n=threshold, replace=True, random_state=random_state)
        else:
            # keep original rows (optionally shuffle without replacement)
            if keep_original_order:
                return x  # keep as-is
            else:
                return x.sample(frac=1, replace=False, random_state=random_state)

    # 5. apply groupby
    balanced_df = (
        df.groupby(color_col, group_keys=False)
          .apply(balance_group)
          .reset_index(drop=True)
    )

    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # 6. new counts
    new_counts = balanced_df[color_col].value_counts().sort_index()

    # 7. save results
    balanced_df.to_csv(outfile, index=False, header=False)

    # 8. save frequency comparison
    with open(freq_report, "w", encoding="utf-8") as f:
        f.write("=== Original frequency distribution ===\n")
        f.write(orig_counts.to_string())
        f.write("\n\n=== Upsampled frequency distribution ===\n")
        f.write(new_counts.to_string())

    print("Upsampling done.")
    print("Saved upsampled dataset to:", outfile)
    print("Saved frequency comparison to:", freq_report)

    return orig_counts, new_counts



upsample_colors(
    infile="condition_b.csv",
    outfile="condition_bupsampled200.csv",
    freq_report="color_frequency_comparison.txt",
    threshold=200,
)