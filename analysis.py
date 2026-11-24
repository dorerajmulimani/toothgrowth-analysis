from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    data_path = base_dir / "data" / "ToothGrowth.csv"
    output_dir = base_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)

    print("=== Dataset Preview ===")
    print(df.head())
    print("\n=== Shape & Missing Values ===")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print(df.isnull().sum())

    print("\n=== Mean Tooth Length by Supplement & Dose ===")
    summary = df.groupby(["supp", "dose"])["len"].agg(["count", "mean", "std"])
    print(summary)

    # Compare tooth length between OJ and VC supplements
    oj_lengths = df.loc[df["supp"] == "OJ", "len"]
    vc_lengths = df.loc[df["supp"] == "VC", "len"]
    t_stat, p_val = stats.ttest_ind(oj_lengths, vc_lengths, equal_var=False)
    print(f"\nWelch's t-test (OJ vs VC): t = {t_stat:.4f}, p = {p_val:.4f}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    for supp, group in df.groupby("supp"):
        means = group.groupby("dose")["len"].mean()
        ax.plot(means.index, means.values, marker="o", label=f"Supplement: {supp}")

    ax.set_title("Mean Guinea Pig Tooth Length by Dose and Supplement")
    ax.set_xlabel("Dose (mg/day)")
    ax.set_ylabel("Tooth Length (len)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    chart_path = output_dir / "tooth_growth_summary.png"
    fig.savefig(chart_path, dpi=150)
    print(f"\nSaved chart to: {chart_path}")


if __name__ == "__main__":
    main()