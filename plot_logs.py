import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
CSV = os.path.join(RESULT_DIR, "summary.csv")
SIZES = [100, 500, 1000, 5000, 10000]
# matches test.py test_episodes per size (block totals -> per-sample)
SAMPLES_PER_SIZE = {100: 200, 500: 200, 1000: 200, 5000: 20, 10000: 20}

# the 9 ablation configs: stage_1 x stage_2
CONFIGS = [
    "Neural_None",
    "Neural_Neural-RC",
    "Neural_BeamSearch-RC",
    "BeamSearch_None",
    "BeamSearch_Neural-RC",
    "BeamSearch_BeamSearch-RC",
    "knn-BeamSearch_None",
    "knn-BeamSearch_Neural-RC",
    "knn-BeamSearch_BeamSearch-RC",
]


def load():
    # (config, size) -> list of gaps / times across distributions
    gap = defaultdict(list)
    tim = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            key = (r["config"], int(r["size"]))
            gap[key].append(float(r["gap_pct"]))
            if r["time_min"]:
                tim[key].append(float(r["time_min"]))
    return gap, tim


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def plot_axes(ax1, ax2, gap, tim, color, style):
    for c in CONFIGS:
        s2 = c.split("_", 1)[1]
        ls = style.get(s2, "-")
        g = [mean(gap[(c, s)]) for s in SIZES]
        # block total minutes -> seconds per sample
        t = [mean(tim[(c, s)]) * 60 / SAMPLES_PER_SIZE[s] for s in SIZES]
        ax1.plot(SIZES, g, ls, marker="o", color=color[c], label=c)
        ax2.plot(SIZES, t, ls, marker="o", color=color[c], label=c)

    for ax in (ax1, ax2):
        ax.set_xscale("log")
        ax.set_xticks(SIZES)
        ax.set_xticklabels(SIZES)
        ax.set_xlabel("TSP size (nodes)")
        ax.grid(True, which="both", alpha=0.3)

    ax1.set_ylabel("Optimality gap (%)")
    ax1.set_title("Gap vs. problem size")
    ax2.set_yscale("log")
    ax2.set_ylabel("Solving time (s, per sample)")
    ax2.set_title("Solving time vs. problem size")

    ax1.legend(fontsize=8, title="Decoding_Refinement")
    ax2.legend(fontsize=8, title="Decoding_Refinement")


def main(separate=False):
    gap, tim = load()

    cmap = plt.get_cmap("tab10")
    color = {c: cmap(i % 10) for i, c in enumerate(CONFIGS)}
    style = {"None": ":", "Neural-RC": "-", "BeamSearch-RC": "--"}

    if separate:
        fig1, ax1 = plt.subplots(figsize=(7, 5.5))
        fig2, ax2 = plt.subplots(figsize=(7, 5.5))
        plot_axes(ax1, ax2, gap, tim, color, style)
        fig1.tight_layout()
        fig2.tight_layout()
        out1 = os.path.join(RESULT_DIR, "comparison_gap.png")
        out2 = os.path.join(RESULT_DIR, "comparison_time.png")
        fig1.savefig(out1)
        fig2.savefig(out2)
        print(f"wrote {out1}")
        print(f"wrote {out2}")
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
        plot_axes(ax1, ax2, gap, tim, color, style)
        fig.tight_layout()
        out = os.path.join(RESULT_DIR, "comparison.png")
        fig.savefig(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--separate", action="store_true", help="save gap and time as separate files")
    args = parser.parse_args()
    main(separate=args.separate)
