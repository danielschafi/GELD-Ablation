"""Pivot ablation summary.csv → mean-over-distributions tables (CSV + LaTeX)."""

import os

import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
CSV_IN = os.path.join(RESULT_DIR, "summary.csv")
SIZES = [100, 500, 1000, 5000, 10000]

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


def load_ablation(path: str = CSV_IN) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["config"].isin(CONFIGS)].copy()
    df["config"] = pd.Categorical(df["config"], categories=CONFIGS, ordered=True)
    return df


def mean_by_config_size(df: pd.DataFrame) -> pd.DataFrame:
    """Mean gap (%) and time (min) over the 4 distributions, per (config, size)."""
    agg = (
        df.groupby(["config", "size"], observed=True)[["gap_pct", "time_min"]]
        .mean()
        .reset_index()
    )
    stage = agg["config"].astype(str).apply(_split_stages)
    agg["stage_1"] = [s[0] for s in stage]
    agg["stage_2"] = [s[1] for s in stage]
    return agg.sort_values(["config", "size"]).reset_index(drop=True)


def _split_stages(config: str) -> tuple[str, str]:
    # knn-BeamSearch_* has a hyphen in stage_1; split on last underscore before stage_2 tokens
    for s2 in ("BeamSearch-RC", "Neural-RC", "None"):
        suffix = "_" + s2
        if config.endswith(suffix):
            return config[: -len(suffix)], s2
    raise ValueError(f"unrecognized config: {config}")


def pivot_metric(agg: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = agg.pivot(index="config", columns="size", values=metric)
    wide = wide.reindex(index=CONFIGS, columns=SIZES)
    wide.index.name = "config"
    wide.columns = [str(s) for s in wide.columns]
    stage = [_split_stages(c) for c in wide.index.astype(str)]
    wide.insert(0, "stage_1", [s[0] for s in stage])
    wide.insert(1, "stage_2", [s[1] for s in stage])
    return wide.reset_index(drop=True)


def to_latex_tabular(
    wide: pd.DataFrame,
    metric_label: str,
    float_fmt: str = "{:.2f}",
) -> str:
    """Plain tabular (no booktabs) suitable for \\begin{tabular}{...}."""
    size_cols = [str(s) for s in SIZES]
    col_spec = "ll" + "r" * len(size_cols)
    header = (
        "Stage 1 & Stage 2 & "
        + " & ".join(f"$N={s}$" for s in size_cols)
        + r" \\"
    )

    lines = [
        r"\begin{tabular}{" + col_spec + "}",
        r"\hline",
        header,
        r"\hline",
    ]
    for _, row in wide.iterrows():
        vals = [float_fmt.format(row[c]) for c in size_cols]
        lines.append(
            f"{row['stage_1']} & {row['stage_2']} & " + " & ".join(vals) + r" \\"
        )
    lines += [
        r"\hline",
        r"\end{tabular}",
    ]
    # caption hint as TeX comment
    lines.insert(0, f"% {metric_label} (mean over uniform/clustered/explosion/implosion)")
    return "\n".join(lines) + "\n"


def main():
    df = load_ablation()
    agg = mean_by_config_size(df)

    long_csv = os.path.join(RESULT_DIR, "summary_mean.csv")
    agg.to_csv(long_csv, index=False, float_format="%.4f")

    gap_wide = pivot_metric(agg, "gap_pct")
    time_wide = pivot_metric(agg, "time_min")

    gap_csv = os.path.join(RESULT_DIR, "summary_gap_table.csv")
    time_csv = os.path.join(RESULT_DIR, "summary_time_table.csv")
    gap_wide.to_csv(gap_csv, index=False, float_format="%.4f")
    time_wide.to_csv(time_csv, index=False, float_format="%.4f")

    gap_tex = os.path.join(RESULT_DIR, "summary_gap_table.tex")
    time_tex = os.path.join(RESULT_DIR, "summary_time_table.tex")
    with open(gap_tex, "w") as f:
        f.write(to_latex_tabular(gap_wide, "Optimality gap (\\%)", "{:.2f}"))
    with open(time_tex, "w") as f:
        f.write(to_latex_tabular(time_wide, "Solving time (min)", "{:.2f}"))

    print(f"wrote {long_csv}")
    print(f"wrote {gap_csv}")
    print(f"wrote {time_csv}")
    print(f"wrote {gap_tex}")
    print(f"wrote {time_tex}")
    print("\n=== Gap (%) ===")
    print(gap_wide.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print("\n=== Time (min) ===")
    print(time_wide.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
