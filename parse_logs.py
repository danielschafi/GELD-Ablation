import os
import re
import glob
import csv

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")
DISTRIBUTIONS = ["uniform", "clustered", "explosion", "implosion"]
SIZES = [100, 500, 1000, 5000, 10000]

# block order is distribution-major (see test.py __main__)
EXPECTED = [(d, s) for d in DISTRIBUTIONS for s in SIZES]

re_episode = re.compile(r"episode\s+\d+/\s*\d+, Elapsed\[(\d+\.\d+)m\]")
re_gap = re.compile(r"Gap:\s*([\d.]+)%")
re_done = re.compile(r"\*\*\* Test Done \*\*\*")
re_ablation = re.compile(r"ablation_params\{'stage_1': '([^']*)', 'stage_2': '?([^'}]*)'?\}")


def parse_log(path):
    """Return list of (gap_percent, time_minutes) per finished block, in order."""
    blocks = []
    cur_elapsed = None
    with open(path) as f:
        for line in f:
            m = re_episode.search(line)
            if m:
                cur_elapsed = float(m.group(1))
            if re_done.search(line):
                # gap is reported a few lines later; capture on next gap match
                pass
            mg = re_gap.search(line)
            if mg:
                blocks.append((float(mg.group(1)), cur_elapsed))
                cur_elapsed = None
    return blocks


def label_from_dirname(name):
    # e.g. 20260521_082854_test_BeamSearch_BeamSearch-RC
    parts = name.split("_test_", 1)
    return parts[1] if len(parts) == 2 else name


def main():
    rows = []
    for d in sorted(glob.glob(os.path.join(RESULT_DIR, "*"))):
        log = os.path.join(d, "log.txt")
        if not os.path.isfile(log):
            continue
        config = label_from_dirname(os.path.basename(d))
        blocks = parse_log(log)
        for i, (gap, t) in enumerate(blocks):
            dist, size = EXPECTED[i] if i < len(EXPECTED) else ("?", "?")
            rows.append({
                "config": config,
                "distribution": dist,
                "size": size,
                "gap_pct": round(gap, 4),
                "time_min": t,
            })

    out_csv = os.path.join(RESULT_DIR, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "distribution", "size", "gap_pct", "time_min"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} rows)")

    # also print a markdown table
    print("\n| Config | Distribution | Size | Gap (%) | Time (min) |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['config']} | {r['distribution']} | {r['size']} | {r['gap_pct']} | {r['time_min']} |")


if __name__ == "__main__":
    main()
