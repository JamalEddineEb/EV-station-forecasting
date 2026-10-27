"""Calendar-feature ablation table: test accuracy with vs without calendar features.

Run the benchmark twice per site (second time with --ablate-calendar and another --outdir), then:

    python src/compare_ablation.py \
        --site Caltech outputs/caltech/benchmark/benchmark_results.json outputs/caltech/benchmark_nocal/benchmark_results.json \
        --site JPL     outputs/jpl/benchmark/benchmark_results.json     outputs/jpl/benchmark_nocal/benchmark_results.json

Uses each model's CPU FP32 runtime. Negative "change" means removing the calendar features made it better.
"""
import argparse
import json

MODELS = ("dlinear", "gru", "ridge")


def load(path):
    with open(path) as f:
        res = json.load(f)
    return res["models"], res["settings"].get("ablate_calendar", False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--site", nargs=3, action="append", required=True, metavar=("NAME", "WITH_CAL", "WITHOUT_CAL"))
    p.add_argument("--out", default=None, help="optional markdown file to write")
    a = p.parse_args()

    lines = ["| Site | Model | MAE with calendar | MAE without | MAE change | RMSE with | RMSE without | "
             "Calibrated coverage with | Calibrated coverage without |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, with_path, without_path in a.site:
        (with_cal, with_flag), (without_cal, without_flag) = load(with_path), load(without_path)
        if with_flag or not without_flag:
            raise SystemExit("{}: first file must be the normal run and second the --ablate-calendar run".format(name))
        for model in MODELS:
            w = with_cal[model]["runtimes"]["cpu_fp32"]["accuracy"]
            o = without_cal[model]["runtimes"]["cpu_fp32"]["accuracy"]
            lines.append("| {} | {} | {:.5f} | {:.5f} | {:+.2f}% | {:.5f} | {:.5f} | {:.3f} | {:.3f} |".format(
                name, model, w["mae"], o["mae"], 100 * (o["mae"] - w["mae"]) / w["mae"],
                w["rmse"], o["rmse"], w["coverage_80"], o["coverage_80"]))
    table = "\n".join(lines)
    print(table)
    if a.out:
        with open(a.out, "w") as f:
            f.write(table + "\n")


if __name__ == "__main__":
    main()