"""Audit same-tree spring budbreak and fall color coverage."""
import argparse
import json
from pathlib import Path

import pandas as pd

from ohio_foliage_backtest import OUTPUT, ROOT
from ohio_foliage_species import tree_windows


def prepare(path):
    columns = ["site_id", "individual_id", "observation_date", "phenophase_status"]
    records = []
    with path.open() as source:
        for line in source:
            if line.strip():
                decoded = json.loads(line)
                records.extend(decoded if isinstance(decoded, list) else [decoded])
    observations = pd.DataFrame(records)
    assert set(observations.phenophase_id).issubset({371, 373})
    observations = observations[columns].drop_duplicates()
    observations.to_csv(ROOT / "data/npn_ohio_spring_bud_visits.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path)
    args = parser.parse_args()
    if args.raw:
        prepare(args.raw)
    obs = pd.read_csv(ROOT / "data/npn_ohio_spring_bud_visits.csv",
                      parse_dates=["observation_date"])
    obs["year"] = obs.observation_date.dt.year
    obs = obs.loc[obs.observation_date.dt.month.le(6)]
    rows = []
    for (site, tree, year), group in obs.groupby(["site_id", "individual_id", "year"]):
        first_yes = group.loc[group.phenophase_status.eq(1), "observation_date"].min()
        last_no = group.loc[group.phenophase_status.eq(0) &
                            group.observation_date.lt(first_yes), "observation_date"].max()
        if pd.notna(last_no) and pd.notna(first_yes):
            rows.append(dict(site=site, tree=tree, year=year, spring_lower=last_no,
                             spring_upper=first_yes, spring_gap_days=(first_yes-last_no).days))
    spring = pd.DataFrame(rows)
    autumn = tree_windows()
    paired = autumn.merge(spring, on=["site", "tree", "year"], how="left", validate="one_to_one")
    paired["usable_spring"] = paired.spring_gap_days.le(14)
    paired.to_csv(OUTPUT / "spring_fall_tree_matches.csv", index=False)
    coverage = paired.groupby(["location", "year"]).agg(
        autumn_tree_windows=("tree", "size"), any_spring_window=("spring_gap_days", "count"),
        spring_window_within_14_days=("usable_spring", "sum")).reset_index()
    coverage.to_csv(OUTPUT / "spring_fall_coverage.csv", index=False)
    print(f"{len(obs)} spring visits; {len(spring)} spring intervals; "
          f"{paired.usable_spring.sum()} autumn matches with a spring gap of 14 days or less")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
