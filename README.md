# Ohio foliage model

An IM2C 2026 Problem 1 research prototype. The model accumulates temperature stress weighted by day length, then simulates uncertainty in the timing of the threshold crossing.

## Run the notebook

Open [the notebook in Google Colab](https://colab.research.google.com/github/Naanman-git/ohio-foliage-model/blob/main/Ohio_Foliage_Model.ipynb) with private GitHub access, or download the notebook and use **File → Upload notebook**. Select **Runtime → Run all**. The notebook is self-contained for the synthetic demonstration. For the real weather coverage table, upload `ohio_noaa_fall_daily.csv` to the Colab runtime via **Files → Upload** (normally into `/content`) before running.

To run locally from the repository directory, install `numpy pandas scipy matplotlib` and run `python ohio_foliage_model.py`. Figures are generated in `figs/`.

## Data and provenance

- `ohio_noaa_fall_daily.csv` contains 16,637 September–November rows from six NOAA GHCN-Daily stations. The request spans 1996-09-01 through 2026-11-14 and the latest available row is 2026-11-10. TMIN is in °C, PRCP in mm and elevation in meters. Values carrying a NOAA quality flag were set to missing. Source is the [NCEI Daily Summaries service](https://www.ncei.noaa.gov/access/search/data-search/daily-summaries/). The [download query](https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00094830,USW00093812,USW00004804,USW00093824,USW00014891,USW00014895&startDate=1996-09-01&endDate=2026-11-14&dataTypes=TMIN,PRCP&includeStationLocation=true&includeStationName=true&includeAttributes=true&units=metric&format=csv) is reproducible.
- `foliage_evidence.csv` documents five dated ODNR status observations. The October 2026 reports describe broad regional conditions and the September 2025 rows describe pre-peak conditions. Report dates and categories are recorded as published and do not specify the day a tier reached 50% canopy color.
- The reference pack documents the original design. The notebook and Python file are the runnable sources.

## Model review

- The notebook defaults to synthetic mode because calibration requires comparable historical peak observations across all three tiers. Add weather-matched, sourced peak dates in CONFIG and switch off synthetic mode to fit real foliage. Expect missing test-season days to be filled from historical weather. Synthetic results demonstrate execution only.
- The two stations in each tier are grouped by station elevation. The groups do not implement the reference pack's county elevation terciles and mix different geographic regions. Equal tier weights are illustrative. For a defensible statewide aggregate, derive county tiers, match observations and station coverage, and use measured area weights.
- The model predicts a stress threshold crossing. Its stated 50% canopy definition needs observations measured on that definition, or an explicit mapping from weekly status categories to crossing intervals. The current calibration expects point dates. A separate held-out years assessment against a simple historical median would show whether weather adds predictive skill.
- The temperature and elevation effects may be hard to identify from only three tier averages. Pre-season mode uses historical temperatures and omits precipitation. The default model also ignores rain in the hindcast.
- The decade analysis extrapolates local fall temperature trends and explores scenarios. The error calculation combines an ensemble spread, fitted residuals, and a drift term. Since the ensemble already includes threshold noise derived from residuals, it may count some error twice. It cannot establish ten-year accuracy.

## Observed-data backtest and paper figures

The observed-data entry point is `python3 ohio_foliage_backtest.py`. Run `python3 ohio_foliage_diagnostics.py` for the paired elevation comparison, historical score checks, and ten-year temperature scenarios. Together they use NumPy, pandas, Matplotlib, and SciPy. The data sources, limitations, results, and figure captions are in [the paper results](paper_results/RESULTS.md).

Run `python3 ohio_foliage_extensions.py` for the matched summer temperature comparison and Perry County daily canopy-color curve. The checked-in input extracts are in `data/`. To regenerate them from the three public source downloads linked in the paper results, run `python3 prepare_foliage_extensions.py COLOR_JSON LEAF_DROP_JSON SUMMER_CSV`.
