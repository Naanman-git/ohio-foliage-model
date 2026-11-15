# Ohio fall color backtest and 2026 holdout

Run `python3 ohio_foliage_backtest.py` from the project folder. The script uses NumPy, pandas, and Matplotlib and writes the figures and numerical tables in this folder. The rebuilt source was edited in VS Code. PNG files are ready for insertion, and SVG files can be scaled for print.

## Study target and data

Peak is defined for this analysis as the date when a monitored tree first reaches 50% non-green color across its potential canopy. The observation interval runs from the last below-50% visit to the first visit in the 50–74%, 75–94%, or 95%+ category. Intervals longer than 21 days and records without a prior lower reading are excluded. Eligible trees come from the deciduous genera represented in the extracted CSV. We take the median of tree lower bounds and of tree upper bounds for each location and year, then score against the midpoint of that window. This represents the monitored trees, with species and trees changing across years.

The Ohio USA-NPN request was made on November 15, 2026 for phenophase 498. It returned 13,802 records. The table at `../data/npn_ohio_tree_crossings.csv` contains filtered tree crossings and site IDs. Matched observations came from Columbus (859), Perry County (35743), and Oxford (34167 and 57318, adjacent monitoring sites combined). These locations were selected for repeated observations and available station records. Some site-years have one tree, so site coverage and sampling stability are limited. Sources include the [USA-NPN data service](https://data.usanpn.org/) and [phenophase definitions](https://www.usanpn.org/files/reports/usa-npn_plant_and_animal_phenophase_definitions_v2.0.pdf). [Observation query](https://services.usanpn.org/npn_portal/observations/getObservations.ndjson?request_src=OpenAI%20Codex&start_date=2009-09-01&end_date=2026-11-14&state=OH&phenophase_id=498).

NOAA stations are Ohio State University Airport (USW00004804), New Lexington 2 NW (USC00335857), and Hamilton Butler County Regional Airport (USW00053855). The files contain daily minimum temperature in °C, with NOAA quality-flagged readings treated as missing. The script interpolates short internal gaps and excludes incomplete historical weather seasons. For the 2026 hindcast, it uses available readings through November 10 and fills missing dates with historical draws. Daily temperature is adjusted for the difference between station and site elevation. [NOAA Daily Summaries](https://www.ncei.noaa.gov/access/search/data-search/daily-summaries/).

## Historical model comparison

The model combines cool temperatures below a fixed 15 °C base with shortening daylight. Each test uses only earlier years to calibrate the threshold at the location, with at least two prior matching site-years. The Monte Carlo ensemble samples training stress values within the observed intervals and uses the spread in those values across sites after centering by site. Preseason runs sample whole historical weather seasons. Hindcasts use the test season's observed weather. The baseline predicts the median date from the site's prior observed window midpoints.

| Mode | Historical cases | Model MAE | Baseline MAE | Model median in observed window | P10–P90 overlaps observed window |
|---|---:|---:|---:|---:|---:|
| Hindcast | 8 | 9.69 days | 9.58 days | 1/8 | 6/8 |
| Preseason | 8 | 7.69 days | 9.58 days | 2/8 | 7/8 |

The preseason model reduced mean absolute error by 1.89 days on eight cases. The observed-weather hindcast was 0.11 days worse than the historical-date baseline. The two comparisons use the same site-years. Interval overlap means the simulated band shares at least one date with the observed window. It gives no direct estimate of how often an exact crossing falls inside the band. The sample is too small for a strong claim that the model improves forecasts.

## 2026 holdout

| Location | Observed monitored-tree window | Preseason median | Hindcast median | Published county estimate |
|---|---|---|---|---|
| Columbus | Oct 7–14 (2 trees) | Oct 21 | Oct 22 | Oct 18–20, Franklin County |
| Oxford | Oct 15–23 (5 trees) | Oct 19 | Oct 23 | Around Oct 29, Butler County |
| Perry County | No qualifying interval | Oct 20 | Oct 24 | Oct 21–27, Perry County |

The Columbus preseason prediction was 10.25 days later than the observed midpoint. The Oxford preseason prediction matched the window midpoint, and its hindcast landed at the end of the observed window. Both modeled bands overlapped the observed windows at both locations. Perry County has a prediction and no eligible 2026 tree window to score.

The [More Than Just Parks county table](https://morethanjustparks.com/foliage-tracker/ohio) was issued November 7. The published values are county estimates, some adjusted using field reports, and use a travel-oriented peak. They provide context for the timing differences across scales. Ohio's [September 24 ODNR preseason forecast](https://ohiodnr.gov/discover-and-learn/safety-conservation/about-ODNR/news/fall-color-forecast-2026) expected central Ohio peak in mid-to-late October and southern Ohio peak in late October to early November. The [October 22 ODNR observations](https://ohiodnr.gov/discover-and-learn/safety-conservation/about-ODNR/news/fall-color-update-4-2026) described generally near-peak to peak statewide, followed by [widespread peak on October 30](https://ohiodnr.gov/discover-and-learn/safety-conservation/about-ODNR/news/fall-color-update-5-final).

## Monte Carlo checks and figures

The script checks that the 5,000 draws per case land in the modeled season or the no-crossing category, that probabilities sum to one, that identical seeds reproduce 2026 results, and that adding 1 °C cannot advance the crossing with the same inputs and sampled threshold. The 2026 output assigned zero probability to no crossing by November 30 in all six runs. Wide bands and sparse field coverage limit the strength of the accuracy conclusions.

1. **Historical backtest.** Predicted medians versus observed midpoints. Horizontal bars show the observed visit windows and the diagonal indicates exact midpoint agreement.
2. **2026 peak probabilities.** Daily probabilities across 5,000 draws per curve. Shading shows the two observed monitoring windows.
3. **Model versus baseline.** Historical mean absolute error across the same eight site-years in the two weather modes.

## Accuracy checks for the final paper

Run `python3 ohio_foliage_diagnostics.py` to regenerate these checks. The script also uses SciPy.

**Elevation.** We refit each historical case twice using the same sites, years, and simulation seed. One run adjusted station temperatures by −0.65 °C per 100 m of elevation difference and the other used the station temperatures directly. The largest change in any median prediction was one day. Hindcast MAE was 9.69 days both ways. Preseason MAE was 7.69 days with the adjustment and 7.56 days without it. With three sites, this comparison gives little evidence that this station-to-site correction improved accuracy. It does not estimate elevation effects across Ohio's full range of terrain.

**Sensitivity to the historical sample.** After dropping each evaluation year in turn from the score, the preseason model had 1.10–2.71 fewer days of MAE than the historical-median baseline. The hindcast ranged from 1.86 days better to 1.06 days worse. This check recomputes the score on the remaining held-out predictions. The predictions themselves still come from their original training windows.

**Seven-day accuracy and simulation probabilities.** Both model modes and the historical-median baseline were within seven days of the observed midpoint in four of eight cases. Although the central 80% simulation bands overlapped both observed 2026 windows, the preseason runs assigned only 15.7% probability to the Columbus window and 35.2% to the Oxford window. The corresponding hindcasts assigned 7.0% and 28.6%. Reporting these probabilities alongside overlap makes it clear how much of each simulation distribution actually landed inside the field window.

[Figure 4, historical accuracy checks](04_accuracy_checks.svg) · [Paired elevation cases](elevation_paired_cases.csv) · [Year omission results](omit_one_year_score_sensitivity.csv) · [Seven-day scores](historical_seven_day_accuracy.csv) · [2026 probability inside the observed window](2026_observed_window_probability.csv)

## Ten-year sensitivity scenarios

We added 3,856 usable daily minimum temperature observations from earlier years at the Hamilton and New Lexington NOAA stations. The trend analysis combines those records with the existing data and uses only complete September–November seasons through 2025 after short interior gaps are interpolated. The temperature slope and its 95% interval use ordinary least squares and assume independent errors. The weather records describe the stations and may reflect local changes such as station moves as well as climate. [NOAA data source](https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily) · [Additional NOAA download query](https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00053855,USC00335857&startDate=1996-09-01&endDate=2018-08-31&dataTypes=TMIN&includeStationLocation=true&includeAttributes=true&units=metric&format=csv)

| Location | Fall seasons used | Fitted change in minimum temperature per decade | Median modeled shift at fitted trend | Median shifts at lower and upper slope bounds |
|---|---:|---:|---:|---:|
| Columbus | 28 | +0.56 °C | 1 day later | 0 to 3 days later |
| Oxford | 25 | +0.32 °C | 1 day later | 1 day earlier to 2 days later |
| Perry County | 28 | +0.39 °C | 1 day later | 0 to 2 days later |

For each of 5,000 Monte Carlo draws, we reran the same historical weather season and threshold after shifting every day by the selected decadal temperature change. The table shows the median of the paired differences. The graph's bars show how that median changes at the lower and upper slope bounds. These scenarios hold the other model inputs fixed. The Oxford and Perry County slope intervals include zero. Future fall weather, species composition, and observation coverage can change, and the available tests do not establish a reliable ten-year accuracy horizon. With only four of eight historical predictions within seven days, the model has yet to demonstrate dependable week-level precision even on the available nearer-term cases.

[Figure 5, fall temperature trends and ten-year scenarios](05_ten_year_scenarios.svg) · [Station season means](noaa_fall_station_means.csv) · [Station trends and intervals](noaa_fall_temperature_trends.csv) · [Scenario results](decade_scenarios.csv)

The most valuable additional data for a stronger paper would be repeated field observations at the same sites, especially Perry County in 2026 and sites at substantially different elevations. Recording the same trees weekly would narrow observation windows and help distinguish differences in tree species from changes in weather.

## Daily forecast percentages

Figure 6 shows the seven consecutive days with the highest total preseason crossing probability at each location. Each percentage is the share of 5,000 simulations whose **first** 50% canopy-color crossing fell on that date. The seven displayed probabilities sum to the probability for that window, subject to rounding. The full-season table includes both weather modes and a separate outcome for simulations with no crossing by November 30. This display measures uncertainty in the crossing date. Measuring percent canopy colored on every date would require a model of the full coloration curve, with observations at multiple levels.

[Figure 6, daily forecast percentages](06_daily_forecast_percent.svg) · [Full daily forecast table](2026_daily_forecast_percent.csv) · [Related research and dataset notes](RESEARCH_NOTES.md)

## Additional model experiments

Run `python3 ohio_foliage_extensions.py` to reproduce Figures 7 and 8 and the comparison tables.

### Repeated foliage observations

The expanded Ohio dataset has 2,205 usable fall canopy-color visits at 53 sites and 2,351 fall leaf-drop visits at 58 sites. The records span 2011–2026 and represent visits to individual trees. Many sites have too few observations to validate a site forecast. In Perry County, the same ten trees were monitored in all five falls from 2020 through 2024. They include three silver maples, three sugar maples, and four northern red oaks. Coverage is reported by site and year in the companion table. The curve evaluation uses only visit dates when at least seven of the ten trees had usable measurements.

The extracts use the USA-NPN *Colored leaves* phase (498) and *Falling leaves* phase (471). Each recorded color category is translated into lower and upper percentage bounds, explicit absence is recorded as zero, and unknown intensity is excluded. The field value represents the fraction of the tree's potential canopy occupied by colored leaves. A leaf-drop reading indicates presence or absence of falling or recently fallen leaves. It does not measure how much foliage has fallen. [USA-NPN definitions](https://www.usanpn.org/files/reports/usa-npn_plant_and_animal_phenophase_definitions_v2.0.pdf) · [Color observations query](https://services.usanpn.org/npn_portal/observations/getObservations.ndjson?request_src=OpenAI%20Codex&start_date=2009-09-01&end_date=2026-11-14&state=OH&phenophase_id=498) · [Leaf-drop observations query](https://services.usanpn.org/npn_portal/observations/getObservations.ndjson?request_src=OpenAI%20Codex&start_date=2009-09-01&end_date=2026-11-14&state=OH&phenophase_id=471)

[Ohio canopy-color visits](../data/npn_ohio_color_visits.csv) · [Ohio leaf-drop visits](../data/npn_ohio_leaf_drop_visits.csv) · [Site and year coverage](expanded_foliage_coverage.csv)

### Does summer temperature improve date predictions?

The extra predictor is June–August mean daily temperature, calculated from the daily minimum and maximum at each matched NOAA station. A summer required at least 88 valid days. The input contains 2,474 valid daily temperature pairs at three stations over 2018–2026. The experiment estimates a single regularized relationship between summer temperature and the logarithm of the cold-stress threshold. It removes average differences between sites when estimating that relationship. Each historical fit uses prior years, and the held-out summer temperatures are available by the September 1 forecast date. Both variants use the same eight site-years and the same 5,000 simulations. [NOAA summer download](https://www.ncei.noaa.gov/access/services/data/v1?dataset=daily-summaries&stations=USW00004804,USW00053855,USC00335857&startDate=2018-06-01&endDate=2026-08-31&dataTypes=TMIN,TMAX&includeAttributes=true&units=metric&format=csv)

| Mode | Current MAE | With summer temperature | Historical median MAE |
|---|---:|---:|---:|
| Preseason | 7.69 days | 7.44 days | 9.58 days |
| Hindcast | 9.69 days | 9.56 days | 9.58 days |

The summer term improved three preseason cases by one day, worsened one by one day, and left four dates unchanged. This is a small exploratory improvement. The original model remains the primary forecast because eight cases give limited evidence of a reliable benefit. The candidate results include the separately held-out 2026 runs.

[Figure 7, summer temperature backtest](07_summer_temperature_backtest.svg) · [Matched case results](summer_temperature_cases.csv) · [NOAA summer means](summer_station_means.csv)

### Daily canopy color in Perry County

We fit a three-parameter curve that reaches 50% at the estimated threshold date, rises to a maximum, then declines. The curve was fit to field category midpoints from earlier years, giving each training year equal total weight. The weather-based curve uses the modeled preseason threshold date. Its comparison uses the same curve shape and the median threshold date from prior years. The plotted 80% band propagates simulated timing variation conditional on a crossing and holds the curve's shape fixed. Field leaf-drop observations are shown separately below the curve.

| Held-out year | Scored visit dates | Weather curve MAE | Historical-date curve MAE |
|---|---:|---:|---:|
| 2022 | 13 | 24.5 percentage points | 17.0 percentage points |
| 2023 | 12 | 15.4 percentage points | 18.0 percentage points |
| 2024 | 11 | 9.4 percentage points | 13.5 percentage points |

The 2024 model estimated a maximum near 78% on October 30. The highest recorded visit was November 4, in the 95–100% band. The model captured the general rise and decline but underestimated the maximum in that year. These results apply to the ten monitored trees at one site. The error is measured against observation band midpoints, and Perry County had no usable visits for this evaluation in 2026.

[Figure 8, daily Perry County canopy color](08_perry_daily_canopy_color.svg) · [Daily percentages and timing band](perry_2024_daily_canopy_color.csv) · [Held-out curve scores](perry_color_curve_metrics.csv)
