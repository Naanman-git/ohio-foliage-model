# Predicting the timing of Ohio fall color

**IM2C 2026, Problem 1 — Peak Peeping**  
**Technical report draft for the news outlet’s analytics team**  
Cover sheet for submission: add verified team names here. Keep names and institutions off the subsequent pages.

## Abstract

We estimate the first date on which monitored trees reach 50% canopy color in Columbus, Oxford, and Perry County, Ohio. A cold accumulation model weighted by daylight uses nearby NOAA weather and USA National Phenology Network observations. A preseason historical evaluation gave 7.69 days mean absolute error against observed window midpoints across eight site-years, versus 9.58 days using the local historical median. In 2026 the principal preseason medians were October 21 in Columbus, October 19 in Oxford, and October 20 in Perry County. Columbus was later than the monitored-tree window, Oxford was within its window, and Perry County lacked a qualifying 2026 window. Exploratory species and rainfall terms reduced error on their respective historical comparisons, with mixed 2026 results. Ten-year warming scenarios suggest a typical shift of around one day later under fixed model assumptions. Available observations leave the future accuracy horizon unresolved.

## 1. Study area and definition

Columbus, Oxford, and Perry County represent three monitored settings in central, southwestern, and southeastern Ohio. The forecasts apply to monitored settings and provide context for nearby sightseeing. Field observers score the fraction of a plant’s potential canopy showing non-green leaves. For each tree-season, we bracket first reaching 50% between the last qualifying below-threshold visit and the first qualifying visit at or above it. We take the median lower and upper bounds across the trees at each site and use their midpoint for scoring. Fifty percent gives a common marker for the transition into strong color. Maximum visual color, leaf retention, and the color of the wider landscape can peak on different dates. [USA-NPN protocol](https://www.usanpn.org/files/reports/usa-npn_plant_and_animal_phenophase_definitions_v2.0.pdf)

## 2. Data and model

We matched the monitored sites to Ohio State University Airport, Hamilton Butler County Airport, and New Lexington 2 NW. Daily minimum temperatures were aligned to September 1 through November 30. Short internal gaps were interpolated. Historical climate realizations require complete fall series after interpolation. The 2026 hindcast incorporates available weather through November 10. The station-to-site temperature correction was −0.0065 °C per meter of elevation difference. The base temperature was set at 15 °C. Data and source queries are documented in the [results notes](RESULTS.md), [research notes](RESEARCH_NOTES.md), and [project readme](../README.md).

For latitude φ and ordinal solar day j, approximate solar declination δ and daylight L in hours are

`δ(j) = (23.44π/180) sin[2π(284+j)/365]`  
`L(j) = (24/π) arccos[clip(−tan(φ)tan(δ(j)), −1, 1)]`.

The fall day index d starts on September 1. Using the June solstice reference L₀, corrected temperature T(d), and stress threshold C, the model is

`S(d) = Σ_{k=0}^d max(0, 15−T(k)) [1−L(k)/L₀]`  
`D = first d such that S(d) ≥ C`.

For each earlier site-year, we interpolate S at the observed window bounds and use their mean as that case’s stress midpoint. The site median of those stress midpoints centers the forecast threshold. For each simulation we sample an earlier site-year, draw uniformly between its bound stresses, subtract that sampled site’s center, and add the resulting deviation to the target site center. The threshold is clipped below at 0.01. Preseason weather comes from sampled whole earlier autumns at the target station. Hindcasts overwrite available days with observed temperatures. The approach follows the general cold accumulation and photoperiod structure studied by [Delpierre et al.](https://doi.org/10.1016/j.agrformet.2008.11.014) and [Archetti et al.](https://doi.org/10.1371/journal.pone.0057373).

For 5,000 draws we record the date of first crossing or a separate no-crossing outcome. The daily probability is the count for the day divided by 5,000. We report the simulated 10th, 50th, and 90th percentiles. The simulations reproduced with a fixed seed, normalized to total probability one, and showed no earlier crossing when every temperature was raised by 1 °C using matched draws.

## 3. Historical accuracy

Each evaluation used earlier years and required at least two prior local observations and seasons. The comparison predicts the median local date from earlier observed windows. Error is the absolute difference between prediction and observed midpoint. The midpoint itself is uncertain when visits are far apart.

| Mode | Site-years | Model MAE | Historical median MAE | Within 7 days |
|---|---:|---:|---:|---:|
| Preseason | 8 | 7.69 d | 9.58 d | 4 of 8 |
| Observed-weather hindcast | 8 | 9.69 d | 9.58 d | 4 of 8 |

The preseason median landed inside two of eight field windows. Its central 80% simulation interval overlapped seven of eight. These events differ because a broad band can overlap a field window while its median misses. With the elevation correction disabled, preseason MAE was 7.56 days. The largest median change was one day. The observed-weather hindcast scored about the same as the calendar baseline, suggesting that uncertainty in thresholds, timing observations, and model structure remains important. [Historical backtest](01_historical_backtest.svg) · [Accuracy checks](04_accuracy_checks.svg)

## 4. Predictions and published 2026 comparisons

| Location | Preseason median | Hindcast median | Monitored-tree window | Published county estimate |
|---|---|---|---|---|
| Columbus | Oct 21 | Oct 22 | Oct 7–14, 2 trees | Franklin, Oct 18–20 |
| Oxford | Oct 19 | Oct 23 | Oct 15–23, 5 trees | Butler, around Oct 29 |
| Perry County | Oct 20 | Oct 24 | No qualifying interval | Perry, Oct 21–27 |

The Columbus preseason median was 10.25 days late relative to its observed midpoint and one day later than the end of the county estimate. The Oxford median matched the monitored-tree midpoint and preceded the county estimate by about ten days. Perry’s hindcast fell within the published county estimate. The September New Lexington article identified local routes and referenced the statewide forecast. The October 8 Perry-area near-peak claim is a secondary reading of the ODNR map. ODNR reported mostly near-peak or peak statewide on October 22 and widespread peak on October 30. WVXU and Spectrum provide regional context, while the published county dates use a tourism definition of peak. The geographic scale and definitions preclude scoring them as exact tree crossings. The preseason probability mass in the field windows was 15.7% in Columbus and 35.2% in Oxford. [Dated news and official source log](2026_news_validation.csv) · [Daily probabilities](06_daily_forecast_percent.svg)

## 5. Additional tests

**Species.** Within each earlier mixed-species site-year, we subtract the median crossing date from each tree’s midpoint and take the median residual by species. If a species is observed in m earlier mixed site-years, its offset is the mean of those residuals multiplied by m/(m+2). The adjustment is added to the pooled preseason median for each tree. Species without earlier estimates use zero offset for the comparison. Across eight historical site-years, the mean of site-year tree MAEs fell from 10.72 to 8.04 days. Six site-years improved and two worsened. In 2026, Columbus individual-tree MAE fell from 10.75 to 9.04 days and Oxford rose from 6.60 to 7.85 days. Perry’s sugar maples preceded northern red oaks in five of six years with usable windows and followed them in 2024. These repeated trees show local variation. Regional species abundance and canopy area would be needed to use these offsets in a landscape prediction. [Species windows](09_perry_species_windows.svg) · [Tree-level cases](species_tree_backtest.csv)

**Summer precipitation and temperature.** We obtained 9,518 NOAA precipitation rows and filtered flagged daily values. At least 88 of 92 summer days were required. The summer precipitation estimate is 92 times the valid daily mean. We fit the log threshold to site-centered precipitation, in 100 mm units, using a ridge penalty of four. The threshold multiplier is exp(βΔP), where ΔP is the site’s summer anomaly in 100 mm units and β is fitted on earlier years. Paired simulations gave historical site MAE of 6.94 days versus 7.69 days. The slope changed sign across fits. In 2026 Columbus error fell by five days, Oxford error increased by three days, and Perry’s prediction shifted from October 20 to October 11 with no measured crossing to judge the change. A separate summer temperature term gave 7.44 days MAE. These exploratory variants were tested on a small shared set of seasons, so the primary forecast remains the original model. [Rainfall cases](summer_precipitation_backtest.csv) · [Summer temperature cases](summer_temperature_cases.csv)

**Spring budbreak.** We downloaded Nature’s Notebook breaking-leaf-buds observations and matched individual plant IDs, sites, and years. The audit yielded 1,656 spring visits, 83 bracketed spring windows, and 39 spring windows at most 14 days wide with an autumn match. Perry supplied 32 of these pairs and Oxford supplied none. The available overlap did not support fitting and validating a predictor across all three locations. [Spring coverage](spring_fall_coverage.csv) · [Scientific context](RESEARCH_NOTES.md)

**Daily canopy color.** A Perry County curve anchored at 50% uses a logistic rise before the modeled crossing and a bounded rise and decline afterward. It beat a fixed historical timing baseline in two of three evaluated years. In 2024 it estimated a maximum near 78% on October 30. The highest recorded category was 95–100% on November 4. Leaf drop indicates whether observers saw leaves falling. The available field data do not quantify total leaf loss. [Daily color curve and leaf-drop observations](08_perry_daily_canopy_color.svg)

## 6. Ten-year outlook and limits

We estimated linear trends in September–November station mean minimum temperature using 25–28 complete seasons per site. The slopes were +0.56 °C per decade for Columbus, +0.32 for Oxford, and +0.39 for Perry County. Rerunning the same simulated years and thresholds after adding each station’s projected warming moved the median paired crossing one day later at each site. Across slope bounds the modeled site medians shifted from one day earlier to three days later. This is a conditional weather perturbation. A ten-year prediction error or reliable accuracy horizon cannot be estimated with the available records. Near-term evidence supports a broad seasonal window, with only four of eight historical median dates within seven days. Annual recalibration as observations accumulate would let the outlet reassess performance. [Ten-year scenarios](05_ten_year_scenarios.svg)

The project’s limited time and ecological expertise constrain the scope of the analysis. The inputs do not include a project-run field campaign or local canopy inventory. The team’s reported time constraints and limited tree biology experience should be described in its own words before submission. The largest technical limitations are small and uneven tree samples, repeated trees within years and sites, sparse visits, unmeasured site moisture and canopy structure, station-to-site differences, and a fixed 50% event that approximates a transition instead of scenic maximum. A future program could monitor fixed routes weekly with photos and identify species and leaf retention consistently.

## 7. Source and reproducibility record

- Primary observations and weather queries are documented in the [data and results notes](RESULTS.md) and [project readme](../README.md).
- Published 2026 comparisons are tagged by source type in the [news cross-check](2026_news_validation.csv).
- The [research notes](RESEARCH_NOTES.md) retain supporting papers, model precedents, and additional data options alongside the user-supplied context.
- The challenge PDF and the user-provided project reference pack are the original task and design references. The earlier notebook documents the synthetic demonstration.
- AI use disclosure draft: OpenAI Codex was used to assist with literature and news discovery, data processing, code development, interpretation, and drafting. Before submitting, the team should confirm the record of tools and prompts, verify all claims and citations, and complete the disclosure under the [COMAP contest AI policy](https://www.contest.comap.com/undergraduate/contests/mcm/flyer/Contest_AI_Policy.pdf).
