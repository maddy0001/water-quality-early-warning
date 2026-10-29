# Sample files for the dashboard

All rows are real data from the project's datasets (nothing made up), except the 10 rows marked
SPIKED in files 2 and 3, which are real samples with two parameters deliberately shifted by 3 standard
deviations, the same method as the paper's synthetic contamination test.

| File | Page | What it is |
|---|---|---|
| `1_classification_test_set_570.csv` | 🔍 Water Classification | The 570 held-out test samples (never used in training), raw values, with the true `Potability` label. The page reports how many it gets right. |
| `2_anomaly_spike_test.csv` | ⚠️ Anomaly Detection | 40 clean test samples + 10 spiked copies, mixed. The `note` column says which are spiked. |
| `3_early_warning_batch.csv` | 🚨 Alert History | Same 50 samples with `timestamp` and `location` columns, for the alert log. |
| `4_forecasting_usgs_station_2198840.csv` | 📈 pH Forecasting | 120 days of real USGS sensor data (test period) for one station, with the actual pH in `pH_actual`. |

The older files (`sample_water_quality_small/batch/labeled.csv`) also work on the Classification,
Anomaly and Alert pages, but their values were typed by hand, and the `Expected_Result` column in the
labeled file follows a simple pH rule, not the dataset's labels, so it is not a test of accuracy.
`sample_water_quality_timeseries.csv` cannot be used for forecasting: the BiLSTM needs the 11 USGS
sensor columns, not the 9 Kaggle columns.
