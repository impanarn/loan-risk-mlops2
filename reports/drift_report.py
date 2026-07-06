import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

ref = pd.read_csv("data/processed/X_train.csv")
cur = pd.read_csv("data/processed/X_test.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)
report.save_html("reports/drift_report.html")
print("Drift report saved to reports/drift_report.html")