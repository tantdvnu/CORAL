README — CORAL: Code and Processing Scripts
Gradient-Boosted Quantile Regression With Context-Aware Online Pinball-Loss Descent for Assured-Rate Link Adaptation in 5G NR
Subir Gupta, Duc-Tan Tran
Data and Code Availability
The dataset used is publicly available (Raca et al., ACM MMSys 2020, GPL-v3), and the experimental protocol is fully described. To support reproducibility, the code, processing scripts and a README that regenerate every figure and table are available at https://github.com/tantdvnu/CORAL.
Dataset
D. Raca, D. Leahy, C. J. Sreenan, and J. J. Quinlan, “Beyond throughput, the next generation: A 5G dataset with channel and context metrics,” Proc. ACM MMSys, 2020, pp. 303–308, doi: 10.1145/3339825.3394938. Source: https://github.com/uccmisl/5Gdataset (83 traces, 188,711 records, 190 cells). The script downloads and extracts it automatically; no sample is synthetic.
Requirements
Python 3.9+ with numpy, pandas, scipy, scikit-learn, matplotlib and git. Optional: lightgbm (GBDT backend; otherwise scikit-learn HistGradientBoosting is used) and numba (faster online loops).
How to Run
git clone https://github.com/tantdvnu/CORAL
cd CORAL
python coral_spl.py
What the Script Does
•	Builds 1-s samples with lagged log-rate, KPI and context features (horizon H = 3 s).
•	Creates 10 leakage-free trace partitions: P1 = unseen download traces; P2 = video traces (traffic shift).
•	Trains GBDT quantile/MSE models and runs HM-5, GBDT-MSE, GBDT-QR, Split CQR, ACI, OLLA, the two ablations and CORAL at α = 0.1 (also 0.05, 0.2, 0.3).
•	Computes violation, worst-trace and per-context violation, utilization, PIT calibration, log-score and goodput, plus Wilcoxon and bootstrap tests.
Outputs (folder results/)
•	table1.csv — Table I (assurance and utilization)
•	table2.csv — Table II (predictive density and goodput)
•	fig2.pdf / fig2.png — Fig. 2(a)–(c)
•	stats.txt — significance tests;  summary.json — configuration and all results
Licence
Code released for research use; the dataset remains under GPL-v3 by its authors.
