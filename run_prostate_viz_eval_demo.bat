@echo off
python "%~dp0code/viz_eval.py" ^
  --data "%~dp0data/Task05_Prostate" ^
  --weights "%~dp0data/models/prostate_baseline.pth" ^
  --normalization zscore ^
  --split Tr ^
  --outdir "%~dp0data/reports/prostate_viz_eval" ^
  --base 16
