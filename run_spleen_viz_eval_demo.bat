@echo off
python "%~dp0code/viz_eval.py" ^
  --data "%~dp0data/Task09_Spleen" ^
  --weights "%~dp0data/models/spleen_baseline.pth" ^
  --normalization zscore ^
  --split Tr ^
  --base 32 ^
  --outdir "%~dp0data/reports/spleen_viz_eval"
