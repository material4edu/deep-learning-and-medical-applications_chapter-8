set DATA_DIR=data\Task05_Prostate
set OUT_DIR=data\reports\exp2
set MODEL_DIR=data\models

REM Train 3 models (lightweight run)
python code\train_prostate_reg.py --data "%DATA_DIR%" --reg baseline --epochs 3 --max_cases 6 --base 32
python code\train_prostate_reg.py --data "%DATA_DIR%" --reg dropout  --epochs 3 --max_cases 6 --base 32
python code\train_prostate_reg.py --data "%DATA_DIR%" --reg l2       --epochs 3 --max_cases 6 --base 32

REM Evaluate HD95 + heatmaps (reuse viz_eval.py utilities)
python code\exp2_prostate_hd95_heatmap.py ^
  --data "%DATA_DIR%" ^
  --baseline "%MODEL_DIR%\prostate_baseline.pth" ^
  --dropout "%MODEL_DIR%\prostate_dropout.pth" ^
  --l2 "%MODEL_DIR%\prostate_l2.pth" ^
  --base 32 ^
  --outdir "%OUT_DIR%"

echo.
echo Done. Check outputs in %OUT_DIR%
pause