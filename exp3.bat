@echo off

set REPORT_DIR=data\reports\exp3
set MODEL_DIR=data\models

REM === 数据路径（按你实际情况改） ===
set SPLEEN_DIR=data\Task09_Spleen
set PROSTATE_DIR=data\Task05_Prostate

if not exist %MODEL_DIR% mkdir %MODEL_DIR%
if not exist %REPORT_DIR% mkdir %REPORT_DIR%
if not exist %REPORT_DIR%\spleen mkdir %REPORT_DIR%\spleen
if not exist %REPORT_DIR%\prostate mkdir %REPORT_DIR%\prostate

echo.
echo ===============================
echo Part A: Spleen (optimizer vs reduced channels)
echo base: 16 -> 8
echo ===============================

REM 1) 运行 exp1（base减半）
REM !!! 如果你的 exp1 脚本不在 code/ 目录，请把下面这行改成实际路径 !!!
python code\exp1_gradnorm_sgd_vs_adam.py ^
  --data "%SPLEEN_DIR%" ^
  --epochs 10 ^
  --lr 0.001 ^
  --base 8 ^
  --max_cases 3 ^
  --batch 4 ^
  --outdir "%REPORT_DIR%\spleen"

echo.
echo ===============================
echo Part B: Prostate (regularization vs reduced channels)
echo base: 32 -> 16
echo ===============================

REM 2) 训练 3 个正则化策略（base减半）
python code\train_prostate_reg.py --data "%PROSTATE_DIR%" --reg baseline --epochs 3 --max_cases 6 --base 16
copy /Y "%MODEL_DIR%\prostate_baseline.pth" "%MODEL_DIR%\prostate_baseline_b16.pth" >nul

python code\train_prostate_reg.py --data "%PROSTATE_DIR%" --reg dropout --epochs 3 --max_cases 6 --base 16
copy /Y "%MODEL_DIR%\prostate_dropout.pth" "%MODEL_DIR%\prostate_dropout_b16.pth" >nul

python code\train_prostate_reg.py --data "%PROSTATE_DIR%" --reg l2 --epochs 3 --max_cases 6 --base 16
copy /Y "%MODEL_DIR%\prostate_l2.pth" "%MODEL_DIR%\prostate_l2_b16.pth" >nul

REM 3) 复用 exp2 的评估脚本（base=16），输出HD95趋势+热力图到 exp3/prostate
python code\exp2_prostate_hd95_heatmap.py ^
  --data "%PROSTATE_DIR%" ^
  --baseline "%MODEL_DIR%\prostate_baseline_b16.pth" ^
  --dropout "%MODEL_DIR%\prostate_dropout_b16.pth" ^
  --l2 "%MODEL_DIR%\prostate_l2_b16.pth" ^
  --base 16 ^
  --outdir "%REPORT_DIR%\prostate"

echo.
echo Done.
echo Check outputs:
echo   %REPORT_DIR%\spleen
echo   %REPORT_DIR%\prostate
pause
