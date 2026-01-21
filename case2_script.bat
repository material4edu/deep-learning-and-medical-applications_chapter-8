@REM # Train three variants
python code/train_prostate_reg.py --data data/Task05_Prostate --reg baseline
python code/train_prostate_reg.py --data data/Task05_Prostate --reg dropout
python code/train_prostate_reg.py --data data/Task05_Prostate --reg l2

@REM # Generate report (CSV + PNG)
python code/train_prostate_reg.py --report --data data/Task05_Prostate --models data/models/prostate_baseline.pth data/models/prostate_dropout.pth data/models/prostate_l2.pth
