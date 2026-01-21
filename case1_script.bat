%train%
python code/train_spleen_opt.py --data data/Task09_Spleen --opt adam
python code/train_spleen_opt.py --data data/Task09_Spleen --opt sgd  --lr 0.01
python code/train_spleen_opt.py --data data/Task09_Spleen --opt rmsprop

%visualize%
python code/train_spleen_opt.py --report --data data/Task09_Spleen --models data/models/spleen_adam.pth data/models/spleen_sgd.pth data/models/spleen_rmsprop.pth