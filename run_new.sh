# python3 main.py --dataset magic04s --model linear --lr 1e-2 --num_epochs 10 --reg 1.5e-5 --algorithm=subgd --batch_size 32
python3 main.py --dataset magic04s --model linear --lr 1e-5 --num_epochs 20 --q_norm 1.01 --reg 1e-3 --algorithm=smd --batch_size 32 --wandb --seed 0
# python3 main.py --dataset magic04s --model linear --lr 1e-5 --num_epochs 20 --q_norm 1.01 --reg 1e-3 --algorithm=prox_smd --batch_size 32
# 
