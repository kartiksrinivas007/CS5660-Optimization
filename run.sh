python3 main.py --dataset magic04s --model linear --lr 0.000001 --num_epochs 20 --q_norm 1.01 --reg 1e-3

python3 main.py --dataset magic04s --model linear --lr 0.000001 --num_epochs 20 --q_norm 1.01 --reg 1e-3 --algorithm=smd

python3 main.py --dataset magic04s --model linear --lr 0.000001 --num_epochs 20 --q_norm 1.01 --reg 1e-3 --algorithm=prox_smd
