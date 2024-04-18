python3 plot_figure_1.py # no regularization as in Sahin Lale

# Tried fine-tuning learning rate and regularization parameter but did not get good results.
# Learning rate as per Sahin Lale is 1e-2.
python3 main.py --q_norm 10.0 --batch_size 128 --lr 0.0001 --num_epochs 20 --reg 1e-4 --algorithm=smd

python3 main.py --q_norm 10.0 --batch_size 128 --lr 0.0001 --num_epochs 20 --reg 1e-4 --algorithm=prox_smd

python3 main.py --q_norm 10.0 --batch_size 128 --lr 0.0001 --num_epochs 20 --reg 1e-4 --algorithm=accelerated_prox_smd

# For MAGIC datasets, lambda=1e-3 and num_epochs=1000 as per https://dl.acm.org/doi/pdf/10.1145/1553374.1553493
# Learning rate was 0.001 in code but that lead to explosion.
python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3

python3 main.py --dataset magic04s --model linear --lr 0.000001 --num_epochs 200 --q_norm 10.0 --reg 1e-3 --algorithm=smd

python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3 --algorithm=prox_smd

python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3 --algorithm=accelerated_prox_smd

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3 --algorithm=smd

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3 --algorithm=prox_smd

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 200 --q_norm 10.0 --reg 1e-3 --algorithm=accelerated_prox_smd