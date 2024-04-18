python3 plot_figure_1.py # no regularization as in Sahin Lale

# rest of the experiments involve a regularization term lambda = 1e-3 that is non-smooth
python3 main.py --q_norm 1.1 --batch_size 128 --num_epochs 4500 --algorithm=smd

python3 main.py --q_norm 3.0 --batch_size 128 --num_epochs 4500 --algorithm=prox_smd

python3 main.py --q_norm 3.0 --batch_size 128 --num_epochs 4500 --algorithm=accelerated_prox_smd

python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3

python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3 --algorithm=smd

python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3 --algorithm=prox_smd

python3 main.py --dataset magic04s --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3 --algorithm=accelerated_prox_smd

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3 --algorithm=smd

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3 --algorithm=prox_smd

python3 main.py --dataset magic04d --model linear --lr 0.0001 --num_epochs 100 --q_norm 1.1 --reg 1e-3 --algorithm=accelerated_prox_smd