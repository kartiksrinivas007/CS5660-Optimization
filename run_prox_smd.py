import os 


# make a set of values that you want to run the subgd algorithm for 


if __name__ == "__main__":
    # make a set of values that you want to run the subgd algorithm for 
    dataset = ["magic04d"]
    model = ["linear"]
    batch_size = [32]
    num_epochs = [20]
    lr = [1e-4]
    device = ["cuda:1"]
    q_norm = [1.5]
    reg = [1e-2]
    test = [0.2]
    val = [0.1]
    momentum = [0.9, 0.99, 0.999]
    seed = [0]
    algorithm = ["prox_smd"]    
    
    for d in dataset:
        for m in model:
            for b in batch_size:
                for n in num_epochs:
                    for l in lr:
                        for dev in device:
                            for q in q_norm:
                                for r in reg:
                                    for t in test:
                                        for v in val:
                                                for s in seed:
                                                    for a in algorithm:
                                                        os.system(f"python main.py --dataset {d} --model {m} --batch_size {b} --num_epochs {n} --lr {l} --device {dev} --q_norm {q} --reg {r} --test {t} --val {v} --seed {s} --algorithm {a} --wandb")
