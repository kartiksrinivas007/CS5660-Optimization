import os 


# make a set of values that you want to run the subgd algorithm for 


if __name__ == "__main__":
    # make a set of values that you want to run the subgd algorithm for 
    dataset = ["magic04s"]
    model = ["linear"]
    batch_size = [32]
    num_epochs = [20]
    lr = [1e-4, 1.5e-5 , 1e-5, 1e-6, 1.5e-6, 1.5e-7, 1e-7]
    device = ["cuda"]
    q_norm = [1.001, 1.5 , 2, 2.5, 3, 4]
    reg = [1e-3]
    test = [0.2]
    val = [0.1]
    momentum = [0.9, 0.99, 0.999]
    seed = [0]
    algorithm = ["accelerated_prox_smd"]    
    
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
                                            for mom in momentum:
                                                for s in seed:
                                                    for a in algorithm:
                                                        os.system(f"python main.py --dataset {d} --model {m} --batch_size {b} --num_epochs {n} --lr {l} --device {dev} --q_norm {q} --reg {r} --test {t} --val {v} --momentum {mom} --seed {s} --algorithm {a} --wandb")