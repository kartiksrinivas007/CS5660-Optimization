import os 


# make a set of values that you want to run the subgd algorithm for 


if __name__ == "__main__":
    # make a set of values that you want to run the subgd algorithm for 
    dataset = ["magic04s", "magic04d"]
    model = ["linear"]
    batch_size = [32]
    num_epochs = [10]
    lr = [1.5e-5 , 1e-5]
    device = ["cuda"]
    reg = [1e-3]
    test = [0.2]
    val = [0.1]
    seed = [0]
    algorithm = ["subgd"]
    
    # make a set of values that you want to run the subgd algorithm for 
    #parallelize this
    for d in dataset:
        for m in model:
            for b in batch_size:
                for n in num_epochs:
                    for l in lr:
                        for dev in device:
                                for r in reg:
                                    for t in test:
                                        for v in val:
                                                for s in seed:
                                                    for a in algorithm:
                                                        os.system(f"python main.py --dataset {d} --model {m} --batch_size {b} --num_epochs {n} --lr {l} --device {dev} --reg {r} --test {t} --val {v}  --seed {s} --algorithm {a} --wandb")