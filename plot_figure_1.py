from main import main
from utils.parser import argument_parser
from multiprocessing import Process, Manager
import matplotlib.pyplot as plt

num_experiments = 5
num_epochs = 4500
batch_size = 128

def gradient(accs, device):
    for i in range(num_experiments):
        args = argument_parser()
        args.num_epochs = num_epochs
        args.batch_size = batch_size
        args.device = "cuda:"+str(device)
        args.reg = 0
        accs.append(main(args))
    print(f"SGD: {accs}")

def mirror_descent(q, accs, device):
    for i in range(num_experiments):
        args = argument_parser()
        args.algorithm = "smd"
        args.q_norm = q
        args.batch_size = batch_size
        args.num_epochs = num_epochs
        args.device = "cuda:" + str(device)
        args.reg = 0
        accs.append(main(args))
    print(f"SMD q={q}: {accs}")

if __name__ == "__main__":
    d1, d2, d3, d4, d5, d6 = 0, 0, 0, 0, 0, 0
    manager = Manager()
    smd1 = manager.list()
    smd3 = manager.list()
    smd8 = manager.list()
    smd10 = manager.list()
    smd14 = manager.list()
    sgd = manager.list()

    p1 = Process(target=gradient, args=(sgd, d1,))
    p2 = Process(target=mirror_descent, args=(1.1, smd1, d2,))
    p3 = Process(target=mirror_descent, args=(3, smd3, d3,))
    p4 = Process(target=mirror_descent, args=(8, smd8, d4,))
    p5 = Process(target=mirror_descent, args=(10, smd10, d5,))
    p6 = Process(target=mirror_descent, args=(14, smd14, d6,))

    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()

    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()

    # create box plot of accuracies
    plt.boxplot([smd1, sgd, smd3, smd8, smd10, smd14], labels=["SMD q=1", "SGD", "SMD q=3", "SMD q=8", "SMD q=10", "SMD q=14"])
    plt.ylabel("Accuracy")
    plt.xlabel("Algorithm")
    plt.legend()
    plt.title("Comparison of SGD and SMD")
    plt.savefig("figure_1.png")
    plt.show()
