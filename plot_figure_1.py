from main import main
from utils.test import test
from utils.parser import argument_parser
from multiprocessing import Process
import matplotlib.pyplot as plt

num_experiments = 5

def gradient():
    accs=[]
    for i in range(num_experiments):
        args = argument_parser()
        model, test_loader = main(args)
        args.num_epochs = 10
        accs.append(test(model, test_loader, args))
    print(f"SGD: {accs}")

def mirror_descent(q):
    accs=[]
    for i in range(num_experiments):
        args = argument_parser()
        args.algorithm = "smd"
        args.q_norm = q
        args.num_epochs = 10
        model, test_loader = main(args)
        accs.append(test(model, test_loader, args))
    print(f"SMD q={q}: {accs}")

if __name__ == "__main__":
    p1 = Process(target=gradient)
    p2 = Process(target=mirror_descent, args=(1.01,))
    p3 = Process(target=mirror_descent, args=(3,))
    p4 = Process(target=mirror_descent, args=(8,))
    p5 = Process(target=mirror_descent, args=(10,))
    p6 = Process(target=mirror_descent, args=(14,))

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
    # plt.boxplot([smd1, sgd, smd3, smd8, smd10, smd14], labels=["SMD q=1", "SGD", "SMD q=3", "SMD q=8", "SMD q=10", "SMD q=14"])
    # plt.ylabel("Accuracy")
    # plt.xlabel("Algorithm")
    # plt.legend()
    # plt.title("Comparison of SGD and SMD")
    # plt.savefig("figure_1.png")
    # plt.show()
