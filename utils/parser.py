import argparse

def argument_parser():
    parser = argparse.ArgumentParser(description='Run the algorithms')
    parser.add_argument('--algorithm', type=str, default='sgd', help='The algorithm to run')
    parser.add_argument('--dataset', type=str, default='cifar10', help='The dataset to use')
    parser.add_argument('--model', type=str, default='resnet18', help='The model to use')
    parser.add_argument('--batch_size', type=int, default=32, help='The batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='The number of epochs')
    parser.add_argument('--lr', type=float, default=0.01, help='The learning rate')
    parser.add_argument('--device', type=str, default="cuda", help='The device to run on')
    parser.add_argument('--q_norm', type=float, default=2, help='The q norm')
    parser.add_argument('--reg', type=float, default=0.001, help='The regularization parameter for the one-norm')
    parser.add_argument('--test', type=float, default=0.2, help='Test size split')
    parser.add_argument('--val', type=float, default=0.2, help='Validation size split')
    parser.add_argument('--momentum', type=float, default=0.9, help='The momentum for the optimizer')
    args = parser.parse_args()
    return args

def test():
    args = argument_parser()
    print(args)