import torch
from tqdm import tqdm
from torchmetrics import Accuracy

def test(model, test_loader, args):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(test_loader)):
            x,y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            acc = Accuracy(task="multiclass", num_classes=10).to(device=args.device)(output, y)
            total_acc += acc.item()
    return total_acc /len(test_loader)