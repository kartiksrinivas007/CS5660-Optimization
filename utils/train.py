import torch.nn as nn
from tqdm import tqdm
import torch 


def train(model, train_loader, val_loader, optim, args):
    model.train()
    train_hist = []
    val_hist = []
    for epoch in range(args.num_epochs):
        train_loss = 0
        for index, batch in tqdm(enumerate(train_loader)):
            optim.zero_grad()
            x,y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            loss.backward()
            optim.step()
            train_loss += loss.item()
            train_hist.append(train_loss/x.shape[0]) # train_loss per SAMPLE
            
        val_loss = validate(model, val_loader, args)
        val_hist.append(val_loss) # val_loss per BATCH
        print(f"Epoch {epoch}: Validation Loss = {val_loss}")

    return model, train_hist, val_hist

def validate(model, val_loader, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(val_loader)):
            x,y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            total_loss += loss.item()
    return total_loss /len(val_loader)