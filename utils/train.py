import torch.nn as nn
from tqdm import tqdm
import torch 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def train(model, train_loader, val_loader, optim, args):
    model.train()
    train_time_hist = []
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
            for param in model.parameters():
                loss += args.reg * torch.norm(param, p=1)
            
            loss.backward()
            optim.step()
            train_loss += loss.item()
            train_time_hist.append(train_loss/x.shape[0]) # train_loss per SAMPLE
        
        # dump a trainign plot
        plt.plot(train_time_hist)
        plt.savefig(f"Plots/training_plot_{epoch}.png")
        plt.close()  
        val_loss = validate(model, val_loader, args)
        train_loss_per_batch = validate(model, train_loader, args)
        val_hist.append(val_loss) # val_loss per BATCH
        train_hist.append(train_loss_per_batch)
        print(f"Epoch {epoch}: Validation Loss = {val_loss}")
        print(f"Epoch {epoch}: Training Loss = {train_loss_per_batch}")

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