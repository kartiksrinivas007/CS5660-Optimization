import torch.nn as nn
from tqdm import tqdm
import torch 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def train(model, train_loader, val_loader, test_loader, optim, args, metrics):
    model.train()
    # train_time_hist = []
    train_hist = []
    val_hist = []
    test_hist = []
    computed_metrics = [[[], [], []] for _ in range(len(metrics))]
    for epoch in range(args.num_epochs):
        train_loss_step = 0
        for index, batch in tqdm(enumerate(train_loader)):
            optim.zero_grad()
            x,y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)

            # If we are using prox_smd or accelerated_prox_smd, we handle regularizer differently
            # Thus we compute the gradient of the original loss function and not the
            # gradient of the loss + regularizer
            if args.algorithm not in ["prox_smd", "accelerated_prox_smd"] and args.reg > 0:
                for param in model.parameters():
                    loss += args.reg * torch.norm(param, p=1)
            
            loss.backward()
            optim.step()
            train_loss_step += loss.item()
            # train_time_hist.append(train_loss_step/x.shape[0]) # train_loss per SAMPLE
        
        # dump a training plot
        # plt.plot(train_time_hist)
        # plt.savefig(f"Plots/training_plot_{epoch}.png")
        # plt.close()
        val_loss = calc_loss(model, val_loader, args)
        train_loss = calc_loss(model, train_loader, args)
        test_loss = calc_loss(model, test_loader, args)
        val_hist.append(val_loss)
        train_hist.append(train_loss)
        test_hist.append(test_loss)
        print(f"Epoch {epoch}: Training Loss = {train_loss}")
        print(f"Epoch {epoch}: Validation Loss = {val_loss}")
        print(f"Epoch {epoch}: Test Loss = {test_loss}")

        for i, metric in enumerate(metrics):
            computed_metrics[i][0].append(compute_metric(model, train_loader, args, metric))
            computed_metrics[i][1].append(compute_metric(model, val_loader, args, metric))
            computed_metrics[i][2].append(compute_metric(model, test_loader, args, metric))

    return train_hist, val_hist, test_hist, computed_metrics

def calc_loss(model, loader, args):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for index, batch in tqdm(enumerate(loader)):
            x,y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            loss = nn.CrossEntropyLoss()(output, y)
            total_loss += loss.item()
    return total_loss /len(loader)

def compute_metric(model, loader, args, metric):
    model.eval()
    total_acc = 0
    with torch.no_grad():
        for index, batch in enumerate(loader):
            x,y = batch
            x = x.to(args.device)
            y = y.to(args.device)
            output = model(x)
            acc = metric(output, y)
            total_acc += acc.item()
    return total_acc/len(loader)
