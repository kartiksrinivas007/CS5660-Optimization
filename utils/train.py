import torch.nn as nn
from tqdm import tqdm
import torch 
import matplotlib.pyplot as plt
import matplotlib
import time
matplotlib.use('Agg')

def train(model, train_loader, val_loader, test_loader, optim, args, metrics):
    model.train()
    train_time_hist = []
    train_hist = []
    val_hist = []
    test_hist = []
    start_time = time.time()
    computed_metrics = [[[], [], []] for _ in range(len(metrics))]
    train_loss_prev = 1e8
    val_loss = calc_loss(model, val_loader, args)
    train_loss = calc_loss(model, train_loader, args)
    test_loss = calc_loss(model, test_loader, args)
    epoch = -1
    for i, metric in enumerate(metrics):
        computed_metrics[i][0].append(compute_metric(model, train_loader, args, metric))
        computed_metrics[i][1].append(compute_metric(model, val_loader, args, metric))
        computed_metrics[i][2].append(compute_metric(model, test_loader, args, metric))
        
        if (args.wandb):
            import wandb
            wandb.log(
                {
                    
                    f"Training {metric.__class__.__name__}": computed_metrics[i][0][-1],
                    f"Validation {metric.__class__.__name__}": computed_metrics[i][1][-1],
                    f"Test {metric.__class__.__name__}": computed_metrics[i][2][-1],
                    f"Training loss": train_loss,
                    f"Validation loss": val_loss,
                    f"Test loss": test_loss,
                    f"epoch": epoch,
                    f"CPU_Time": start_time-start_time
                }
            )
    # loss_reg_loss_ratio_hist = []
    # loss_ratio_hist = []
    for epoch in range(args.num_epochs):
        train_loss_step = 0
        # if(train_loss_prev - train_loss_step < 1e-5):
        #     break
        # else:
        #     train_loss_prev = train_loss_step
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
            reg_loss = torch.tensor(0.0).to(args.device)
            if args.algorithm not in ["prox_smd", "accelerated_prox_smd"] and args.reg > 0:
                for param in model.parameters():
                    reg_loss += args.reg * torch.norm(param, p=1)
            else:
                reg_loss = torch.tensor(0.0, requires_grad=False).to(args.device)
                for param in model.parameters():
                    reg_loss += args.reg * torch.norm(param, p=1)
            

            loss = loss + reg_loss
            if (index == 0):
                print("The ratio of reg_loss to loss is: ", reg_loss.item()/loss.item())
    
            loss.backward()
            optim.step()
            train_loss_step += loss.item()
            train_time_hist.append(train_loss_step/x.shape[0]) # train_loss per SAMPLE
            # loss_reg_loss_ratio_hist.append(loss.item()/reg_loss.item())
        

        val_loss = calc_loss(model, val_loader, args)
        train_loss = calc_loss(model, train_loader, args)
        test_loss = calc_loss(model, test_loader, args)
        # avg_loss_ratio = sum(loss_reg_loss_ratio_hist)/len(loss_reg_loss_ratio_hist)
        val_hist.append(val_loss)
        train_hist.append(train_loss)
        test_hist.append(test_loss)
        # loss_ratio_hist.append(avg_loss_ratio)
        print(f"Epoch {epoch}: Training Loss = {train_loss}")
        print(f"Epoch {epoch}: Validation Loss = {val_loss}")
        print(f"Epoch {epoch}: Test Loss = {test_loss}")
        # print(f"Epoch {epoch}: Average Loss/Reg Loss Ratio = {avg_loss_ratio}")

        for i, metric in enumerate(metrics):
            computed_metrics[i][0].append(compute_metric(model, train_loader, args, metric))
            computed_metrics[i][1].append(compute_metric(model, val_loader, args, metric))
            computed_metrics[i][2].append(compute_metric(model, test_loader, args, metric))
            
            if (args.wandb):
                import wandb
                wandb.log(
                    {
                        
                        f"Training {metric.__class__.__name__}": computed_metrics[i][0][-1],
                        f"Validation {metric.__class__.__name__}": computed_metrics[i][1][-1],
                        f"Test {metric.__class__.__name__}": computed_metrics[i][2][-1],
                        f"Training loss": train_loss,
                        f"Validation loss": val_loss,
                        f"Test loss": test_loss,
                        f"epoch": epoch,
                        f"CPU_Time": time.time()-start_time
                        
                    }
                )
            
            fig, ax = plt.subplots(3, 1)
            ax[0].plot(computed_metrics[i][0], label="Training metric vs epoch", color="red")
            ax[1].plot(computed_metrics[i][1], label="Validation metric vs epoch", color="blue")
            ax[2].plot(computed_metrics[i][2], label="Test metric vs epoch", color="green")
            
            # ax[3].plot(loss_ratio_hist, label="Average Loss/Reg Loss Ratio vs epoch", color="purple")
            fig.legend()
            fig.savefig(f"Plots/Metrics/metric_plot_{i}_{epoch}.png")
            plt.close(fig)
            

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
