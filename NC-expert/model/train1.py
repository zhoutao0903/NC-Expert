import argparse
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import datetime
import time
import matplotlib.pyplot as plt
import yaml
import sys
import copy
from lib.mean import cluster_data
sys.path.append("..")
from lib.utils import (
    MaskedMAELoss,
    print_log,
    seed_everything,

    set_cpu_num,
    CustomJSONEncoder,
)
from lib.metrics import RMSE_MAE_MAPE
from lib.dataprepare_cluster import  data_provider
from model.DFExpert import STAEformer
from model.SFExpert import GCNformer


# ! X shape: (B, T, N, C)

@torch.no_grad()
def eval_model(model, valset_loader, criterion, SCALER):
    model.eval()
    batch_loss_list = []
    for i, (x_batch, y_batch) in enumerate(valset_loader):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)
        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

    return np.mean(batch_loss_list)

@torch.no_grad()
def predict(model, loader, SCALER):
    model.eval()
    y = []
    out = []

    for i, (x_batch, y_batch) in enumerate(loader):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        out_batch = out_batch.cpu().numpy()
        y_batch = y_batch.cpu().numpy()
        out.append(out_batch)
        y.append(y_batch)

    out = np.vstack(out).squeeze()  # (samples, out_steps, num_nodes)
    y = np.vstack(y).squeeze()

    return y, out

def train_one_epoch(
    model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=None, SCALER=None
):
    model.train()
    batch_loss_list = []
    for i, (x_batch, y_batch) in enumerate(trainset_loader):
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        out_batch = model(x_batch)
        out_batch = SCALER.inverse_transform(out_batch)

        loss = criterion(out_batch, y_batch)
        batch_loss_list.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        if clip_grad:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

    epoch_loss = np.mean(batch_loss_list)
    scheduler.step()

    return epoch_loss

def train(
    model,
    trainset_loader,
    valset_loader,
    optimizer,
    scheduler,
    criterion,
    clip_grad=0,
    max_epochs=200,
    early_stop=30,
    verbose=1,
    plot=False,
    log=None,
    save=None,
    SCALER=None
):
    model = model.to(DEVICE)

    wait = 0
    min_val_loss = np.inf

    train_loss_list = []
    val_loss_list = []

    for epoch in range(max_epochs):
        train_loss = train_one_epoch(
            model, trainset_loader, optimizer, scheduler, criterion, clip_grad, log=log, SCALER=SCALER
        )
        train_loss_list.append(train_loss)

        val_loss = eval_model(model, valset_loader, criterion, SCALER)
        val_loss_list.append(val_loss)

        if (epoch + 1) % verbose == 0:
            print_log(
                datetime.datetime.now(),
                "Epoch",
                epoch + 1,
                " \tTrain Loss = %.5f" % train_loss,
                "Val Loss = %.5f" % val_loss,
                log=log,
            )

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            wait += 1
            if wait >= early_stop:
                break

    model.load_state_dict(best_state_dict)
    train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*predict(model, trainset_loader, SCALER))
    val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*predict(model, valset_loader, SCALER))

    out_str = f"Early stopping at epoch: {epoch+1}\n"
    out_str += f"Best at epoch {best_epoch+1}:\n"
    out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
    out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        train_rmse,
        train_mae,
        train_mape,
    )
    out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
    out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (
        val_rmse,
        val_mae,
        val_mape,
    )
    print_log(out_str, log=log)

    if plot:
        plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
        plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
        plt.title("Epoch-Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    if save:
        torch.save(best_state_dict, save)
    return model

@torch.no_grad()
def test_model(model, testset_loader, log=None, SCALER=None):
    model.eval()
    print_log("--------- Test ---------", log=log)

    start = time.time()
    y_true, y_pred = predict(model, testset_loader, SCALER)
    end = time.time()

    rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
    out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    out_steps = y_pred.shape[1]
    for i in range(out_steps):
        rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    print_log(out_str, log=log, end="")
    print_log("Inference time: %.2f s" % (end - start), log=log)


if __name__ == "__main__":
    # -------------------------- set running environment ------------------------- #

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="PEMS08")
    parser.add_argument("-g", "--gpu_num", type=int, default=0)
    parser.add_argument('--seq_len', type=int, default=12, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=12, help='prediction sequence length')
    args = parser.parse_args()

    # Set random seed
    seed = 60
    seed_everything(seed)  # Set the random seed for reproducibility
    set_cpu_num(1)

    GPU_ID = args.gpu_num
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = args.dataset.upper()
    data_path = f"../data/{dataset}"
    model_name = DFExpert.__name__
    model_name1 = SFExpert.__name__

    # Load configuration files
    with open(f"{model_name}.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    cfg = cfg[dataset]
    with open(f"{model_name1}.yaml", "r") as f:
        cfg1 = yaml.safe_load(f)
    cfg1 = cfg1[dataset]

    # Create log directory
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = f"../logs/{now}"
    os.makedirs(log_path, exist_ok=True)

    # Initialize log files for each cluster
    log_0 = open(os.path.join(log_path, f"{model_name}-{dataset}-cluster_0.log"), "w")
    log_1 = open(os.path.join(log_path, f"{model_name1}-{dataset}-cluster_1.log"), "w")

    # ------------------------------- load dataset ------------------------------- #

    print_log(dataset, log=log_0)
    print_log(dataset, log=log_1)

    # Cluster data
    X1, labels, id  = cluster_data(data_path)

    # Split data based on clustering results
    clusters = {}
    for i in range(len(np.unique(labels))):
        clusters[i] = X1[labels == i, :, :]

    # Save clustered data
    cluster_dir = os.path.join(data_path, 'clusters')
    os.makedirs(cluster_dir, exist_ok=True)
    for i, cluster_data in clusters.items():
        np.savez(os.path.join(cluster_dir, f'cluster_{i}.npz'), data=cluster_data)
        print_log(f'Saved cluster_{i}.npz', log=log_0)
        print_log(f'Saved cluster_{i}.npz', log=log_1)

    # Load clustered data
    (
        (trainset_loader_0, valset_loader_0, testset_loader_0, SCALER_0, num_nodes_0),
        (trainset_loader_1, valset_loader_1, testset_loader_1, SCALER_1, num_nodes_1)
    ) = data_provider(
        args,
        os.path.join(data_path, 'clusters'),
        tod=cfg.get("time_of_day", False),
        dow=cfg.get("day_of_week", False),
        batch_size=cfg.get("batch_size", 16),
        log=log_0
    )
    # Set random seed
    seed = 57
    seed_everything(seed)  # Set the random seed for reproducibility
    # ------------------------------- load model ------------------------------- #
    cfg["model_args"]["num_nodes"] = num_nodes_0
    model_0 = DFExpert(**cfg["model_args"])

    cfg1["model_args"]["num_nodes"] = num_nodes_1
    model_1 = SFExpert(**cfg1["model_args"])

    # Set loss, optimizer, and scheduler
    if dataset in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08","XMBRT"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer_0 = torch.optim.Adam(
        model_0.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler_0 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_0,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    optimizer_1 = torch.optim.Adam(
        model_1.parameters(),
        lr=cfg1["lr"],
        weight_decay=cfg1.get("weight_decay", 0),
        eps=cfg1.get("eps", 1e-8),
    )
    scheduler_1 = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_1,
        milestones=cfg1["milestones"],
        gamma=cfg1.get("lr_decay_rate", 0.1),
        verbose=False,
    )

    print_log("--------- Training Cluster 0 ---------", log=log_0)
    seed_everything(seed)  # Set the random seed for reproducibility
    set_cpu_num(1)
    model_0 = train(
        model_0,
        trainset_loader_0,
        valset_loader_0,
        optimizer_0,
        scheduler_0,
        criterion,
        clip_grad=cfg.get("clip_grad"),
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        verbose=1,
        log=log_0,
        save=None,  # Do not save model
        SCALER=SCALER_0
    )
    y_true_0, y_pred_0 = predict(model_0, testset_loader_0, SCALER_0)
    # 直接输出模型0在测试集上的结果
    print_log("--------- Testing Cluster 0 ---------", log=log_0)
    test_model(model_0, testset_loader_0, log=log_0, SCALER=SCALER_0)
    log_0.close()

    # 继续训练模型1
    print_log(f"Loss: {criterion._get_name()}", log=log_1)
    print_log("--------- Training Cluster 1 ---------", log=log_1)
    seed_everything(seed)  # Set the random seed for reproducibility
    set_cpu_num(1)
    model_1 = train(
        model_1,
        trainset_loader_1,
        valset_loader_1,
        optimizer_1,
        scheduler_1,
        criterion,
        clip_grad=cfg1.get("clip_grad"),
        max_epochs=cfg1.get("max_epochs", 200),
        early_stop=cfg1.get("early_stop", 10),
        verbose=1,
        log=log_1,
        save=None,  # Do not save model
        SCALER=SCALER_1
    )
    y_true_1, y_pred_1 = predict(model_1, testset_loader_1, SCALER_1)
    # 输出模型1在测试集上的结果
    print_log("--------- Testing Cluster 1 ---------", log=log_1)
    test_model(model_1, testset_loader_1, log=log_1, SCALER=SCALER_1)
    log_1.close()

    # --------------------------- Concatenate Results --------------------------- #

    # Concatenate the true and predicted values from both clusters
    y_true_concat = np.concatenate((y_true_0, y_true_1), axis=2)  # Concatenate along the node axis
    y_pred_concat = np.concatenate((y_pred_0, y_pred_1), axis=2)  # Concatenate along the node axis


    from lib.metrics import RMSE_MAE_MAPE as metric_fn

    out_steps = y_pred_concat.shape[1]
    rmse_all, mae_all, mape_all = metric_fn(y_true_concat, y_pred_concat)
    out_str = "ALL Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
        rmse_all,
        mae_all,
        mape_all,
    )
    for i in range(out_steps):
        rmse, mae, mape = metric_fn(y_true_concat[:, i, :], y_pred_concat[:, i, :])
        out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (
            i + 1,
            rmse,
            mae,
            mape,
        )

    # Save the concatenated results to a log file
    unified_log_path = os.path.join(log_path, f"{model_name}-{dataset}-unified.log")
    with open(unified_log_path, "w") as unified_log:
        unified_log.write(out_str)

    print(out_str)






