import argparse
import json
import os
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import pickle
from lsnn_model import LSNNet,CNN
from spikingjelly.activation_based import functional


@dataclass
class TrainConfig:
    data_path: str
    save_dir: str = "checkpoints_lsnn"
    num_classes: int = 2
    in_leads: int = 3
    signal_length: int = 1500
    batch_size: int = 64
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.0
    val_ratio: float = 0.2
    random_seed: int = 42
    filters: int = 16
    sim_steps: int = 8
    dropout: float = 0.1
    tau: float = 2.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MODMAset(Dataset):
    def __init__(self, root,files):
        self.root = root
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        try:
            sample = pickle.load(open(os.path.join(self.root, self.files[index]), "rb"))
        except:
            sample = pickle.load(open(os.path.join(self.root, self.files[index-1]), "rb"))
        X = sample["X"]
        Y = sample["y"]
        data = torch.FloatTensor(X / 10)
        return data,Y


class StandardizePerSample:
    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x: [N, C, T]
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        return (x - mean) / std


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz_dataset(data_path: str):
    data = np.load(data_path)
    required = {"x", "y"}
    missing = required - set(data.files)
    if missing:
        raise KeyError(f"NPZ must contain keys {required}, missing: {missing}")
    x = data["x"]
    y = data["y"]
    return x, y


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_y_true, all_y_pred = [], []
    all_prob = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits, _ = model(x)
        loss = criterion(logits, y)
        losses.append(loss.item())

        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)

        all_y_true.extend(y.cpu().numpy().tolist())
        all_y_pred.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob[:, 1].cpu().numpy().tolist() if prob.shape[1] == 2 else prob.cpu().numpy().tolist())
        functional.reset_net(model)
    # print(all_y_true)
    # print(all_y_pred)
    acc = accuracy_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred, average="macro")
    uar = recall_score(all_y_true, all_y_pred, average="macro")
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "acc": acc,
        "macro_f1": f1,
        "uar": uar,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "score": all_prob,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        functional.reset_net(model)

    return float(np.mean(losses)) if losses else 0.0


def main():
    parser = argparse.ArgumentParser(description="Train LSNNet with PyTorch + SpikingJelly")
    parser.add_argument("--data_path", type=str,default='/home/data', help="NPZ file containing x [N,C,T] and y [N]")
    parser.add_argument("--save_dir", type=str, default="checkpoints_lsnn")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--sim_steps", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--tau", type=float, default=1.5)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--in_leads", type=int, default=19)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = TrainConfig(
        data_path=args.data_path,
        save_dir=args.save_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        filters=args.filters,
        sim_steps=args.sim_steps,
        dropout=args.dropout,
        tau=args.tau,
        num_classes=args.num_classes,
        in_leads=args.in_leads,
        val_ratio=args.val_ratio,
        random_seed=args.seed,
    )

    set_seed(cfg.random_seed)

    train_path = '/home/data/data/NeuroLM_data/dataset/MODMA/train/'
    val_path = '/home/data/data/NeuroLM_data/dataset/MODMA/train/'

    train_ds = MODMAset(train_path,os.listdir(train_path))
    val_ds = MODMAset(val_path,os.listdir(val_path))

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # model = CNN(19,2).to(cfg.device)
    model = LSNNet(
        in_leads=cfg.in_leads,
        num_classes=cfg.num_classes,
        filters=cfg.filters,
        sim_steps=cfg.sim_steps,
        dropout=cfg.dropout,
        tau=cfg.tau,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_acc = -1.0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.device)
        val_metrics = evaluate(model, val_loader, criterion, cfg.device)

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_uar": val_metrics["uar"],
        }
        print(log)



if __name__ == "__main__":
    main()
