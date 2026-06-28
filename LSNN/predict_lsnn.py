import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lsnn_model import LSNNet


class EEGOnlyDataset(Dataset):
    def __init__(self, x: np.ndarray):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 3:
            raise ValueError(f"x must be [N, C, T], got {x.shape}")
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True) + 1e-6
        self.x = (x - mean) / std

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    all_pred = []
    all_prob = []

    for x in loader:
        x = x.to(device)
        logits, _ = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(dim=1)
        all_pred.extend(pred.cpu().numpy().tolist())
        all_prob.extend(prob.cpu().numpy().tolist())
        model.reset()

    return all_pred, all_prob


def build_model_from_ckpt(ckpt, device):
    cfg = ckpt["config"]
    model = LSNNet(
        in_leads=cfg["in_leads"],
        num_classes=cfg["num_classes"],
        filters=cfg["filters"],
        sim_steps=cfg["sim_steps"],
        dropout=cfg["dropout"],
        tau=cfg["tau"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    return model


def main():
    parser = argparse.ArgumentParser(description="Predict with trained LSNNet")
    parser.add_argument("--checkpoint", type=str, required=True, help="best_model.pth from training")
    parser.add_argument("--input_npz", type=str, required=True, help="NPZ file containing x [N,C,T]")
    parser.add_argument("--output_json", type=str, default="predictions.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = build_model_from_ckpt(ckpt, device)

    data = np.load(args.input_npz)
    if "x" not in data.files:
        raise KeyError("Input NPZ must contain key 'x'")
    ds = EEGOnlyDataset(data["x"])
    loader = DataLoader(ds, batch_size=64, shuffle=False)

    pred, prob = predict(model, loader, device)
    out = [{"index": i, "pred": int(p), "prob": [float(v) for v in pr]} for i, (p, pr) in enumerate(zip(pred, prob))]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved predictions to: {args.output_json}")
    if out:
        print("First 5 results:")
        for item in out[:5]:
            print(item)


if __name__ == "__main__":
    main()
