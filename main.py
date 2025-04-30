import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import StockSequenceDataset, StockDDPM
import pytorch_lightning as pl
import os
import pandas as pd
import argparse
import json
from pathlib import Path
import random
import numpy as np
from tqdm import tqdm

def read_stock_and_sentiment(file_path, stock_cols, sentiment_cols, use_conditioning):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=stock_cols + (sentiment_cols if use_conditioning else [])).reset_index(drop=True)

    stock_data = df[stock_cols].values
    stock_min = stock_data.min(axis=0)
    stock_max = stock_data.max(axis=0)
    stock_data = (stock_data - stock_min) / (stock_max - stock_min + 1e-8)

    sentiment_data = None
    if use_conditioning:
        sentiment_data = df[sentiment_cols].values
        sentiment_data = (sentiment_data - sentiment_data.min(axis=0)) / (sentiment_data.max(axis=0) - sentiment_data.min(axis=0) + 1e-8)

    return (
        torch.tensor(stock_data, dtype=torch.float32),
        torch.tensor(sentiment_data, dtype=torch.float32) if use_conditioning else None,
        stock_min, stock_max
    )

def denormalize(data, data_min, data_max):
    return data * (data_max - data_min + 1e-8) + data_min

def plot_forecast(context_close, true_future_close, predicted_close, stock_min, stock_max, close_idx, save_path):
    full_true = torch.cat([context_close, true_future_close], dim=0)
    full_pred = torch.cat([context_close, predicted_close], dim=0)

    full_true = denormalize(full_true, stock_min[close_idx], stock_max[close_idx])
    full_pred = denormalize(full_pred, stock_min[close_idx], stock_max[close_idx])

    plt.figure(figsize=(14, 6))
    plt.plot(full_true, label="Ground Truth Close", linestyle="--", color="gray")
    plt.plot(range(len(context_close), len(context_close) + len(predicted_close)), full_pred[-len(predicted_close):], label="Predicted Close", color="blue")
    plt.axvline(x=len(context_close) - 1, linestyle=":", color="black", label="Forecast Start")
    plt.title("Close Price: Forecast + Context")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_loss(losses, save_path):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def get_unique_model_dir(base_name):
    base_path = Path("models")
    base_path.mkdir(exist_ok=True)
    i = 1
    while (base_path / f"{base_name}_{i}").exists():
        i += 1
    model_dir = base_path / f"{base_name}_{i}"
    model_dir.mkdir()
    return model_dir

def save_config(path, stock_cols, sentiment_cols, context_len, pred_len):
    config = {
        "STOCK_COLS": stock_cols,
        "SENTIMENT_COLS": sentiment_cols,
        "CONTEXT_LEN": context_len,
        "PRED_LEN": pred_len
    }
    with open(path, "w") as f:
        json.dump(config, f, indent=2)

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

def compute_metrics(predicted, ground_truth):
    gt_mean = ground_truth.mean()
    gt_std = ground_truth.std()
    norm_pred = (predicted - gt_mean) / gt_std
    norm_true = (ground_truth - gt_mean) / gt_std
    norm_dev = torch.mean(torch.abs(norm_pred - norm_true)).item()
    norm_rmse = torch.sqrt(torch.mean((norm_pred - norm_true) ** 2)).item()
    accuracy = (torch.round(norm_pred * 100) == torch.round(norm_true * 100)).float().mean().item()
    gt_diff = ground_truth[1:] - ground_truth[:-1]
    pred_diff = predicted[1:] - predicted[:-1]
    direction_match = ((gt_diff > 0) == (pred_diff > 0)).float().mean().item()
    return norm_dev, norm_rmse, accuracy, direction_match

def evaluate_on_random_windows(model, stock_data, sentiment_data, context_len, pred_len, stock_min, stock_max, close_idx, cond_flag, model_dir):
    results = []
    total_windows = len(stock_data) - context_len - pred_len
    indices = random.sample(range(total_windows), 10)

    print("\n🔍 Evaluating on 10 random windows:")
    for idx in tqdm(indices, desc="Evaluating"):
        context = stock_data[idx:idx + context_len].unsqueeze(0)
        target = stock_data[idx + context_len:idx + context_len + pred_len, close_idx].unsqueeze(-1)
        sentiment = sentiment_data[idx + context_len:idx + context_len + pred_len].unsqueeze(0) if cond_flag else None

        pred = model.sample(context, sentiment, device=model.device, num_samples=5).squeeze(0)
        pred_denorm = denormalize(pred, stock_min[close_idx], stock_max[close_idx])
        target_denorm = denormalize(target, stock_min[close_idx], stock_max[close_idx])
        results.append(compute_metrics(pred_denorm, target_denorm))

    results = np.array(results)
    metrics = {
        "Normalized Deviation (mean ± std)": [float(results[:, 0].mean()), float(results[:, 0].std())],
        "Normalized RMSE (mean ± std)": [float(results[:, 1].mean()), float(results[:, 1].std())],
        "Accuracy (mean ± std)": [float(results[:, 2].mean()), float(results[:, 2].std())],
        "Direction Match (mean ± std)": [float(results[:, 3].mean()), float(results[:, 3].std())]
    }

    with open(Path(model_dir) / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n📊 Evaluation Results:")
    for k, v in metrics.items():
        print(f"{k}: {v[0]:.4f} ± {v[1]:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", action="store_true", help="Use sentiment conditioning")
    parser.add_argument("--kan", action="store_true", help="Use KAN layers")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    parser.add_argument("--eval", action="store_true", help="Only evaluate a trained model")
    parser.add_argument("--ckpt_path", type=str, help="Path to model folder (for --eval)")
    args = parser.parse_args()

    FILE_PATH = "data/AAPL_combined.csv"

    if args.eval:
        if not args.ckpt_path:
            raise ValueError("Please provide --ckpt_path when using --eval")
        config_path = Path(args.ckpt_path) / "config.json"
        checkpoint_path = Path(args.ckpt_path) / "last.ckpt"
        if not config_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing config.json or last.ckpt in {args.ckpt_path}")
        config = load_config(config_path)
        STOCK_COLS = config["STOCK_COLS"]
        SENTIMENT_COLS = config["SENTIMENT_COLS"]
        CONTEXT_LEN = config["CONTEXT_LEN"]
        PRED_LEN = config["PRED_LEN"]
    else:
        STOCK_COLS = ["Close"]
        SENTIMENT_COLS = ["ts_polarity"]
        CONTEXT_LEN = 360
        PRED_LEN = 30

    close_idx = STOCK_COLS.index("Close")

    stock_data, sentiment_data, stock_min, stock_max = read_stock_and_sentiment(
        FILE_PATH, STOCK_COLS, SENTIMENT_COLS, use_conditioning=args.cond
    )

    if args.eval:
        model = StockDDPM.load_from_checkpoint(
            str(Path(args.ckpt_path) / "last.ckpt"),
            context_dim=len(STOCK_COLS),
            hidden_dim=128,
            use_conditioning=args.cond,
            kan = args.kan,
            sentiment_dim=len(SENTIMENT_COLS) if args.cond else 0,
            context_len=CONTEXT_LEN,
            pred_len=PRED_LEN
        )
        model.eval()
        evaluate_on_random_windows(model, stock_data, sentiment_data, CONTEXT_LEN, PRED_LEN, stock_min, stock_max, close_idx, args.cond, args.ckpt_path)
        return

    dataset = StockSequenceDataset(
        stock_data, sentiment_data,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN,
        use_conditioning=args.cond,
        stock_cols=STOCK_COLS
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = StockDDPM(
        context_dim=len(STOCK_COLS),
        hidden_dim=128,
        use_conditioning=args.cond,
        kan = args.kan,
        sentiment_dim=len(SENTIMENT_COLS) if args.cond else 0,
        context_len=CONTEXT_LEN,
        pred_len=PRED_LEN
    )

    trainer = pl.Trainer(max_epochs=args.epochs, gpus=1 if torch.cuda.is_available() else 0)
    trainer.fit(model, dataloader)

    model_type = "cond" if args.cond else "norm"
    model_type = model_type + "_kan" if args.kan else model_type 
    model_dir = get_unique_model_dir(model_type)
    trainer.save_checkpoint(str(model_dir / "last.ckpt"))
    save_config(model_dir / "config.json", STOCK_COLS, SENTIMENT_COLS, CONTEXT_LEN, PRED_LEN)

    eval_context = stock_data[-(CONTEXT_LEN + PRED_LEN):-PRED_LEN].unsqueeze(0)
    eval_sentiment = sentiment_data[-PRED_LEN:].unsqueeze(0) if args.cond else None
    context_close = stock_data[-(CONTEXT_LEN + PRED_LEN):-PRED_LEN, close_idx].unsqueeze(-1)
    true_future_close = stock_data[-PRED_LEN:, close_idx].unsqueeze(-1)

    predicted = model.sample(eval_context, eval_sentiment, device=model.device, num_samples=5)
    plot_forecast(context_close, true_future_close, predicted.squeeze(0), stock_min, stock_max, close_idx, save_path=model_dir / "forecast.png")
    plot_training_loss(model.training_losses, save_path=model_dir / "train_loss.png")

    print(f"\n✅ Model saved to: {model_dir}")

if __name__ == "__main__":
    main()
