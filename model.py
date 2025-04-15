import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import pytorch_lightning as pl
import math

def get_beta_schedule(timesteps, start=1e-4, end=0.02):
    return torch.linspace(start, end, timesteps)

class DiffusionUtils(nn.Module):
    def __init__(self, timesteps):
        super().__init__()
        betas = get_beta_schedule(timesteps)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1)
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise

def get_sinusoidal_embedding(timesteps, embedding_dim):
    device = timesteps.device
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class StockSequenceDataset(Dataset):
    def __init__(self, stock_data, sentiment_data=None, context_len=360, pred_len=30, use_conditioning=False, stock_cols=None):
        self.stock_data = stock_data
        self.sentiment_data = sentiment_data
        self.context_len = context_len
        self.pred_len = pred_len
        self.use_conditioning = use_conditioning
        self.total_len = context_len + pred_len
        if stock_cols is None or "Close" not in stock_cols:
            raise ValueError("STOCK_COLS must include 'Close'")
        self.close_index = stock_cols.index("Close")

    def __len__(self):
        return len(self.stock_data) - self.total_len

    def __getitem__(self, idx):
        context = self.stock_data[idx:idx + self.context_len]
        target_close = self.stock_data[idx + self.context_len:idx + self.total_len, self.close_index].unsqueeze(-1)
        if self.use_conditioning:
            sentiment = self.sentiment_data[idx + self.context_len:idx + self.total_len]
            return context, target_close, sentiment
        else:
            return context, target_close

class DiffusionModel(nn.Module):
    def __init__(self, context_dim, hidden_dim, context_len, pred_len, output_dim=1, sentiment_dim=0, time_emb_dim=32):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        self.output_dim = output_dim
        self.sentiment_dim = sentiment_dim

        self.context_encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(context_len * context_dim, hidden_dim),
            nn.ReLU()
        )

        input_size = output_dim + time_emb_dim + hidden_dim + (sentiment_dim if sentiment_dim > 0 else 0)

        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.prediction_net = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.out_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_t, t, context, sentiment=None):
        B, L, _ = x_t.shape
        t_emb = get_sinusoidal_embedding(t, self.time_emb_dim).unsqueeze(1).expand(B, L, -1)
        context_encoded = self.context_encoder(context).unsqueeze(1).expand(-1, L, -1)

        inputs = [x_t, t_emb, context_encoded]
        if self.sentiment_dim > 0 and sentiment is not None:
            inputs.append(sentiment)

        x_input = torch.cat(inputs, dim=-1).view(B * L, -1)
        h = self.prediction_net(x_input)
        gated = self.tanh(h) * self.sigm(h)
        return self.out_layer(gated).view(B, L, self.output_dim)

class StockDDPM(pl.LightningModule):
    def __init__(self, context_dim, hidden_dim, use_conditioning=False, sentiment_dim=0,
                 context_len=360, pred_len=30, timesteps=1000, lr=1e-3):
        super().__init__()
        self.model = DiffusionModel(
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            context_len=context_len,
            pred_len=pred_len,
            output_dim=1,
            sentiment_dim=sentiment_dim
        )
        self.diffusion = DiffusionUtils(timesteps)
        self.criterion = nn.MSELoss()
        self.use_conditioning = use_conditioning
        self.lr = lr
        self.pred_len = pred_len
        self.training_losses = []

    def training_step(self, batch, batch_idx):
        if self.use_conditioning:
            context, target, sentiment = batch
        else:
            context, target = batch
            sentiment = None

        B = context.size(0)
        t = torch.randint(0, self.diffusion.betas.size(0), (B,), device=self.device)

        last_close = context[:, -1, -1].unsqueeze(1).unsqueeze(-1).expand(-1, self.pred_len, 1)
        residual_target = target - last_close

        noise = torch.randn_like(residual_target)
        x_t = self.diffusion.q_sample(residual_target, t, noise)
        noise_pred = self.model(x_t, t, context, sentiment)

        noise_loss = self.criterion(noise_pred, noise)
        beta_t = self.diffusion.betas[t].view(-1, 1, 1)
        x0_pred = x_t - beta_t.sqrt() * noise_pred

        aux_loss = self.criterion(x0_pred, residual_target)
        smooth_loss = torch.mean((x0_pred[:, 1:] - x0_pred[:, :-1]) ** 2)

        loss = noise_loss + 0.5 * aux_loss + 0.5 * smooth_loss
        self.training_losses.append(loss.item())
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.model.parameters(), lr=self.lr)

    def sample(self, context, sentiment=None, device="cpu", num_samples=5):
        self.eval()
        context = context.to(device)
        if sentiment is not None:
            sentiment = sentiment.to(device)

        predictions = []
        for _ in range(num_samples):
            B = context.size(0)
            x = torch.randn(B, self.pred_len, 1).to(device)
            for t in reversed(range(self.diffusion.betas.size(0))):
                t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
                pred_noise = self.model(x, t_tensor, context, sentiment)

                beta = self.diffusion.betas[t]
                alpha = self.diffusion.alphas[t]
                alpha_bar = self.diffusion.alpha_bars[t]

                sqrt_recip_alpha = (1 / alpha.sqrt())
                sqrt_one_minus_alpha_bar = (1 - alpha_bar).sqrt()

                noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
                x = sqrt_recip_alpha * (x - beta / sqrt_one_minus_alpha_bar * pred_noise) + beta.sqrt() * noise

            last_close = context[:, -1, -1].unsqueeze(1).unsqueeze(-1).expand(-1, self.pred_len, 1)
            predictions.append((x + last_close).detach().cpu())

        return torch.stack(predictions).mean(dim=0)

