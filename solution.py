import os
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================================
# 1. HYPERPARAMETERS & CONFIGURATION
# ==========================================
CONFIG = {
    # Kaggle:  '/kaggle/input/competitions/previsao-climatica-de-precipitacao-sobre-a-america-do-sul'
    # Local:   pasta onde os arquivos .nc foram baixados
    'data_dir': r'C:\Users\vinic\OneDrive\Desktop\WorCap_RAIN-PREDICTION\data',
    'batch_size': 4,
    'epochs': 30,
    'learning_rate': 2e-4,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'seq_len': 6,
    'timesteps': 1000,
    'hidden_dim': 64,
    'ddim_steps': 20,
    'num_inference_samples': 20,
    'lambda_x0': 0.5,
    'num_workers': 0,  # 0 no Windows para evitar erro de multiprocessing
}

# ==========================================
# 2. DATASET
# ==========================================
class ClimateDataset(Dataset):
    """
    Carrega os dados ERA5 do conjunto de treino.
    O alvo é a ANOMALIA de precipitação: y - climatologia_mensal(m).
    Isso reduz a variância do sinal a ser aprendido pelo modelo.
    """
    NORM_STATS = None  # Preenchido pelo primeiro dataset criado

    def __init__(self, data_dir, seq_len=3, is_train=True):
        self.seq_len = seq_len
        self.is_train = is_train
        self.data_dir = data_dir

        def carregar_var(nome_arquivo):
            caminho = os.path.join(data_dir, nome_arquivo)
            ds = xr.open_dataset(caminho)
            return ds[list(ds.data_vars)[0]].values.astype(np.float32)

        if is_train:
            print("Carregando arquivos NetCDF em memória...")
            self.alvo = carregar_var("treino_tp_alvo.nc")   # (T, H, W)

            # Para calcular a climatologia, precisamos saber o mês de cada timestep.
            # O alvo tem T meses. Inferimos o índice de mês (0-11) ciclicamente.
            # Os dados de treino cobrem de Jan/1979 a Dez/2022 = 528 meses.
            T = self.alvo.shape[0]

            self.vars_data = [
                carregar_var("treino_t2.nc"),
                carregar_var("treino_cloud_cover.nc"),
                carregar_var("treino_geopotential_850.nc"),
                carregar_var("treino_rel_hum_850.nc"),
                carregar_var("treino_shum_850.nc"),
                carregar_var("treino_surface_pressure.nc"),
                carregar_var("treino_temperature_850.nc"),
                carregar_var("treino_u_850.nc"),
                carregar_var("treino_v_850.nc"),
            ]

            if ClimateDataset.NORM_STATS is None:
                print("Calculando climatologia mensal e estatísticas z-score...")

                # Climatologia mensal: média de cada mês (Jan...Dez) sobre todos os anos
                # Shape: (12, H, W)
                clim = np.zeros((12, self.alvo.shape[1], self.alvo.shape[2]), dtype=np.float32)
                count = np.zeros(12, dtype=np.int32)
                for t_idx in range(T):
                    m = t_idx % 12
                    valid = ~np.isnan(self.alvo[t_idx])
                    clim[m][valid] += self.alvo[t_idx][valid]
                    count[m] += 1
                for m in range(12):
                    clim[m] /= count[m]

                # Anomalia = alvo - climatologia
                anomalias = []
                for t_idx in range(T):
                    m = t_idx % 12
                    anomalias.append(self.alvo[t_idx] - clim[m])
                anomalias = np.stack(anomalias, axis=0)  # (T, H, W)

                # Z-score sobre as anomalias
                anom_flat = anomalias[~np.isnan(anomalias)]
                anom_mean = float(anom_flat.mean())
                anom_std  = float(anom_flat.std()) + 1e-8

                # Z-score das features de entrada
                means, stds = [], []
                for v in self.vars_data:
                    flat = v[~np.isnan(v)]
                    means.append(float(flat.mean()))
                    stds.append(float(flat.std()) + 1e-8)

                ClimateDataset.NORM_STATS = {
                    'means': means, 'stds': stds,
                    'tp_mean': anom_mean, 'tp_std': anom_std,
                    'climatology': clim,   # (12, H, W) — persistido para a inferência
                }
                print(f"  anom_mean={anom_mean:.4f}, anom_std={anom_std:.4f}")

            self.clim = ClimateDataset.NORM_STATS['climatology']  # (12, H, W)

            self.valid_indices = [
                i for i in range(seq_len - 1, T - 1)
                if not np.isnan(self.alvo[i]).any()
            ]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        t = self.valid_indices[idx]
        stats = ClimateDataset.NORM_STATS

        def slice_and_norm(var_array, mean, std):
            sliced = var_array[t - self.seq_len + 1: t + 1]
            return (sliced - mean) / std

        channels = [
            slice_and_norm(v, stats['means'][i], stats['stds'][i])
            for i, v in enumerate(self.vars_data)
        ]

        x = np.stack(channels, axis=1)  # (seq_len, 9, H, W)

        # Alvo é a ANOMALIA normalizada
        m = t % 12
        anomalia = self.alvo[t] - self.clim[m]
        y = (anomalia - stats['tp_mean']) / stats['tp_std']

        x = np.nan_to_num(x, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)


        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        )


# ==========================================
# 3. ARQUITETURA
# ==========================================
class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            input_dim + hidden_dim, 4 * hidden_dim, kernel_size, padding=padding
        )

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.split(gates, gates.size(1) // 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.GroupNorm(8, out_c),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalUNet(nn.Module):
    """
    U-Net com 3 níveis de downsampling condicional ao contexto do ConvLSTM
    e ao embedding de timestep. Usa GroupNorm em vez de BatchNorm para
    estabilidade com batch_size pequeno.
    """
    def __init__(self, c_in=1, c_cond=64, c_out=1):
        super().__init__()
        base = 32

        # Encoder
        self.inc = DoubleConv(c_in + c_cond, base)        # → base
        self.down1 = DoubleConv(base, base * 2)            # → base*2
        self.down2 = DoubleConv(base * 2, base * 4)        # → base*4

        # Bottleneck
        self.bottleneck = DoubleConv(base * 4, base * 4)

        # Decoder
        self.up2 = DoubleConv(base * 4 + base * 2, base * 2)
        self.up1 = DoubleConv(base * 2 + base, base)
        self.out_conv = nn.Conv2d(base, c_out, 1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Timestep embedding: sinusoidal → MLP → adaptado para cada nível
        t_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(1, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim)
        )
        self.t_proj2 = nn.Linear(t_dim, base * 2)
        self.t_proj3 = nn.Linear(t_dim, base * 4)

    @staticmethod
    def _pad_to(x, ref):
        dY = ref.size(2) - x.size(2)
        dX = ref.size(3) - x.size(3)
        return F.pad(x, [dX // 2, dX - dX // 2, dY // 2, dY - dY // 2])

    def forward(self, x, t, cond):
        t_emb = self.time_mlp(t)  # (B, t_dim)

        x_in = torch.cat([x, cond], dim=1)

        # Encoder
        s0 = self.inc(x_in)                               # (B, base, H, W)
        s1 = self.down1(self.pool(s0))                    # (B, base*2, H/2, W/2)
        s1 = s1 + self.t_proj2(t_emb).unsqueeze(-1).unsqueeze(-1)
        s2 = self.down2(self.pool(s1))                    # (B, base*4, H/4, W/4)
        s2 = s2 + self.t_proj3(t_emb).unsqueeze(-1).unsqueeze(-1)

        # Bottleneck
        b = self.bottleneck(s2)                           # (B, base*4, H/4, W/4)

        # Decoder
        u2 = self._pad_to(self.upsample(b), s1)
        d2 = self.up2(torch.cat([u2, s1], dim=1))        # (B, base*2, H/2, W/2)

        u1 = self._pad_to(self.upsample(d2), s0)
        d1 = self.up1(torch.cat([u1, s0], dim=1))        # (B, base, H, W)

        return self.out_conv(d1)


class LSTMDiffusionModel(nn.Module):
    def __init__(self, in_channels, hidden_dim, timesteps):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.timesteps = timesteps

        self.convlstm = ConvLSTMCell(in_channels, hidden_dim)
        self.unet = ConditionalUNet(c_in=1, c_cond=hidden_dim, c_out=1)

        # Cosine noise schedule (melhor que o linear para preservar detalhes finos)
        betas = self._cosine_betas(timesteps)
        alphas = 1.0 - betas
        alpha_hat = torch.cumprod(alphas, dim=0)

        self.register_buffer('beta', betas)
        self.register_buffer('alpha', alphas)
        self.register_buffer('alpha_hat', alpha_hat)

    @staticmethod
    def _cosine_betas(T, s=0.008):
        steps = torch.arange(T + 1, dtype=torch.float64)
        f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi * 0.5) ** 2
        f = f / f[0]
        betas = 1 - (f[1:] / f[:-1])
        return betas.clamp(0, 0.999).float()

    def get_context(self, x_seq):
        B, seq_len, C, H, W = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        c = torch.zeros(B, self.hidden_dim, H, W, device=x_seq.device)
        for t in range(seq_len):
            h, c = self.convlstm(x_seq[:, t], (h, c))
        return h

    def forward(self, x_seq, y_target):
        B = x_seq.size(0)
        t = torch.randint(0, self.timesteps, (B,), device=x_seq.device).long()
        noise = torch.randn_like(y_target)

        alpha_hat_t = self.alpha_hat[t].view(B, 1, 1, 1)
        x_noisy = torch.sqrt(alpha_hat_t) * y_target + torch.sqrt(1 - alpha_hat_t) * noise

        cond = self.get_context(x_seq)
        t_norm = t.float().view(B, 1) / self.timesteps
        noise_pred = self.unet(x_noisy, t_norm, cond)

        # Loss híbrido: erro no ruído + erro de reconstrução direta de x0
        l_noise = F.mse_loss(noise_pred, noise)

        x0_pred = (x_noisy - torch.sqrt(1 - alpha_hat_t) * noise_pred) / torch.sqrt(alpha_hat_t)
        l_x0 = F.mse_loss(x0_pred, y_target)

        return l_noise + CONFIG['lambda_x0'] * l_x0

    @torch.no_grad()
    def sample_ddim(self, x_seq, ddim_steps=20):
        """
        Amostragem via DDIM (determinística, sem ruído extra).
        Permite reduzir o número de avaliações de rede de T para ddim_steps.
        """
        B, _, _, H, W = x_seq.shape
        cond = self.get_context(x_seq)
        x = torch.randn(B, 1, H, W, device=x_seq.device)

        step_size = self.timesteps // ddim_steps
        time_steps = list(reversed(range(0, self.timesteps, step_size)))
        time_steps_next = time_steps[1:] + [-1]

        for t_curr, t_next in zip(time_steps, time_steps_next):
            t_tensor = torch.full((B,), t_curr, device=x_seq.device, dtype=torch.long)
            t_norm = t_tensor.float().view(B, 1) / self.timesteps
            noise_pred = self.unet(x, t_norm, cond)

            a_hat = self.alpha_hat[t_curr].view(B, 1, 1, 1)
            a_hat_next = (
                self.alpha_hat[t_next].view(B, 1, 1, 1)
                if t_next >= 0
                else torch.ones(B, 1, 1, 1, device=x.device)
            )

            x0_pred = (x - torch.sqrt(1 - a_hat) * noise_pred) / torch.sqrt(a_hat)
            x0_pred = x0_pred.clamp(-3, 3)  # limitar estimativas de x0 instáveis
            dir_xt = torch.sqrt(1 - a_hat_next) * noise_pred
            x = torch.sqrt(a_hat_next) * x0_pred + dir_xt

        return x  # retorna em espaço normalizado z-score (sem clamp fixo)


# ==========================================
# 4. TREINAMENTO
# ==========================================
def train():
    print("Carregando dados...")
    dataset = ClimateDataset(CONFIG['data_dir'], seq_len=CONFIG['seq_len'])
    loader = DataLoader(
        dataset, batch_size=CONFIG['batch_size'],
        shuffle=True, num_workers=CONFIG['num_workers'], pin_memory=(CONFIG['device'] == 'cuda')
    )

    model = LSTMDiffusionModel(
        in_channels=9,
        hidden_dim=CONFIG['hidden_dim'],
        timesteps=CONFIG['timesteps']
    ).to(CONFIG['device'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CONFIG['epochs'], eta_min=1e-6
    )

    for epoch in range(CONFIG['epochs']):
        model.train()
        pbar = tqdm(loader, desc=f"Época {epoch + 1}/{CONFIG['epochs']}")
        for x, y in pbar:
            x, y = x.to(CONFIG['device']), y.to(CONFIG['device'])
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            pbar.set_postfix({'loss': f'{loss.item():.5f}', 'lr': f'{scheduler.get_last_lr()[0]:.2e}'})
        scheduler.step()

    torch.save(model.state_dict(), 'modelo_final.pth')
    # Salvar stats de normalização para uso na inferência
    np.save('norm_stats.npy', ClimateDataset.NORM_STATS)
    return model


# ==========================================
# 5. VALIDAÇÃO LOCAL
# ==========================================
def validate_rmse(model, num_val_months=12):
    """
    Avalia o RMSE nos últimos num_val_months do conjunto de treino.
    Desnormaliza a anomalia e soma a climatologia mensal para obter a precipitação absoluta.
    """
    print(f"\nValidação Local (RMSE) — últimos {num_val_months} meses...")
    model.eval()
    dataset = ClimateDataset(CONFIG['data_dir'], seq_len=CONFIG['seq_len'], is_train=True)
    stats = ClimateDataset.NORM_STATS
    clim = stats['climatology']  # (12, H, W)

    total_samples = len(dataset)
    val_indices = range(total_samples - num_val_months, total_samples)

    rmses = []
    n_samples = CONFIG['num_inference_samples']

    for idx in tqdm(val_indices, desc="Validando"):
        x, y_norm = dataset[idx]
        t_abs = dataset.valid_indices[idx]
        m = t_abs % 12

        x = x.unsqueeze(0).to(CONFIG['device'])
        # Desnormaliza anomalia e soma climatologia
        y_real = y_norm.numpy() * stats['tp_std'] + stats['tp_mean'] + clim[m]

        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                p = model.sample_ddim(x, ddim_steps=CONFIG['ddim_steps'])
                p = p.squeeze().cpu().numpy() * stats['tp_std'] + stats['tp_mean'] + clim[m]
                p = np.clip(p, 0, None)
                preds.append(p)

        pred_mean = np.mean(preds, axis=0)
        rmse = np.sqrt(np.mean((pred_mean - y_real.squeeze()) ** 2))
        rmses.append(rmse)

    rmse_final = np.mean(rmses)
    print(f"-> RMSE Médio de Validação: {rmse_final:.4f} mm/dia")
    return rmse_final


# ==========================================
# 6. INFERÊNCIA E SUBMISSÃO
# ==========================================
def infer_and_submit(model):
    print("Iniciando inferência para submissão...")
    model.eval()
    data_dir = CONFIG['data_dir']
    seq_len = CONFIG['seq_len']
    stats = ClimateDataset.NORM_STATS
    clim = stats['climatology']  # (12, H, W)

    sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))

    def load_feature_unified(filename, mean, std):
        ds_train = xr.open_dataset(os.path.join(data_dir, f"treino_{filename}.nc"))
        ds_test = xr.open_dataset(os.path.join(data_dir, "teste_features.nc"))
        var_name = list(ds_train.data_vars)[0]
        train_tail = ds_train[var_name].values[-(seq_len - 1):].astype(np.float32)
        test_full = ds_test[var_name].values.astype(np.float32)
        merged = np.concatenate([train_tail, test_full], axis=0)
        return (merged - mean) / std

    var_files = [
        ("t2", 0), ("cloud_cover", 1), ("geopotential_850", 2),
        ("rel_hum_850", 3), ("shum_850", 4), ("surface_pressure", 5),
        ("temperature_850", 6), ("u_850", 7), ("v_850", 8),
    ]

    vars_unified = [
        load_feature_unified(fname, stats['means'][i], stats['stds'][i])
        for fname, i in var_files
    ]

    n_samples = CONFIG['num_inference_samples']
    device = CONFIG['device']
    all_preds = []

    # O período de teste começa em Janeiro de 2023 (mês 0 do ciclo anual)
    # Os dados de treino terminam em Dezembro de 2022
    # → o alvo do step 0 é Janeiro 2023 → mês índice 0
    test_start_month = 0  # Janeiro

    for step in tqdm(range(24), desc="Processando meses"):
        channels = [v[step: step + seq_len] for v in vars_unified]
        x = np.stack(channels, axis=1)  # (seq_len, 9, H, W)
        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

        # Índice do mês-alvo (0=Jan, 1=Fev, ..., 11=Dez)
        m = (test_start_month + step) % 12

        preds = []
        for _ in range(n_samples):
            p = model.sample_ddim(x_tensor, ddim_steps=CONFIG['ddim_steps'])
            # Desnormaliza anomalia e soma climatologia do mês correto
            p = p.squeeze().cpu().numpy() * stats['tp_std'] + stats['tp_mean'] + clim[m]
            p = np.clip(p, 0, None)
            preds.append(p)

        pred_mean = np.mean(preds, axis=0)
        all_preds.extend(pred_mean.flatten())

    sub['tp_mm_day'] = all_preds
    sub.to_csv('submission.csv', index=False)
    print("Arquivo submission.csv gerado.")


if __name__ == '__main__':
    model = train()
    validate_rmse(model, num_val_months=12)
    infer_and_submit(model)

