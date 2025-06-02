import uproot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.cuda.amp as amp

# Load datasets
# #paths to samples
zero_t_samples = [
    'data/WW.Flz.part1.root',
    'data/WW.Flz.part2.root',
    'data/Wjets-incl.INb.part1.root',
    'data/Wjets-incl.INb.part2.root',
    'data/Wjets-incl.INb.part3.root',
    'data/Wjets-incl.INb.part4.root',
    'data/Wjets-incl.INb.part5.root',
]

one_t_samples = [
    'data/t-channel_tbar_4f.leB.part1.root',
    'data/t-channel_tbar_4f.leB.part2.root',
    'data/t-channel_top_4f.qlh.part1.root',
    'data/t-channel_top_4f.qlh.part2.root',
]

two_t_samples = [
    'data/ttbar_doublelep.KQP.part1.root',
    'data/ttbar_doublelep.KQP.part2.root',
    'data/ttbar_doublelep.KQP.part3.root',
    'data/ttbar_doublelep.KQP.part4.root',
    'data/ttbar_semilep.vRt.part1.root',
    'data/ttbar_semilep.vRt.part2.root',
]

three_t_samples = [
    'data/TTTJ_UL16Summer20_RoO.root',
    'data/TTTW_UL16Summer20_CgT.root',
]

four_t_samples = [
    'data/TTTT_UL16Summer20_eIE.part1.root',
    'data/TTTT_UL16Summer20_eIE.part2.root',
    'data/TTTT_UL16Summer20_eIE.part3.root',
    'data/TTTT_UL16Summer20_eIE.part4.root',
]

# create dataframe
total_data = []
for i, samples in tqdm(enumerate([zero_t_samples,one_t_samples,two_t_samples,three_t_samples,four_t_samples])):
    list_of_samples = []
    for path in tqdm(samples):
        print('processing',i, path)
        file = uproot.open(path)
        a = file['LHE'].arrays(file['LHE'].keys(), library="np")
        df = pd.DataFrame(a)
        list_of_samples.append(df)
        print('successful')
    df_sample = pd.concat(list_of_samples).reset_index()
    df_sample['class'] = i
    total_data.append(df_sample)
data = pd.concat(total_data).reset_index()

data = data.drop(columns = ['level_0', 'index'])   
print('shape:', data.shape)

def split_input_target(df):
    # Выбираем ВСЕ переменные с "Gen" (включая генераторные глобальные переменные)
    target_columns = [col for col in df.columns if 'Gen' in col]
    
    # Выбираем ВСЕ переменные БЕЗ "Gen" (конечные частицы + обычные глобальные переменные)
    input_columns = [col for col in df.columns if 'Gen' not in col]
    
    # Проверка на случай, если есть колонки, не связанные ни с чем
    remaining = set(df.columns) - set(target_columns + input_columns)
    if remaining:
        raise ValueError(f'Unaccounted columns: {remaining}')
    
    
    X = df[input_columns].copy()
    y = df[target_columns].copy()
    
    return X, y
data, _ = split_input_target(data)


def preprocess_df(df, particle_types, features, scaler=None, fit_scaler=True):
    df = df.copy()

    # Заменяем значения Phi на cos(Phi) прямо в тех же колонках
    for ptype, indices in particle_types.items():
        for idx in indices:
            phi_col = f'Phi_{ptype}{idx}'
            if phi_col in df.columns:
                df[phi_col] = np.cos(df[phi_col].values)

    # Список признаков для масштабирования (все указанные, включая Phi — теперь это cos(Phi))
    cols_to_scale = []
    for ptype, indices in particle_types.items():
        for idx in indices:
            for feat in features:
                col = f'{feat}_{ptype}{idx}'
                if col in df.columns:
                    cols_to_scale.append(col)

    # Добавляем глобальные числовые переменные (начинаются с N_ или MeT)
    global_num_cols = [col for col in df.columns if col.startswith('N_') or col == 'MeT']
    cols_to_scale += global_num_cols

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler, cols_to_scale


particle_types = {
    'BJet': [1, 2, 3, 4],
    'Jet': list(range(1, 13)),
    'Lep': [1, 2, 3, 4],
}

particle_features = ["Pt", "Eta", "Phi", "Px", "Py", "Pz"]

particle_cols = []
for ptype, indices in particle_types.items():
    for idx in indices:
        for feat in particle_features:
            col = f"{feat}_{ptype}{idx}"
            particle_cols.append(col)

global_cols = ["N_J", "N_BJ", "N_Nu", "N_Lep", "MeT"]

label_col = "class"
data, scaler, cols = preprocess_df(data, particle_types, particle_features)
train_df, val_df = train_test_split(data, test_size=0.2, stratify=data['class'], shuffle=True, random_state=42)

class AEDataset(Dataset):
    def __init__(self, df, particle_cols, global_cols, num_particles=20, particle_feat_dim=6):
        """
        df: pandas DataFrame с данными
        particle_cols: список колонок с признаками частиц (в правильном порядке)
        global_cols: список колонок с глобальными признаками
        num_particles: число частиц в событии
        particle_feat_dim: число признаков на частицу
        """
        self.df = df.reset_index(drop=True)
        self.particle_cols = particle_cols
        self.global_cols = global_cols
        self.num_particles = num_particles
        self.particle_feat_dim = particle_feat_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        particles = torch.tensor(row[self.particle_cols].values.reshape(self.num_particles, self.particle_feat_dim), dtype=torch.float32)
        global_feats = torch.tensor(row[self.global_cols].values, dtype=torch.float32)
        return particles, global_feats

class Masker(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        weights = self.net(x).squeeze(-1)  # [B, N]
        return weights.unsqueeze(-1)       # [B, N, 1]

class AEEncoder(nn.Module):
    def __init__(self, num_particles=20, particle_feat_dim=6, global_feat_dim=5, emb_dim=64, num_heads=4):
        super().__init__()

        self.masker = Masker(particle_feat_dim)

        self.particle_proj = nn.Sequential(
            nn.Linear(particle_feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, emb_dim)
        )

        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)

        self.particle_pool = nn.AdaptiveAvgPool1d(1)

        self.global_net = nn.Sequential(
            nn.Linear(global_feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.combined_proj = nn.Linear(emb_dim * 2, emb_dim)

    def forward(self, particles, global_feats):
        weights = self.masker(particles)             # [B, N, 1]
        x = self.particle_proj(particles)            # [B, N, emb_dim]
        x = x * weights                              # soft masking

        attn_out, _ = self.attn(x, x, x)             # attention без key_padding_mask
        pooled_particles = self.particle_pool(attn_out.transpose(1, 2)).squeeze(-1)  # [B, emb_dim]

        global_emb = self.global_net(global_feats)   # [B, emb_dim]

        combined = torch.cat([pooled_particles, global_emb], dim=1)  # [B, emb_dim*2]

        emb = self.combined_proj(combined)           # [B, emb_dim]

        return emb, weights

class AEDecoder(nn.Module):
    def __init__(self, num_particles=20, particle_feat_dim=6, global_feat_dim=5, emb_dim=64):
        super().__init__()

        output_dim = num_particles * particle_feat_dim + global_feat_dim

        self.decoder = nn.Sequential(
            nn.Linear(emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

        self.num_particles = num_particles
        self.particle_feat_dim = particle_feat_dim
        self.global_feat_dim = global_feat_dim

    def forward(self, emb):
        out = self.decoder(emb)
        particles = out[:, :self.num_particles * self.particle_feat_dim]
        global_feats = out[:, self.num_particles * self.particle_feat_dim:]
        particles = particles.view(-1, self.num_particles, self.particle_feat_dim)
        return particles, global_feats

class Autoencoder(nn.Module):
    def __init__(self, num_particles=20, particle_feat_dim=6, global_feat_dim=5, emb_dim=64, num_heads=4):
        super().__init__()

        self.encoder = AEEncoder(num_particles, particle_feat_dim, global_feat_dim, emb_dim, num_heads)
        self.decoder = AEDecoder(num_particles, particle_feat_dim, global_feat_dim, emb_dim)

    def forward(self, particles, global_feats):
        emb, weights = self.encoder(particles, global_feats)
        particles_recon, global_recon = self.decoder(emb)
        return particles_recon, global_recon, emb, weights

train_dataset = AEDataset(train_df, particle_cols, global_cols, num_particles=20, particle_feat_dim=6)
val_dataset = AEDataset(val_df, particle_cols, global_cols, num_particles=20, particle_feat_dim=6)

def loss_fn(particles_recon, particles_true, global_recon, global_true):
    loss_particles = F.mse_loss(particles_recon, particles_true)
    loss_global = F.mse_loss(global_recon, global_true)
    return loss_particles + loss_global

def train_autoencoder(model, train_dataset, val_dataset, config):
    device = config["device"]
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=6,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=6,
        pin_memory=True
    )

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")

        for batch in loop:
            particles, global_feats = batch  # вот здесь батч
            particles = particles.to(device, non_blocking=True)
            global_feats = global_feats.to(device, non_blocking=True)

            optimizer.zero_grad()
            particles_recon, global_recon, emb, weights= model(particles, global_feats)
            loss = loss_fn(particles_recon, particles, global_recon, global_feats)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                particles, global_feats = batch
                particles = particles.to(device, non_blocking=True)
                global_feats = global_feats.to(device, non_blocking=True)

                particles_recon, global_recon, emb, weights = model(particles, global_feats)
                loss = loss_fn(particles_recon, particles, global_recon, global_feats)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"[{epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config["model_save_path"])
            print(">> Best model saved.")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config["loss_plot_path"])
    plt.close()

config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 1e-4,
    "batch_size": 256,
    "epochs": 30,
    "model_save_path": "best_ae_model_new_data_v3.pt",
    "loss_plot_path": "ae_atten_loss_lr_e-4_v3.png"
}
model = Autoencoder(
    num_particles=20,
    particle_feat_dim=6,
    global_feat_dim=5,
    emb_dim=64,
    num_heads=4
).to('cuda')

# weights_path = '/scratch/vasil/reconstruction/models/best_ae_model_new_data_v3.pt'
# state_dict = torch.load(weights_path)
# model.load_state_dict(state_dict)

train_autoencoder(model, train_dataset, val_dataset, config)