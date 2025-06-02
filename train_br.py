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
from tqdm.auto import tqdm
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
df = data


def preprocess_df(df, particle_types, features,
                         gen_particle_types, gen_features,
                         scaler=None, fit_scaler=True):
    df = df.copy()

   
  

    
    cols_to_scale = []
    for ptype, indices in particle_types.items():
        for idx in indices:
            for feat in features:
                col = f'{feat}_{ptype}{idx}'
                if col in df.columns:
                    cols_to_scale.append(col)


    for ptype, indices in gen_particle_types.items():
        for idx in indices:
            for feat in gen_features:
                col = f'{feat}_{ptype}{idx}'
                if col in df.columns:
                    cols_to_scale.append(col)

    
    global_num_cols = [col for col in df.columns if col.startswith('N_') or col == 'MeT']
    cols_to_scale += global_num_cols

    # Масштабируем
    if scaler is None:
        scaler = StandardScaler()
    if fit_scaler:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    else:
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler, cols_to_scale



class ParticleMasker(nn.Module):
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
        mask = self.net(x).squeeze(-1)
        return mask

class SelfAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, emb_dim, num_heads):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 4),
            nn.GELU(),
            nn.Linear(emb_dim * 4, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)
    def forward(self, query, key, value, key_padding_mask=None):
        attn_out, _ = self.cross_attn(query, key, value, key_padding_mask=key_padding_mask)
        x = self.norm1(query + attn_out)
        ff_out = self.ff(x)
        out = self.norm2(x + ff_out)
        return out

class BasicRegression(nn.Module):
    def __init__(self,
                 particle_feat_dim=6,
                 num_particles=20,
                 emb_dim=64,
                 num_heads=4,
                 num_gen_tops=4,
                 num_gen_nus=4,
                 input_global_dim=5,    # конечные глобальные признаки (вход)
                 target_global_dim=2):  # генераторные глобальные признаки (таргет)
        super().__init__()
        self.particle_feat_dim = particle_feat_dim
        self.num_particles = num_particles
        self.emb_dim = emb_dim

        self.masker = ParticleMasker(particle_feat_dim)

        self.particle_processor = nn.Sequential(
            nn.Linear(particle_feat_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        self.particle_self_attn = SelfAttentionBlock(emb_dim, num_heads)

        self.gen_top_embedding = nn.Parameter(torch.randn(1, num_gen_tops, emb_dim))

        self.cross_attn = CrossAttentionBlock(emb_dim, num_heads)

        self.global_processor = nn.Sequential(
            nn.Linear(input_global_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim)
        )

        output_dim = 6 * num_gen_tops + 6 * num_gen_nus + target_global_dim
        self.regressor = nn.Sequential(
            nn.Linear(emb_dim * num_gen_tops + emb_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, particles, global_feats):
        mask = self.masker(particles)
        key_padding_mask = (mask < 0.5).squeeze(-1)

        x = self.particle_processor(particles)
        x = self.particle_self_attn(x, key_padding_mask=key_padding_mask)

        batch_size = particles.size(0)
        gen_top_emb = self.gen_top_embedding.expand(batch_size, -1, -1)

        top_rep = self.cross_attn(gen_top_emb, x, x, key_padding_mask=key_padding_mask)

        top_rep_pooled = top_rep.reshape(batch_size, -1)

        global_emb = self.global_processor(global_feats)
        combined = torch.cat([top_rep_pooled, global_emb], dim=1)

        output = self.regressor(combined)

        return output


# Исходные данные: словари с типами частиц и их номерами
particle_types = {
    'BJet': [1, 2, 3, 4],
    'Jet': list(range(1, 13)),
    'Lep': [1, 2, 3, 4],
}

gen_particle_types = {
    'GenTop': [1, 2, 3, 4],
    'GenNu': [1, 2, 3, 4],
}

features = ['Pt', 'Eta', 'Phi', 'Px', 'Py', 'Pz']

# 1. Формируем список колонок для входа (конечные частицы)
input_particle_cols = []
for ptype, indices in particle_types.items():
    for idx in indices:
        for feat in features:
            col = f"{feat}_{ptype}{idx}"
            input_particle_cols.append(col)

# 2. Глобальные входные признаки
input_global_cols = ["N_J", "N_BJ", "N_Nu", "N_Lep", "MeT"]

# 3. Целевые колонки — только для GenTop и GenNu
target_cols = []
for ptype, indices in gen_particle_types.items():
    for idx in indices:
        for feat in features:
            col = f"{feat}_{ptype}{idx}"
            target_cols.append(col)

# Добавим целевые глобальные переменные
target_cols += ['N_GenTop', 'N_GenNu']
df, scaler, cols = preprocess_df(df, particle_types, features,
                         gen_particle_types, features,
                         scaler=None, fit_scaler=True)

class HEPEvtDataset(Dataset):
    def __init__(self, df, input_particle_cols, input_global_cols, target_cols, num_particles=20, particle_feat_dim=6):
        """
        df: pandas DataFrame с данными
        input_particle_cols: список колонок для признаков частиц (в правильном порядке)
        input_global_cols: список колонок глобальных переменных (например, N_GenTop, N_GenNu)
        target_cols: список колонок с целевыми генераторными признаками
        """
        self.df = df.reset_index(drop=True)
        self.input_particle_cols = input_particle_cols
        self.input_global_cols = input_global_cols
        self.target_cols = target_cols
        self.num_particles = num_particles
        self.particle_feat_dim = particle_feat_dim

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        particles = torch.tensor(row[self.input_particle_cols].values.reshape(self.num_particles, self.particle_feat_dim), dtype=torch.float32)
        global_feats = torch.tensor(row[self.input_global_cols].values, dtype=torch.float32)
        targets = torch.tensor(row[self.target_cols].values, dtype=torch.float32)
        return particles, global_feats, targets

def train_model(model, train_dataset, val_dataset, config):
   
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    epochs = config.get("epochs", 20)
    batch_size = config.get("batch_size", 64)
    lr = config.get("lr", 1e-3)
    num_workers = config.get("num_workers", 4)
    save_model_path = config.get("save_model_path", None)
    save_plot_path = config.get("save_plot_path", None)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False)
        for particles, global_feats, targets in progress_bar:
            particles = particles.to(device)
            global_feats = global_feats.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(particles, global_feats)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * particles.size(0)
            avg_loss = running_loss / ((progress_bar.n + 1) * particles.size(0))
            progress_bar.set_postfix(loss=avg_loss)

        epoch_train_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Val]  ", leave=False)
            for particles, global_feats, targets in progress_bar_val:
                particles = particles.to(device)
                global_feats = global_feats.to(device)
                targets = targets.to(device)

                outputs = model(particles, global_feats)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item() * particles.size(0)

                avg_val_loss = running_val_loss / ((progress_bar_val.n + 1) * particles.size(0))
                progress_bar_val.set_postfix(val_loss=avg_val_loss)

        epoch_val_loss = running_val_loss / len(val_dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch}/{epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Сохраняем модель, если валид. лосс улучшился
        if save_model_path is not None and epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), save_model_path)
            print(f"Model saved to {save_model_path}")

    # Построение графика потерь
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    if save_plot_path is not None:
        plt.savefig(save_plot_path)
        print(f"Loss plot saved to {save_plot_path}")

    plt.show()

    return model

config = {
        "epochs": 30,
        "batch_size": 512,
        "lr": 1e-4,
        "device": "cuda",
        "num_workers": 4,
        "save_model_path": './models/best_br_model_v2.pth',
        "save_plot_path": './plots/loss_br_e-4_v2.png'
    }
model = BasicRegression()
model.to('cuda')


df_train, df_val = train_test_split(df.drop(columns= 'class'), test_size=0.2, stratify=df['class'], shuffle=True, )


num_particles = 20       # число конечных частиц
particle_feat_dim = 6    # число признаков на частицу
global_feat_dim = 2      # N_GenTop и N_GenNu
num_gen_tops = 4
num_gen_nus = 4

train_dataset = HEPEvtDataset(df_train, input_particle_cols, input_global_cols, target_cols,
                             num_particles=num_particles, particle_feat_dim=particle_feat_dim)
val_dataset = HEPEvtDataset(df_val, input_particle_cols, input_global_cols, target_cols,
                           num_particles=num_particles, particle_feat_dim=particle_feat_dim)


model = BasicRegression()


trained_model = train_model(model, train_dataset, val_dataset, config)