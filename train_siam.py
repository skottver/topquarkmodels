import uproot
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
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

#paths to samples
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
        df = df.loc[:, ~df.columns.str.startswith('SP_')]
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
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_df(df, particle_types, features, scaler=None, fit_scaler=True):
    df = df.copy()

    
    for ptype, indices in particle_types.items():
        for idx in indices:
            phi_col = f'Phi_{ptype}{idx}'
            if phi_col in df.columns:
                df[phi_col] = np.cos(df[phi_col].values)

    
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



df = data

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
df, scaler, cols = preprocess_df(df, particle_types, particle_features)
train_df, val_df = train_test_split(df, test_size=0.2, shuffle=True, stratify=df['class'], random_state=42)
class TripletHEPDataset(Dataset):
    def __init__(self, df, label_col, particle_cols, global_cols, num_particles=20, particle_feat_dim=6):
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.particle_cols = particle_cols
        self.global_cols = global_cols
        self.num_particles = num_particles
        self.particle_feat_dim = particle_feat_dim

        # Предварительно преобразуем данные в numpy для скорости
        self.particles_data = self.df[self.particle_cols].values.reshape(-1, num_particles, particle_feat_dim)
        self.global_data = self.df[self.global_cols].values
        self.labels = self.df[self.label_col].values

        # Создаём словарь: класс -> индексы строк
        self.label_to_indices = {}
        for label in np.unique(self.labels):
            self.label_to_indices[label] = np.where(self.labels == label)[0]

    def __len__(self):
        return len(self.df)

    def get_triplet(self, anchor_idx):
        anchor_label = self.labels[anchor_idx]
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice(positive_indices[positive_indices != anchor_idx])

        negative_labels = list(self.label_to_indices.keys())
        negative_labels.remove(anchor_label)
        negative_label = np.random.choice(negative_labels)
        negative_idx = np.random.choice(self.label_to_indices[negative_label])

        return anchor_idx, positive_idx, negative_idx

    def __getitem__(self, idx):
        a_idx, p_idx, n_idx = self.get_triplet(idx)

        def extract(i):
            particles = torch.tensor(self.particles_data[i], dtype=torch.float32)
            global_feats = torch.tensor(self.global_data[i], dtype=torch.float32)
            return particles, global_feats

        anchor = extract(a_idx)
        positive = extract(p_idx)
        negative = extract(n_idx)

        return anchor, positive, negative


class ParticleMasker(nn.Module):
    def __init__(self, particle_input_dim=6):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(particle_input_dim, particle_input_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1)
        )
        
    def forward(self, particles):
        # particles: [batch, num_particles, 6]
        mask = (particles.abs().sum(dim=-1, keepdim=True) != 0).float()  # [B, N, 1]
        projected = self.proj(particles * mask)
        return {
            "masked_particles": projected,
            "mask": mask  # [B, N, 1]
        }

class GlobalProcessor(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
    def forward(self, global_feats):
        return self.net(global_feats)

class ParticleMasker(nn.Module):
    def __init__(self, particle_input_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(particle_input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, particles):
        # particles: [B, N, 6]
        mask = self.net(particles).squeeze(-1)  # [B, N]
        return mask.unsqueeze(-1)                # [B, N, 1]

class ParticleProcessor(nn.Module):
    def __init__(self, particle_input_dim=6, emb_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(particle_input_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, particles):
        # particles: [B, N, 6]
        B, N, _ = particles.shape
        x = particles.view(B * N, -1)
        x = self.net(x)
        x = x.view(B, N, -1)
        return x  # [B, N, emb_dim]

class GlobalProcessor(nn.Module):
    def __init__(self, input_dim=5, emb_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    def forward(self, global_feats):
        # global_feats: [B, input_dim]
        return self.net(global_feats)  # [B, emb_dim]

class EventEncoder(nn.Module):
    def __init__(self, num_particles=20, particle_dim=6, global_dim=5, emb_dim=64, num_heads=4):
        super().__init__()
        self.particle_masker = ParticleMasker(particle_dim)
        self.particle_processor = ParticleProcessor(particle_dim, emb_dim=32)
        self.particle_attn = nn.MultiheadAttention(embed_dim=32, num_heads=num_heads, batch_first=True)
        self.global_processor = GlobalProcessor(global_dim, emb_dim=16)
        self.final_proj = nn.Linear(32 + 16, emb_dim)

    def forward(self, x):
        particles = x["particles"]            # [B, N, 6]
        global_feats = x["global"]            # [B, global_dim]

        weights = self.particle_masker(particles)         # [B, N, 1], soft mask
        processed_particles = self.particle_processor(particles)  # [B, N, 32]
        masked_particles = processed_particles * weights          # soft masking

        attn_out, _ = self.particle_attn(masked_particles, masked_particles, masked_particles)
        # Внимание по всем частицам

        pooled_particles = (attn_out * weights).sum(dim=1) / (weights.sum(dim=1) + 1e-6)  # [B, 32]

        global_emb = self.global_processor(global_feats)  # [B, 16]

        combined = torch.cat([pooled_particles, global_emb], dim=1)  # [B, 48]

        emb = F.normalize(self.final_proj(combined), dim=-1)  # [B, emb_dim]

        return emb


def triplet_loss(anchor, positive, negative, margin=1.0):
    d_ap = F.pairwise_distance(anchor, positive)
    d_an = F.pairwise_distance(anchor, negative)
    return F.relu(d_ap - d_an + margin).mean()

def train_encoder_model(model, train_dataset, val_dataset, config):
    device = config["device"]
    model = model.to(device)
    print(f'model is located on {next(model.parameters()).device}')
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=6, pin_memory=6)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=6, pin_memory=6)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(config["epochs"]):
        model.train()
        total_loss = 0
        for (pa, ga), (pp, gp), (pn, gn) in tqdm(train_loader):
            pa, ga = pa.to(device), ga.to(device)
            pp, gp = pp.to(device), gp.to(device)
            pn, gn = pn.to(device), gn.to(device)

            anchor = model({"particles": pa, "global": ga})
            positive = model({"particles": pp, "global": gp})
            negative = model({"particles": pn, "global": gn})

            loss = triplet_loss(anchor, positive, negative, margin=config["margin"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for (pa, ga), (pp, gp), (pn, gn) in val_loader:
                pa, ga = pa.to(device), ga.to(device)
                pp, gp = pp.to(device), gp.to(device)
                pn, gn = pn.to(device), gn.to(device)

                anchor = model({"particles": pa, "global": ga})
                positive = model({"particles": pp, "global": gp})
                negative = model({"particles": pn, "global": gn})
                loss = triplet_loss(anchor, positive, negative, margin=config["margin"])
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
    plt.ylabel("Triplet Loss")
    plt.title("Encoder Training Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(config["loss_plot_path"])
    plt.close()

encoder = EventEncoder(
    num_particles=20,
    particle_dim=6,
    global_dim=5,
    emb_dim=64
)
weights_path = 'siam_encoder_new.pt'
state_dict = torch.load(weights_path)
encoder.load_state_dict(state_dict)
train_dataset = TripletHEPDataset(train_df, label_col, particle_cols, global_cols, num_particles=20)
val_dataset = TripletHEPDataset(val_df, label_col, particle_cols, global_cols, num_particles=20)
config = {
    'device': 'cuda',
    'lr': 1e-4,
    'batch_size': 256,
    'epochs': 20,
    'margin': 1.0,
    'num_workers': 6,
    'model_save_path': 'siam_encoder_new.pt',
    'loss_plot_path': 'triplet_loss_new_lr_1e-4.png'
}
train_encoder_model(encoder, train_dataset, val_dataset, config)