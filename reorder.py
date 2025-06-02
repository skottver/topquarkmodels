import numpy as np
import pandas as pd
import uproot
import os

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi

def delta_r(eta1, phi1, eta2, phi2):
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)

def group_particles_by_top(row, particle_prefixes, gen_top_prefix='GenTop', features=['Pt', 'Eta', 'Phi', 'Px', 'Py', 'Pz']):
    n_tops = int(row.get(f'N_{gen_top_prefix}', 0))
    if n_tops == 0:
        # Нет топов — сортируем все частицы по Pt в убывании
        sorted_particles = sorted(particle_prefixes, key=lambda p: row.get(f'Pt_{p}', 0), reverse=True)
        return sorted_particles

    top_etas = [row.get(f'Eta_{gen_top_prefix}{i+1}', np.nan) for i in range(n_tops)]
    top_phis = [row.get(f'Phi_{gen_top_prefix}{i+1}', np.nan) for i in range(n_tops)]

    assignments = []
    for pfx in particle_prefixes:
        eta = row.get(f'Eta_{pfx}', np.nan)
        phi = row.get(f'Phi_{pfx}', np.nan)
        drs = [delta_r(eta, phi, top_etas[i], top_phis[i]) for i in range(n_tops)]
        closest_top = int(np.argmin(drs))
        assignments.append((pfx, closest_top, min(drs)))
particle_types = {
    'BJet': [1, 2, 3, 4],
    'Jet': list(range(1, 13)),
    'Lep': [1, 2, 3, 4],
}
gen_particle_types = {
    'GenBJet': [1, 2, 3, 4],
    'GenJet': list(range(1, 13)),
    'GenLep': [1, 2, 3, 4],
    'GenNu': [1, 2, 3, 4],
    'GenTop': [1, 2, 3, 4],
}

root_files = [
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/WW.Flz.part1.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/WW.Flz.part2.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part1.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part2.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part3.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part4.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part5.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_tbar_4f.leB.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_tbar_4f.leB.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_top_4f.qlh.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_top_4f.qlh.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part3.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part4.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_semilep.vRt.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_semilep.vRt.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTJ_UL16Summer20_RoO.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTW_UL16Summer20_CgT.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part3.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part4.root',
]
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

process_and_save_root_files(root_files, particle_types, gen_particle_types, output_dir)



    groups = {i: [] for i in range(n_tops)}
    for pfx, top_idx, dr_val in assignments:
        groups[top_idx].append((pfx, dr_val))

    for i in groups:
        groups[i].sort(key=lambda x: x[1])  # сортируем по dR внутри топа

    ordered = []
    for top_idx in range(n_tops):
        ordered.extend([p[0] for p in groups[top_idx]])

    return ordered

def reorder_row_inplace(row, particle_types, gen_top_prefix='GenTop', features=['Pt', 'Eta', 'Phi', 'Px', 'Py', 'Pz']):
    for ptype, indices in particle_types.items():
        particle_prefixes = [f'{ptype}{i}' for i in indices]
        ordered_particles = group_particles_by_top(row, particle_prefixes, gen_top_prefix, features=features)

        new_values = {idx: {feat: 0.0 for feat in features} for idx in range(1, len(indices)+1)}

        for i, old_pfx in enumerate(ordered_particles):
            if i+1 > len(indices):
                # Если частиц больше, чем предусмотрено в индексе, игнорируем лишние
                break
            new_idx = i + 1
            for feat in features:
                old_col = f"{feat}_{old_pfx}"
                if old_col in row:
                    new_values[new_idx][feat] = row[old_col]

        for idx in range(1, len(indices)+1):
            for feat in features:
                col = f"{feat}_{ptype}{idx}"
                row[col] = new_values[idx][feat]

    return row

def reorder_row_inplace_gen(row, gen_particle_types, gen_top_prefix='GenTop', features=['Pt', 'Eta', 'Phi', 'Px', 'Py', 'Pz']):
    for ptype, indices in gen_particle_types.items():
        particle_prefixes = [f'{ptype}{i}' for i in indices]  # ptype уже с 'Gen' в названии
        ordered_particles = group_particles_by_top(row, particle_prefixes, gen_top_prefix, features=features)

        new_values = {idx: {feat: 0.0 for feat in features} for idx in range(1, len(indices)+1)}

        for i, old_pfx in enumerate(ordered_particles):
            if i+1 > len(indices):
                break
            new_idx = i + 1
            for feat in features:
                old_col = f"{feat}_{old_pfx}"
                if old_col in row:
                    new_values[new_idx][feat] = row[old_col]

        for idx in range(1, len(indices)+1):
            for feat in features:
                col = f"{feat}_{ptype}{idx}"
                row[col] = new_values[idx][feat]

    return row

def reorder_dataframe_rows_inplace(df, particle_types, gen_particle_types, gen_top_prefix='GenTop', features=['Pt', 'Eta', 'Phi', 'Px', 'Py', 'Pz']):
    df = df.copy()
    df = df.apply(lambda row: reorder_row_inplace(row, particle_types, gen_top_prefix, features), axis=1)
    df = df.apply(lambda row: reorder_row_inplace_gen(row, gen_particle_types, gen_top_prefix, features), axis=1)
    return df

def process_and_save_root_files(root_files, particle_types, gen_particle_types, output_dir):
    for filepath in root_files:
        with uproot.open(filepath) as file:
            tree_name = 'LHE'
            tree = file[tree_name]
            df = tree.arrays(library="pd")
        df = df.loc[:, ~df.columns.str.startswith('SP_')]
        print(df.shape)
        df_reordered = reorder_dataframe_rows_inplace(df, particle_types, gen_particle_types)
        df_reordered = df_reordered.reset_index(drop=True)
        print(df_reordered.shape)
        filename = os.path.basename(filepath)
        output_path = os.path.join(output_dir, filename)
        with uproot.recreate(output_path) as f:
            f[tree_name] = df_reordered.to_dict(orient="list")
        print(f"Processed and saved: {output_path}")


particle_types = {
    'BJet': [1, 2, 3, 4],
    'Jet': list(range(1, 13)),
    'Lep': [1, 2, 3, 4],
}
gen_particle_types = {
    'GenBJet': [1, 2, 3, 4],
    'GenJet': list(range(1, 13)),
    'GenLep': [1, 2, 3, 4],
    'GenNu': [1, 2, 3, 4],
    'GenTop': [1, 2, 3, 4],
}

root_files = [
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/WW.Flz.part1.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/WW.Flz.part2.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part1.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part2.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part3.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part4.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/Wjets-incl.INb.part5.root',
    '/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_tbar_4f.leB.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_tbar_4f.leB.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_top_4f.qlh.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/t-channel_top_4f.qlh.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part3.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_doublelep.KQP.part4.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_semilep.vRt.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/ttbar_semilep.vRt.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTJ_UL16Summer20_RoO.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTW_UL16Summer20_CgT.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part1.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part2.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part3.root',
'/scratch/pvolkov/DM/samples/MultiLepton/grid_gen/TTTT_UL16Summer20_eIE.part4.root',
]
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

process_and_save_root_files(root_files, particle_types, gen_particle_types, output_dir)


