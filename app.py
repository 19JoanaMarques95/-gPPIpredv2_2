import os
import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATv2Conv, global_mean_pool

# --- 1. CONFIGURATION & PHYSICS ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AA_PHYSICS_5D_SCALED = {
    'A': [0.70, 0.17, 0.11, 0.40, 0.00], 'R': [0.00, 0.67, 0.71, 1.00, 1.00],
    'N': [0.11, 0.32, 0.33, 0.33, 0.00], 'D': [0.11, 0.30, 0.26, 0.00, 0.31],
    'C': [0.78, 0.29, 0.31, 0.29, 0.67], 'Q': [0.11, 0.50, 0.44, 0.36, 0.00],
    'E': [0.11, 0.47, 0.37, 0.06, 0.34], 'G': [0.46, 0.00, 0.00, 0.40, 0.00],
    'H': [0.14, 0.55, 0.56, 0.60, 0.48], 'I': [1.00, 0.64, 0.45, 0.41, 0.00],
    'L': [0.92, 0.64, 0.45, 0.40, 0.00], 'K': [0.07, 0.65, 0.54, 0.87, 0.84],
    'M': [0.66, 0.61, 0.54, 0.37, 0.00], 'F': [0.81, 0.77, 0.72, 0.34, 0.00],
    'P': [0.32, 0.31, 0.32, 0.44, 0.00], 'S': [0.41, 0.17, 0.15, 0.36, 0.00],
    'T': [0.42, 0.33, 0.26, 0.35, 0.00], 'W': [0.40, 1.00, 1.00, 0.39, 0.00],
    'Y': [0.36, 0.79, 0.73, 0.36, 0.81], 'V': [0.97, 0.48, 0.34, 0.40, 0.00],
    'X': [0.00, 0.00, 0.00, 0.00, 0.00]
}

# --- 2. ARCHITECTURE ---
class SiameseGAT_v3(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=64, num_layers=8):
        super(SiameseGAT_v3, self).__init__()
        self.node_lin = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GATv2Conv(hidden_channels, hidden_channels // 4, heads=4))
            self.batch_norms.append(nn.BatchNorm1d(hidden_channels))
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_channels * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward_once(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_lin(x)
        for conv, bn in zip(self.convs, self.batch_norms):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.elu(h)
            x = x + h 
        return global_mean_pool(x, batch)

    def forward(self, g1, g2):
        out1 = self.forward_once(g1)
        out2 = self.forward_once(g2)
        combined = torch.cat([out1, out2], dim=1)
        return self.fc(combined)

# --- 3. HOTSPOT LOGIC ---
def get_hotspots(model, g_bait, g_prey, bait_seq, prey_seq, top_n=10):
    g_bait.x.requires_grad = True
    g_prey.x.requires_grad = True
    logits = model(Batch.from_data_list([g_bait]), Batch.from_data_list([g_prey]))
    model.zero_grad()
    logits.backward()
    
    def extract_top(grad, seq):
        importance = grad.abs().sum(dim=1).cpu().numpy()
        indices = np.argsort(importance)[-top_n:][::-1]
        return ", ".join([f"{seq[i]}{i+1}" for i in indices if i < len(seq)])

    return extract_top(g_bait.x.grad, bait_seq), extract_top(g_prey.x.grad, prey_seq)

# --- 4. DATA UTILS ---
CACHE_DIR = "protein_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_protein_data(uniprot_id):
    uniprot_id = uniprot_id.strip().upper()
    cache_path = os.path.join(CACHE_DIR, f"{uniprot_id}.pt")
    if os.path.exists(cache_path): return torch.load(cache_path)
    
    # Fetch FASTA for sequence
    res_fasta = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta")
    # Fetch JSON for metadata (Protein Name)
    res_json = requests.get(f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json")
    
    if res_fasta.status_code != 200: raise ValueError(f"ID {uniprot_id} not found.")
    
    seq = "".join(res_fasta.text.split("\n")[1:])
    
    # Extract Protein Name
    p_name = "Unknown Protein"
    if res_json.status_code == 200:
        json_data = res_json.json()
        p_name = json_data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', "Unknown")

    L = len(seq)
    adj = np.eye(L)
    for i in range(L - 1):
        adj[i, i+1] = 1; adj[i+1, i] = 1
    edge_index = torch.from_numpy(adj).nonzero().t().contiguous().long()
    
    x = torch.tensor([AA_PHYSICS_5D_SCALED.get(a, AA_PHYSICS_5D_SCALED['X']) for a in seq], dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index)
    
    package = (data, seq, p_name)
    torch.save(package, cache_path)
    return package

# --- 5. PREDICTION CORE ---
def run_prediction(bait_id, prey_raw, threshold):
    try:
        prey_ids = [p.strip().upper() for p in prey_raw.replace(',', ' ').split() if p.strip()]
        g_bait, bait_seq, bait_name = get_protein_data(bait_id)
        
        results = []
        for p_id in prey_ids:
            g_prey, prey_seq, prey_name = get_protein_data(p_id)
            
            model.eval()
            with torch.no_grad():
                logits = model(Batch.from_data_list([g_bait.to(DEVICE)]), 
                               Batch.from_data_list([g_prey.to(DEVICE)]))
                prob = torch.sigmoid(logits).item()
            
            b_hot, p_hot = get_hotspots(model, g_bait.to(DEVICE), g_prey.to(DEVICE), bait_seq, prey_seq)
            
            results.append({
                "Bait ID": bait_id,
                "Bait Name": bait_name,
                "Prey ID": p_id,
                "Prey Name": prey_name,
                "Probability": round(prob, 4),
                "Binds": "Yes" if prob >= threshold else "No",
                "Bait Hotspots": b_hot,
                "Prey Hotspots": p_hot,
                "Bait Sequence": bait_seq,
                "Prey Sequence": prey_seq
            })
            torch.cuda.empty_cache()

        df = pd.DataFrame(results)
        
        # Plotting logic
        plt.figure(figsize=(8, max(4, 0.5 * len(df))))
        sns.barplot(data=df, x='Probability', y='Prey ID', palette='viridis')
        plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold {threshold}')
        plt.title(f"PPI Analysis for Bait: {bait_id}")
        plt.xlim(0, 1)
        plt.legend()
        plt.tight_layout()
        
        tmp_csv = os.path.join(tempfile.gettempdir(), "results.csv")
        df.to_csv(tmp_csv, index=False)
        return plt.gcf(), df, tmp_csv, f"✅ Analyzed {len(df)} preys."
    except Exception as e:
        return None, None, None, f"❌ Error: {str(e)}"

# --- 6. INITIALIZATION & UI ---
model = SiameseGAT_v3(in_channels=5).to(DEVICE)
MODEL_PATH = "GATv3_FINAL_CORRECTED.pth"
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))

custom_theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Inter"), "sans-serif"])

with gr.Blocks() as demo:
    gr.Markdown("# 🧬 gPPIpred v3 – Advanced Siamese GAT Predictor")
    
    with gr.Row():
        with gr.Column(scale=2):
            b_in = gr.Textbox(label="Bait ID (UniProt)", value="P04637")
            p_in = gr.Textbox(label="Prey IDs (comma separated)", value="O15169, P00519")
        with gr.Column(scale=1):
            t_slider = gr.Slider(0.1, 0.95, value=0.90, step=0.01, label="Sensitivity Threshold")
            btn = gr.Button("🔍 Predict", variant="primary")
            status = gr.Textbox(label="Status")
            
    with gr.Tabs():
        with gr.TabItem("Chart"): plot = gr.Plot()
        with gr.TabItem("Data"): 
            table = gr.Dataframe()
            csv_file = gr.File(label="Download CSV")
            
    btn.click(run_prediction, inputs=[b_in, p_in, t_slider], outputs=[plot, table, csv_file, status])

if __name__ == "__main__":
    demo.launch(theme=custom_theme)
