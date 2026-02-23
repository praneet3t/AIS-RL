import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ==========================================
# 1. HYPERPARAMS & CONFIG
# ==========================================
CSV_FOLDER_PATH = "./your_3k_csv_folder/*.csv" # UPDATE THIS
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Using device: {DEVICE}")

# Features we use to predict
FEATURE_COLS = [
    'Speed_Norm', 'Draft_Norm', 'Wave_H_Norm', 'Wave_P_Norm', 
    'COG_sin', 'COG_cos', 'Wave_Dir_sin', 'Wave_Dir_cos'
]

# ==========================================
# 2. FAST DATA LOADER & SHIFTER
# ==========================================
def load_and_prep_data(folder_path):
    all_x = []
    all_y = []
    
    file_list = glob.glob(folder_path)
    print(f"📂 Found {len(file_list)} files. Processing...")
    
    for file in tqdm(file_list):
        df = pd.read_csv(file)
        
        # Skip garbage files with less than 2 rows
        if len(df) < 2:
            continue
            
        # A. Trig Encodings (Angles to sin/cos)
        df['COG_sin'] = np.sin(np.radians(df['COG_Heading']))
        df['COG_cos'] = np.cos(np.radians(df['COG_Heading']))
        df['Wave_Dir_sin'] = np.sin(np.radians(df['Wave_Dir']))
        df['Wave_Dir_cos'] = np.cos(np.radians(df['Wave_Dir']))
        
        # B. The "Next Action" Shift
        # We want to predict the turn angle the human made to get to the NEXT state
        df['Target_Next_Turn_Angle'] = df['Turn_Angle'].shift(-1)
        
        # C. Drop the last row (it has no 'next' action) and any NaNs
        df = df.dropna(subset=['Target_Next_Turn_Angle', 'Wave_Height', 'Instant_Speed_Knots'])
        
        if len(df) == 0:
            continue
            
        # Keep raw features for normalization later
        features_raw = df[['Instant_Speed_Knots', 'Draft', 'Wave_Height', 'Wave_Period',
                           'COG_sin', 'COG_cos', 'Wave_Dir_sin', 'Wave_Dir_cos']].values
        targets = df['Target_Next_Turn_Angle'].values
        
        all_x.append(features_raw)
        all_y.append(targets)

    # Stack everything into giant numpy arrays
    X_raw = np.vstack(all_x)
    y = np.concatenate(all_y).reshape(-1, 1)
    
    print(f"✅ Extracted {len(X_raw)} total state-action pairs.")
    return X_raw, y

# ==========================================
# 3. NORMALIZATION & TENSORS
# ==========================================
X_raw, y = load_and_prep_data(CSV_FOLDER_PATH)

print("⚖️ Normalizing continuous features...")
scaler = StandardScaler()
# We only scale the non-trig continuous columns (first 4 columns)
X_continuous_scaled = scaler.fit_transform(X_raw[:, :4])
X_final = np.hstack((X_continuous_scaled, X_raw[:, 4:])) # Re-attach sin/cos

# Convert to PyTorch tensors and push to GPU if available
X_tensor = torch.tensor(X_final, dtype=torch.float32).to(DEVICE)
y_tensor = torch.tensor(y, dtype=torch.float32).to(DEVICE)

dataset = TensorDataset(X_tensor, y_tensor)
# pin_memory speeds up CPU to GPU transfer if using standard CPU tensors initially
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 4. THE BC MODEL (MLP BASELINE)
# ==========================================
class BehavioralCloningMLP(nn.Module):
    def __init__(self, input_dim):
        super(BehavioralCloningMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Output is a single value: Next_Turn_Angle
        )

    def forward(self, x):
        return self.net(x)

model = BehavioralCloningMLP(input_dim=X_final.shape[1]).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
criterion = nn.MSELoss() # Mean Squared Error for predicting a continuous turn angle



# ==========================================
# 5. FAST TRAINING LOOP
# ==========================================
print("\n🔥 Starting Fast Training on GPU...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS} | MSE Loss: {avg_loss:.4f}")

print("\n🎯 Training Complete! You have a baseline BC model.")
# torch.save(model.state_dict(), "bc_human_prior.pth")