import numpy as np
import json

print("Loading grid data for IRL math...")
data = np.load("irl_grid_data.npz")
phi_s = data['phi_s']
phi_E = data['phi_E']
bounds = data['bounds']
GRID_H, GRID_W, FEAT_DIM = phi_s.shape

# Hyperparameters
EPOCHS = 500
LEARNING_RATE = 0.05
L2_REG = 0.01 
weights = np.zeros(FEAT_DIM)

# Adam params
m, v = np.zeros(FEAT_DIM), np.zeros(FEAT_DIM)
beta1, beta2, epsilon = 0.9, 0.999, 1e-8

print("Starting IRL Gradient Descent...")

for epoch in range(EPOCHS):
    reward_grid = np.einsum('ijk,k->ij', phi_s, weights)
    
    # Softmax probabilities (Boltzmann)
    exp_r = np.exp(reward_grid - np.max(reward_grid))
    p_s = exp_r / np.sum(exp_r)
    
    # Model expectations
    phi_model = np.einsum('ij,ijk->k', p_s, phi_s)
    
    # Gradient: Expert - Model - Regularization
    gradient = phi_E - phi_model - (L2_REG * weights)
    
    # Adam Update
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** (epoch + 1))
    v_hat = v / (1 - beta2 ** (epoch + 1))
    
    weights += LEARNING_RATE * m_hat / (np.sqrt(v_hat) + epsilon)
    
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:03d} | Grad Norm: {np.linalg.norm(gradient):.4f} | Weights: {np.round(weights, 3)}")

# Generate and save the final cost matrix
final_reward_grid = np.einsum('ijk,k->ij', phi_s, weights)
np.save("final_reward_grid.npy", final_reward_grid)

# Save metadata for the router
metadata = {
    "weights": {"wave_h": float(weights[0]), "wave_p": float(weights[1]), "traffic": float(weights[2])},
    "bounds": {"min_lat": float(bounds[0]), "max_lat": float(bounds[1]), "min_lon": float(bounds[2]), "max_lon": float(bounds[3])}
}
with open("prior_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("Saved 'final_reward_grid.npy' and 'prior_metadata.json'.")