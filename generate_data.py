import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
from observations import DummyObs

def generate_circular_trajectory(n_steps=100, radius=0.6, height=0.7):
    """Generate a circular trajectory in 3D space with rotation"""
    # Use 3 full rotations to make the trajectory longer
    t = np.linspace(0, 6*np.pi, n_steps)  # Changed from 2π to 6π for 3 rotations
    
    # Position trajectory (circular in xy-plane with larger radius)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height * np.ones_like(t)
    
    # Simple rotation trajectory (rotate around z-axis)
    rx = np.zeros_like(t)
    ry = np.zeros_like(t)
    rz = t/3  # Slower rotation (divided by 3 since we're doing 3 rotations)
    
    # Combine into 6D trajectory
    actions = np.stack([x, y, z, rx, ry, rz], axis=1)
    return actions

def create_synthetic_dataset(n_trajectories=1000, n_steps=200, out_path='data/synthetic'):
    """Create a synthetic dataset of circular trajectories"""
    os.makedirs(out_path, exist_ok=True)
    
    out_buffer = []
    for _ in tqdm(range(n_trajectories)):
        # Generate trajectory
        actions = generate_circular_trajectory(n_steps)
        
        # Create trajectory
        traj = []
        prev_obs = None
        for t in range(n_steps):
            # Create observation with proper image format
            obs = DummyObs()
            obs.prev = prev_obs  # Link to previous observation
            
            # Get current action
            action = actions[t].astype(np.float32)
            
            # Dummy reward
            reward = 0.0
            
            traj.append((obs, action, reward))
            prev_obs = obs
        
        out_buffer.append(traj)
    
    # Save dataset
    with open(os.path.join(out_path, 'buf.pkl'), 'wb') as f:
        pkl.dump(out_buffer, f)
    print(f"Dataset saved to {out_path}/buf.pkl")

if __name__ == "__main__":
    # Generate dataset
    create_synthetic_dataset(
        n_trajectories=70,  # number of trajectories
        n_steps=200,         # steps per trajectory
        out_path='data/synthetic_circular_long'
    )