import numpy as np
import pickle as pkl
import os
from tqdm import tqdm

def generate_circular_trajectory(n_steps=100, radius=0.3, height=0.5):
    """Generate a circular trajectory in 3D space with rotation"""
    t = np.linspace(0, 2*np.pi, n_steps)
    
    # Position trajectory (circular in xy-plane)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = height * np.ones_like(t)
    
    # Simple rotation trajectory (rotate around z-axis)
    rx = np.zeros_like(t)
    ry = np.zeros_like(t)
    rz = t  # rotate from 0 to 2π
    
    # Combine into 6D trajectory
    actions = np.stack([x, y, z, rx, ry, rz], axis=1)
    return actions

def create_synthetic_dataset(n_trajectories=1000, n_steps=100, out_path='data/synthetic'):
    """Create a synthetic dataset of circular trajectories"""
    os.makedirs(out_path, exist_ok=True)
    
    out_buffer = []
    for _ in tqdm(range(n_trajectories)):
        # Generate trajectory
        actions = generate_circular_trajectory(n_steps)
        
        # Create trajectory
        traj = []
        for t in range(n_steps):
            # Create dummy state and image
            obs = {
                'state': np.zeros(7, dtype=np.float32),  # dummy robot state
                'enc_cam_0': np.zeros((256, 256, 3), dtype=np.uint8)  # dummy encoded image
            }
            
            # Get current action
            action = actions[t].astype(np.float32)
            
            # Dummy reward
            reward = 0.0
            
            traj.append((obs, action, reward))
        
        out_buffer.append(traj)
    
    # Save dataset
    with open(os.path.join(out_path, 'buf.pkl'), 'wb') as f:
        pkl.dump(out_buffer, f)
    print(f"Dataset saved to {out_path}/buf.pkl")

if __name__ == "__main__":
    # Generate dataset
    create_synthetic_dataset(
        n_trajectories=1,  # number of trajectories
        n_steps=100,         # steps per trajectory
        out_path='data/synthetic_circular'
    )