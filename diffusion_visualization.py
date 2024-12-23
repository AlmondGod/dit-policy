import argparse
import numpy as np
import genesis as gs
from time import time, sleep
import torch
from data4robotics.models.diffusion import DiffusionTransformerAgent
from data4robotics import load_resnet18, load_vit

class DiffusionVisualizer:
    def __init__(self, model_path, device='cuda'):
        # Load pretrained vision model first
        self.transform, self.vision_model = load_resnet18()  # or load_vit() depending on your model
        self.vision_model.to(device)
        self.vision_model.eval()
        
        # Load pretrained diffusion model
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model with checkpoint config
        model_kwargs = checkpoint['model_kwargs']
        # Override the features with our pretrained vision model
        model_kwargs['features'] = self.vision_model
        
        self.model = DiffusionTransformerAgent(**model_kwargs)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(device)
        self.model.eval()
        
        # Set up diffusion parameters
        self.model._eval_diffusion_steps = 100  # We want to visualize all steps
        self.diffusion_schedule = self.model.diffusion_schedule
        self.noise_net = self.model.noise_net

    def run_diffusion_step(self, obs_dict, noise_actions, timestep):
        """Run a single step of the diffusion process"""
        with torch.no_grad():
            # Transform images using the pretrained model's transform
            if 'images' in obs_dict:
                obs_dict['images'] = self.transform(obs_dict['images'])
            
            B = noise_actions.shape[0]
            s_t = self.model.tokenize_obs(obs_dict['images'], obs_dict['states'])
            
            # For transformer model, we need encoder cache
            if not hasattr(self, 'enc_cache'):
                self.enc_cache = self.noise_net.forward_enc(s_t)
            
            # Predict noise and take diffusion step
            batched_timestep = timestep.unsqueeze(0).repeat(B).to(self.device)
            noise_pred = self.noise_net.forward_dec(
                noise_actions, batched_timestep, self.enc_cache
            )
            
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred,
                timestep=timestep,
                sample=noise_actions
            ).prev_sample
            
            return noise_actions

def run_sim(scene, visualizer, obs_dict):
    n_particles = 100  # Should match your action sequence length
    n_steps = visualizer.model._train_diffusion_steps
    
    # Initialize noise
    B = 1  # batch size
    noise_actions = torch.randn(
        B, visualizer.model.ac_chunk, visualizer.model.ac_dim, 
        device=visualizer.device
    )
    
    # Setup diffusion schedule
    visualizer.diffusion_schedule.set_timesteps(n_steps)
    visualizer.diffusion_schedule.alphas_cumprod = (
        visualizer.diffusion_schedule.alphas_cumprod.to(visualizer.device)
    )
    
    t_prev = time()
    for t, timestep in enumerate(visualizer.diffusion_schedule.timesteps):
        # Run diffusion step
        noise_actions = visualizer.run_diffusion_step(obs_dict, noise_actions, timestep)
        
        # Convert actions to particle positions
        # Assuming actions are end-effector positions in 3D space
        positions = noise_actions[0].detach().cpu().numpy()  # [T, 3]
        
        # Update particle positions in simulator
        # Note: You'll need to implement these methods based on your Genesis setup
        scene.update_particle_positions(positions)
        scene.step()
        
        # FPS tracking
        t_now = time()
        print(1 / (t_now - t_prev), "FPS")
        t_prev = t_now
        sleep(0.0005)  # Small delay to control simulation speed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-m", "--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("-o", "--obs_path", required=True, help="Path to observation file")
    args = parser.parse_args()

    # Initialize Genesis
    gs.init(backend=gs.cpu)

    # Create scene with appropriate camera settings
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=4e-3, substeps=10),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-1.0, -1.0, 0.0),
            upper_bound=(1.0, 1.0, 1.0),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        show_viewer=args.vis
    )

    # Initialize particles for visualization
    particles = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.5),
            size=(0.1, 0.1, 0.1)
        ),
        surface=gs.surfaces.Default(
            color=(0.4, 0.4, 1.0),
            vis_mode="particle"
        )
    )

    scene.build()

    # Load model and observation
    visualizer = DiffusionVisualizer(args.model_path)
    obs_dict = torch.load(args.obs_path)  # You'll need to implement loading your observations

    # Run simulation in another thread
    gs.tools.run_in_another_thread(
        fn=run_sim, 
        args=(scene, visualizer, obs_dict)
    )
    
    if args.vis:
        scene.viewer.start()

if __name__ == "__main__":
    main()