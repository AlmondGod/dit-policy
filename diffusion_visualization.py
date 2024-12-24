import argparse
import numpy as np
import genesis as gs
from time import time, sleep
import torch
from data4robotics.models.diffusion import DiffusionTransformerAgent
from data4robotics import load_resnet18
from observations import DummyObs
from pyvirtualdisplay import Display
from IPython import display
import imageio

class DiffusionVisualizer:
    def __init__(self, model_path, device='cuda'):
        # Load pretrained vision model first
        self.transform, self.vision_model = load_resnet18()
        self.vision_model.to(device)
        self.vision_model.eval()
        
        # Load pretrained diffusion model
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model with checkpoint config
        model_kwargs = checkpoint['model_kwargs']
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
            # Create proper observation format
            obs = DummyObs()
            images = obs.image(0)[None, None]  # Add batch and time dimensions
            states = obs.state[None, None]  # Add batch and time dimensions
            
            # Convert to tensors and move to device
            images = torch.from_numpy(images).float().to(self.device) / 255.0
            states = torch.from_numpy(states).float().to(self.device)
            
            B = noise_actions.shape[0]
            s_t = self.model.tokenize_obs(images, states)
            
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

def run_sim(scene, visualizer, frames):
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

        print(f"running diffusion step {t}")

        # Run diffusion step
        noise_actions = visualizer.run_diffusion_step({}, noise_actions, timestep)
        
        # Convert actions to particle positions (xyz only)
        positions = noise_actions[0].detach().cpu().numpy()  # [T, 3]

        print(f"getting positions {positions}")
        
        # Update particle positions in simulator
        scene.update_particle_positions(positions)
        scene.step()
        
        print(f"rendering frame {t}")
        # Capture frame
        frame = scene.render()
        frames.append(frame)
        
        # FPS tracking
        t_now = time()
        print(f"Step {t}/{n_steps}, {1/(t_now - t_prev):.1f} FPS")
        t_prev = t_now
        sleep(0.0005)

def main():

    print("starting main")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    print("starting virtual display")
    # Start virtual display for headless rendering
    virtual_display = Display(visible=0, size=(800, 600))
    virtual_display.start()

    print("starting genesis")
    # Initialize Genesis
    gs.init(backend=gs.cpu)

    print("creating scene")
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
        show_viewer=False  # Headless mode
    )

    print("adding particles")
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
    print("building scene")
    scene.build()

    print("loading model")
    # Load model and run visualization
    visualizer = DiffusionVisualizer(args.model_path)
    
    # Create list to store frames
    frames = []
    
    # Run simulation and collect frames
    run_sim(scene, visualizer, frames)
    
    # Save animation
    print("Saving animation...")
    imageio.mimsave('diffusion_visualization.gif', frames, fps=30)
    
    # Display the animation in Colab
    with open('diffusion_visualization.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), format='gif'))

if __name__ == "__main__":
    main()