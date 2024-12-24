print("Script starting - top level", flush=True)

import argparse
import numpy as np
import torch
import imageio

try:
    print("Importing genesis...", flush=True)
    import genesis as gs
    print("Genesis imported successfully", flush=True)
except Exception as e:
    print("Failed to import genesis:", e, flush=True)
    raise

try:
    print("Importing display modules...", flush=True)
    from pyvirtualdisplay import Display
    from IPython import display
    print("Display modules imported successfully", flush=True)
except Exception as e:
    print("Failed to import display modules:", e, flush=True)
    raise

try:
    print("Importing custom modules...", flush=True)
    from data4robotics.models.diffusion import DiffusionTransformerAgent
    from data4robotics import load_resnet18
    from observations import DummyObs
    print("Custom modules imported successfully", flush=True)
except Exception as e:
    print("Failed to import custom modules:", e, flush=True)
    raise

class DiffusionVisualizer:
    def __init__(self, model_path, device='cpu'):  # Changed default to CPU
        print(f"Initializing DiffusionVisualizer with device: {device}", flush=True)
        try:
            print("Loading vision model...", flush=True)
            self.transform, self.vision_model = load_resnet18()
            self.vision_model.to(device)
            self.vision_model.eval()
            print("Vision model loaded successfully", flush=True)
            
            print("Loading checkpoint...", flush=True)
            self.device = torch.device(device)
            checkpoint = torch.load(model_path, map_location=device)
            print("Checkpoint loaded successfully", flush=True)
            
            print("Initializing diffusion model...", flush=True)
            model_kwargs = checkpoint['model_kwargs']
            model_kwargs['features'] = self.vision_model
            self.model = DiffusionTransformerAgent(**model_kwargs)
            self.model.load_state_dict(checkpoint['state_dict'])
            self.model.to(device)
            self.model.eval()
            print("Diffusion model initialized successfully", flush=True)
            
            self.model._eval_diffusion_steps = 100
            self.diffusion_schedule = self.model.diffusion_schedule
            self.noise_net = self.model.noise_net
        except Exception as e:
            print("Error in DiffusionVisualizer initialization:", e, flush=True)
            raise

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

        print(f"running diffusion step {t}", flush=True)

        # Run diffusion step
        noise_actions = visualizer.run_diffusion_step({}, noise_actions, timestep)
        
        # Convert actions to particle positions (xyz only)
        positions = noise_actions[0].detach().cpu().numpy()  # [T, 3]

        print(f"getting positions {positions}", flush=True)
        
        # Update particle positions in simulator
        scene.update_particle_positions(positions)
        scene.step()
        
        print(f"rendering frame {t}", flush=True)
        # Capture frame
        frame = scene.render()
        frames.append(frame)
        
        # FPS tracking
        t_now = time()
        print(f"Step {t}/{n_steps}, {1/(t_now - t_prev):.1f} FPS", flush=True)
        t_prev = t_now
        sleep(0.0005)

def main():
    print("\n=== Starting main() ===", flush=True)
    
    print("Parsing arguments...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    args = parser.parse_args()
    print(f"Arguments parsed: {args}", flush=True)
    
    print("\nInitializing virtual display...")
    virtual_display = Display(visible=0, size=(800, 600))
    virtual_display.start()
    print("Virtual display started")
    
    print("\nInitializing Genesis...")
    gs.init(backend=gs.cpu)
    print("Genesis initialized")
    
    print("\nCreating scene...")
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
        show_viewer=False
    )
    print("Scene created")
    
    print("\nSetting up visualization...")
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
    print("Particles added")
    
    print("Building scene...")
    scene.build()
    print("Scene built successfully")
    
    print("\nInitializing visualizer...")
    visualizer = DiffusionVisualizer(args.model_path)
    print("Visualizer initialized")
    
    frames = []
    print("\nStarting simulation...")
    run_sim(scene, visualizer, frames)
    
    print("\nSaving animation...")
    imageio.mimsave('diffusion_visualization.gif', frames, fps=30)
    print("Animation saved")
    
    print("\nDisplaying result...")
    with open('diffusion_visualization.gif', 'rb') as f:
        display.display(display.Image(data=f.read(), format='gif'))
    print("Display complete")

if __name__ == "__main__":
    print("\n=== Script entry point ===", flush=True)
    main()