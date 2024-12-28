import argparse
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from IPython.display import clear_output
clear_output(wait=True)

import genesis as gs
from data4robotics.models.diffusion import DiffusionTransformerAgent
from observations import DummyObs
from data4robotics.models.resnet import ResNet
from time import time, sleep
from datetime import datetime
    

class DiffusionVisualizer:
    def __init__(self, model_path, device='cuda'):
        print(f"Initializing DiffusionVisualizer with device: {device}")
        try:
            print("Loading checkpoint...")
            self.device = torch.device(device)
            checkpoint = torch.load(model_path, map_location=device)
            print("Checkpoint loaded successfully")
            print("Available keys in checkpoint:", list(checkpoint.keys()))
            
            print("Loading ResNet18...")
            # Create custom ResNet with GroupNorm
            resnet = ResNet(
                size=18,
                norm_cfg={"name": "group_norm", "num_groups": 32},
                weights="IMAGENET1K_V1",
                avg_pool=False  # Following resnet_gn_nopool config
            )
            
            # Load your custom weights with strict=False
            resnet_state_dict = torch.load('/content/IN_1M_resnet18.pth', map_location=device)
            if 'model' in resnet_state_dict:
                resnet_state_dict = resnet_state_dict['model']
            
            # Fix the state dict keys
            fixed_state_dict = {}
            for k, v in resnet_state_dict.items():
                if k.startswith('_model.'):
                    fixed_state_dict[k.replace('_model.', '')] = v
                else:
                    fixed_state_dict[k] = v
                    
            # Load with strict=False to allow missing keys
            resnet.load_state_dict(fixed_state_dict, strict=False)
            print("ResNet18 loaded successfully")
            
            print("Loading state dict keys:", checkpoint['model'].keys())
            
            print("Initializing diffusion model...")
            # Create a new model instance with parameters from diffusion.yaml
            self.model = DiffusionTransformerAgent(
                features=resnet,  # Use properly initialized ResNet
                odim=7,  # From task.obs_dim in config
                n_cams=1,  # From task.n_cams in config
                use_obs="add_token",  # From config
                ac_dim=6,  # From task.ac_dim in config
                ac_chunk=40,  # From config
                train_diffusion_steps=40,  # From config
                eval_diffusion_steps=8,  # From config
                imgs_per_cam=1,  # Default value
                dropout=0.1,  # From config
                share_cam_features=False,  # From config
                early_fusion=False,  # From config
                feat_norm=None,  # From config
                noise_net_kwargs={
                    'time_dim': 256,
                    'hidden_dim': 512,
                    'num_blocks': 6,
                    'dim_feedforward': 2048,
                    'dropout': 0.1,
                    'nhead': 8,
                    'activation': "gelu"
                }
            )
            
            # Load the state dictionary
            self.model.load_state_dict(checkpoint['model'])
            self.model.to(device)
            self.model.eval()
            print("Diffusion model initialized successfully")
            
            # Get the vision model from the loaded model
            # The feature extractor is likely stored in visual_features
            self.vision_model = self.model.visual_features[0]  # Based on the checkpoint keys
            self.vision_model.eval()
            
            self.model._eval_diffusion_steps = 8
            self.diffusion_schedule = self.model.diffusion_schedule
            self.noise_net = self.model.noise_net
            
            # Set up diffusion timesteps
            self.diffusion_schedule.set_timesteps(num_inference_steps=20)
            
        except Exception as e:
            print("Error in DiffusionVisualizer initialization:", e)
            print("Exception type:", type(e))
            import traceback
            traceback.print_exc()
            raise

    def run_diffusion_step(self, obs_dict, noise_actions, timestep):
        """Run a single step of the diffusion process"""
        with torch.no_grad():

            # add print statements below
            print("running diffusion step")
            # Create proper observation format
            obs = DummyObs()
            raw_image = obs.image(0)  # [H, W, C]
            states = obs.state  # [state_dim]
            
            # Add batch and time dimensions and convert to tensors
            images = torch.from_numpy(raw_image.copy()).float().to(self.device) / 255.0  # [H, W, C]
            images = images.permute(2, 0, 1)  # [C, H, W]
            images = images.unsqueeze(0)  # [1, C, H, W]
            
            print("processing states")
            states = torch.from_numpy(states.copy()).float().to(self.device)
            states = states.unsqueeze(0)  # [1, state_dim]
            
            # Format images as dictionary with camera keys
            image_dict = {"cam0": images}  # Model expects dict with camera keys
            
            B = noise_actions.shape[0]
            s_t = self.model.tokenize_obs(image_dict, states)

            print("tokenized obs")
            
            # For transformer model, we need encoder cache
            if not hasattr(self, 'enc_cache'):
                self.enc_cache = self.noise_net.forward_enc(s_t)
            
            # Handle both tensor and integer timesteps
            if isinstance(timestep, int):
                batched_timestep = torch.tensor([timestep] * B).to(self.device)
            else:
                batched_timestep = torch.tensor([timestep.item()] * B).to(self.device)
            
            noise_pred = self.noise_net.forward_dec(
                noise_actions, batched_timestep, self.enc_cache
            )
            
            print("forward dec")
            
            noise_actions = self.diffusion_schedule.step(
                model_output=noise_pred,
                timestep=timestep if isinstance(timestep, torch.Tensor) else torch.tensor(timestep),
                sample=noise_actions
            ).prev_sample
            
            print("stepped")
            
            return noise_actions

def run_sim(scene, visualizer, frames, cam, particles):
    """Run simulation with visualization"""
    print("Starting simulation...")
    
    # Single noise action
    noise_actions = torch.randn(1, 6).to(visualizer.device)
    n_steps = 10 # Total number of steps
    
    n_particles = particles._n_particles
    
    t_prev = time()
    
    # Process one timestep at a time
    for timestep in range(n_steps):
        print(f"Processing timestep {timestep}")
        
        with torch.no_grad():
            # Run single diffusion step
            noise_actions = visualizer.run_diffusion_step({}, noise_actions, timestep)
            
            print("ran diffusion step")
            
            # Get positions from noise actions - only take first 3 dimensions
            action = noise_actions.cpu().numpy()[0, :3]  # Take only XYZ coordinates [3]
            
            print("got noise actions")
            # Create particle positions - reshape to match number of particles
            positions = np.tile(action, (n_particles, 1))  # Shape will be (n_particles, 3)
            noise = np.random.normal(0, 0.05, (n_particles, 3))

            print("got positions")

            positions = positions[:n_particles, :3]
            positions = positions + noise
            
            print("updated positions with noise, setting particle positions")

            # Update particle positions
            particles.set_position(positions)

            print("set particle positions, stepping scene")
            
            # Before scene step
            torch.cuda.synchronize()  # Make sure all CUDA operations are done
            scene.step()
            torch.cuda.synchronize()  # Wait for step to complete
            print("stepped scene")

            # Before render
            torch.cuda.synchronize()  # Ensure previous CUDA operations are done
            torch.cuda.empty_cache()  # Clear CUDA memory before render
            
            try:
                rgb, depth, seg, normal = cam.render(
                    rgb=True,
                    depth=False, 
                    segmentation=False,
                    normal=False
                )
                print("rendered frame")
            except Exception as e:
                print(f"Render failed with error: {e}")
                rgb, depth, seg, normal = None, None, None, None

            t_now = time()
            print(t_now - t_prev, "time for step")
            t_prev = t_now

            # Add small delay to allow system to stabilize
            sleep(0.1)

    if scene.viewer is not None:
        scene.viewer.stop()

def main():
    print("\n=== Starting main() ===", flush=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", required=True)
    args = parser.parse_args()
    
    # virtual_display = Display(
    #     visible=0, 
    #     size=(800, 600),
    #     color_depth=24,  # Specify color depth
    #     backend='xvfb',  # Explicitly set backend
    #     extra_args=['-ac', '-screen', '0', '800x600x24']  # Additional X server args
    # )
    # virtual_display.start()
    
    print("torch.cuda.is_available()", torch.cuda.is_available())
    gs.init(backend=gs.cuda)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=4e-3, 
            substeps=20
        ),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-2.0, -2.0, -2.0),  # Reduced from -2.0
            upper_bound=(2.0, 2.0, 2.0),    # Reduced from 2.0
            grid_density=2,  # Added to reduce memory usage
        ),
        # viewer_options=gs.options.ViewerOptions(
        #     camera_pos=(1.5, 1.5, 2.5),
        #     camera_lookat=(0.0, 0.0, 0.0),
        #     camera_fov=30,
        # ),
        show_viewer=False
    )

    # add robot arm to scene
    r0 = scene.add_entity(
        gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
    )

    # add plane to scene
    plane = scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))


    # First create the particles with a default color
    particles = scene.add_entity(
        material=gs.materials.MPM.Liquid(),
        morph=gs.morphs.Box(
            pos=(0.0, 0.0, 0.0),
            size=(1.4, 1.4, 1.4)
        ),
        surface=gs.surfaces.Default(
            color=(0.2, 0.6, 1.0, 1.0),  # Default color
            vis_mode="particle"
        )
    )

    # Get the actual number of particles that were created
    n_particles = particles._n_particles
    print(f"Number of particles: {n_particles}")
    
    # # Create rainbow gradient colors similar to render_particles.py circular trajectory
    # theta = np.linspace(0, 2*np.pi, n_particles)
    # colors = np.zeros((n_particles, 4))
    # colors[:, 0] = 0.5 + 0.5 * np.sin(theta)  # Red channel
    # colors[:, 1] = 0.5 + 0.5 * np.sin(theta + 2*np.pi/3)  # Green channel
    # colors[:, 2] = 0.5 + 0.5 * np.sin(theta + 4*np.pi/3)  # Blue channel
    # colors[:, 3] = 1.0  # Alpha channel
    
    # # Set the colors after particle creation
    # particles.surface.color = colors

    cam = scene.add_camera(
        # res=(512, 384),
        # res=(256, 192),
        res=(128, 96),
        # res=(64, 48),
        pos=(1.5, 1.5, 2.5),
        lookat=(0, 0, 0),
        fov=60,
        GUI=False,
        denoise=False
    )

    print("building scene")
    
    scene.build()

    print("built scene")

    # render rgb, depth, segmentation, normal
    # rgb, depth, segmentation, normal = cam.render(rgb=True, depth=True, segmentation=True, normal=True)
    
    cam.start_recording()

    print("started recording")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    visualizer = DiffusionVisualizer(args.model_path, device=device)
    
    print("visualizer built")

    frames = []
    print("\nStarting simulation...")
    try:
        run_sim(scene, visualizer, frames, cam, particles)
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if scene.viewer is not None:
            scene.viewer.stop()
        # virtual_display.stop()

    #include time in name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cam.stop_recording(save_to_filename=f"diffusion_visualization_{timestamp}.mp4")

    #save frames
    # np.save("diffusion_visualization.mov", frames)
    
    # print("\nSaving animation...")
    # imageio.mimsave('diffusion_visualization.gif', frames, fps=30)
    # print("Animation saved")
    
    # print("\nDisplaying result...")
    # with open('diffusion_visualization.gif', 'rb') as f:
    #     display.display(display.Image(data=f.read(), format='gif'))
    # print("Display complete")

if __name__ == "__main__":
    main()