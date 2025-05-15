import hydra

from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    
    import logging
    import os
    
    import gym
    import isaacgym
    import isaacgymenvs
    from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
    from isaacgymenvs.utils.utils import set_np_formatting, set_seed
    import torch
    
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only
    set_np_formatting()

    # global rank of the GPU
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)

    num_envs = 256

    envs = isaacgymenvs.make(
        seed=0, 
        task="Ant", 
        num_envs=num_envs, 
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0,
        headless=False,
        multi_gpu=False,
        virtual_screen_capture=True,
        force_render=False,
    )
    envs.is_vector_env = True
    envs = gym.wrappers.RecordVideo(
        envs,
        "./videos",
        step_trigger=lambda step: step % 10000 == 0, # record the videos every 10000 steps
        video_length=100  # for each video record up to 100 steps
    )
    envs.reset()
    image = envs.render(mode="rgb_array")
    print("the image of Isaac Gym viewer is an array of shape", image.shape)
    for _ in range(10):
        actions = torch.zeros((num_envs,) + envs.action_space.shape, device = 'cuda:0') # 2.0 * torch.rand((num_envs,) + envs.action_space.shape, device = 'cuda:0') - 1.0
        envs.step(actions)
    envs.close()
 
if __name__ == "__main__":
    launch_rlg_hydra()