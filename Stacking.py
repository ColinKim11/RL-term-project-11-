print("\n\n\n__   __ _____  _   _  _____  _   _ __   __ _____  _____  _   _  _           _    _  _____ ______  _   __  \n\ \ / /|  _  || \ | ||  __ \| | | |\ \ / /|  ___||  _  || \ | |( )         | |  | ||  _  || ___ \| | / /  \n \ V / | | | ||  \| || |  \/| |_| | \ V / | |__  | | | ||  \| ||/  ___     | |  | || | | || |_/ /| |/ /   \n  \ /  | | | || . ` || | __ |  _  |  \ /  |  __| | | | || . ` |   / __|    | |/\| || | | ||    / |    \   \n  | |  \ \_/ /| |\  || |_\ \| | | |  | |  | |___ \ \_/ /| |\  |   \__ \    \  /\  /\ \_/ /| |\ \ | |\  \  \n  \_/   \___/ \_| \_/ \____/\_| |_/  \_/  \____/  \___/ \_| \_/   |___/     \/  \/  \___/ \_| \_|\_| \_/  \n\n\n")                                                                                                        

import gymnasium as gym
import panda_gym
from stable_baselines3 import DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np
import torch
import torch.nn as nn

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

policy_kwargs = dict(
    net_arch = [64, 64, 64, 32, 16, 8], #[256,128,64,32,16,8,4],
    activation_fn = nn.ReLU,
    optimizer_class = torch.optim.Adam, # AdamW, Adam, NAG
    optimizer_kwargs = dict(weight_decay=1e-3)
)

def make_env():
    # env = gym.make("PandaReach-v3", render_mode="human")
    env = gym.make("PandaStack-v3", render_mode="human")
    
    n_actions = env.action_space.shape[-1]
    print(env.action_space)  # Box(-1.0, 1.0, (4,), float32)
    print("Action Space Low:", env.action_space.low)
    print("Action Space High:", env.action_space.high)
    print("Action Space Shape:", env.action_space.shape)
    print("Action Space dtype:", env.action_space.dtype)

    sigma = np.array([0.00, 0.00, 0.4, 0.4]) # (x, y, z, EE)

    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=sigma)
    model = SAC("MultiInputPolicy", 
                env, 
                buffer_size=int(1e7), 
                device=device, 
                verbose=1, 
                learning_starts=200,
                learning_rate=2e-3, 
                action_noise=action_noise, 
                policy_kwargs=policy_kwargs,
                batch_size=500,
                )
    
    print(model.policy.actor)
    print(model.policy.critic)
    
    model.learn(total_timesteps=int(1e7), log_interval=5)
    model.save("new")
    env.close()

make_env()

env = gym.make("PandaStack-v3", render_mode="human")
model = SAC.load("new")
obs, info = env.reset()

try:
    for _ in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            obs, info = env.reset()
            if terminated:
                print("Episode finished")
finally:
    env.close()
