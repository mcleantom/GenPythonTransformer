# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import json
import datetime as dt

# from stable_baselines3.common.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

from custom_envs import StockTradingEnv

import pandas as pd

df = pd.read_csv('AAPL.csv')
df = df.sort_values('Date')

env = DummyVecEnv([lambda: StockTradingEnv(df)])

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps = 20000)

obs = env.reset()
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    env.render()
    