from stable_baselines3 import PPO

model = PPO.load('logs/rl_model_505000_steps.zip')
print(model.get_parameters())