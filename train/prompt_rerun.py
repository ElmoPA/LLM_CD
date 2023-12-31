import os
import subprocess

num_timesteps_iter = 100000
name = "prompt"
path = "logs/"
os.makedirs(path + name, exist_ok=True)
path = os.path.join(path, name)

# Initial Training with Base Parameters
subprocess.run(["python3", "train/base_train.py",
                "--d", path,
                "--n", name,
                "--ts", str(num_timesteps_iter),
                "--lr", "1e-4",  # Default learning rate
                "--ent", "0.01"])  # Slightly higher entropy coefficient

# Adaptive Training with Variable Gap Sizes and Learning Rates
gap = 0
learning_rate = 1e-4
intervals = {30: 0.5, 40: 1.0, 50: 1.5, 60: 2.0, 70: 2.5}
for i in range(69):
    if i in intervals:
        gap = intervals[i]
    if i % 10 == 0 and i > 0:
        learning_rate /= 2  # Reduce learning rate every 10 iterations

    subprocess.run(["python3", "train/load_train.py",
                    "--ld", os.path.join(path, name + str(i)),
                    "--s", os.path.join(path, name + str(i + 1)),
                    "--d", path,
                    "--n", name,
                    "--g", gap,
                    "--ts", str(num_timesteps_iter),
                    ])
    subprocess.run(["python3", "train/test.py",
                    "--ld", os.path.join(path, name + str(i)),
                    "--g", gap,
                    "--d", path])

# Optionally, you can add more sophisticated logic to adjust learning rate and entropy based on performance metrics.
