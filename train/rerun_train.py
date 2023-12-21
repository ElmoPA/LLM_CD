import os
import subprocess
import time

num_timesteps_iter = 100000
name = "enter"
path = "logs/"
os.makedirs(path + name, exist_ok=True)
path = os.path.join(path, name)
subprocess.run(["python3", "train/base_train.py",
                "--d", path,
                "--n", name,
                "--ts", str(num_timesteps_iter)])
for i in range(29):
    subprocess.run(["python3", "train/load_train.py",
                    "--ld", os.path.join(path, name + str(i)),
                    "--s", os.path.join(path, name + str(i+1)),
                    "--d", path,
                    "--n", name,
                    "--ts", str(num_timesteps_iter)])