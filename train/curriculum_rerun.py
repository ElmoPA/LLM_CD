import os
import subprocess
import time

num_timesteps_iter = 100000
name = "curr"
path = "logs/"
os.makedirs(path + name, exist_ok=True)
path = os.path.join(path, name)
gap = 0.0
subprocess.run(["python3", "train/base_train.py",
                "--d", path,
                "--n", name,
                "--g", str(gap),
                "--ts", str(num_timesteps_iter)])
intervals = {30:0.5, 40:1.0, 50:1.5, 60: 2.0, 70: 2.5}
for i in range(69):
    if i in intervals:
        gap = intervals[i]

    subprocess.run(["python3", "train/load_train.py",
                    "--ld", os.path.join(path, name + str(i)),
                    "--s", os.path.join(path, name + str(i+1)),
                    "--d", path,
                    "--n", name,
                    "--g", str(gap),
                    "--ts", str(num_timesteps_iter)])
    subprocess.run(["python3", "train/test.py",
                    "--ld", os.path.join(path, name + str(i)),
                    "--d", path])