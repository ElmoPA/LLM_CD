import subprocess
import time

num_timesteps_iter = 10000
name = "main"
subprocess.run(["python3", "base_train.py", "--n", name + "0", "--ts", num_timesteps_iter])
for i in range(9):
    subprocess.run(["python3", "load_train.py",
                    "--ld", name + i,
                    "--n", name + (i+1),
                    "--ts", num_timesteps_iter])