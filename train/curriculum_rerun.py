import os
import subprocess
import time

num_timesteps_iter = 100000
name = "fast"
path = "logs/"
os.makedirs(path + name, exist_ok=True)
path = os.path.join(path, name)
subprocess.run(["python3", "train/base_train.py",
                "--d", path,
                "--n", name,
                "--g", str(0.0),
                "--ts", str(num_timesteps_iter)])
ex = False
for i in range(99):
    try:
        subprocess.run(["python3", "train/load_train.py",
                        "--ld", os.path.join(path, name + str(i)),
                        "--s", os.path.join(path, name + str(i+1)),
                        "--d", path,
                        "--n", name,
                        "--g", str(0.0),
                        "--ts", str(num_timesteps_iter)], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Subprocess error: {e}")
        ex = True
        break
    except Exception as e:
        print(f"Other error: {e}")
        ex = True
        break
    if ex:
        break