import numpy as np
import os
alpha_s = np.linspace(0.2,2.0,5)
l1_ratio = np.linspace(0.2,2.0,5)

for alpha in alpha_s:
    for l1 in l1_ratio:
        os.system(f"python simpl-model.py -a {alpha} -l1 {l1}")