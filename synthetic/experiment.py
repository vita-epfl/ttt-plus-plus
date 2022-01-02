"""System module."""
import time
import numpy as np
import pandas as pd
import main

start = time.time()
Results = np.zeros((5 * 5 * 5, 7))
i = 0
for tran in (0.2, 0.4, 0.6, 0.8, 1.0):
    for rot in (30, 40, 50, 60, 70):
        for sep in (-0.3, -0.4, -0.5, -0.6, -0.7):
            print("tran:{0}   rot:{1},  sep:{2}".format(tran, rot, sep))
            Results[i] = main.run_experiment(tran=tran, rot=rot, sep=sep, seed=2021, compare_tent_shot=True)
            i += 1
data = pd.DataFrame(Results, columns=['tran', 'rot', 'sep', 'test', 'ttt++', 'tent', 'shot'])
data.to_csv("docs/result.csv")
