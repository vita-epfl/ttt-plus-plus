"""System module."""
import time
import numpy as np
import pandas as pd
import main as main

start = time.time()
Results = np.zeros((9 * 5 * 6, 12))
i = 0
for tran in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
    for rot in (10, 30, 50, 70, 90):
        for sep in (-0.3, -0.4, -0.5, -0.6, -0.7, -0.8):
            print("tran:{0}   rot:{1},  sep:{2}".format(tran, rot, sep))
            Results[i] = main.run_experiment(tran=tran, rot=rot, sep=sep, seed=2021)
            i += 1
data = pd.DataFrame(Results, columns=['tran', 'rot', 'sep', 'Relation', 'test', 'TTT', 'TTT++',
                                      'TENT', 'SHOT', 'SHOT_div', 'SHOT_pseudo', 'pseudo+FA'])
data.to_csv("docs/result.csv")
