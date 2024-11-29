import numpy as np
import pandas as pd
np.random.seed(0)
df = pd.DataFrame()
df["seeds"] = np.random.randint(1000, size=100)
df.to_csv("random_seeds.csv")
