import pandas as pd
import numpy as np

PATH = "chrisPP/saved_distances.csv"
# Read the csv with distances saved for each keypoint
df = pd.read_csv(PATH)

for col in df.columns:
    print(col)
    if col!="personID":
        vals =df.sort_values(by=col, ascending=False).personID.values
        print("person ID : {}".format(vals))
        vals =df.sort_values(by=col, ascending=False)[col].values
        print("score : {}".format(vals))
        print()
