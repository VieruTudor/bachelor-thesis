import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()
df.drop('epoch', inplace=True, axis=1)


df_train = df[["train/box_loss", "train/obj_loss", "train/cls_loss"]]
df_metrics = df[["metrics/precision", "metrics/recall", "metrics/mAP_0.5"]]
df_val = df[["val/box_loss", "val/obj_loss", "val/cls_loss"]]

df_train.plot()
df_metrics.plot()
df_val.plot()

# plt.tight_layout()
plt.xlabel("epoch")
plt.show()