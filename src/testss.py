import numpy as np
d = np.load("outputs/caltech_occupancy_data.npz")
for s in ("train", "val", "test"):
    h, y = d[s + "_X_hist"], d[s + "_y"]
    print(s, "persistence MAE", np.abs(y - h[:, -1:]).mean().round(3),
          "| mean occupancy", y.mean().round(2), "| std", y.std().round(2))