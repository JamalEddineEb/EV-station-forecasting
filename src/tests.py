import numpy as np

path = "outputs/benchmark/forecasts/dlinear_cuda_fp32.npz"

with np.load(path) as d:
    actual = d["actual"]
    pred = d["prediction"]
    lower = d["lower"]
    upper = d["upper"]

err = pred - actual

print("Example 90")
print("Actual:    ", actual[90])
print("Predicted: ", pred[90])
print("Error:     ", err[90])

print("\nOverall test-set diagnostics")
print("Mean error (bias):", err.mean())
print("MAE:", np.abs(err).mean())
print("RMSE:", np.sqrt(np.mean(err**2)))
print("Interval coverage:", np.mean((actual >= lower) & (actual <= upper)))

print("\nMetrics by forecast horizon")
for h in [0, 3, 7, 15]:
    print(
        f"Step {h + 1:2d}: "
        f"MAE={np.abs(err[:, h]).mean():.3f}, "
        f"bias={err[:, h].mean():.3f}, "
        f"coverage={np.mean((actual[:, h] >= lower[:, h]) & (actual[:, h] <= upper[:, h])):.3f}"
    )