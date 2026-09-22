import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy.fft import dct, idct
from sklearn.linear_model import Lasso

DATA_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL"

N = 784
M = 196

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 100

ALPHA = 0.001

def load_images(filename):
    with open(filename, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows, cols)

def load_labels(filename):
    with open(filename, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data

print("Loading MNIST...")

x_train = load_images(
    DATA_PATH + r"\train-images.idx3-ubyte"
)

x_test = load_images(
    DATA_PATH + r"\t10k-images.idx3-ubyte"
)

x_train = x_train[:TRAIN_SAMPLES]
x_test = x_test[:TEST_SAMPLES]

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

print("Creating DCT basis...")

D = np.zeros((N, N), dtype=np.float32)

for i in range(N):
    unit = np.zeros(N)
    unit[i] = 1
    D[:, i] = idct(unit, norm="ortho")

print("DCT basis:", D.shape)

print("Creating measurement matrix...")

np.random.seed(42)

A = (
    np.random.randn(M, N)
    .astype("float32")
    / np.sqrt(M)
)

print("Measurement matrix A:", A.shape)

print("Creating measurements...")

y_train = x_train @ A.T
y_test = x_test @ A.T

print("Measurements:", y_train.shape)

print("Creating DCT sensing matrix...")

B = A @ D

print("DCT sensing matrix:", B.shape)

print("Starting DCT-LASSO reconstruction...")

reconstructed_images = []

for i in range(TEST_SAMPLES):

    measurement = y_test[i]

    model = Lasso(
        alpha=ALPHA,
        max_iter=10000,
        fit_intercept=False
    )

    model.fit(B, measurement)

    dct_coefficients = model.coef_

    reconstructed = D @ dct_coefficients

    reconstructed = np.clip(
        reconstructed,
        0,
        1
    )

    reconstructed_images.append(
        reconstructed
    )

reconstructed_images = np.array(
    reconstructed_images
)

print("Reconstruction completed.")

mse_values = []
nmse_values = []
nmse_db_values = []

for i in range(TEST_SAMPLES):

    original = x_test[i]

    reconstructed = reconstructed_images[i]

    error = original - reconstructed

    mse = np.mean(error ** 2)

    nmse = (
        np.sum(error ** 2)
        /
        np.sum(original ** 2)
    )

    nmse_db = 10 * np.log10(nmse)

    mse_values.append(mse)
    nmse_values.append(nmse)
    nmse_db_values.append(nmse_db)

mean_mse = np.mean(mse_values)
mean_nmse = np.mean(nmse_values)
mean_nmse_db = np.mean(nmse_db_values)

print()
print("DCT-LASSO Results")
print("----------------------")

print("M:", M)
print("N:", N)

print("Average MSE:", mean_mse)

print("Average NMSE:", mean_nmse)

print(
    "Average NMSE (dB):",
    mean_nmse_db
)

index = 0

original = x_test[index]

reconstructed = reconstructed_images[index]

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)

plt.imshow(
    original.reshape(28, 28),
    cmap="gray"
)

plt.title("Original")

plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(
    reconstructed.reshape(28, 28),
    cmap="gray"
)

plt.title("DCT-LASSO")

plt.axis("off")

plt.tight_layout()

plt.show()
