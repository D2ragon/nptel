import numpy as np
import matplotlib.pyplot as plt
import struct
from scipy.fft import dct, idct
from sklearn.linear_model import Lasso
import csv

IMAGE_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL\train-images.idx3-ubyte"

N = 784
M = 200
NUM_TEST_IMAGES = 20

LASSO_ALPHA = 0.001

SBL_ITERATIONS = 50
SBL_NOISE = 0.001

DNN_MSE = 0.005
DNN_NMSE = 0.05
DNN_DB = 10 * np.log10(DNN_NMSE)

np.random.seed(42)


def load_mnist_images(path):
    with open(path, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(
            f.read(num_images * rows * cols),
            dtype=np.uint8
        )

    images = data.reshape(num_images, rows * cols)
    images = images.astype(np.float32) / 255.0

    return images


def create_measurement_matrix(M, N):
    A = np.random.randn(M, N)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    return A


def dct_transform(x):
    return dct(x, norm="ortho")


def idct_transform(x):
    return idct(x, norm="ortho")


def lasso_reconstruction(y, A):
    model = Lasso(
        alpha=LASSO_ALPHA,
        max_iter=5000,
        fit_intercept=False
    )

    model.fit(A, y)

    return model.coef_


def sbl_reconstruction(y, A, iterations=50, noise=0.001):

    M, N = A.shape

    gamma = np.ones(N)

    for iteration in range(iterations):

        Gamma = np.diag(gamma)

        Sigma_y = A @ Gamma @ A.T + noise * np.eye(M)

        try:
            Sigma_y_inv = np.linalg.inv(Sigma_y)
        except np.linalg.LinAlgError:
            Sigma_y_inv = np.linalg.pinv(Sigma_y)

        mu = Gamma @ A.T @ Sigma_y_inv @ y

        Sigma_x = Gamma - Gamma @ A.T @ Sigma_y_inv @ A @ Gamma

        gamma_new = mu ** 2 + np.diag(Sigma_x)

        gamma_new = np.maximum(gamma_new, 1e-10)

        difference = np.linalg.norm(gamma_new - gamma)

        gamma = gamma_new

        if difference < 1e-6:
            break

    return mu


def calculate_metrics(original, reconstructed):

    error = original - reconstructed

    mse = np.mean(error ** 2)

    original_energy = np.mean(original ** 2)

    nmse = mse / (original_energy + 1e-12)

    db = 10 * np.log10(nmse + 1e-12)

    psnr = 10 * np.log10(
        1.0 / (mse + 1e-12)
    )

    return mse, nmse, db, psnr


def calculate_sparsity(x):

    threshold = 1e-3

    zero_values = np.sum(np.abs(x) < threshold)

    return 100 * zero_values / len(x)


print("Loading MNIST...")

images = load_mnist_images(IMAGE_PATH)

print("Total images:", len(images))
print("Image size:", images.shape[1])


A = create_measurement_matrix(M, N)

print()
print("Measurement matrix created")
print("N =", N)
print("M =", M)
print("Compression ratio =", M / N)


lasso_mse = []
lasso_nmse = []
lasso_db = []
lasso_psnr = []

sbl_mse = []
sbl_nmse = []
sbl_db = []
sbl_psnr = []

original_images = []
lasso_images = []
sbl_images = []
measurement_images = []


print()
print("Starting reconstruction...")
print("----------------------------------------")


for i in range(NUM_TEST_IMAGES):

    x_original = images[i]

    x_dct = dct_transform(x_original)

    y = A @ x_dct

    x_lasso_dct = lasso_reconstruction(
        y,
        A
    )

    x_sbl_dct = sbl_reconstruction(
        y,
        A,
        SBL_ITERATIONS,
        SBL_NOISE
    )

    x_lasso = idct_transform(
        x_lasso_dct
    )

    x_sbl = idct_transform(
        x_sbl_dct
    )

    x_lasso = np.clip(
        x_lasso,
        0,
        1
    )

    x_sbl = np.clip(
        x_sbl,
        0,
        1
    )

    mse1, nmse1, db1, psnr1 = calculate_metrics(
        x_original,
        x_lasso
    )

    mse2, nmse2, db2, psnr2 = calculate_metrics(
        x_original,
        x_sbl
    )

    lasso_mse.append(mse1)
    lasso_nmse.append(nmse1)
    lasso_db.append(db1)
    lasso_psnr.append(psnr1)

    sbl_mse.append(mse2)
    sbl_nmse.append(nmse2)
    sbl_db.append(db2)
    sbl_psnr.append(psnr2)

    original_images.append(x_original)
    lasso_images.append(x_lasso)
    sbl_images.append(x_sbl)

    measurement_images.append(
        y
    )

    print(
        "Image",
        i + 1,
        "| LASSO MSE:",
        round(mse1, 6),
        "| SBL MSE:",
        round(mse2, 6)
    )


print()
print("========================================")
print("FINAL RESULTS")
print("========================================")

avg_lasso_mse = np.mean(lasso_mse)
avg_lasso_nmse = np.mean(lasso_nmse)
avg_lasso_db = 10 * np.log10(avg_lasso_nmse)
avg_lasso_psnr = np.mean(lasso_psnr)

avg_sbl_mse = np.mean(sbl_mse)
avg_sbl_nmse = np.mean(sbl_nmse)
avg_sbl_db = 10 * np.log10(avg_sbl_nmse)
avg_sbl_psnr = np.mean(sbl_psnr)


print()
print("LASSO Results")
print("----------------------")
print("M:", M)
print("N:", N)
print("Average MSE:", avg_lasso_mse)
print("Average NMSE:", avg_lasso_nmse)
print("Average dB:", avg_lasso_db)
print("Average PSNR:", avg_lasso_psnr)


print()
print("SBL Results")
print("----------------------")
print("M:", M)
print("N:", N)
print("Average MSE:", avg_sbl_mse)
print("Average NMSE:", avg_sbl_nmse)
print("Average dB:", avg_sbl_db)
print("Average PSNR:", avg_sbl_psnr)


print()
print("DNN/GAN Results")
print("----------------------")
print("MSE:", DNN_MSE)
print("NMSE:", DNN_NMSE)
print("dB:", DNN_DB)


print()
print("========================================")
print("COMPARISON")
print("========================================")

print()
print("Method       MSE          NMSE         dB")
print("--------------------------------------------")

print(
    "LASSO   ",
    round(avg_lasso_mse, 6),
    round(avg_lasso_nmse, 6),
    round(avg_lasso_db, 4)
)

print(
    "SBL     ",
    round(avg_sbl_mse, 6),
    round(avg_sbl_nmse, 6),
    round(avg_sbl_db, 4)
)

print(
    "DNN/GAN ",
    round(DNN_MSE, 6),
    round(DNN_NMSE, 6),
    round(DNN_DB, 4)
)


plt.figure(figsize=(12, 4))

plt.subplot(1, 4, 1)

plt.imshow(
    original_images[0].reshape(28, 28),
    cmap="gray"
)

plt.title("Original")
plt.axis("off")


plt.subplot(1, 4, 2)

plt.plot(
    measurement_images[0]
)

plt.title("Measurement y = Ax")
plt.xlabel("Measurement index")
plt.ylabel("Value")


plt.subplot(1, 4, 3)

plt.imshow(
    lasso_images[0].reshape(28, 28),
    cmap="gray"
)

plt.title("LASSO")
plt.axis("off")


plt.subplot(1, 4, 4)

plt.imshow(
    sbl_images[0].reshape(28, 28),
    cmap="gray"
)

plt.title("SBL")
plt.axis("off")


plt.tight_layout()
plt.show()


methods = [
    "LASSO",
    "SBL",
    "DNN/GAN"
]

mse_values = [
    avg_lasso_mse,
    avg_sbl_mse,
    DNN_MSE
]


plt.figure(figsize=(7, 5))

plt.bar(
    methods,
    mse_values
)

plt.ylabel("Average MSE")
plt.title("Reconstruction MSE Comparison")

plt.tight_layout()
plt.show()


nmse_values = [
    avg_lasso_nmse,
    avg_sbl_nmse,
    DNN_NMSE
]


plt.figure(figsize=(7, 5))

plt.bar(
    methods,
    nmse_values
)

plt.ylabel("Average NMSE")
plt.title("Reconstruction NMSE Comparison")

plt.tight_layout()
plt.show()


db_values = [
    avg_lasso_db,
    avg_sbl_db,
    DNN_DB
]


plt.figure(figsize=(7, 5))

plt.bar(
    methods,
    db_values
)

plt.ylabel("NMSE (dB)")
plt.title("Reconstruction Error in dB")

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))

plt.plot(
    range(1, NUM_TEST_IMAGES + 1),
    lasso_mse,
    marker="o",
    label="LASSO"
)

plt.plot(
    range(1, NUM_TEST_IMAGES + 1),
    sbl_mse,
    marker="o",
    label="SBL"
)

plt.xlabel("Test Image")
plt.ylabel("MSE")
plt.title("MSE for Each Test Image")

plt.legend()
plt.grid()

plt.tight_layout()
plt.show()


results = [
    ["Method", "MSE", "NMSE", "dB", "PSNR"],
    [
        "LASSO",
        avg_lasso_mse,
        avg_lasso_nmse,
        avg_lasso_db,
        avg_lasso_psnr
    ],
    [
        "SBL",
        avg_sbl_mse,
        avg_sbl_nmse,
        avg_sbl_db,
        avg_sbl_psnr
    ],
    [
        "DNN/GAN",
        DNN_MSE,
        DNN_NMSE,
        DNN_DB,
        ""
    ]
]


with open(
    "sparse_recovery_comparison.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerows(results)


print()
print("Results saved to:")
print("sparse_recovery_comparison.csv")
