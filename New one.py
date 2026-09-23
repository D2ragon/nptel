import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import struct

from scipy.fft import idct
from sklearn.linear_model import Lasso
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


DATA_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL"

N = 784
M = 196

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 50

EPOCHS = 20
BATCH_SIZE = 64

LASSO_ALPHA = 0.001

SBL_ITERATIONS = 30
SBL_NOISE = 0.001

np.random.seed(42)
tf.random.set_seed(42)


def load_images(filename):

    with open(filename, "rb") as f:

        magic, num, rows, cols = struct.unpack(
            ">IIII",
            f.read(16)
        )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return data.reshape(
            num,
            rows,
            cols
        )


def load_labels(filename):

    with open(filename, "rb") as f:

        magic, num = struct.unpack(
            ">II",
            f.read(8)
        )

        data = np.frombuffer(
            f.read(),
            dtype=np.uint8
        )

        return data


print("Loading MNIST...")


x_train = load_images(
    DATA_PATH + r"\train-images.idx3-ubyte"
)

x_test = load_images(
    DATA_PATH + r"\t10k-images.idx3-ubyte"
)


y_train_labels = load_labels(
    DATA_PATH + r"\train-labels.idx1-ubyte"
)

y_test_labels = load_labels(
    DATA_PATH + r"\t10k-labels.idx1-ubyte"
)


x_train = x_train[:TRAIN_SAMPLES]
x_test = x_test[:TEST_SAMPLES]


x_train = x_train.astype(
    "float32"
) / 255.0

x_test = x_test.astype(
    "float32"
) / 255.0


x_train = x_train.reshape(
    -1,
    N
)

x_test = x_test.reshape(
    -1,
    N
)


print(
    "Training data:",
    x_train.shape
)

print(
    "Testing data:",
    x_test.shape
)


print()
print("Creating common measurement matrix...")


A = (
    np.random.randn(
        M,
        N
    ).astype("float32")
    / np.sqrt(M)
)


print(
    "Measurement matrix A:",
    A.shape
)


print()
print("Creating measurements...")


y_train = x_train @ A.T

y_test = x_test @ A.T


print(
    "Compressed training data:",
    y_train.shape
)

print(
    "Compressed testing data:",
    y_test.shape
)


print()
print("Creating DCT basis...")


D = np.zeros(
    (N, N),
    dtype=np.float32
)


for i in range(N):

    unit = np.zeros(N)

    unit[i] = 1.0

    D[:, i] = idct(
        unit,
        norm="ortho"
    )


print(
    "DCT basis:",
    D.shape
)


print()
print("Creating DCT sensing matrix...")


B = A @ D


print(
    "DCT sensing matrix:",
    B.shape
)


print()
print("==============================")
print("DCT-LASSO")
print("==============================")


lasso_reconstructed = []


for i in range(TEST_SAMPLES):

    measurement = y_test[i]


    model = Lasso(
        alpha=LASSO_ALPHA,
        max_iter=10000,
        fit_intercept=False
    )


    model.fit(
        B,
        measurement
    )


    dct_coefficients = model.coef_


    reconstructed = D @ dct_coefficients


    reconstructed = np.clip(
        reconstructed,
        0,
        1
    )


    lasso_reconstructed.append(
        reconstructed
    )


lasso_reconstructed = np.array(
    lasso_reconstructed
)


print(
    "DCT-LASSO reconstruction completed."
)


print()
print("==============================")
print("DCT-SBL")
print("==============================")


def sbl_reconstruction(
    B,
    y,
    iterations=30,
    noise=0.001
):

    m, n = B.shape


    gamma = np.ones(
        n,
        dtype=np.float64
    )


    for iteration in range(iterations):


        BG = B * gamma[np.newaxis, :]


        C = (
            BG @ B.T
            +
            noise * np.eye(m)
        )


        try:

            C_inv = np.linalg.inv(C)

        except np.linalg.LinAlgError:

            C_inv = np.linalg.pinv(C)


        mu = (
            gamma[:, None]
            *
            B.T
            @
            C_inv
            @
            y
        )


        Sigma = (
            gamma
            -
            gamma**2
            *
            np.sum(
                B.T
                *
                (
                    C_inv @ B.T
                ),
                axis=1
            )
        )


        Sigma = np.maximum(
            Sigma,
            0
        )


        gamma_new = (
            mu**2
            +
            Sigma
        )


        gamma_new = np.maximum(
            gamma_new,
            1e-12
        )


        difference = np.mean(
            np.abs(
                gamma_new - gamma
            )
        )


        gamma = gamma_new


        if difference < 1e-6:

            break


    return mu


sbl_reconstructed = []


for i in range(TEST_SAMPLES):

    measurement = y_test[i]


    sbl_coefficients = sbl_reconstruction(
        B,
        measurement,
        iterations=SBL_ITERATIONS,
        noise=SBL_NOISE
    )


    reconstructed = D @ sbl_coefficients


    reconstructed = np.clip(
        reconstructed,
        0,
        1
    )


    sbl_reconstructed.append(
        reconstructed
    )


sbl_reconstructed = np.array(
    sbl_reconstructed
)


print(
    "DCT-SBL reconstruction completed."
)


print()
print("==============================")
print("GAN")
print("==============================")


generator = tf.keras.Sequential([

    Input(
        shape=(M,)
    ),

    Dense(
        256,
        activation="relu"
    ),

    Dense(
        512,
        activation="relu"
    ),

    Dense(
        784,
        activation="sigmoid"
    )
])


discriminator = tf.keras.Sequential([

    Input(
        shape=(N,)
    ),

    Dense(
        512,
        activation="relu"
    ),

    Dense(
        256,
        activation="relu"
    ),

    Dense(
        1,
        activation="sigmoid"
    )
])


g_optimizer = Adam(
    learning_rate=0.0002
)

d_optimizer = Adam(
    learning_rate=0.0002
)


bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(
    measurements,
    real_images
):


    with tf.GradientTape() as d_tape:


        fake_images = generator(
            measurements,
            training=True
        )


        real_output = discriminator(
            real_images,
            training=True
        )


        fake_output = discriminator(
            fake_images,
            training=True
        )


        d_real_loss = bce(
            tf.ones_like(
                real_output
            ),
            real_output
        )


        d_fake_loss = bce(
            tf.zeros_like(
                fake_output
            ),
            fake_output
        )


        d_loss = (
            d_real_loss
            +
            d_fake_loss
        )


    d_gradients = d_tape.gradient(
        d_loss,
        discriminator.trainable_variables
    )


    d_optimizer.apply_gradients(
        zip(
            d_gradients,
            discriminator.trainable_variables
        )
    )


    with tf.GradientTape() as g_tape:


        fake_images = generator(
            measurements,
            training=True
        )


        fake_output = discriminator(
            fake_images,
            training=True
        )


        reconstruction_loss = tf.reduce_mean(
            tf.square(
                real_images
                -
                fake_images
            )
        )


        adversarial_loss = bce(
            tf.ones_like(
                fake_output
            ),
            fake_output
        )


        g_loss = (
            reconstruction_loss
            +
            0.001
            *
            adversarial_loss
        )


    g_gradients = g_tape.gradient(
        g_loss,
        generator.trainable_variables
    )


    g_optimizer.apply_gradients(
        zip(
            g_gradients,
            generator.trainable_variables
        )
    )


    return (
        d_loss,
        g_loss,
        reconstruction_loss
    )


dataset = tf.data.Dataset.from_tensor_slices(
    (
        y_train,
        x_train
    )
)


dataset = dataset.shuffle(
    TRAIN_SAMPLES
).batch(
    BATCH_SIZE
)


print()
print("Starting GAN training...")


for epoch in range(EPOCHS):


    d_losses = []

    g_losses = []

    reconstruction_losses = []


    for measurements, real_images in dataset:


        d_loss, g_loss, rec_loss = train_step(
            measurements,
            real_images
        )


        d_losses.append(
            d_loss.numpy()
        )


        g_losses.append(
            g_loss.numpy()
        )


        reconstruction_losses.append(
            rec_loss.numpy()
        )


    print(
        "Epoch",
        epoch + 1,
        "/",
        EPOCHS,
        "| D Loss:",
        round(
            np.mean(d_losses),
            6
        ),
        "| G Loss:",
        round(
            np.mean(g_losses),
            6
        ),
        "| MSE:",
        round(
            np.mean(
                reconstruction_losses
            ),
            6
        )
    )


print()
print("GAN training completed.")


gan_reconstructed = generator(
    y_test,
    training=False
).numpy()


gan_reconstructed = np.clip(
    gan_reconstructed,
    0,
    1
)


print()
print("==============================")
print("Calculating Results")
print("==============================")


def calculate_metrics(
    original,
    reconstructed
):


    mse_values = []

    nmse_values = []

    nmse_db_values = []


    for i in range(
        len(original)
    ):


        error = (
            original[i]
            -
            reconstructed[i]
        )


        mse = np.mean(
            error ** 2
        )


        nmse = (
            np.sum(
                error ** 2
            )
            /
            np.sum(
                original[i] ** 2
            )
        )


        nmse_db = (
            10
            *
            np.log10(
                nmse
            )
        )


        mse_values.append(
            mse
        )

        nmse_values.append(
            nmse
        )

        nmse_db_values.append(
            nmse_db
        )


    return (
        np.mean(mse_values),
        np.mean(nmse_values),
        np.mean(nmse_db_values)
    )


lasso_mse, lasso_nmse, lasso_db = calculate_metrics(
    x_test,
    lasso_reconstructed
)


sbl_mse, sbl_nmse, sbl_db = calculate_metrics(
    x_test,
    sbl_reconstructed
)


gan_mse, gan_nmse, gan_db = calculate_metrics(
    x_test,
    gan_reconstructed
)


print()
print("======================================")
print("FINAL COMPARISON")
print("======================================")


print()
print("DCT-LASSO")
print("MSE:", lasso_mse)
print("NMSE:", lasso_nmse)
print("NMSE (dB):", lasso_db)


print()
print("DCT-SBL")
print("MSE:", sbl_mse)
print("NMSE:", sbl_nmse)
print("NMSE (dB):", sbl_db)


print()
print("GAN")
print("MSE:", gan_mse)
print("NMSE:", gan_nmse)
print("NMSE (dB):", gan_db)


methods = [
    "DCT-LASSO",
    "DCT-SBL",
    "GAN"
]


nmse_db_results = [
    lasso_db,
    sbl_db,
    gan_db
]


print()
print("Creating final comparison graph...")


plt.figure(
    figsize=(9, 6)
)


plt.plot(
    methods,
    nmse_db_results,
    marker="o",
    linewidth=2,
    markersize=8
)


for i in range(
    len(methods)
):

    plt.text(
        i,
        nmse_db_results[i],
        f"{nmse_db_results[i]:.2f} dB",
        ha="center",
        va="bottom"
    )


plt.xlabel(
    "Reconstruction Method"
)


plt.ylabel(
    "NMSE (dB)"
)


plt.title(
    "Comparison of DCT-LASSO, DCT-SBL and GAN"
)


plt.grid(
    True
)


plt.tight_layout()


plt.show()
