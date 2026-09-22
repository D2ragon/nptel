import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import struct

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.linear_model import Lasso

np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL"

N = 784
M = 196

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 50

EPOCHS = 20
BATCH_SIZE = 64

LASSO_ALPHA = 0.001

SBL_ITERATIONS = 20
SBL_NOISE = 0.001


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

        return data.reshape(num, rows, cols)


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

y_train_labels = load_labels(
    DATA_PATH + r"\train-labels.idx1-ubyte"
)

x_test = load_images(
    DATA_PATH + r"\t10k-images.idx3-ubyte"
)

y_test_labels = load_labels(
    DATA_PATH + r"\t10k-labels.idx1-ubyte"
)

print("Full training data:", x_train.shape)
print("Testing data:", x_test.shape)

x_train = x_train[:TRAIN_SAMPLES]

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

x_test_used = x_test[:TEST_SAMPLES]

print("Training data used:", x_train.shape)
print("Testing data used:", x_test_used.shape)

print("Creating measurement matrix...")

A = (
    np.random.randn(M, N)
    .astype("float32")
    / np.sqrt(M)
)

print("Measurement matrix A:", A.shape)

print("Creating measurements...")

y_train = x_train @ A.T
y_test = x_test_used @ A.T

print("Compressed training data:", y_train.shape)
print("Compressed testing data:", y_test.shape)

print("Creating Generator...")

generator = tf.keras.Sequential([

    Input(shape=(M,)),

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


print("Creating Discriminator...")

discriminator = tf.keras.Sequential([

    Input(shape=(784,)),

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
            tf.ones_like(real_output),
            real_output
        )

        d_fake_loss = bce(
            tf.zeros_like(fake_output),
            fake_output
        )

        d_loss = (
            d_real_loss +
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
                real_images -
                fake_images
            )
        )

        adversarial_loss = bce(
            tf.ones_like(fake_output),
            fake_output
        )

        g_loss = (
            reconstruction_loss +
            0.001 *
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


print("Creating dataset...")

dataset = tf.data.Dataset.from_tensor_slices(
    (
        y_train,
        x_train
    )
)

dataset = dataset.shuffle(
    10000
).batch(
    BATCH_SIZE
)

print("Starting GAN training...")

nmse_db_history = []
mse_history = []

for epoch in range(EPOCHS):

    d_losses = []
    g_losses = []
    reconstruction_losses = []
    nmse_values = []

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

        fake_images = generator(
            measurements,
            training=False
        )

        error = tf.reduce_sum(
            tf.square(
                real_images -
                fake_images
            )
        )

        signal = tf.reduce_sum(
            tf.square(
                real_images
            )
        )

        nmse = error / signal

        nmse_values.append(
            nmse.numpy()
        )

    epoch_mse = np.mean(
        reconstruction_losses
    )

    epoch_nmse = np.mean(
        nmse_values
    )

    epoch_nmse_db = (
        10 *
        np.log10(
            epoch_nmse
        )
    )

    mse_history.append(
        epoch_mse
    )

    nmse_db_history.append(
        epoch_nmse_db
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
            epoch_mse,
            6
        ),
        "| NMSE (dB):",
        round(
            epoch_nmse_db,
            3
        )
    )


print()
print("GAN training completed.")


print()
print("Generating GAN reconstructions...")

gan_reconstructed = generator(
    y_test,
    training=False
).numpy()


gan_error = np.sum(
    (
        x_test_used -
        gan_reconstructed
    ) ** 2
)

gan_signal = np.sum(
    x_test_used ** 2
)

gan_mse = np.mean(
    (
        x_test_used -
        gan_reconstructed
    ) ** 2
)

gan_nmse = (
    gan_error /
    gan_signal
)

gan_nmse_db = (
    10 *
    np.log10(
        gan_nmse
    )
)


print()
print("GAN Results")
print("----------------------")
print("Average MSE:", gan_mse)
print("Average NMSE:", gan_nmse)
print("Average NMSE(dB):", gan_nmse_db)


print()
print("Running LASSO...")

lasso_reconstructed = np.zeros(
    (TEST_SAMPLES, N)
)

for i in range(TEST_SAMPLES):

    model = Lasso(
        alpha=LASSO_ALPHA,
        fit_intercept=False,
        max_iter=5000
    )

    model.fit(
        A,
        y_test[i]
    )

    lasso_reconstructed[i] = model.coef_

    if (i + 1) % 10 == 0:

        print(
            "LASSO image",
            i + 1,
            "/",
            TEST_SAMPLES
        )


lasso_mse = np.mean(
    (
        x_test_used -
        lasso_reconstructed
    ) ** 2
)

lasso_error = np.sum(
    (
        x_test_used -
        lasso_reconstructed
    ) ** 2
)

lasso_signal = np.sum(
    x_test_used ** 2
)

lasso_nmse = (
    lasso_error /
    lasso_signal
)

lasso_nmse_db = (
    10 *
    np.log10(
        lasso_nmse
    )
)


print()
print("LASSO Results")
print("----------------------")
print("Average MSE:", lasso_mse)
print("Average NMSE:", lasso_nmse)
print("Average NMSE(dB):", lasso_nmse_db)


def sbl_reconstruction(
    A,
    y,
    iterations=20,
    noise=0.001
):

    M_local, N_local = A.shape

    gamma = np.ones(
        N_local,
        dtype=np.float64
    )

    A_double = A.astype(
        np.float64
    )

    y_double = y.astype(
        np.float64
    )

    for iteration in range(iterations):

        AG = A_double * gamma

        C = (
            AG @ A_double.T
            +
            noise *
            np.eye(M_local)
        )

        try:

            C_inv_y = np.linalg.solve(
                C,
                y_double
            )

        except np.linalg.LinAlgError:

            C = C + 1e-6 * np.eye(M_local)

            C_inv_y = np.linalg.solve(
                C,
                y_double
            )

        x = (
            gamma *
            (
                A_double.T @ C_inv_y
            )
        )

        try:

            C_inv_A = np.linalg.solve(
                C,
                A_double
            )

        except np.linalg.LinAlgError:

            C = C + 1e-6 * np.eye(M_local)

            C_inv_A = np.linalg.solve(
                C,
                A_double
            )

        posterior_variance = (
            gamma -
            gamma *
            gamma *
            np.sum(
                A_double *
                C_inv_A,
                axis=0
            )
        )

        posterior_variance = np.maximum(
            posterior_variance,
            0
        )

        gamma_new = (
            x ** 2 +
            posterior_variance
        )

        gamma_new = np.maximum(
            gamma_new,
            1e-12
        )

        if np.linalg.norm(
            gamma_new - gamma
        ) / (
            np.linalg.norm(gamma) + 1e-12
        ) < 1e-4:

            gamma = gamma_new

            break

        gamma = gamma_new

    x = np.clip(
        x,
        0,
        1
    )

    return x.astype(
        np.float32
    )


print()
print("Running SBL...")

sbl_reconstructed = np.zeros(
    (TEST_SAMPLES, N),
    dtype=np.float32
)

for i in range(TEST_SAMPLES):

    sbl_reconstructed[i] = sbl_reconstruction(
        A,
        y_test[i],
        SBL_ITERATIONS,
        SBL_NOISE
    )

    print(
        "SBL image",
        i + 1,
        "/",
        TEST_SAMPLES
    )


sbl_mse = np.mean(
    (
        x_test_used -
        sbl_reconstructed
    ) ** 2
)

sbl_error = np.sum(
    (
        x_test_used -
        sbl_reconstructed
    ) ** 2
)

sbl_signal = np.sum(
    x_test_used ** 2
)

sbl_nmse = (
    sbl_error /
    sbl_signal
)

sbl_nmse_db = (
    10 *
    np.log10(
        sbl_nmse
    )
)


print()
print("SBL Results")
print("----------------------")
print("Average MSE:", sbl_mse)
print("Average NMSE:", sbl_nmse)
print("Average NMSE(dB):", sbl_nmse_db)


print()
print("===================================")
print("FINAL COMPARISON")
print("===================================")

print()
print("Method       MSE          NMSE        NMSE(dB)")
print(
    "GAN       ",
    round(gan_mse, 6),
    " ",
    round(gan_nmse, 6),
    " ",
    round(gan_nmse_db, 3)
)

print(
    "LASSO     ",
    round(lasso_mse, 6),
    " ",
    round(lasso_nmse, 6),
    " ",
    round(lasso_nmse_db, 3)
)

print(
    "SBL       ",
    round(sbl_mse, 6),
    " ",
    round(sbl_nmse, 6),
    " ",
    round(sbl_nmse_db, 3)
)


methods = [
    "GAN",
    "LASSO",
    "SBL"
]

mse_values = [
    gan_mse,
    lasso_mse,
    sbl_mse
]

nmse_values = [
    gan_nmse,
    lasso_nmse,
    sbl_nmse
]

nmse_db_values = [
    gan_nmse_db,
    lasso_nmse_db,
    sbl_nmse_db
]


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    methods,
    mse_values
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Average MSE"
)

plt.title(
    "GAN vs LASSO vs SBL - MSE"
)

plt.grid(
    axis="y"
)

plt.show()


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    methods,
    nmse_values
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "Average NMSE"
)

plt.title(
    "GAN vs LASSO vs SBL - NMSE"
)

plt.grid(
    axis="y"
)

plt.show()


plt.figure(
    figsize=(7, 5)
)

plt.bar(
    methods,
    nmse_db_values
)

plt.xlabel(
    "Method"
)

plt.ylabel(
    "NMSE (dB)"
)

plt.title(
    "GAN vs LASSO vs SBL - NMSE (dB)"
)

plt.grid(
    axis="y"
)

plt.show()


index = 0

plt.figure(
    figsize=(12, 3)
)

plt.subplot(
    1,
    4,
    1
)

plt.imshow(
    x_test_used[index].reshape(28, 28),
    cmap="gray"
)

plt.title(
    "Original"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    4,
    2
)

plt.imshow(
    gan_reconstructed[index].reshape(28, 28),
    cmap="gray"
)

plt.title(
    "GAN"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    4,
    3
)

plt.imshow(
    lasso_reconstructed[index].reshape(28, 28),
    cmap="gray"
)

plt.title(
    "LASSO"
)

plt.axis(
    "off"
)


plt.subplot(
    1,
    4,
    4
)

plt.imshow(
    sbl_reconstructed[index].reshape(28, 28),
    cmap="gray"
)

plt.title(
    "SBL"
)

plt.axis(
    "off"
)

plt.tight_layout()

plt.show()


plt.figure(
    figsize=(7, 5)
)

plt.plot(
    range(1, EPOCHS + 1),
    mse_history,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "MSE"
)

plt.title(
    "GAN MSE vs Epoch"
)

plt.grid()

plt.show()


plt.figure(
    figsize=(7, 5)
)

plt.plot(
    range(1, EPOCHS + 1),
    nmse_db_history,
    marker="o"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Normalized MSE (dB)"
)

plt.title(
    "GAN NMSE vs Epoch"
)

plt.grid()

plt.show()
