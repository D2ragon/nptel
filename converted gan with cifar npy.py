import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

DATA_PATH = r"C:\Users\Tintu\OneDrive\Desktop\NPOL\archive(4)\cifar-10-batches-py\cifar10_gray_28"

N = 784
M = 196

TRAIN_SAMPLES = 10000
TEST_SAMPLES = 50
EPOCHS = 20
BATCH_SIZE = 64

np.random.seed(42)
tf.random.set_seed(42)


def load_cifar_npy(data_path):
    files = [
        os.path.join(data_path, "data_batch_1_gray28.npy"),
        os.path.join(data_path, "data_batch_2_gray28.npy"),
        os.path.join(data_path, "data_batch_3_gray28.npy"),
        os.path.join(data_path, "data_batch_4_gray28.npy"),
        os.path.join(data_path, "data_batch_5_gray28.npy")
    ]

    image_arrays = []

    for filename in files:
        if not os.path.exists(filename):
            raise FileNotFoundError(
                "Could not find: " + filename +
                "\nUpdate the filenames in load_cifar_npy() to match your files."
            )

        images = np.load(filename)

        if images.ndim == 4 and images.shape[-1] == 1:
            images = images[..., 0]

        if images.ndim == 3 and images.shape[1:] == (28, 28):
            images = images.reshape(-1, N)
        elif images.ndim == 2 and images.shape[1] == N:
            pass
        else:
            raise ValueError(
                f"Unexpected image shape in {filename}: {images.shape}. "
                "Expected (count, 28, 28), (count, 28, 28, 1), "
                "or (count, 784)."
            )

        images = images.astype(np.float32)

        if images.max() > 1.0:
            images /= 255.0

        image_arrays.append(images)

    return np.concatenate(image_arrays, axis=0)


print("Loading converted CIFAR-10 images...")

all_images = load_cifar_npy(DATA_PATH)

print("All image data:", all_images.shape)

if len(all_images) <= TEST_SAMPLES:
    raise ValueError("Not enough images. Check the five .npy files.")

random_indices = np.random.permutation(len(all_images))
all_images = all_images[random_indices]

test_count = min(TEST_SAMPLES, len(all_images) - 1)
train_count = min(TRAIN_SAMPLES, len(all_images) - test_count)

x_train = all_images[:train_count]
x_test = all_images[train_count:train_count + test_count]

print("Training data:", x_train.shape)
print("Testing data:", x_test.shape)

print("Creating measurement matrix...")

A = np.random.randn(M, N).astype(np.float32) / np.sqrt(M)

print("Measurement matrix A:", A.shape)

print("Creating compressed measurements...")

y_train = x_train @ A.T
y_test = x_test @ A.T

print("Compressed training data:", y_train.shape)
print("Compressed testing data:", y_test.shape)

print("Creating Generator...")

generator = tf.keras.Sequential([
    Input(shape=(M,)),
    Dense(256, activation="relu"),
    Dense(512, activation="relu"),
    Dense(N, activation="sigmoid")
])

print("Creating Discriminator...")

discriminator = tf.keras.Sequential([
    Input(shape=(N,)),
    Dense(512, activation="relu"),
    Dense(256, activation="relu"),
    Dense(1, activation="sigmoid")
])

g_optimizer = Adam(learning_rate=0.0002)
d_optimizer = Adam(learning_rate=0.0002)

bce = tf.keras.losses.BinaryCrossentropy()


@tf.function
def train_step(measurements, real_images):
    with tf.GradientTape() as d_tape:
        fake_images = generator(measurements, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        d_real_loss = bce(tf.ones_like(real_output), real_output)
        d_fake_loss = bce(tf.zeros_like(fake_output), fake_output)
        d_loss = d_real_loss + d_fake_loss

    d_gradients = d_tape.gradient(
        d_loss, discriminator.trainable_variables
    )
    d_optimizer.apply_gradients(
        zip(d_gradients, discriminator.trainable_variables)
    )

    with tf.GradientTape() as g_tape:
        fake_images = generator(measurements, training=True)
        fake_output = discriminator(fake_images, training=True)

        reconstruction_loss = tf.reduce_mean(
            tf.square(real_images - fake_images)
        )

        adversarial_loss = bce(
            tf.ones_like(fake_output), fake_output
        )

        g_loss = reconstruction_loss + 0.001 * adversarial_loss

    g_gradients = g_tape.gradient(
        g_loss, generator.trainable_variables
    )
    g_optimizer.apply_gradients(
        zip(g_gradients, generator.trainable_variables)
    )

    return d_loss, g_loss, reconstruction_loss


print("Creating dataset...")

dataset = tf.data.Dataset.from_tensor_slices((y_train, x_train))
dataset = dataset.shuffle(
    len(x_train), seed=42, reshuffle_each_iteration=True
).batch(BATCH_SIZE)

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
            measurements, real_images
        )

        d_losses.append(float(d_loss.numpy()))
        g_losses.append(float(g_loss.numpy()))
        reconstruction_losses.append(float(rec_loss.numpy()))

        fake_images = generator(measurements, training=False)

        error = tf.reduce_sum(
            tf.square(real_images - fake_images)
        )
        signal = tf.reduce_sum(tf.square(real_images))
        nmse = error / (signal + 1e-12)

        nmse_values.append(float(nmse.numpy()))

    epoch_mse = float(np.mean(reconstruction_losses))
    epoch_nmse = float(np.mean(nmse_values))
    epoch_nmse_db = float(
        10 * np.log10(max(epoch_nmse, 1e-12))
    )

    mse_history.append(epoch_mse)
    nmse_db_history.append(epoch_nmse_db)

    print(
        "Epoch", epoch + 1, "/", EPOCHS,
        "| D Loss:", round(float(np.mean(d_losses)), 6),
        "| G Loss:", round(float(np.mean(g_losses)), 6),
        "| MSE:", round(epoch_mse, 6),
        "| NMSE (dB):", round(epoch_nmse_db, 3)
    )

print("Training completed.")

index = 0

measurement = y_test[index:index + 1]
original = x_test[index:index + 1]

reconstructed = generator(
    measurement, training=False
).numpy()

mse = np.mean((original - reconstructed) ** 2)
error = np.sum((original - reconstructed) ** 2)
signal = np.sum(original ** 2)
nmse = error / (signal + 1e-12)
nmse_db = 10 * np.log10(max(nmse, 1e-12))

print()
print("Test image:", index)
print("Reconstruction MSE:", mse)
print("Normalized MSE:", nmse)
print("Normalized MSE (dB):", nmse_db, "dB")

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original.reshape(28, 28), cmap="gray", vmin=0, vmax=1)
plt.title("Original CIFAR-10 Grayscale")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(
    reconstructed.reshape(28, 28),
    cmap="gray",
    vmin=0,
    vmax=1
)
plt.title("GAN Reconstructed")
plt.axis("off")

plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(1, EPOCHS + 1), nmse_db_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Normalized MSE (dB)")
plt.title("NMSE vs Epoch")
plt.grid()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.plot(range(1, EPOCHS + 1), mse_history, marker="o")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("MSE vs Epoch")
plt.grid()
plt.tight_layout()
plt.show()
