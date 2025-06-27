import os
import numpy as np
import torch
from PIL import Image
from src.pca import PCA
from src.autoencoder import Autoencoder, DenoisingAutoencoder
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# TODO: change to your data path
DATA_PATH = "./data"
PLOT_PATH = "./output"

# set random seed
np.random.seed(0)

"""
Tips for debugging:
- Use `print` to check the shape of your data. Shape mismatch is a common error.
- Use `ipdb` to debug your code
    - `ipdb.set_trace()` to set breakpoints and check the values of your variables in interactive mode
    - `python -m ipdb -c continue hw3.py` to run the entire script in debug mode. Once the script is paused, you can use `n` to step through the code line by line.
"""


def read_image():
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    file_path = DATA_PATH + "/subject_05_17.png"  # TODO: change to your path
    img = Image.open(file_path).convert("L")
    img_array = np.array(img)
    img_vector = img_array.flatten()
    img_vector = img_vector / 255.0
    return np.array(img_vector, dtype="float")


def load_data(split: str) -> tuple[np.ndarray, np.ndarray]:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    data_path = DATA_PATH + "/" + split
    files = os.listdir(data_path)
    image_vectors = []
    label_vectors = []

    for f in files:
        # Read the image using PIL
        img = Image.open(data_path + "/" + f).convert("L")
        f_name, f_type = os.path.splitext(f)
        label = int(f_name[-2:]) - 1
        label_vectors.append(label)

        # Convert the image to a numpy array
        img_array = np.array(img)

        # Reshape the image into a vector
        img_vector = img_array.flatten()
        img_vector = img_vector / 255.0
        image_vectors.append(img_vector)

    return np.array(image_vectors), np.array(label_vectors)


def compute_acc(y_pred: np.ndarray, y_val: np.ndarray):
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    return np.sum(y_pred == y_val) / len(y_val)


def reconstruction_loss(
    img_vec: np.ndarray, img_vec_reconstructed: np.ndarray
) -> float:
    """
    DO NOT MODIFY THIS FUNCTION.
    """
    return ((img_vec - img_vec_reconstructed) ** 2).mean()


def plot_image_vector(image_vector: np.ndarray, title: str, filename: str):
    plt.figure(figsize=(5, 5))

    # Rescale the image vector to the range [0, 1] for plotting
    min_val = image_vector.min()
    max_val = image_vector.max()
    if max_val - min_val > 1e-6:  # Avoid division by zero for constant vectors
        scaled_image_vector = (image_vector - min_val) / (max_val - min_val)
    else:
        scaled_image_vector = image_vector  # Or handle as a constant color

    plt.imshow(scaled_image_vector.reshape(61, 80), cmap="gray", vmin=0, vmax=1)
    plt.title(title)
    plt.axis("off")
    plt.savefig(PLOT_PATH + "/" + filename)
    plt.close()


def train_models(X_train, type):
    if type == "pca":
        print("Training PCA...")
        pca = PCA(n_components=40)
        pca.fit(X_train)

        return pca, None

    elif type == "ae":
        print("Training Autoencoder...")
        autoencoder = Autoencoder(input_dim=4880, encoding_dim=488)
        ae_loss = autoencoder.fit(X_train, epochs=500, batch_size=135)

        return autoencoder, ae_loss

    elif type == "dae":
        print("Training Denoising Autoencoder...")
        deno_autoencoder = DenoisingAutoencoder(input_dim=4880, encoding_dim=488)
        dae_loss = deno_autoencoder.fit(X_train, epochs=500, batch_size=135)

        return deno_autoencoder, dae_loss

    else:
        raise ValueError("Invalid model type. Choose 'pca', 'ae', or 'dae'.")


def transform_features(models, X_train, X_val):
    print("Transforming features...")
    pca, ae, dae = models
    return {
        "pca": (pca.transform(X_train), pca.transform(X_val)),
        "ae": (ae.transform(X_train), ae.transform(X_val)),
        "dae": (dae.transform(X_train), dae.transform(X_val)),
    }


def evaluate_models(features, y_train, y_val):
    print("Evaluating models...")
    print("Training classifiers...")
    results = {}
    for key, (X_tr, X_val_t) in features.items():
        clf = LogisticRegression(max_iter=10000, random_state=0)
        clf.fit(X_tr, y_train)
        y_pred = clf.predict(X_val_t)
        acc = compute_acc(y_pred, y_val)
        results[key] = acc
    return results


def reconstruct_and_plot(model, type):
    img_vec = read_image()

    if type == "pca":
        z_pca = model.transform(img_vec)
        img_pca = model.reconstruct(z_pca)
        loss_pca = reconstruction_loss(img_vec, img_pca)

        plot_reconstruction_comparison(
            img_vec, img_pca, "PCA", "pca_reconstruction.png"
        )

        if model.mean is not None:
            plot_image_vector(model.mean, "Mean Face", "mean_face.png")
        if model.components is not None:
            for i in range(min(4, model.components.shape[1])):
                plot_image_vector(
                    model.components[:, i], f"Eigenface {i+1}", f"eigenface_{i+1}.png"
                )

        return loss_pca

    elif type == "ae":
        z_ae = model.transform(img_vec)
        img_ae = model.reconstruct(torch.tensor(z_ae, dtype=torch.float32))
        loss_ae = reconstruction_loss(img_vec, img_ae)

        plot_reconstruction_comparison(
            img_vec, img_ae, "Autoencoder", "autoencoder_reconstruction.png"
        )

        return loss_ae

    elif type == "dae":
        z_dae = model.transform(img_vec)
        img_dae = model.reconstruct(torch.tensor(z_dae, dtype=torch.float32))
        loss_dae = reconstruction_loss(img_vec, img_dae)

        plot_reconstruction_comparison(
            img_vec, img_dae, "Denoising AE", "denoising_autoencoder_reconstruction.png"
        )

        return loss_dae

    else:
        raise ValueError("Invalid model type. Choose 'pca', 'ae', or 'dae'.")


def plot_reconstruction_comparison(original, reconstructed, title, filename):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original.reshape(61, 80), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.reshape(61, 80), cmap="gray", vmin=0, vmax=1)
    plt.title(f"{title} Reconstructed")
    plt.axis("off")

    plt.savefig(f"{PLOT_PATH}/{filename}")
    plt.close()


def plot_loss_curve(loss_history, title, filename):
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title(title)
    plt.savefig(f"{PLOT_PATH}/{filename}")
    plt.close()


def plot_multiple_loss_curves(loss_histories, title, filename):
    plt.figure(figsize=(10, 5))
    for label, loss_history in loss_histories.items():
        plt.plot(loss_history, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{PLOT_PATH}/{filename}")
    plt.close()


def main():
    print("Loading data...")
    X_train, y_train = load_data("train")
    X_val, y_val = load_data("val")

    # Train models
    pca, _ = train_models(X_train, "pca")
    ae, ae_loss = train_models(X_train, "ae")
    dae, dae_loss = train_models(X_train, "dae")

    plot_loss_curve(ae_loss, "Autoencoder Training Loss", "autoencoder_loss.png")
    plot_loss_curve(
        dae_loss,
        "Denoising Autoencoder Training Loss",
        "denoising_autoencoder_loss.png",
    )

    # Transform features and evaluate models
    features = transform_features((pca, ae, dae), X_train, X_val)
    acc_results = evaluate_models(features, y_train, y_val)

    for method, acc in acc_results.items():
        print(f"Accuracy from {method}: {acc:.4f}")

    # Reconstruct and plot images
    loss_pca = reconstruct_and_plot(pca, "pca")
    loss_ae = reconstruct_and_plot(ae, "ae")
    loss_dae = reconstruct_and_plot(dae, "dae")

    print(f"Reconstruction Loss (PCA): {loss_pca:.6f}")
    print(f"Reconstruction Loss (Autoencoder): {loss_ae:.6f}")
    print(f"Reconstruction Loss (Denoising Autoencoder): {loss_dae:.6f}")


def dae_arch_test():
    print("Testing Denoising Autoencoder...")
    X_train, y_train = load_data("train")

    dae_losses = {}
    for arch_type in ["default", "shallow", "deep"]:
        print(f"Training Denoising Autoencoder with architecture {arch_type}...")
        dae = DenoisingAutoencoder(input_dim=4880, encoding_dim=488, arch=arch_type)
        dae_loss = dae.fit(X_train, epochs=500, batch_size=135)
        dae_losses[arch_type] = dae_loss

        img_vec = read_image()
        z_dae = dae.transform(img_vec)
        img_dae = dae.reconstruct(torch.tensor(z_dae, dtype=torch.float32))
        loss_dae = reconstruction_loss(img_vec, img_dae)

        print(
            f"Reconstruction Loss (Denoising Autoencoder Arch {arch_type}): {loss_dae:.6f}"
        )

    plot_multiple_loss_curves(
        dae_losses,
        "Denoising Autoencoder Training Loss (Different Architectures)",
        "dae_architecture_test_loss.png",
    )


def dae_optimizer_test():
    print("Testing Denoising Autoencoder with different optimizers...")
    X_train, y_train = load_data("train")

    dae_losses = {}
    for optimizer_type in ["adam", "adagrad", "adadelta", "sgd"]:
        print(f"Training Denoising Autoencoder with {optimizer_type} optimizer...")
        dae = DenoisingAutoencoder(input_dim=4880, encoding_dim=488)
        dae_loss = dae.fit(
            X_train, epochs=500, batch_size=135, optimizer_name=optimizer_type, lr=0.001
        )
        dae_losses[optimizer_type] = dae_loss

    plot_multiple_loss_curves(
        dae_losses,
        "Denoising Autoencoder Training Loss (Different Optimizers)",
        "dae_optimizer_test_loss.png",
    )


if __name__ == "__main__":
    # main()
    dae_arch_test()
    dae_optimizer_test()
