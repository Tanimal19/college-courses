# Background

Your goal is to create a simplified facial recognition system that can identify individuals from human face images. The main aim is to provide you with a comprehensive understanding of traditional and deep learning models used to extract valuable (dimension-reduced) features for this purpose. We will focus on two unsupervised learning models covered in the course: principal component analysis and autoencoder.

Additionally, we will expand our exploration by introducing the denoising autoencoder as an advanced variation of the autoencoder.

The dataset is provided in @data/

The data we used here is processed from so-called Yale face data, but you are not allowed to download the original one. There are 11 images per subject. The 11 images, each of dimension 80×61, are taken under conditions of center-light, w/glasses, happy, left-light, w/o glasses, normal, right-light, sad, sleepy, surprised, and wink. We will take the last two images per subject (surprised, wink) for validation and the other 9 images for training.

# Homework Description

You MUST use the provided sample code @hw4.py, @pca.py and @autoencoder.py as a
starting point for your implementation.

## 1. Principal Component Analysis

(a) Implement fit method for the class PCA in pca.py. The fit method should calculate the mean of the training data and the top n components eigenvectors of the mean-shifted training data. Run fit on the training dataset (of 135 images) with any n components, where n is larger than 4.

(b) Implement the reconstruct methods for PCA. Transform the given file subject_05_17.png with PCA and reconstruct the image with PCA. Plot the original image and the reconstructed image side by side and put it in your report.

(c) Implement the transform methods for PCA. Then use the transformed features to train a classifier of LogisticRegression, as done in hw4.py

## 2. Autoencoder

(a) Implement fit method for the class Autoencoder in autoencoder.py. The fit method should optimize the reconstruction error, i.e., the averaged squared error between $x_n$ and $g(x_n)$, where $g(x_n)$ is the reconstructed example of xn after going through the encoder and decoder of the associated Autoencoder. Please take the default architecture in the constructor of the Autoencoder, and train with any proper optimization routine in PyTorch. Plot the averaged squared error when running fit on the training dataset as a function of number of iterations (or epochs) and put it in your report.

(b) Implement the reconstruct methods for Autoencoder. Transform the given file subject_05_17.png into lower dimension and reconstruct the image with Autoencoder. Plot the original image and the reconstructed image side by side and put it in your report.

(c) Implement the transform methods for Autoencoder. Then use the transformed features to train a LogisticRegression classifier, as done in hw4.py.

## 3. Denoising Autoencoder

(a) Implement fit method for the class DenoisingAutoencoder in autoencoder.py. The fit method should optimize the averaged squared error between $x_n$ and $g(x_n + ϵ)$, where $g(x_n + ϵ)$ is the reconstructed example of $x_n$ plus a Gaussian noise $ϵ$. Each component of $ϵ$ is assumed to come from an independent Gaussian distribution of standard deviation noise factor, which is by default set to 0.2. Please take the default architecture in the constructor of the Autoencoder, and train with any proper optimization routine in PyTorch. Plot the averaged squared error when running fit on the training dataset as a function of number of iterations (or epochs) and put it in your report.

# Report

A detailed report answering the following questions:

(a) Plot the mean vector as an image as well as the top 4 eigenvectors, each as an image, by calling
the given plot component routine. Each of those top eigenvectors is usually called an eigenface that
physically means an informative “face ingredient.”

(b) Plot the training curve of Autoencoder and DenoisingAutoencoder

(c) Plot the original image and the images reconstructed with PCA, Autoencoder, and DenoisingAutoencoder
side by side, ideally as large as possible. Then, list the mean squared error between the original image and each reconstructed image

(d) Modify the architecture in Autoencoder in its constructor. Try at least two different network architectures for the denoising autoencoder. You can consider trying a deeper or shallower or fatter
or thinner network. You can also consider adding convolutional layers and/or other activation functions. Draw the architecture that you have tried and discuss your findings, particularly in terms of the reconstruction error that the architecture can achieve after decent optimization.

(e) Test at least 2 different optimizers, compare the training curve of DenoisingAutoencoder and discuss what you have found in terms of the convergence speed and overall performance of the model.
