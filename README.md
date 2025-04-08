# Classifier Guided DDPM on CheXpert Dataset

This repository contains an implementation of a Classifier Guided Denoising Diffusion Probabilistic Model (DDPM) trained on the CheXpert dataset. The goal is to leverage the power of DDPMs combined with a classifier to generate realistic chest X-ray images conditioned on specific pathologies.

## Overview

The Classifier Guided DDPM is a deep generative model that learns to generate high-quality images by iteratively denoising Gaussian noise. In this project, we extend the standard DDPM by incorporating a classifier that guides the generation process based on desired pathology labels from the CheXpert dataset.

## Dataset

The [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/) dataset is a large public dataset for chest radiograph interpretation, consisting of 224,316 chest radiographs of 65,240 patients. Each radiograph is labeled with the presence of 14 observations as positive, negative, or uncertain. This dataset is used to train the classifier and the DDPM.

## Model Architecture

The model consists of two main components:

1. **Classifier**: A convolutional neural network (CNN) classifier is trained to predict the presence of pathologies in chest X-ray images. The classifier is used to guide the generation process of the DDPM.

2. **DDPM**: The Denoising Diffusion Probabilistic Model is implemented using a U-Net architecture. It learns to generate realistic chest X-ray images by iteratively denoising Gaussian noise conditioned on the classifier's predictions.

## Usage

Instructions on how to install dependencies, preprocess the dataset, train the model, and generate chest X-ray images will be provided in the repository's documentation.

## Results

The generated chest X-ray images will be evaluated using both quantitative metrics (e.g., FID, IS) and qualitative visual assessments. Sample generated images showcasing the model's performance will be included in the repository.

## Contributions

Contributions to the project are welcome! If you encounter any issues, have suggestions for improvements, or would like to extend the model, please feel free to open an issue or submit a pull request.
