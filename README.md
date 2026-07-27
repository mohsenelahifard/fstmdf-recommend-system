# Deep Matrix Factorization via Feature Subspace Transfer (FSTDMF) 🎬🧠
**Unofficial PyTorch Implementation**

**Important Note on Authorship and Attribution:**
This repository contains an unofficial PyTorch implementation of the **FSTDMF** framework. The core methodology, theoretical foundations, and architectural design are entirely derived from the work of Weichen Wang and Jing Wang. The purpose of this repository is to provide a practical, from-scratch implementation to reproduce their findings. All scientific credit for the approach belongs solely to the original authors.

**Original Paper:**
> Weichen Wang, Jing Wang "Deep matrix factorization via feature subspace transfer for recommendation system" Complex & Intelligent Systems, Volume 10, April 2024, 4939–4954 DOI: 10.1007/s40747-024-01414-2

---

## Project Overview 🔍
This project provides a robust, heavily customized deep learning pipeline to reproduce the FSTDMF recommendation system, explicitly designed to tackle the data sparsity problem. It implements a Deep Matrix Factorization (DMF) model enhanced with a Semi-Autoencoder (SA) for intelligent initialization and a Feature Subspace Transfer (FST) mechanism to seamlessly incorporate item side information (e.g., genres and release years).

## Implementation Features 🚀

### 1. Data Preparation & Preprocessing
- **Dataset Parsing**: Automated loading and formatting of MovieLens-100K and MovieLens-1M datasets.
- **Side Information Encoding**: One-hot encoding of movie attributes (release year and genres).
- **Feature Orthogonalization**: Utilizing QR decomposition to project auxiliary features into a partial orthonormal subspace.

### 2. Deep Learning Architecture (PyTorch)
- **Deep Matrix Factorization (DMF)**: A multi-layer non-linear latent variable model built dynamically using `torch.nn`.
- **Semi-Autoencoder (SA)**: An unsupervised pre-training module programmed to extract robust initial latent representations from the sparse rating matrix.
- **Feature Subspace Transfer (FST)**: A custom regularization mechanism measuring the subspace projection distance to transfer knowledge from side information to the main model.

### 3. Training & Optimization Engineering
- **Custom Scaled Loss Function**: A meticulously scaled loss function integrating Mean Squared Error (MSE), L2 regularization, and subspace projection penalty to ensure stable gradient descent and prevent early network death.
- **Data Leakage Prevention**: Strict separation of Inner-Train and Validation sets during the Grid Search hyperparameter tuning for $\alpha$ and $\beta$.

### 4. Evaluation & Analysis
- **Sensitivity Analysis**: Automated evaluation of the model's performance across different latent dimensions ($r$) and hidden layer depths.
- **Ablation Study**: Modular execution to isolate and measure the impact of the SA and FST components.
- **Visualization**: Generates high-quality comparative plots for MAE and RMSE metrics using `matplotlib`.

## Requirements ⚙️

The project was developed and tested in an environment with the following dependencies:

- `numpy==1.26.4`
- `pandas==1.5.3`
- `matplotlib==3.7.3`
- `torch==2.8.0+cpu` *(CPU-only version)*

Install them using:

```bash
pip install -r requirements.txt
```

## Usage 🔧
1. Clone the repository and navigate to the project directory.

2. Download the datasets and place them in the appropriate directories (`../ml-100k` and `../ml-1m`):
   - [MovieLens 100K Dataset (ml-100k.zip)](http://files.grouplens.org/datasets/movielens/ml-100k.zip)
   - [MovieLens 1M Dataset (ml-1m.zip)](http://files.grouplens.org/datasets/movielens/ml-1m.zip)

3. Run the main Python script:

   ```bash
   python fstdmf.py
   ```

4. The script will automatically:
   - Train the DMF and FSTDMF models across different train/test ratios (70%, 50%, 30%).
   - Perform dimension and layer sensitivity analyses.
   - Run the ablation study.
   - Save the comparative charts as PNG files.

## Reproduction Results & Insights 📊
- **Successful Reproduction**: The implementation successfully reproduces the core claims of the original paper, confirming that the FSTDMF model consistently outperforms the standard DMF model, particularly in highly sparse scenarios (e.g., 30% training ratio).
- **Ablation Insights**: The custom ablation pipeline empirically demonstrates that intelligent initialization (SA) and feature transfer (FST) work synergistically to prevent overfitting.
- **Optimization Trade-offs**: In this implementation, the `Adam` optimizer was utilized for PyTorch compatibility and memory efficiency, offering a robust alternative to the `iRprop+` algorithm used in the original paper for large-scale sparse matrices.

## Acknowledgements ✨

This reproduction project was developed by Mohsen Elahifard as part of the Big Data Analytics coursework under the supervision of Dr. Mostafa Haghir Chehreghani.

Feel free to contribute by submitting issues or pull requests! 🎉
