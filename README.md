# pLM4CPPs

pLM4CPPs predicts cell-penetrating peptides (CPPs) using deep learning and protein language models. It integrates CNNs for feature extraction, achieving high accuracy and reliability. The project evaluates multiple embeddings from sources like BEPLER, CPCProt, SeqVec, ESM, and ProtT5-XL. The implementation is based on Kumar et al., 2024 (Submitted).

## Overview
This repository contains pre-trained models, datasets, and Jupyter notebooks for predicting CPPs. It provides tools for embedding generation, model training, and evaluation.

## Directory Structure
- **`dataset/`**: Contains datasets for training and evaluation.
- **`models/`**: Pre-trained models organized by type.
- **`notebooks/`**: Jupyter notebooks for experimentation and analysis.
- **`src/`**: Source code files for model training and prediction.
- **`embedded_data/`**: Pre-computed embeddings for use with the models.

## Models
- **`ESM2-1280/`**: Models trained with ESM2-1280 embeddings.
- **`ESM2-320/`**: Models trained with ESM2-320 embeddings.
- **`ESM2-480/`**: Models trained with ESM2-480 embeddings.
- **`ESM2-640/`**: Models trained with ESM2-640 embeddings.
- **`Prot-T5-BFD/`**: Models trained with Prot-T5-BFD embeddings.
- **`SeqVec/`**: Models trained with SeqVec embeddings.

## Datasets
- **`pLM4CPPs_dataset_CPP.xlsx`** and **`pLM4CPPs_dataset_Non-CPP.xlsx`**: Datasets for training and evaluation.
- **`kelm_dataset_CPP.csv`** and **`kelm_dataset_Non-CPP.csv`**: Independent datasets for evaluation.

## Embedded Data
- **`Final_non_redundant_sequences.xlsx`**: Non-redundant sequences dataset.
- **`kelm_dataset.csv`**: KELM dataset with embeddings.
- **`prot_t5_xl_bfd_per_protein_embeddings.csv`**: Prot-T5 embeddings with BFD.
- **`seqvec_whole_sample_reduced_embeddings_file_ordered.csv`**: SeqVec embeddings.
- **`whole_sample_dataset_esm2_t12_35M_UR50D_unified_480_dimension.csv`**: ESM2 480-dimensional embeddings.
- **`whole_sample_dataset_esm2_t30_150M_UR50D_unified_640_dimension.csv`**: ESM2 640-dimensional embeddings.
- **`whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv`**: ESM2 1280-dimensional embeddings.

## Usage

### Preparing Your Data
1. Save your peptide sequences in an XLSX file named `user.xlsx`.
2. Ensure the file contains a column named `sequence` with your peptide sequences.

### Generating Embeddings
1. Open `scripts/ESM2_320_embeddings.ipynb`.
2. Replace `Final_non_redundant_sequences.xlsx` with your dataset file.
3. Run the notebook to generate embeddings.
4. Save the output as `user_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv`.

### Making Predictions
Use the following Python script to make predictions:

```python
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
saved_model = load_model('models/ESM2-320/model_file.h5')

# Load the user's dataset
dataset_external = pd.read_excel('user.xlsx', na_filter=False)

# Load the embedded data
X_external_data_name = 'user_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_external_data = pd.read_csv(X_external_data_name, header=0, index_col=0, delimiter=',')
X_external = np.array(X_external_data)

# Normalize the external dataset
scaler = StandardScaler().fit(X_external)  # Fit scaler on the external data if training data is not available
X_external_normalized = scaler.transform(X_external)

# Predict probabilities for external dataset
predicted_probas_ext = saved_model.predict(X_external_normalized, batch_size=32)

# Convert probabilities to class labels
predicted_classes_ext = (predicted_probas_ext > 0.5).astype(int)

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predicted_classes_ext, columns=['Prediction'])
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions have been saved to 'predictions.csv'.")
