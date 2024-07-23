# pLM4CPPs

## Overview
This repository contains pre-trained models, datasets, and Jupyter notebooks for predicting cell-penetrating peptides (CPPs) using various embeddings and neural network architectures.

## Directory Structure
- **`dataset/`**: Contains datasets for training and evaluation.
- **`models/`**: Contains pre-trained models organized by type.
- **`notebooks/`**: Jupyter notebooks for experimentation and analysis.
- **`src/`**: Source code files for model training and prediction.

## Models
- `ESM2-1280/`: Models trained with ESM2-1280 embeddings.
- `ESM2-320/`: Models trained with ESM2-320 embeddings.
- `ESM2-480/`: Models trained with ESM2-480 embeddings.
- `ESM2-640/`: Models trained with ESM2-640 embeddings.
- `Port-T5-BFD/`: Models trained with Port-T5-BFD embeddings.
- `SeqVec/`: Models trained with SeqVec embeddings.

## Datasets
- `pLM4CPPs_dataset_CPP.xlsx` and `pLM4CPPs_dataset_Non-CPP.xlsx`: Datasets used for model training and evaluation.
- `kelm_dataset_CPP.csv` and `kelm_dataset_Non-CPP.csv`: Independent Datasets used for model evaluation.

## Embedded Data
We provide pre-computed embeddings for ease of use. These files are located in the `embedded_data/` directory.

- **`Final_non_redundant_sequences.xlsx`**: A dataset containing non-redundant sequences.
- **`kelm_dataset.csv`**: KELM dataset with embeddings.
- **`prot_t5_xl_bfd_per_protein_embeddings.csv`**: Prot-T5 embeddings with BFD.
- **`seqvev_whole_smaple_reduced_embeddings_file_ordered.csv`**: SeqVec embeddings.
- **`whole_sample_dataset_esm2_t12_35M_UR50D_unified_480_dimension.csv`**: ESM2 480-dimensional embeddings.
- **`whole_sample_dataset_esm2_t30_150M_UR50D_unified_640_dimension.csv`**: ESM2 640-dimensional embeddings.
- **`whole_sample_dataset_esm2_t33_650M_UR50D_unified_1280_dimension.csv`**: ESM2 1280-dimensional embeddings.
- **`whole_sample_dataset_esm2_t6_8M_UR50D_unif

## Notebooks
Contains Jupyter notebooks for each model type, demonstrating how to use the models for predictions.

## Usage

1. Requirements (For example ESM-320), If you find something missing, please visit the embeddings.ipynb for respective language models.

pip install fair-esm torch tensorflow sklearn biopython h5py

2. Prepared your Data
Save your peptide sequences in a xlsx file named `user.xlsx`. The file should contain at least one column named `sequence` with your peptide sequences.

3. Generating Embeddings
	Open `scripts/ESM2_320_embeddings.ipynb`.
	Replace `Final_non_redundant_sequences.xlsx` with your dataset file.
	Run the notebook to generate embeddings.
	Save the output as `user_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv`.

4. ing Predictions
Use the following Python script to make predictions on your embedded data:

```python
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the trained model
saved_model = load_model('models/ESM2-320/model_file.h5')

# Load the user's external dataset
dataset_external = pd.read_excel('user.xlsx', na_filter=False)

# Load the embedded data
X_external_data_name = 'user_dataset_esm2_t6_8M_UR50D_unified_320_dimension.csv'
X_external_data = pd.read_csv(X_external_data_name, header=0, index_col=0, delimiter=',')
X_external = np.array(X_external_data)

# Normalize the external dataset
scaler = StandardScaler().fit(X_external)  # Fit scaler on training data if available
X_external_normalized = scaler.transform(X_external)

# Predict probabilities for external dataset
predicted_probas_ext = saved_model.predict(X_external_normalized, batch_size=32)

# Convert probabilities to class labels
predicted_classes_ext = (predicted_probas_ext > 0.5).astype(int)  # Use 0.5 as a default threshold

# Save predictions to a CSV file
predictions_df = pd.DataFrame(predicted_classes_ext, columns=['Prediction'])
predictions_df.to_csv('predictions.csv', index=False)

print("Predictions have been saved to 'predictions.csv'.")


#### **7. Results**

How to interpret the results:

```markdown
The `predictions.csv` file will contain the following columns:
- `Prediction`: `1` for Cell Penetrating Peptides and `0` for Non-Cell Penetrating Peptides.

8. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
