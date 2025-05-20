# GTVelo
**_RNA velocity inference based on graph transformer_**

## Installation
The environment can be set up using either of the following methods:
### Using conda
conda env create -f environment.yml

### Using pip
pip install -r requirements.txt

# data
https://doi.org/10.5281/zenodo.15251792

```
data/             
├── raw/          # Original experimental data
└── processed/    # Preprocessed adata and latent_adata objects
```

# GTvelo - Usage Steps

## Installation: 

Install the package dependencies using either the requirements.txt file or environment.yml file to set up the necessary environment.

## Data Acquisition:

Obtain the single-cell RNA sequencing datasets from the Zenodo database for RNA velocity analysis.

## Model Training and Analysis:

Use the gtvelo package to initialize a VAE model with models.vae_model, train it with train_vae.py, and then infer RNA velocity with output_results for downstream trajectory analysis.

```
notebook/
├── fig2/
├── fig3/
│── fig4/
└── fig5/
```
