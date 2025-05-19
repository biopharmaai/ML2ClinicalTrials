# TrialBench: Multi-modal AI-ready Clinical Trial Datasets (R Interface)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Installation 

### 1.1 Install R Dependencies

```r
install.packages("reticulate") 
library(reticulate)
conda_create("r_trialbench", python_version = "3.10")
use_condaenv("r_trialbench")
reticulate::py_install("trialbench", pip = TRUE)
```

### 1.2 Manully Download  `mesh_embeddings.txt.gz` File

Due to the package size limitation, please download `mesh_embeddings.txt.gz` from [here](https://github.com/ML2Health/ML2ClinicalTrials/blob/main/AI4Trial/data/mesh-embeddings/mesh_embeddings.txt.gz). Then copy it to your `trialbench` path as

```bash
cp mesh_embeddings.txt.gz your_path_to/miniconda3/envs/r_trialbench/lib/python3.10/site-packages/trialbench/data/mesh-embeddings/
```

## 2. Supported Tasks & Phases

| Task                         | Task Name                                            | Phase Name |
| ---------------------------- | ---------------------------------------------------- | ---------- |
| Mortality Prediction         | `mortality_rate`/`mortality_rate_yn`             | 1-4        |
| Adverse Event Prediction     | `serious_adverse_rate`/`serious_adverse_rate_yn` | 1-4        |
| Patient Retention Prediction | `patient_dropout_rate`/`patient_dropout_rate_yn` | 1-4        |
| Trial Duration Prediction    | `duration`                                         | 1-4        |
| Trial Outcome Prediction     | `outcome`                                          | 1-4        |
| Trial Failure Analysis       | `failure_reason`                                   | 1-4        |
| Dosage Prediction            | `dose`/`dose_cls`                                | All        |

### Clinical Trial Phase Definitions

```
Phase 1: Safety Evaluation
Phase 2: Efficacy Assessment
Phase 3: Large-scale Testing
Phase 4: Post-marketing Surveillance
```

## 3. Quick Start

### 3.1 Basic Usage

```r
# Source the <function.R> first
source("your_path_to/r.trialbench/R/function.R", encoding = "UTF-8")

# Download datasets (optional)
save_path <- "data/"
download_all_data(save_path)

# Load dataset
task <- "dose"
phase <- "All"

# Retrieve DataFrames
data_list <- load_data(task, phase)
train_df <- data_list$train_df
valid_df <- data_list$valid_df
test_df <- data_list$test_df
```

### 3.2 Data Schema

Dosage Prediction tasks provide:

- `nctid_lst`: ClinicalTrials.gov identifiers
- `smiles_lst`: Molecular structures in SMILES notation
- `mesh_lst`: Medical Subject Headings (MeSH) terms

Other tasks additionally include:

- `icdcode_lst`: ICD diagnosis codes
- `criteria_lst`: Eligibility criteria
- `tabular_lst`: Structured trial metadata
- `text_lst`: Unstructured protocol text

Labels are accessible via:

```r
# Access label vectors
train_labels <- train_df$label_lst  # Returns [nctid, max, min, avg] dosage values, e.g. []
```

## 4. API Documentation

### `load_data()`

| Parameter | Type   | Description                            |
| --------- | ------ | -------------------------------------- |
| `task`  | string | Target prediction task identifier      |
| `phase` | string | Clinical trial phase (e.g., 'Phase 2') |

Returns a list containing:

- `train_df`, `valid_df`, `test_df`: Split datasets as DataFrames
- `num_classes`: Number of output classes (classification tasks)
- `tabular_input_dim`: Dimensionality of tabular features

## 5. Citation

If using TrialBench in research, cite:

```bibtex
@article{chen2024trialbench,
  title={Trialbench: Multi-modal artificial intelligence-ready clinical trial datasets},
  author={Chen, Jintai and Hu, Yaojun and Wang, Yue and Lu, Yingzhou and Cao, Xu and Lin, Miao and Xu, Hongxia and Wu, Jian and Xiao, Cao and Sun, Jimeng and others},
  journal={arXiv preprint arXiv:2407.00631},
  year={2024}
}
```

## 6. Troubleshooting

Validate your environment:

```r
# Check Python connectivity
reticulate::py_module_available("trialbench")  # Should return TRUE

# Verify package version
reticulate::py_package_version("trialbench")   # Should be ≥0.3.0
```
