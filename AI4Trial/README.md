# TrialBench: Multi-modal AI-ready Clinical Trial Datasets

[![PyPI version](https://img.shields.io/pypi/v/trialbench.svg?color=brightgreen)](https://pypi.org/project/trialbench/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 1. Installation

```bash
pip install trialbench
```

## 2. Tasks & Phases

| Supported Tasks              | Task Name                                            | Phase Name |
| ---------------------------- | ---------------------------------------------------- | ---------- |
| Mortality Prediction         | `mortality_rate`/`mortality_rate_yn`             | 1-4        |
| Adverse Event Prediction     | `serious_adverse_rate`/`serious_adverse_rate_yn` | 1-4        |
| Patient Retention Prediction | `patient_dropout_rate`/`patient_dropout_rate_yn` | 1-4        |
| Trial Duration Prediction    | `duration`                                         | 1-4        |
| Trial Outcome Prediction     | `outcome`                                          | 1-4        |
| Trial Failure Analysis       | `failure_reason`                                   | 1-4        |
| Dosage Prediction            | `dose`/`dose_cls`                                | All        |

### Clinical Trial Phases

```
Phase 1: Safety Evaluation
Phase 2: Efficacy Assessment
Phase 3: Large-scale Testing
Phase 4: Post-marketing Surveillance
```

### 3. Quick Start

#### 3.1 Usage of `trialbench`

```python
import trialbench

# Download all datasets at once (optional)
save_path = 'data/'
trialbench.function.download_all_data(save_path)

# Load dataset
task = 'dose'
phase = 'All'

# Load dataloader.Dataloader 
train_loader, valid_loader, test_loader, num_classes, tabular_input_dim = trialbench.function.load_data(task, phase, data_format='dl')
# or Load pd.Dataframe
train_df, valid_df, test_df, num_classes, tabular_input_dim = trialbench.function.load_data(task, phase, data_format='df')
```

#### 3.2 Attributes of Each Task

Each task provides different feature sets. The Dosage Prediction task returns `nctid_lst`, `smiles_lst`, and `mesh_lst`, while all other tasks provide `nctid_lst`, `icdcode_lst`, `smiles_lst`, `criteria_lst`, `tabular_lst`, `text_lst`, and `mesh_lst`.

All tasks use `label_lst` as the label variable. Please refer to the guide documentation for detailed feature as well as label descriptions.

```
# Demo for accessing data elements
task = 'dose'
phase = 'All'

# When using DataLoader objects:
# Features
nctid_list = train_loader.dataset.nctid_lst
smiles_list = train_loader.dataset.smiles_lst
mesh_list = train_loader.dataset.mesh_lst
# Labels
# return [datatset_name, label_max, label_min, label_avg], e.g. ['NCT03422510', 2, 2, 2]
label_list = train_loader.dataset.label_lst 

# When using DataFrames:
# Features
nctid_list = train_df.nctid_lst
smiles_list = train_df.smiles_lst
mesh_list = train_df.mesh_lst
# Labels
label_list = train_df.label_lst
```

4. Data Loading

### `load_data` Parameters

| Parameter       | Type | Description                                              |
| --------------- | ---- | -------------------------------------------------------- |
| `task`        | str  | Target prediction task (e.g., 'mortality_rate_yn')       |
| `phase`       | int  | Clinical trial phase (1-4)                               |
| `data_format` | str  | Data format ('dl' for Dataloader, 'df' for pd.DataFrame) |

## 5. Citation

If you use TrialBench in your research, please cite:

```bibtex
@article{chen2024trialbench,
  title={Trialbench: Multi-modal artificial intelligence-ready clinical trial datasets},
  author={Chen, Jintai and Hu, Yaojun and Wang, Yue and Lu, Yingzhou and Cao, Xu and Lin, Miao and Xu, Hongxia and Wu, Jian and Xiao, Cao and Sun, Jimeng and others},
  journal={arXiv preprint arXiv:2407.00631},
  year={2024}
}
```
