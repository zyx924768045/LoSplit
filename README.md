# LoSplit: Loss-Guided Dynamic Split for Training-Time Defense Against Graph Backdoor Attacks [NeurIPS 25] [paper](https://openreview.net/forum?id=3Cpw7YftBm)
This repository provides the implementation of **LoSplit**, the first training-time defense for mitigating graph backdoor attacks. Our method supports various attack settings, including **GTA**, **UGBA**, **DPGBA**, and **SPEAR**.

## 🚀 Reproducing Results
1. Download the weights and pre-training parameters from here: https://www.dropbox.com/scl/fo/iw2nouqgqx7nz7y32043w/AMAJNKdEMfBz0v-H6omERS0?rlkey=2d3ane6ak6kzhv79xsvhx2j72&e=1&st=4d7rd61z&dl=0

2. Modify the address for these **2** args: '**(1) --trigger_generator_address**', '(2) **--pre_train_param (induding poison_x, poison_edge_index, poison_edge_weights, poison_labels, idx_poison_found, idx_clean_found, target_label)**'

To reproduce the results reported in the paper, run the following scripts:

bash install.sh
bash defense.sh [--device_id=]


- If you want to run the code on a specific GPU, specify it via the optional argument `--device_id=`

- if you want to specify the device for a specific task,you should modify the device_id in the `defense.sh` file.

This Implementation is built upon [SPEAR](github.com/yhDing/SPEAR)