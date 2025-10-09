# LoSplit: Loss-Guided Dynamic Split for Training-Time Defense Against Graph Backdoor Attacks [NeurIPS 25 Poster]
This repository provides the implementation of **LoSplit**, a novel defense in graph for mitigating graph backdoor attacks. Our method supports various attack settings, including **GTA**, **UGBA**, **DPGBA**, and **SPEAR**.

## 🚀 Reproducing Results

To reproduce the results reported in the paper, run the following scripts:

bash install.sh
bash defense.sh [--device_id=]


- If you want to run the code on a specific GPU, specify it via the optional argument `--device_id=`

- if you want to specify the device for a specific task,you should modify the device_id in the `defense.sh` file.

This Implementation is built upon [SPEAR](github.com/yhDing/SPEAR)