conda create -n py38-cu113 python=3.8.13
conda activate py38-cu113

sudo apt-get --purge remove "*cuda*"
conda install -c conda-forge cudatoolkit=11.3
sudo sh cuda_11.3.1_465.19.01_linux.run --silent --toolkit

conda install pytorch==1.12.1 torchvision=0.13.1 torchaudio=0.12.1 pytorch-cuda=11.3 -c pytorch -c nvidia
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.1+cu113.html
pip install scikit-learn-extra