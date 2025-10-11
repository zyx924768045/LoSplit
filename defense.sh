# 默认设备ID为0
DEVICE_ID=0

# 解析命令行参数
for arg in "$@"; do
    if [[ $arg == --device_id=* ]]; then
        DEVICE_ID="${arg#*=}"
    fi
done


#!/bin/bash

# GTA
python GTA_LoSplit.py --dataset=Cora --vs_number=48 --trigger_generator_address=./weights/GTA/Cora/GTA_Cora_weights.pth --pre_train_param=./weights/GTA/Cora/GTA_Cora.pt --split_lr=0.01 --split_epoch=10
python GTA_LoSplit.py --dataset=Citeseer --vs_number=43 --trigger_generator_address=./weights/GTA/Citeseer/GTA_Citeseer_weights.pth --pre_train_param=./weights/GTA/Citeseer/GTA_Citeseer.pt --split_lr=0.05 --split_epoch=10
python GTA_LoSplit.py --dataset=Pubmed --vs_number=201 --trigger_generator_address=./weights/GTA/Pubmed/GTA_Pubmed_weights.pth --pre_train_param=./weights/GTA/Pubmed/GTA_Pubmed.pt --split_lr=0.01 --split_epoch=10
python GTA_LoSplit.py --dataset=Flickr --vs_number=167 --trigger_generator_address=./weights/GTA/Flickr/GTA_Flickr_weights.pth --pre_train_param=./weights/GTA/Flickr/GTA_Flickr.pt --split_lr=0.03 --split_epoch=10
python GTA_LoSplit.py --dataset=Physics --vs_number=200 --trigger_generator_address=./weights/GTA/Physics/GTA_Physics_weights.pth --pre_train_param=./weights/GTA/Physics/GTA_Physics.pt --split_lr=0.00001 --split_epoch=20
python GTA_LoSplit.py --dataset=ogbn-arxiv --vs_number=569 --trigger_generator_address=./weights/GTA/OGB-arXiv/GTA_ogbn_arxiv_weights.pth --pre_train_param=./weights/GTA/OGB-arXiv/GTA_ogbn_arxiv.pt --hidden=128 --split_lr=0.03 --split_epoch=15

# UGBA
python UGBA_LoSplit.py --dataset=Cora --vs_number=44 --homo_loss_weight=50 --homo_boost_thrd=0.5 --trigger_generator_address=./weights/UGBA/Cora/UGBA_Cora_weights.pth --pre_train_param=./weights/UGBA/Cora/UGBA_Cora.pt --split_lr=0.01 --split_epoch=20
python UGBA_LoSplit.py --dataset=Citeseer --vs_number=42 --homo_loss_weight=150 --homo_boost_thrd=0.3 --trigger_generator_address=./weights/UGBA/Citeseer/UGBA_Citeseer_weights.pth --pre_train_param=./weights/UGBA/Citeseer/UGBA_Citeseer.pt --split_lr=0.01 --split_epoch=10
python UGBA_LoSplit.py --dataset=Pubmed --vs_number=162 --homo_loss_weight=100 --homo_boost_thrd=0.5 --trigger_generator_address=./weights/UGBA/Pubmed/UGBA_Pubmed_weights.pth --pre_train_param=./weights/UGBA/Pubmed/UGBA_Pubmed.pt --split_lr=0.03 --split_epoch=15
python UGBA_LoSplit.py --dataset=Physics --vs_number=394 --hidden=64 --homo_loss_weight=200 --homo_boost_thrd=0.8 --trigger_generator_address=./weights/UGBA/Physics/UGBA_Physics_weights.pth --pre_train_param=./weights/UGBA/Physics/UGBA_Physics.pt --split_lr=0.001 --split_epoch=10
python UGBA_LoSplit.py --dataset=Flickr --vs_number=169 --homo_loss_weight=100 --homo_boost_thrd=0.5 --trigger_generator_address=./weights/UGBA/Flickr/UGBA_Flickr_weights.pth --pre_train_param=./weights/UGBA/Flickr/UGBA_Flickr.pt --split_lr=0.01 --split_epoch=15
python UGBA_LoSplit.py --dataset=ogbn-arxiv --vs_number=565 --homo_loss_weight=200 --homo_boost_thrd=0.8 --trigger_generator_address=./weights/UGBA/OGB-arXiv/UGBA_ogbn_arxiv_weights.pth --pre_train_param=./weights/UGBA/OGB-arXiv/UGBA_ogbn_arxiv.pt --trojan_epoch=800 --epoch=800 --split_lr=0.03 --split_epoch=15

# DPGBA
python ./DPGBA/DPGBA_LoSplit.py --dataset=Cora --vs_number=50 --k=50 --hidden=32 --weight_target=1 --weight_ood=1 --weight_targetclass=2 --train_lr=0.015 --lr=0.015 --target_class=0 --trojan_epochs=300 --trigger_generator_address=./weights/DPGBA/Cora/DPGBA_Cora_weights.pth --pre_train_param=./weights/DPGBA/Cora/DPGBA_Cora.pt --split_lr=0.01 --split_epoch=15
python ./DPGBA/DPGBA_LoSplit.py --dataset=Citeseer --vs_number=45 --k=50 --hidden=32 --weight_target=1 --weight_ood=1 --weight_targetclass=2 --train_lr=0.015 --lr=0.015 --target_class=0 --trojan_epochs=300 --trigger_generator_address=./weights/DPGBA/Citeseer/DPGBA_Citeseer_weights.pth --pre_train_param=./weights/DPGBA/Citeseer/DPGBA_Citeseer.pt --split_lr=0.01 --split_epoch=10
python ./DPGBA/DPGBA_LoSplit.py --dataset=Pubmed --vs_number=194 --k=50 --hidden=16 --weight_target=1 --weight_ood=2 --weight_targetclass=5 --train_lr=0.015 --lr=0.015 --target_class=0 --trojan_epochs=400 --epochs=200 --trigger_generator_address=./weights/DPGBA/Pubmed/DPGBA_Pubmed_weights.pth --pre_train_param=./weights/DPGBA/Pubmed/DPGBA_Pubmed.pt --split_lr=0.01 --split_epoch=40
python ./DPGBA/DPGBA_LoSplit.py --dataset=Physics --vs_number=200 --k=50 --hidden=128 --weight_target=1 --weight_ood=50 --weight_targetclass=10 --train_lr=0.002 --lr=0.002 --target_class=0 --trojan_epochs=400 --epochs=200 --trigger_generator_address=./weights/DPGBA/Physics/DPGBA_Physics_weights.pth --pre_train_param=./weights/DPGBA/Physics/DPGBA_Physics.pt --split_lr=0.0001 --split_epoch=20 
python ./DPGBA/DPGBA_LoSplit.py --dataset=Flickr --vs_number=166 --k=50 --hidden=128 --weight_target=10 --weight_ood=1 --weight_targetclass=20 --train_lr=0.002 --lr=0.002 --target_class=0 --trojan_epochs=400 --epochs=200 --trigger_generator_address=./weights/DPGBA/Flickr/DPGBA_Flickr_weights.pth --pre_train_param=./weights/DPGBA/Flickr/DPGBA_Flickr.pt --split_lr=0.01 --split_epoch=15
# python ./DPGBA_arxiv/DPGBA_LoSplit.py --dataset=ogbn-arxiv --vs_number=565 --k=50 --hidden=512 --weight_target=25 --weight_ood=1 --weight_targetclass=10 --train_lr=0.015 --lr=0.015 --target_class=0 --trojan_epochs=800 --epoch=800 --trigger_generator_address=./weights/DPGBA/OGB-arXiv/DPGBA_ogbn_arxiv_weights.pth --pre_train_param=./weights/DPGBA/OGB-arXiv/DPGBA_ogbn_arxiv.pt --split_lr=0.03 --split_epoch=15
python ./DPGBA_arxiv/DPGBA_LoSplit.py --dataset=ogbn-arxiv --homo_loss_weight=500 --vs_number=569  --k=10 --hidden=32 --weight_target=20 --weight_ood=1 --weight_targetclass=30 --train_lr=0.01 --lr=0.01 --target_class=0 --trojan_epochs=81 --epoch=500 --rec_epochs=300 --range=0.1 --trigger_generator_address=./weights/DPGBA/OGB-arXiv/DPGBA_ogbn_arxiv_weights.pth --pre_train_param=./weights/DPGBA/OGB-arXiv/DPGBA_ogbn_arxiv.pt --split_lr=0.03 --split_epoch=15


# SPEAR
python SPEAR_LoSplit.py --dataset=Cora --homo_loss_weight=0.1 --target_loss_weight=1 --vs_number=50 --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --alpha_int=30 --hidden=80 --shadow_lr=0.0002 --trojan_lr=0.0002 --trigger_generator_address=./weights/SPEAR/Cora/SPEAR_Cora_weights.pth --pre_train_param=./weights/SPEAR/Cora/SPEAR_Cora.pt --split_lr=0.008 --split_epoch=10
python SPEAR_LoSplit.py --dataset=Citeseer --homo_loss_weight=0.05 --target_loss_weight=1 --vs_number=47 --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --alpha_int=10 --hidden=64 --shadow_lr=0.0002 --trojan_lr=0.0002 --trigger_generator_address=./weights/SPEAR/Citeseer/SPEAR_Citeseer_weights.pth --pre_train_param=./weights/SPEAR/Citeseer/SPEAR_Citeseer.pt --split_lr=0.08 --split_epoch=20
python SPEAR_LoSplit.py --dataset=Pubmed --homo_loss_weight=0.15 --vs_number=222 --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --alpha_int=10 --hidden=64 --target_class=2 --shadow_lr=0.005 --trojan_lr=0.005 --trigger_generator_address=./weights/SPEAR/Pubmed/SPEAR_Pubmed_weights.pth --pre_train_param=./weights/SPEAR/Pubmed/SPEAR_Pubmed.pt --split_lr=0.03 --split_epoch=15
python SPEAR_LoSplit.py --dataset=Flickr --vs_number=166 --homo_loss_weight=0.15  --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --alpha_int=15 --hidden=64 --target_class=0 --shadow_lr=0.005 --trojan_lr=0.005 --trigger_generator_address=./weights/SPEAR/Flickr/SPEAR_Flickr_weights.pth --pre_train_param=./weights/SPEAR/Flickr/SPEAR_Flickr.pt --split_lr=0.03 --split_epoch=15
python SPEAR_LoSplit.py --dataset=Physics --vs_number=207 --homo_loss_weight=0.15  --test_model=GCN --defense_mode=none --epochs=200 --trojan_epochs=200 --alpha_int=15 --hidden=64 --target_class=0 --shadow_lr=0.0005 --trojan_lr=0.0005 --trigger_generator_address=./weights/SPEAR/Physics/GTA_Physics_weights.pth --pre_train_param=./weights/SPEAR/Physics/SPEAR_Physics.pt --split_lr=0.001 --split_epoch=20
python SPEAR_LoSplit.py --dataset=ogbn-arxiv --homo_loss_weight=0 --vs_number=579 --test_model=GCN --defense_mode=none --epochs=800 --trojan_epochs=800 --alpha_int=5 --hidden=80 --outter_size=256 --shadow_lr=0.01 --trojan_lr=0.01 --train_lr=0.02 --trigger_generator_address=./weights/SPEAR/OGB-arXiv/SPEAR_ogbn_arxiv_weights.pth --pre_train_param=./weights/SPEAR/OGB-arXiv/SPEAR_ogbn-arxiv.pt --split_lr=0.1 --split_epoch=15

