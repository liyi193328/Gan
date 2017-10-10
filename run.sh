#!/usr/bin/env bash
cd /home/bigdata/cwl/Gan/

/home/bigdata/anaconda3/bin/python train.py --model_name=h_kolod --train_datapath=/home/bigdata/cwl/Gan/data/cluster/h_kolod.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/cluster/h_kolod.train --epoch=50 --batch_size=32 > ./run_logs/h_kolod.log 2>&1
/home/bigdata/anaconda3/bin/python train.py --model_name=h_usoskin --train_datapath=/home/bigdata/cwl/Gan/data/cluster/h_usoskin.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/cluster/h_usoskin.train --epoch=50 --batch_size=32 > ./run_logs/h_usoskin.log 2>&1
/home/bigdata/anaconda3/bin/python train.py --model_name=h_pollen --train_datapath=/home/bigdata/cwl/Gan/data/h_pollen.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/h_pollen.train --epoch=50 --batch_size=32 > ./run_logs/h_pollen.log 2>&1

/home/bigdata/anaconda3/bin/python train.py --model_name=h_kolod_0.9 --truly_mis_pro=0.9 --train_datapath=/home/bigdata/cwl/Gan/data/cluster/h_kolod.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/cluster/h_kolod.train --epoch=50 --batch_size=32 > ./run_logs/h_kolod_0.9.log 2>&1
/home/bigdata/anaconda3/bin/python train.py --model_name=h_usoskin_0.9 truly_mis_pro=0.9 --train_datapath=/home/bigdata/cwl/Gan/data/cluster/h_usoskin.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/cluster/h_usoskin.train --epoch=50 --batch_size=32 > ./run_logs/h_usoskin_0.9.log 2>&1
/home/bigdata/anaconda3/bin/python train.py --model_name=h_pollen_0.9 truly_mis_pro=0.9 --train_datapath=/home/bigdata/cwl/Gan/data/h_pollen.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/h_pollen.train --epoch=50 --batch_size=32 > ./run_logs/h_pollen_0.9.log 2>&1

/home/bigdata/anaconda3/bin/python train.py --model_name=chu --train_datapath=/home/bigdata/cwl/Gan/data/chu/chu_sc_handle.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/chu/chu_sc_handle.train --epoch=50 --batch_size=32 > ./run_logs/chu_sc_handle.log 2>&1
/home/bigdata/anaconda3/bin/python train.py --model_name=chu_0.9 --train_datapath=/home/bigdata/cwl/Gan/data/chu/chu_sc_handle.train --infer_complete_datapath=/home/bigdata/cwl/Gan/data/chu/chu_sc_handle.train --epoch=50 --batch_size=32 > ./run_logs/chu_sc_handle_0.9.log 2>&1
