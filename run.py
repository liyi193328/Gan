#encoding=utf-8

import os
import subprocess

data_dir = "/home/bigdata/cwl/Gan/data/cluster"
python_path = "/home/bigdata/anaconda3/bin/python"
script_path = "/home/bigdata/cwl/Gan/train.py"
done_list = ["h_pollen_0_0", "h_pollen_0_0.9"]

for file in os.listdir(data_dir):
  path = os.path.join(data_dir, file)
  name = file.split(".")[0]
  for dropout in [0, 0.1]:
    for random_mask in [0, 0.9]:
      drop_flag = dropout
      random_mask_flag = random_mask
      model_name = "{}_{}_{}".format(name, dropout, random_mask)
      if model_name in done_list:
        print("{} done before".format(model_name))
        continue
      par_dict = {
        "python": python_path,
        "script_path": script_path,
        "model_name": model_name,
        "drop_flag": drop_flag,
        "random_mask_flag":random_mask_flag,
        "datapath":path
      }
      cmd = "{python} {script_path} --model_name={model_name} --dropout={drop_flag} --truly_mis_pro={random_mask_flag} --train_datapath={datapath} --infer_complete_datapath={datapath} --epoch=100 --batch_size=32 > ./run_logs/{model_name}.log 2>&1".format(
        **par_dict
      )
      print("running {}...".format(cmd))
      ret = subprocess.check_call(cmd, shell=True, cwd="/home/bigdata/cwl/Gan")