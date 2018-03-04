#encoding=utf-8

import os
import codecs
import subprocess

data_dir = "F:/project/Gan/data"
python_path = "/home/bigdata/anaconda3/bin/python"
script_path = "F:/project/Gan/train.py"
done_path = "F:/project/Gan/done_train.txt"
done_model_names = []

for dropout in [0.2,0.4,0.6,0.8]:
  for random_mask in [0.2, 0.4, 0.6, 0.8]:
    for file in os.listdir(data_dir):
      path = os.path.join(data_dir, file)
      name = file.split(".")[0]
      if name == "h_brain":
        continue
      drop_flag = dropout
      random_mask_flag = random_mask
      model_name = "{}_{}_{}".format(name, dropout, random_mask)
      outDir = "./prediction_epo_500/dropout_{}/random_mask_{}/".format(dropout, random_mask_flag)
      if model_name in done_model_names:
        print("{} has been trained before".format(model_name))
        continue
      par_dict = {
        "python": python_path,
        "script_path": script_path,
        "model_name": model_name,
        "drop_flag": drop_flag,
        "random_mask_flag":random_mask_flag,
        "datapath":path,
        "outDir": outDir
      }
      # cmd = "{python} {script_path} --model_name={model_name} --dropout={drop_flag} --truly_mis_pro={random_mask_flag} --train_datapath={datapath} " \
      #       "--infer_complete_datapath={datapath} --outDir={outDir} --epoch=150 --batch_size=32 > ./run_logs/{model_name}.log 2>&1".format(
      #   **par_dict
      # )

      cmd = "python {script_path} --model_name={model_name} --dropout={drop_flag} --truly_mis_pro={random_mask_flag} --train_datapath={datapath} " \
            "--infer_complete_datapath={datapath} --outDir={outDir} --epoch=500 --batch_size=32 > ./run_logs/{model_name}.log 2>&1".format(
        **par_dict
      )
      print("running {}...".format(cmd))
      ret = subprocess.check_call(cmd, shell=True, cwd="F:/project/Gan")
      print(ret)
