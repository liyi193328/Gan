#encoding=utf-8

import os
import codecs
import subprocess

data_dir = "/home/bigdata/cwl/Gan/data/cluster"
python_path = "/home/bigdata/anaconda3/bin/python"
script_path = "/home/bigdata/cwl/Gan/train.py"
done_path = "/home/bigdata/cwl/Gan/done_train.txt"

mask_dir = "/home/bigdata/cwl/Gan/cluster/mask"
mask_file_suffixs = [v.split(".")[0] for v in os.listdir(mask_dir)]

f = codecs.open(done_path, "r", "utf-8")
done_model_names = [v.strip() for v in f.readlines()]
f.close()
f = codecs.open(done_path, "w", "utf-8")

done_model_names = []

for file in os.listdir(data_dir):
  path = os.path.join(data_dir, file)
  name = file.split(".")[0]
  flag = False
  mask_path = ""
  for mask_file in mask_file_suffixs:
    if mask_file in name:
      flag = True
      mask_path = os.path.join(mask_dir, mask_file)

  if flag is True:
    for dropout in [0]:
      for random_mask in [0]:
        drop_flag = dropout
        random_mask_flag = random_mask
        model_name = "{}_{}_{}".format(name, dropout, random_mask)
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
          "random_mask_path": mask_path
        }
        cmd = "{python} {script_path} --model_name={model_name} --dropout={drop_flag} --truly_mis_pro={random_mask_flag} --train_datapath={datapath} " \
              "--infer_complete_datapath={datapath} --random_mask_path={random_mask_path} --epoch=100 --batch_size=32 > ./run_logs/{model_name}.log 2>&1".format(
          **par_dict
        )
        print("running {}...".format(cmd))
        ret = subprocess.check_call(cmd, shell=True, cwd="/home/bigdata/cwl/Gan")
        print(ret)
        f.write(model_name+"\n")
f.close()
