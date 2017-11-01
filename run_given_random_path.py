#encoding=utf-8

import os
import codecs
import subprocess

data_dir = "/home/bigdata/cwl/Gan/data/cluster"
python_path = "/home/bigdata/anaconda3/bin/python"
script_path = "/home/bigdata/cwl/Gan/train.py"
done_path = "/home/bigdata/cwl/Gan/done_train.txt"

mask_dir = "/home/bigdata/cwl/Gan/cluster/mask"
mask_file_suffixs_dict = {}
for x in os.listdir(mask_dir):
  t = x.split(".")[0]
  q = t.split("_")
  name = q[:-1]
  name = "_".join(name)
  mask_file_suffixs_dict[name] = x

f = codecs.open(done_path, "r", "utf-8")
done_model_names = [v.strip() for v in f.readlines()]
f.close()
f = codecs.open(done_path, "w", "utf-8")

done_model_names = []
print(mask_file_suffixs_dict)

for file in os.listdir(data_dir):
  path = os.path.join(data_dir, file)
  name = file.split(".")[0]
  flag = False
  mask_path = ""
  for mask_file in mask_file_suffixs_dict:
    if mask_file in name:
      flag = True
      mask_path = os.path.join(mask_dir, mask_file_suffixs_dict[mask_file])

  if flag is True:
    for dropout in [0, 0.1]:
        drop_flag = dropout
        random_mask_flag = 0
        model_name = "{}_{}_{}_rp_{}".format(name, dropout, 0, 1)
        outDir = "./prediction_epo_200/dropout_{}/random_path/".format(dropout)
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
          "random_mask_path": mask_path,
          "outDir": outDir
        }
        cmd = "{python} {script_path} --model_name={model_name} --dropout={drop_flag} --truly_mis_pro={random_mask_flag} --train_datapath={datapath} " \
              "--infer_complete_datapath={datapath} --random_mask_path={random_mask_path} --outDir={outDir} --epoch=150 --batch_size=32 > ./run_logs/{model_name}.log 2>&1".format(
          **par_dict
        )
        print("running {}...".format(cmd))
        my_env = os.environ.copy()
        # my_env["CUDA_VISIBLE_DEVICES"] = ""
        ret = subprocess.check_call(cmd, shell=True, cwd="/home/bigdata/cwl/Gan", env=my_env)
        print(ret)
        f.write(model_name+"\n")
f.close()
