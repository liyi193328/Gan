#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf

from gan import Gan

parser = argparse.ArgumentParser()
parser.add_argument('--infer_complete_datapath', default="./data/drop80.infer",type=str)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--missing_val", default=0, type=float)
parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=100)
parser.add_argument('--imgSize', type=int, default=64)
parser.add_argument('--lam', type=float, default=0.1)
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
parser.add_argument('--outDir', type=str, default='./data')
parser.add_argument('--outInterval', type=int, default=50)
parser.add_argument('--maskType', type=str,
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres'],
                    default='center')
parser.add_argument('--centerScale', type=float, default=0.25)
parser.add_argument("--feature_nums",type=int, default=13416, help = "The size of image to use")
parser.add_argument("--model_name", type=str, default="tanh-mak-0", help="model name will be loaded")
args = parser.parse_args()

assert(os.path.exists(args.checkpoint_dir))

model = Gan(args.feature_nums)
model.complete(args)
