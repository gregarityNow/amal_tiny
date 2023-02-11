
from src import train_ViT

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-quickie",type=int,default = 0)
parser.add_argument("-batch_size",type=int,default=16);
parser.add_argument("-targetCompRate",type=int,default=32);
parser.add_argument("-hid_size",type=int,default=50);
parser.add_argument("-ro",type=float,default=0.5);
parser.add_argument("-comp_n_epochs",type=int,default=50);
parser.add_argument("-n_epochs",type=int,default=50);
parser.add_argument("-dsName",type=str,default="mnist");
parser.add_argument("-outPath",type=str,default="./../out/");

opt = parser.parse_args()

allLosses, allAccs, compRates = train_ViT(opt)


