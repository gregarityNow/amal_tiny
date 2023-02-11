
from src import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-quickie",type=int,default = -1)
parser.add_argument("-batch_size",type=int,default=16);
parser.add_argument("-targetCompRate",type=int,default=32);
parser.add_argument("-hid_size",type=int,default=50);
parser.add_argument("-ro",type=float,default=0.5);
parser.add_argument("-lr",type=float,default=0.01);
parser.add_argument("-comp_n_epochs",type=int,default=50);
parser.add_argument("-n_epochs",type=int,default=50);
parser.add_argument("-dsName",type=str,default="mnist");
parser.add_argument("-outPath",type=str,default="./../out/");
parser.add_argument("-runType",type=str,default="train");

opt = parser.parse_args()

if opt.runType == "train":
	for dsName in opt.dsName.split(" "):
		train_ViT(opt, dsName)

elif opt.runType == "viz":
	allLosses, allAccs, compRates, allHiddenSizes = loadRunData(opt)
	visualize_loss_acc(opt, allLosses, allAccs, compRates)




