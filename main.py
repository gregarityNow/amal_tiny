
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
parser.add_argument("-dsName",type=str,default="mnist cifar10");
parser.add_argument("-outPath",type=str,default="./../out/");
parser.add_argument("-runType",type=str,default="train");
parser.add_argument("-fromBaseline",type=int,default=0);
parser.add_argument("-doAdapt",type=int,default=1);
parser.add_argument("-ppti",type=int,default=1);
parser.add_argument("-startSmall",type=int,default=0);
parser.add_argument("-version",type=int,default=0);
parser.add_argument("-stopEarly",type=int,default=10);
parser.add_argument("-randomScores",type=int,default=0);
parser.add_argument("-normType",type=str,default="l1");

opt = parser.parse_args()

for dsName in opt.dsName.split(" "):
	opt.dsName = dsName
	if opt.runType == "train":
		if opt.doAdapt:
			train_ViT(opt)
		else:
			train_ViT_vanilla(opt)
	elif opt.runType == "viz":
		allLosses, allAccs, compRates, epochLengths, allHiddenSizes = loadRunData(opt)
		visualize_loss_acc(opt, allLosses, allAccs, epochLengths, compRates)


