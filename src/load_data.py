import pathlib
from .basis_funcs import *;
#, SwinForImageClassification
import torch
# from datasets import load_dataset

from collections import OrderedDict


from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader


class IncreaseDim:
	"""Rotate by one of the given angles."""

	def __call__(self, x):
		img2 = torch.cat([x, x, x], dim=0)
		return img2


kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

def getRoot(opt):
	if opt.ppti:
		root = "/tmp/f002nb9/datasets/mnist"
		pathlib.Path(root).mkdir(exist_ok=True,parents=True);
	else:
		root = "/home/fherron/sac/tina/data"
		pathlib.Path(root).mkdir(exist_ok=True, parents=True);

	return root

def get_mnist_loaders(opt):
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,)),
									IncreaseDim()
									])

	root = getRoot(opt);
	train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)


	if opt.quickie > -1:
		train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(opt.quickie))
		test_dataset = torch.utils.data.Subset(test_dataset, torch.arange(opt.quickie))

	train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, **kwargs)

	return train_loader, test_loader, 10

def get_cifar10_loaders(opt):

	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
									])

	root = getRoot(opt);
	train_dataset = datasets.CIFAR10(root, train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10(root, train=False, download=True, transform=transform)

	if opt.quickie > -1:
		train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(opt.quickie))
		test_dataset = torch.utils.data.Subset(test_dataset, torch.arange(opt.quickie))

	train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(test_dataset, batch_size=opt.batch_size, **kwargs)

	return train_loader, test_loader, 10



def saveModel(model, opt, compRate = 1):

	outDir = opt.outPath + "/models/"
	pathlib.Path(outDir).mkdir(parents=True, exist_ok=True);
	outPath = outDir + "/" + "adapter_" + str(compRate) + "_" + getOutNameForOpt(opt)  + ".cpkt"
	if opt.quickie > -1:
		print("not saving for quickie",outPath);
		return;

	stateDict = model.state_dict()
	paramsDict = OrderedDict({k: stateDict[k] for k in stateDict.keys() if "adapter" in k or "classifier" in k})
	torch.save(paramsDict, outPath)

	print("saved model to", outPath)


def loadModel(model, opt, compRate=1):
	outDir = opt.outPath + "/models/"
	outPath = outDir + "/" + "adapter_" + str(compRate) + "_" + getOutNameForOpt(opt) + ".cpkt"

	paramsDictLoaded = torch.load(outPath)
	model.load_state_dict(paramsDictLoaded, strict=False)

	print("loaded weights from",outPath)

import pickle


def saveRunData(opt, runData):
	outDir = opt.outPath + "/runData/"
	pathlib.Path(outDir).mkdir(parents=True, exist_ok=True);
	outPath = outDir + "/" + "adapter_" + getOutNameForOpt(opt) + ".pkl"
	with open(outPath,"wb") as fp:
		pickle.dump(runData, fp);

	print("saved run data to",outPath)


def loadRunData(opt):
	outDir = opt.outPath + "/runData/"
	outPath = outDir + "/" + "adapter_" + getOutNameForOpt(opt) + ".pkl"
	with open(outPath,"rb") as fp:
		runData = pickle.load(fp);

	return runData

def getOutNameForOpt(opt):
	return opt.dsName + "_" + str(opt.hid_size) +"_"+ str(opt.ro) + ("_quickie" if opt.quickie > 0 else "") +\
		    ("_vanilla" if opt.doAdapt == 0 else "") + ("_startSmall" if opt.startSmall == 1 else "")\
		   + ("_" + opt.normType + "Norm" if opt.normType != "l1" else "") + ("_randomScores" if opt.randomScores else "") + "_v" + str(opt.version);

def saveFig(opt, plotName):
	outDir = opt.outPath + "/img/"
	pathlib.Path(outDir).mkdir(parents=True, exist_ok=True);
	outPath = outDir + "/" + plotName + "_" + getOutNameForOpt(opt) + ".png"

	plt.savefig(outPath)
	print("Saved fig to",outPath)