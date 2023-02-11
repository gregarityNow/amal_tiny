
from .basis_funcs import *;
#, SwinForImageClassification
import torch
from datasets import load_dataset




from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader


class IncreaseDim:
	"""Rotate by one of the given angles."""

	def __call__(self, x):
		img2 = torch.cat([x, x, x], dim=0)
		return img2


kwargs = {'num_workers': 10, 'pin_memory': True} if use_cuda else {}

def get_mnist_loaders(batch_size=128, quickie=-1):
	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.1307,), (0.3081,)),
									IncreaseDim()
									])

	train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
	test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

	if quickie > -1:
		train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(quickie))
		test_dataset = torch.utils.data.Subset(test_dataset, torch.arange(quickie))

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, **kwargs)

	return train_loader, test_loader

def get_cifar10_loaders(batch_size=128, quickie=-1):

	transform = transforms.Compose([transforms.ToTensor(),
									transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
									])

	train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform)
	test_dataset = datasets.CIFAR10('data', train=False, download=True, transform=transform)

	if quickie > -1:
		train_dataset = torch.utils.data.Subset(train_dataset, torch.arange(quickie))
		test_dataset = torch.utils.data.Subset(test_dataset, torch.arange(quickie))

	train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
	test_loader = DataLoader(test_dataset, batch_size=batch_size, **kwargs)

	return train_loader, test_loader

