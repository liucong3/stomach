
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get10(args, data_root='test_data/cifar10-data', train=True, val=True):
	ds = []
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	if train:
		train_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(
				root=data_root, train=True, download=True,
				transform=transforms.Compose([
					transforms.Pad(4),
					transforms.RandomCrop(32),
					transforms.RandomHorizontalFlip(),
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				])),
			batch_size=args.batch_size, shuffle=True, **kwargs)
		ds.append(train_loader)
	if val:
		test_loader = torch.utils.data.DataLoader(
			datasets.CIFAR10(
				root=data_root, train=False, download=True,
				transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
				])),
			batch_size=args.batch_size, shuffle=False, **kwargs)
		ds.append(test_loader)
	ds = ds[0] if len(ds) == 1 else ds
	return ds
