#!/usr/bin/python

import torch
import torchvision
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def image_data_set(args, image_path):
	transform = transforms.Compose(
		[transforms.Scale(args.image_size), # transforms.Scale((360,310))
		transforms.RandomCrop(args.image_size), 
		transforms.ToTensor(),
	 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	return ImageFolder(root=image_path, transform=transform)

def split_data_set(category_elems, splits):
	# indexes
	splitted_indexes = None
	for category in category_elems:
		elems = category_elems[category]
		elems = [elems[i] for i in torch.randperm(len(elems))]
		split = splits[category]
		if splitted_indexes is None:
			splitted_indexes = []
			for i in range(len(split)):
				splitted_indexes.append([])
		count = 0
		for i in range(len(split)):
			splitted_indexes[i] += elems[count : count + split[i]]
			count += split[i]
	# samplers
	samplers = []
	from torch.utils.data.sampler import SubsetRandomSampler
	for i in range(len(splitted_indexes)):
		samplers.append(SubsetRandomSampler(splitted_indexes[i]))
	return samplers
		
def get_splits_uniform(category_elems, splits_p):
	min_count = None
	lens = [len(category_elems[c]) for c in category_elems]
	min_count = min(lens)
	split = []
	total = 0
	for i in range(len(splits_p)):
		split.append(int(min_count * splits_p[i]))
		total += split[-1]
	split[-1] += min_count - total
	print '  data_len:', lens, 'split:', split
	splits = {}
	for c in category_elems:
		splits[c] = split
	return splits

def get_category_info(data_set):
	from misc import progress_bar
	category_elems = {}
	for i in range(len(data_set)):
		image, category = data_set[i]
		if not category_elems.has_key(category):
			category_elems[category] = []
		category_elems[category].append(i)
		progress_bar(i, len(data_set))
	return category_elems

def get_data_loaders(args, image_path, splits_p, transforms):
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	data_set = image_data_set(args, image_path)
	category_elems = get_category_info(data_set)
	splits = get_splits_uniform(category_elems, splits_p)
	samplers = split_data_set(category_elems, splits)
	loaders = []
	for i in range(len(samplers)):
		loader = DataLoader(data_set, batch_size=args.batch_size, sampler=samplers[i], **kwargs)
		loaders.append(loader)
	return loaders

if __name__ == '__main__':
	from cifar_main import parse_argument
	args = parse_argument(additional_arguments={'image-size':256, 'num_classes':4})
	image_path='Data/Normal'
	splits_p=[0.8, 0.1, 0.1]
	train_transform = transforms.Compose([
		transforms.Pad(args.image_size // 8),
		transforms.RandomCrop(args.image_size),
		transforms.RandomHorizontalFlip(),
	])
	trans = [train_transform, None, None]
	train_loader, eval_loader, test_loader = get_data_loaders(args, image_path, splits_p, trans)

	print('len(train_loader)', len(train_loader))
	print('len(eval_loader)', len(eval_loader))
	print('len(test_loader)', len(test_loader))

	images, labels = iter(train_loader).next()
	print('images.size()', images.size())
	img = torchvision.utils.make_grid(images)
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	plt.show()

