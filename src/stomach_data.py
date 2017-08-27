#!/usr/bin/python

def image_data_set(args, image_path):
	import torchvision.transforms as transforms
	transform = transforms.Compose(
		[transforms.Scale(args.image_size), # transforms.Scale((256,256))
		transforms.RandomCrop(args.image_size), 
		transforms.ToTensor(),
	 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	from torchvision.datasets import ImageFolder
	return ImageFolder(root=image_path, transform=transform)

def split_data_set(category_elems, splits):
	# indexes
	splitted_indexes = None
	for category in category_elems:
		elems = category_elems[category]
		elems = [elems[i] for i in torch.randperm(len(elems))]
		split = splits[category]
		if splitted_indexes is None:
			splitted_indexes = [[] for _ in range(len(split)]
		count = 0
		for i in range(len(split)):
			splitted_indexes[i] += elems[count : count + split[i]]
			count += split[i]
	# samplers 
	from torch.utils.data.sampler import SubsetRandomSampler
	return [SubsetRandomSampler(index) for index in splitted_indexes]

def get_splits_uniform(category_elems, splits_p):
	min_count = None
	lens = [len(category_elems[c]) for c in category_elems]
	min_count = min(lens)
	split = [int(min_count * p) for p in splits_p]
	split[-1] += min_count - sum(split)
	print '  data_len:', lens, 'split:', split
	return {c:split for c in category_elems}

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
	print('Loading data ...')
	kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
	# Read data from folder 'image_path'
	# The 'data_set' is a list of tuple: (image, image_category), 
	# where 'image' is a RGB image and 'image_category' is the label of the image, which ranges from 0-3
	data_set = image_data_set(args, image_path)
	# 'category_elems' is a dict whose keys are the four integer labels (categories) and
	# 	whose values are the indexes of the images belonging to each category.
	# For example category_elems[2] = [0, 3, 10, ...] if image #0, #3, #10, ... have label 2 (belong to categery 2)
	category_elems = get_category_info(data_set)
	# 'splits_p' specifies the how to split the data.
	# For example, if splits_p is [0.8, 0.1, 0.1], the data will be splitted into 3 parts, 
	# 	the first part will contain 80% of the data, the second part contains 10%, etc.
	# 'splits' is a dict, where for each category c splits[c] is a list of len(splits_p), 
	#	and splits[c][0] is the number of elements in the first part for category c.
	splits = get_splits_uniform(category_elems, splits_p)
	# 'samplers' is a list of torch.utils.data.sampler.SubsetRandomSampler, each sampler will be use to control
	#	how a DataLoader read data from the 'data_set'.
	# Specifically, the i-th sampler samples from 'data_set' only those data in splits[c][i], for any c.
	# For instances if 'splits_p' in the previous function is [0.8, 0.1, 0.1],
	#	samplers[0] smaples 80% of data in 'data_set', and samplers[1] samples another 10% of data in 'data_set', etc.
	samplers = split_data_set(category_elems, splits)
	from torch.utils.data import DataLoader
	# Build DataLoader's using 'samplers'
	return [DataLoader(data_set, batch_size=args.batch_size, sampler=s, **kwargs) for s in samplers]

if __name__ == '__main__':
	from cifar_main import parse_argument
	args = parse_argument(additional_arguments={'image-size':256, 'num_classes':4})
	image_path='Data/Normal'
	splits_p=[0.8, 0.1, 0.1]
	import torchvision.transforms as transforms
	train_transform = transforms.Compose([
		transforms.Pad(args.image_size // 8),
		transforms.RandomCrop(args.image_size),
		transforms.RandomHorizontalFlip(),
	])
	trans = [train_transform, None, None]
	train_loader, eval_loader, test_loader = get_data_loaders(args, image_path, splits_p, trans)

	import numpy, torchvision, matplotlib.pyplot as plt
	def inspect_data(loader):
		images, labels = iter(loader).next()
		print labels.view(1,-1)
		print('images.size()', images.size())
		img = torchvision.utils.make_grid(images)
		npimg = img.numpy()
		plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
		plt.show()

	inspect_data(train_loader)
	inspect_data(eval_loader)
	inspect_data(test_loader)

