
from __future__ import print_function

if __name__ == '__main__':
	from cifar_main import parse_argument
	arguments={'image-size':256, 'num_classes':4, 'batch-size':16}
	args = parse_argument(additional_arguments=arguments, description='Location classification for stomach images.')
	import misc
	misc.ensure_dir(args.logdir)
	logger = misc.Logger(args.logdir, 'train_log')
	print = logger.info
	print("=================FLAGS==================")
	for k, v in args.__dict__.items():
		print('{}: {}'.format(k, v))
	print("========================================")

	image_path='Data/Normal'
	splits_p=[0.9, 0.1]
	import torchvision.transforms as transforms
	train_transform = transforms.Compose([
		transforms.Pad(args.image_size // 8),
		transforms.RandomCrop(args.image_size),
		transforms.RandomHorizontalFlip(),
	])
	trans = [train_transform, None]
	from stomach_data import get_data_loaders
	train_loader, test_loader = get_data_loaders(args, image_path, splits_p, trans)
	from models import LocConvNet
	model = LocConvNet(args)
	import train
	train.main(args, model, train_loader, test_loader, print=logger.info)


