
from __future__ import print_function

def run_task():

	arguments={
		'image-size':256, 
		'batch-size':32, 
		'test-batch-size':32, 
		'lr':1e-3, 
		'decreasing-lr':'20,40,60,80,100,120,140',
		'dropout':0.1
		'num_classes':3,
		'task':'type',
		}
	from cifar_main import parse_argument
	args = parse_argument(additional_arguments=arguments, description='Location classification for stomach images.')
	if args.task == 'loc':
		image_path='Data/Normal'
		args.num_classes = 4
	elif args.task == 'type':
		image_path='Data'
		args.num_classes = 3
	else:
		raise

	import misc
	misc.ensure_dir(args.logdir)
	logger = misc.Logger(args.logdir, 'train_log')
	print = logger.info
	print("-----------------FLAGS-----------------")
	for k, v in args.__dict__.items():
		print('{}: {}'.format(k, v))
	print("---------------------------------------\n")

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
	misc.init_params(model)
	print("-----------------MODEL-----------------")
	print(model)
	print("---------------------------------------\n")
	import train
	train.main(args, model, train_loader, test_loader, print=logger.info)


if __name__ == '__main__':
	run_task()