
from __future__ import print_function
import torch

def parse_argument(additional_arguments={}, description=None):
	import args, train
	arguments = train.train_args
	arguments.update(additional_arguments)
	arguments = args.parse_argument(arguments, description=description)
	arguments.cuda = arguments.gpu >= 0 and torch.cuda.is_available()
	torch.manual_seed(arguments.seed)
	if arguments.cuda:
		torch.cuda.manual_seed(arguments.seed)
		torch.cuda.set_device(arguments.gpu)
	return arguments

if __name__ == '__main__':
	args = parse_argument(description='A test using the CIFAR10 dataset.')
	import misc
	misc.ensure_dir(args.logdir)
	logger = misc.Logger(args.logdir, 'train_log')
	print = logger.info
	print("=================FLAGS==================")
	for k, v in args.__dict__.items():
		print('{}: {}'.format(k, v))
	print("========================================")

	import cifar_data, models
	train_loader, test_loader = cifar_data.get10(args)
	model = models.CIFAR()
	import train
	train.main(args, model, train_loader, test_loader)


