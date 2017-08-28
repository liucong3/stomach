from __future__ import print_function

import os, time, torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import misc
from misc import progress_bar

train_args = {
	'batch-size':64, 
	'test-batch-size':256, 
	'epochs':200, 
	'lr':0.01, 
	'decreasing-lr':'80,120',
	'lr-decreasing-rate':0.5,
	'weight-decay':0.01, 
	'momentum':0.5, 
	'gpu':-1, 
	'seed':1,
	'log-interval':10,
	'test-interval':1,
	'logdir':'log',
	'batch-norm':True,
	'dropout':0.0,
}

def main(args, model, train_loader, test_loader, decreasing_lr=[], print=print):
	if args.cuda:
		model = model.cuda()
	if isinstance(args.decreasing_lr, str):
		args.decreasing_lr = list(map(int, args.decreasing_lr.split(',')))
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
	info = {'best_acc':0, 'old_file':None, 't_begin':time.time()}
	try:
		# ready to go
		for epoch in range(args.epochs):
			info['epoch'] = epoch
			run(args, model, train_loader, info, optimizer=optimizer, print=print)

			# elapse_time = time.time() - info['t_begin']
			# speed_epoch = elapse_time / (epoch + 1)
			# speed_batch = speed_epoch / len(train_loader)
			# eta = speed_epoch * args.epochs - elapse_time
			# print("Elapsed {}, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
			# 	misc.format_time(elapse_time), speed_epoch, speed_batch, eta))
	
			if epoch % args.test_interval == 0:
				run(args, model, test_loader, info, optimizer=None, print=print)
	except Exception as e:
		import traceback
		traceback.print_exc()
	except KeyboardInterrupt:
		exit()
	finally:
		print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-info['t_begin'], info['best_acc']))


def run(args, model, data_loader, info, optimizer=None, print=print):
	is_train = (not optimizer is None)
	print('%s %d - %s' % (is_train and 'Train' or 'Eval', info['epoch'], time.ctime()))
	if is_train: 
		model.train()
	else:
		model.eval()
	if is_train and info['epoch'] in args.decreasing_lr:
		optimizer.param_groups[0]['lr'] *= args.lr_decreasing_rate
	msg = None
	total_loss = 0
	total_correct = 0
	data_count = 0

	for batch_idx, (data, target) in enumerate(data_loader):
		data_count += data.size(0)
		indx_target = target.clone()
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)

		if is_train:
			optimizer.zero_grad()
		output = model(data)
		loss = F.cross_entropy(output, target)
		if is_train:
			loss.backward()
			optimizer.step()

		total_loss += loss.data[0]
		pred = output.data.max(1)[1]  # get the index of the max log-probability
		total_correct += pred.cpu().eq(indx_target).sum()

		if (not is_train) or (batch_idx % args.log_interval == 0 and batch_idx > 0):
			loss = total_loss / data_count
			acc = 100. * total_correct / data_count
			msg = 'Loss:{:.3f},Acc:{}/{}({:.3f}%)'.format(total_loss/data_count, total_correct, data_count, 
				100. * total_correct / data_count)
			if is_train:
				msg += 'lr:{:.5f}'.format(optimizer.param_groups[0]['lr'])
		progress_bar(batch_idx, len(data_loader), msg)

	if (not is_train) and acc > info['best_acc']:
		new_file = os.path.join(args.logdir, 'best-{}.pth'.format(info['epoch']))
		misc.save_model(model, new_file, old_file=info['old_file'], verbose=True)
		info['best_acc'] = acc
		info['old_file'] = new_file


