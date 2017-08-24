from __future__ import print_function
import torch, os, time, misc
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

train_args = {
	'batch-size':64, 
	'test-batch-size':1000, 
	'epochs':150, 
	'lr':0.01, 
	'decreasing-lr':'80,120',
	'weight-decay':0.01, 
	'momentum':0.9, 
	'gpu':0, 
	'seed':1,
	'log-interval':10,
	'test-interval':1,
	'logdir':'log',
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
			train(args, model, optimizer, train_loader, info, print)

			elapse_time = time.time() - info['t_begin']
			speed_epoch = elapse_time / (epoch + 1)
			speed_batch = speed_epoch / len(train_loader)
			eta = speed_epoch * args.epochs - elapse_time
			print("Elapsed {}, {:.2f} s/epoch, {:.2f} s/batch, ets {:.2f}s".format(
				misc.format_time(elapse_time), speed_epoch, speed_batch, eta))
	
			if epoch % args.test_interval == 0:
				eval(args, model, test_loader, info, print)
	except Exception as e:
		import traceback
		traceback.print_exc()
	finally:
		print("Total Elapse: {:.2f}, Best Result: {:.3f}%".format(time.time()-info['t_begin'], info['best_acc']))

def train(args, model, optimizer, train_loader, info, print=print):
	print('Train ' + time.ctime())
	model.train()
	if info['epoch'] in args.decreasing_lr:
		optimizer.param_groups[0]['lr'] *= 0.1
	for batch_idx, (data, target) in enumerate(train_loader):
		indx_target = target.clone()
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)

		optimizer.zero_grad()
		output = model(data)
		loss = F.cross_entropy(output, target)
		loss.backward()
		optimizer.step()

		if batch_idx % args.log_interval == 0 and batch_idx > 0:
			pred = output.data.max(1)[1]  # get the index of the max log-probability
			correct = pred.cpu().eq(indx_target).sum()
			acc = correct * 1.0 / len(data)
			print('Elapsed: {} Epoch: {} [{}/{}] Loss: {:.6f} Acc: {:.4f} lr: {:.2e}'.format(
				misc.format_time(time.time() - info['t_begin']), info['epoch'], batch_idx * len(data), len(train_loader),
				loss.data[0], acc, optimizer.param_groups[0]['lr']))

def eval(args, model, test_loader, info, print=print):
	print('Eval ' + time.ctime())
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		indx_target = target.clone()
		if args.cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.cross_entropy(output, target).data[0]
		pred = output.data.max(1)[1]  # get the index of the max log-probability
		correct += pred.cpu().eq(indx_target).sum()

	test_loss = test_loss / len(test_loader) # average over number of mini-batch
	acc = 100. * correct / len(test_loader)
	print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
		test_loss, correct, len(test_loader), acc))
	if acc > info['best_acc']:
		new_file = os.path.join(args.logdir, 'best-{}.pth'.format(info['epoch']))
		misc.save_model(model, new_file, old_file=info['old_file'], verbose=True)
		info['best_acc'] = acc
		info['old_file'] = new_file


