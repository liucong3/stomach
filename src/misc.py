
import os, torch

def ensure_dir(path, erase=False):
	import shutil
	if os.path.exists(path) and erase:
		print("Removing old folder {}".format(path))
		shutil.rmtree(path)
	if not os.path.exists(path):
		print("Creating folder {}".format(path))
		os.makedirs(path)

class Logger(object):
	def __init__(self, logdir, name='log'):
		import logging
		if not os.path.exists(logdir):
			os.makedirs(logdir)
		log_file = os.path.join(logdir, name)
		if os.path.exists(log_file):
			os.remove(log_file)
		self._logger = logging.getLogger()
		self._logger.setLevel('INFO')
		fh = logging.FileHandler(log_file)
		ch = logging.StreamHandler()
		self._logger.addHandler(fh)
		self._logger.addHandler(ch)

	def info(self, str_info):
		self._logger.info(str_info)

def save_model(model, new_file, old_file=None, verbose=False):
	from collections import OrderedDict
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	if old_file and os.path.exists(old_file):
		if verbose:
			print("Removing old model {}".format(old_file))
		os.remove(old_file)
	if verbose:
		print("Saving model to {}".format(new_file))

	state_dict = OrderedDict()
	for k, v in model.state_dict().items():
		if v.is_cuda:
			v = v.cpu()
		state_dict[k] = v
	torch.save(state_dict, new_file)

def init_params(net):
	import torch.nn as nn
	import torch.nn.init as init
	for m in net.modules():
		if isinstance(m, nn.Conv2d):
			init.kaiming_normal(m.weight, mode='fan_out')
			if not m.bias is None:
				init.constant(m.bias, 0)
		elif isinstance(m, nn.BatchNorm2d):
			if not m.weight is None:
				init.constant(m.weight, 1)
				init.constant(m.bias, 0)
		elif isinstance(m, nn.Linear):
			init.normal(m.weight, std=1e-3)
			if not m.bias is None:
				init.constant(m.bias, 0)

def format_time(seconds, with_ms=False):
	days = int(seconds / 3600/24)
	seconds = seconds - days*3600*24
	hours = int(seconds / 3600)
	seconds = seconds - hours*3600
	minutes = int(seconds / 60)
	seconds = seconds - minutes*60
	secondsf = int(seconds)
	seconds = seconds - secondsf
	millis = int(seconds*1000)

	f = ''
	if days > 0:
		f += str(days) + '/'
	if hours > 0:
		f += str(hours) + ':'
	f += str(minutes) + '.' + str(secondsf)
	if with_ms and millis > 0:
		f += '_' + str(millis)
	return f
				
# progress_bar

_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 60.
import time
last_time = time.time()
begin_time = last_time
last_len = -1
def progress_bar(current, total, msg=None):
	global last_time, begin_time, last_len
	if current == 0:
		begin_time = time.time()  # Reset for new bar.
		last_len = -1

	cur_len = int(TOTAL_BAR_LENGTH*current/total)
	rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1
	cur_time = time.time()
	if last_len == cur_len and current < total - 1 and cur_time - last_time < 1 and msg is None:
		return

	import sys
	sys.stdout.write(' [')
	for i in range(cur_len):
		sys.stdout.write('=')
	sys.stdout.write('>')
	for i in range(rest_len):
		sys.stdout.write('.')
	sys.stdout.write(']')

	last_time = cur_time
	tot_time = cur_time - begin_time
	last_len = cur_len

	L = []
	est_time = tot_time / (current + 1) * total
	L.append(' Time:%s/Est:%s' % (format_time(tot_time), format_time(est_time)))
	if msg:
		L.append(' ' + msg)

	msg = ''.join(L)
	sys.stdout.write(msg)
	for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
		sys.stdout.write(' ')

	# Go back to the center of the bar.
	for i in range(term_width-int(TOTAL_BAR_LENGTH/2)):
		sys.stdout.write('\b')
	sys.stdout.write(' %d/%d ' % (current+1, total))

	if current < total - 1:
		sys.stdout.write('\r')
	else:
		sys.stdout.write('\n')
	sys.stdout.flush()

