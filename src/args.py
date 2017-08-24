
import argparse

def parse_argument(arguments, description=None):
	parser = argparse.ArgumentParser(description)
	for name in arguments:
		assert isinstance(name, str)
		value = arguments[name]
		# print(name, value, value.__class__.__name__)
		if isinstance(value, type):
			if value in [int, float, str, bool]:
				parser.add_argument('--' + name, type=value, required=True, help='(type: %s, required)' % value.__name__)
			else:
				raise ValueError('Type %s is not supported in argument.' % value.__name__)
		elif isinstance(value, bool):
			if value == True:
				parser.add_argument('--' + name, action='store_false', default=True, help='(default: True)')
			else:
				parser.add_argument('--' + name, action='store_true', default=False, help='(default: False)')
		elif isinstance(value, int):
			parser.add_argument('--' + name, type=int, default=value, help='(default: %d)'%value)
		elif isinstance(value, float):
			parser.add_argument('--' + name, type=float, default=value, help='(default: %f)'%value)
		elif isinstance(value, str):
			parser.add_argument('--' + name, type=str, default=value, help='(default: %s)'%value)
		else:
			raise KeyError('Supported argument value types: int, float, str, bool.')
	return parser.parse_args()

if __name__ == '__main__':
	argument_spec = {
		'arg1':int,
		'arg2':float,
		'arg3':str,
		'arg4':bool,
		'arg5':'SOME TEXT',
		'arg6':True,
		'arg7':False,
		'arg8':10,
		'arg9':10.5,
	}
	opt = parse_argument(argument_spec, description='ARG USAGE')