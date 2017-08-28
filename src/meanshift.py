
# Meanshit algorithm to perform discontinuity preserving smoothing
# Epanechnikov kernel K(x) is used, i.e. G(x) is a uniform kernel
def meanshit(image, h_s, h_r):
	image = image.transpose(0,2)
	import torch

	def get_input(image):
		size = list(image.size())
		size[-1] += 2
		x = image.new(*size)
		x[:,:,0:3].copy_(image)
		x[:,:,-2].copy_(torch.arange(0, size[0]).unsqueeze(1).expand(size[:2]))
		x[:,:,-1].copy_(torch.arange(0, size[1]).unsqueeze(0).expand(size[:2]))
		h = image.new([h_r] * 3 + [h_s] * 2)
		return x, h

	def uniform_kernel(x):
		gt1 = (x > 1)
		x = 1 - x
		x[gt1] = 0
		return x

	def neighbor(data, x, y, range):
		x, y = int(x), int(y)
		x1, x2 = max(0, x - range), min(data.size(0), x + range + 1)
		y1, y2 = max(1, y - range), min(data.size(1), y + range + 1)
		return data[x1:x2, y1:y2, :]

	x, h = get_input(image)
	h = h.unsqueeze_(0).unsqueeze_(0)

	size = x.size()[0:2]
	while True:
		max_change = 1
		for i in range(size[0]):
			for j in range(size[1]):
				from misc import progress_bar
				progress_bar(i * size[1] + j, size[0] * size[1])
				x2 = neighbor(x, x[i,j,-2], x[i,j,-1], h_s)
				x1 = x[i,j].unsqueeze(0).unsqueeze(0).expand_as(x2)
				g = uniform_kernel((x2 - x1).div_(h.expand_as(x2))).prod(2)
				m = x2.mul(g.unsqueeze_(2).expand_as(x2)).sum(0).sum(0).div_(g.sum()).sub_(x[i,j])
				x[i,j].add_(m)
				max_change = max(max_change, m.abs().max())
		print 'max_change', max_change
		break


	image.copy_(x[:,:,0:3])


if __name__ == '__main__':

	def load_image(path, image_size=None, to_tensor=True):
		from torchvision.datasets.folder import default_loader
		from torchvision import transforms
		image = default_loader(path)
		trans = []
		if not image_size is None:
			trans.append(transforms.Scale(image_size))
			trans.append(transforms.RandomCrop(image_size))
		if to_tensor:
			trans.append(transforms.ToTensor())
			#trans.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
		if len(trans) > 0:
			image = transforms.Compose(trans)(image)
		return image

	def show_image(image):
		import torch
		if torch.is_tensor(image):
			from torchvision import transforms
			image = transforms.ToPILImage()(image)
		image.show()

	image = load_image('Data/Normal/Angular/AA68GCAB0T4S3.JPG')
	#print image.max(), image.min(), image.mean(), image.std()
	show_image(image)
	meanshit(image, 4, 4)
	show_image(image)

