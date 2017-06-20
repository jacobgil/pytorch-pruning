import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import time

def print_convs(model):
	for name, module in model._modules.items():
		if isinstance(module, torch.nn.modules.conv.Conv2d):
			print name, model._modules[name].weight.size()
 
def replace_layers(model, i, indexes, layers):
	if i in indexes:
		return layers[indexes.index(i)]
	return model[i]

def prune_conv_layer(model, layer_index, filter_index):
	name, conv = model._modules.items()[layer_index]
	next_conv = None
	offset = 1

	while layer_index + offset <  len(model._modules.items()):
		res =  model._modules.items()[layer_index+offset]
		if isinstance(res[1], torch.nn.modules.conv.Conv2d):
			next_name, next_conv = res
			break
		offset = offset + 1
	
	new_conv = \
		torch.nn.Conv2d(in_channels = conv.in_channels, \
			out_channels = conv.out_channels - 1,
			kernel_size = conv.kernel_size, \
			stride = conv.stride,
			padding = conv.padding,
			dilation = conv.dilation,
			groups = conv.groups,
			bias = conv.bias)

	old_weights = conv.weight.data.numpy()
	new_weights = new_conv.weight.data.numpy()

	new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
	new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
	new_conv.weight.data = torch.from_numpy(new_weights)

	bias_numpy = conv.bias.data.numpy()
	bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
	bias[:filter_index] = bias_numpy[:filter_index]
	bias[filter_index : ] = bias_numpy[filter_index + 1 :]
	new_conv.bias.data = torch.from_numpy(bias)

	if not next_conv is None:
		next_new_conv = \
			torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
				out_channels =  next_conv.out_channels, \
				kernel_size = next_conv.kernel_size, \
				stride = next_conv.stride,
				padding = next_conv.padding,
				dilation = next_conv.dilation,
				groups = next_conv.groups,
				bias = next_conv.bias)

		old_weights = next_conv.weight.data.numpy()
		new_weights = next_new_conv.weight.data.numpy()

		new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
		new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
		next_new_conv.weight.data = torch.from_numpy(new_weights)

		next_new_conv.bias.data = next_conv.bias.data

	if not next_conv is None:
	 	model = torch.nn.Sequential(
	            *(replace_layers(model, i, [layer_index, layer_index+offset], \
	            	[new_conv, next_new_conv]) for i, _ in enumerate(model)))

	# TBD: For now we don't prune the last conv layer.
	# This requires adjusting the first linear layer of the fully connected part.
	# else:
	#  	model = torch.nn.Sequential(
	#             *(replace_layers(model, i, [layer_index], \
	#             	[new_conv]) for i, _ in enumerate(model)))

	return model

if __name__ == '__main__':
	model = models.vgg16(pretrained=True)
	model.eval()
	features = model.features

	features = prune_conv_layer(features, 2, 10)
	print_convs(features)