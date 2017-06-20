import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time

class Model(torch.nn.Module):
	def __init__(self, vgg_model):
		super(Model, self).__init__()
		self.features = vgg_model.features
		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
		    nn.Dropout(),
		    nn.Linear(25088, 4096),
		    nn.ReLU(inplace=True),
		    nn.Dropout(),
		    nn.Linear(4096, 4096),
		    nn.ReLU(inplace=True),
		    nn.Linear(4096, 2))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x

def train_batch(model, optimizer, criterion, batch, label):
	batch = batch.cuda()
	label = label.cuda()
	output = model(Variable(batch))
	loss = criterion(output, Variable(label))
	model.zero_grad()
	loss.backward()
	optimizer.step()
	return loss

def train_epoch(data_loader, model, optimizer):
	model.train()
	criterion = torch.nn.CrossEntropyLoss()
	for i, (batch, label) in enumerate(data_loader):
		loss = train_batch(model, optimizer, criterion, batch, label)

def test(data_loader, model):
	model.eval()
	correct = 0
	total = 0
	for i, (batch, label) in enumerate(data_loader):
		batch = batch.cuda()
		output = model(Variable(batch))
		pred = output.data.max(1)[1]
 		correct += pred.cpu().eq(label).sum()
 		total += label.size(0)

 	print "Accuracy :", float(correct) / total

def train(train_path, test_path):
	model = Model(torchvision.models.vgg16(pretrained=True))
	model = model.cuda()
	data_loader = dataset.loader(train_path, pin_memory = True)
	test_data_loader = dataset.test_loader(test_path, pin_memory = True)
	optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)

	test(test_data_loader, model)
	print "Starting.. "
	epochs = 10
	for i in range(epochs):
		train_epoch(data_loader, model, optimizer)
		test(test_data_loader, model)
	torch.save(model, "model")

def random_prune(model):
	test_data_loader = dataset.test_loader(args.test_path, pin_memory = True, batch_size=64)
	test(test_data_loader, model)
	model = model.cpu()
	model.eval()

	l = len(model.features._modules.items())
	print "Total number of layers", l
	items = [x for x in enumerate(model.features._modules.items())]
	items = items[::-5]
	for i, (name, module) in items:
		if i >= l - 5:
			continue
		if isinstance(module, torch.nn.modules.conv.Conv2d):
			filters = model.features._modules[name].weight.size(0)
			t0 = time.time()
			for _ in range(filters//2):
				model.features = prune_conv_layer(model.features, i, 0)

	print "After pruning.. "
	model = model.cuda()
	test(test_data_loader, model)

class ImportanceExtractor:
	def __init__(self, model):
		self.model = model
		num_convs = 0
		for name, module in self.model.features._modules.items():
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
		    	num_convs = num_convs + 1

		self.importance_values = {}

	def __call__(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}
		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
		    x = module(x)
		    if isinstance(module, torch.nn.modules.conv.Conv2d):
		    	x.register_hook(self.save_gradient)
		        self.activations += [x]
		        self.activation_to_layer[activation_index] = layer
		        activation_index += 1

		feature_output = x.view(x.size(0), -1)
		final_output = self.model.classifier(feature_output)
		return final_output

	def save_gradient(self, grad):
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]
		values = torch.sum((activation * grad), dim = 0).sum(dim=2).sum(dim=3)[0, :, 0, 0].data
		values = values / (activation.size(0) * activation.size(2) * activation.size(3))

		if activation_index not in self.importance_values:
			self.importance_values[activation_index] = torch.FloatTensor(activation.size(1)).zero_().cuda()

		self.importance_values[activation_index] += values

		self.grad_index += 1

	def find_k_minimum(self, num):
		data = []
		for i in sorted(self.importance_values.keys())[: -1]: #TBD
			for j in range(self.importance_values[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.importance_values[i][j]))

		k_minimim = nsmallest(num, data, itemgetter(2))
		return k_minimim


def get_filters_to_prune(train_data_loader, model, batch_size = 64, num_batches = 15):
	model.eval()
	extractor = ImportanceExtractor(model)
	print "In get_importance_criteron"
	t0 = time.time()

	criterion = torch.nn.CrossEntropyLoss()
	
	# for param in model.features.parameters():
	# 	param.requires_grad = True

	for i, (batch, label) in enumerate(train_data_loader):
		# if i >= num_batches:
		# 	break

		batch = batch.cuda()
		label = label.cuda()
		
		output = extractor(Variable(batch))
		loss = criterion(output, Variable(label))
		model.zero_grad()
		loss.backward()

	#Layerwise normalize
	for i in extractor.importance_values:
		v = torch.abs(extractor.importance_values[i])
		v = v / np.sqrt(torch.sum(v * v))
		extractor.importance_values[i] = v.cpu()

	# for param in model.features.parameters():
	# 	param.requires_grad = False

	filters_to_prune = extractor.find_k_minimum(256)
	# After each of the k filters are prunned,
	# the filter index of the next filters change since the model is smaller.

	filters_to_prune_per_layer = {}
	for (l, f, _) in filters_to_prune:
		if l not in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = []
		filters_to_prune_per_layer[l].append(f)

	for l in filters_to_prune_per_layer:
		filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
		for i in range(len(filters_to_prune_per_layer[l])):
			filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

	del filters_to_prune
	filters_to_prune = []
	for l in filters_to_prune_per_layer:
		for i in filters_to_prune_per_layer[l]:
			filters_to_prune.append((l, i))

	del extractor

	return filters_to_prune

def num_filters(model):
	filters = 0
	for name, module in model.features._modules.items():
		if isinstance(module, torch.nn.modules.conv.Conv2d):
			filters = filters + module.out_channels
	return filters

def taylor_prune(train_path, test_path, model):
	train_data_loader = dataset.loader(train_path, pin_memory = True, batch_size = 16)
	test_data_loader = dataset.test_loader(test_path, pin_memory = True)
	model = model.cuda()
	model.eval()
	test(test_data_loader, model)
	for param in model.features.parameters():
		param.requires_grad = True

	for iteration in range(8):
		filters_to_prune = get_filters_to_prune(train_data_loader, model)
		model = model.cpu()
		for layer_index, filter_index in filters_to_prune:
			model.features = prune_conv_layer(model.features, layer_index, filter_index)
		
		model = model.cuda()
		print "After pruning", iteration, "Number of filters left", num_filters(model)
		test(test_data_loader, model)
		
		model.train()
		for param in model.features.parameters():
			param.requires_grad = True

		optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
		epochs = 5
		for i in range(epochs):
			train_epoch(train_data_loader, model, optimizer)

		print "Finished fine tuning next iteration"
		test(test_data_loader, model)
		torch.save(model, "model_prunned")


	# for param in model.features.parameters():
	# 	param.requires_grad = False

	# train_data_loader = dataset.loader(train_path, pin_memory = True, batch_size = 32)
	# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.95)
	# print "Finished pruning. Now training more.. "
	# for i in range(25):
	# 	train_epoch(train_data_loader, model, optimizer)
	# 	test(test_data_loader, model)
	# torch.save(model, "model_prunned")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train", action="store_true")
    parser.add_argument("--random_prune", dest="random_prune", action="store_true")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "test")
    parser.set_defaults(train=False)
    parser.set_defaults(random_prune=False)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
	args = get_args()

	if args.train:
		train(args.train_path, args.test_path)
	elif args.random_prune:
		model = torch.load("model")
		model.eval()
		model = model.cuda()
		random_prune(model)
	else:
		model = torch.load("model")
		model = model.cuda()
		taylor_prune(args.train_path, args.test_path, model)