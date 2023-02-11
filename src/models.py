

from transformers import ViTForImageClassification

from .basis_funcs import *;
from .load_data import loadModel

class AdapterBlock(nn.Module):
	'''
	Class describing the adapter architecture
	'''

	def __init__(self, in_size, hid_size, activation=nn.ReLU):
		super().__init__()
		self.block = nn.Sequential(
			nn.Linear(in_size, hid_size),
			activation(),
			nn.Linear(hid_size, in_size)
		)

	def forward(self, x):
		adaptOut = self.block(x)
		resOut = x + adaptOut

		return resOut


doAdapt = True


class AdapterLayer(nn.Module):
	'''
	Class using the adapter architecture, to be inserted into Transformer blocks
	'''

	def __init__(self, prev_layer, in_size, hid_size):
		super().__init__()
		self.prev_layer = prev_layer
		self.adapter = AdapterBlock(in_size, hid_size)

	def forward(self, x):
		prev_out = self.prev_layer(x)  # *?
		# print("prev_out.shape",prev_out.shape,"x.shape",x.shape,self.adapter)
		if doAdapt:
			prev_out = self.adapter(prev_out)
		return prev_out


class AdapterLayerOutput(AdapterLayer):
	'''
	Class using the adapter architecture, to be inserted into Transformer blocks
	'''

	def forward(self, hidden_states, input_tensor):
		prev_out = self.prev_layer(hidden_states, input_tensor)
		if doAdapt:
			prev_out = self.adapter(prev_out)
		return prev_out

def initializeModel(opt,numClasses,compRate=1):

	if not opt.doAdapt:
		model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
		model.classifier = nn.Linear(in_features=768, out_features=numClasses, bias=True)
		model.init_weights();
		return model

	model = ViT_TINA(opt.hid_size, n_classes=numClasses)
	if opt.fromBaseline:
		loadModel(model, opt, compRate=compRate)
	return model

class ViT_TINA(nn.Module):
	def __init__(self, hid_sizes, n_classes):
		super().__init__()
		self.n_classes = n_classes
		self.model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
		self.depth = len(self.model.vit.encoder.layer)

		if type(hid_sizes) == int:
			hid_size = hid_sizes
			self.hid_sizes = {"intermediate": [hid_size] * self.depth, "output": [hid_size] * self.depth}
		else:
			self.hid_sizes=hid_sizes

		for layerIndex in range(self.depth):
			# prevLayer_MSA = self.model.vit.encoder.layer[layerIndex].attention.attention
			# prevLayer_MLP = self.model.vit.encoder.layer[layerIndex].attention.output

			self.model.vit.encoder.layer[layerIndex].intermediate = AdapterLayer(self.model.vit.encoder.layer[layerIndex].intermediate,
																				 in_size=self.model.vit.encoder.layer[layerIndex].intermediate.dense.out_features,
																				 hid_size=self.hid_sizes["intermediate"][layerIndex])

			self.model.vit.encoder.layer[layerIndex].output = AdapterLayerOutput(self.model.vit.encoder.layer[layerIndex].output,
																				 in_size=self.model.vit.encoder.layer[layerIndex].output.dense.out_features,
																				 hid_size=self.hid_sizes["output"][layerIndex])

		if doAdapt:
			self.model.classifier = nn.Linear(in_features=768, out_features=n_classes, bias=True)

	def forward(self, input):
		return self.model.forward(**input)

	def get_model(self):
		return self.model

	def reassignWeights(self, oldModel, weightVectorsDown, biasDown, weightVectorsUp, ro1=False):
		if ro1:
			print("ro1 asserted")
		with torch.no_grad():
			for layerIndex in range(self.depth):
				if ro1:
					assert torch.all((oldModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].weight ==
									  weightVectorsDown["intermediate"][layerIndex]))

					assert torch.all((oldModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].weight ==
									  weightVectorsUp["intermediate"][layerIndex]))

					assert torch.all((oldModel.model.vit.encoder.layer[layerIndex].output.adapter.block[0].weight ==
									  weightVectorsDown["output"][layerIndex]))

					assert torch.all((oldModel.model.vit.encoder.layer[layerIndex].output.adapter.block[2].weight ==
									  weightVectorsUp["output"][layerIndex]))

					assert torch.all((oldModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].bias ==
									  biasDown["intermediate"][layerIndex]))

					assert torch.all((oldModel.model.vit.encoder.layer[layerIndex].output.adapter.block[0].bias ==
									  biasDown["output"][layerIndex]))

				assert (self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].weight.shape == weightVectorsDown["intermediate"][layerIndex].shape)
				self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].weight = nn.Parameter(weightVectorsDown["intermediate"][layerIndex]).to(device)
				assert (self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].bias.shape == nn.Parameter(biasDown["intermediate"][layerIndex]).shape)
				self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].bias = nn.Parameter(biasDown["intermediate"][layerIndex]).to(device)

				assert (self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].weight.shape == weightVectorsUp["intermediate"][layerIndex].shape)
				self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].weight = nn.Parameter(weightVectorsUp["intermediate"][layerIndex]).to(device)
				assert (self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].bias.shape == nn.Parameter(oldModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].bias).shape)
				self.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].bias = nn.Parameter(oldModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].bias).to(device)

				assert (self.model.vit.encoder.layer[layerIndex].output.adapter.block[0].weight.shape == weightVectorsDown["output"][layerIndex].shape)
				self.model.vit.encoder.layer[layerIndex].output.adapter.block[0].weight = nn.Parameter(weightVectorsDown["output"][layerIndex]).to(device)
				assert (self.model.vit.encoder.layer[layerIndex].output.adapter.block[0].bias.shape == nn.Parameter(biasDown["output"][layerIndex]).shape)
				self.model.vit.encoder.layer[layerIndex].output.adapter.block[0].bias = nn.Parameter(biasDown["output"][layerIndex]).to(device)

				assert (self.model.vit.encoder.layer[layerIndex].output.adapter.block[2].weight.shape == weightVectorsUp["output"][layerIndex].shape)
				self.model.vit.encoder.layer[layerIndex].output.adapter.block[2].weight = nn.Parameter(weightVectorsUp["output"][layerIndex]).to(device)
				assert (self.model.vit.encoder.layer[layerIndex].output.adapter.block[2].bias.shape == nn.Parameter(oldModel.model.vit.encoder.layer[layerIndex].output.adapter.block[2].bias).shape)
				self.model.vit.encoder.layer[layerIndex].output.adapter.block[2].bias = nn.Parameter(oldModel.model.vit.encoder.layer[layerIndex].output.adapter.block[2].bias).to(device)

			self.model.classifier.weight.copy_(oldModel.model.classifier.weight)
			self.model.classifier.bias.copy_(oldModel.model.classifier.bias)
		self.zero_grad()
