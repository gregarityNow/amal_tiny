
from .basis_funcs import *

from .models import ViT_TINA

def rank_hidden_values(opt, model):
	scores = []
	depth = len(model.model.vit.encoder.layer)
	for layerIndex in range(depth):
		layer = model.model.vit.encoder.layer[layerIndex]

		for att in ["intermediate", "output"]:
			weightsDown = getattr(layer, att).adapter.block[0]
			weightsDownAbs = torch.abs(weightsDown.weight)

			weightsUp = getattr(layer, att).adapter.block[2]
			weightsUpAbs = torch.abs(weightsUp.weight)

			if opt.normType == "l1":
				byRowSummationDown = torch.norm(weightsDownAbs, dim=1,p=1)
				byRowSummationUp = torch.norm(weightsUpAbs, dim=0,p=1)
			elif opt.normType == "l2":
				if layerIndex == 0:
					print("l2 normin")
				byRowSummationDown = torch.norm(weightsDownAbs, dim=1, p=1)
				byRowSummationUp = torch.norm(weightsUpAbs, dim=0, p=1)
			elif opt.normType == "inf":
				if layerIndex == 0:
					print("inf normin")
				byRowSummationDown = torch.norm(weightsDownAbs, dim=1,p=float("inf"))
				byRowSummationUp = torch.norm(weightsUpAbs, dim=0,p=float("inf"))
			elif opt.normType == "minf":
				if layerIndex == 0:
					print("ming normin")
				byRowSummationDown = torch.norm(weightsDownAbs, dim=1, p=-float("inf"))
				byRowSummationUp = torch.norm(weightsUpAbs, dim=0, p=-float("inf"))

			denom = sum(weightsDownAbs.shape)
			downUpScores = (byRowSummationDown + byRowSummationUp) / denom
			for scoreIndex, score in enumerate(downUpScores):
				myScore = score.item()
				if opt.randomScores:
					myScore = np.random.rand()
					if scoreIndex == 0:
						print("randing that boii",myScore);
				d = {"layerIndex": layerIndex, "scoreIndex": scoreIndex,
					 "block": att, "score": myScore,
					 "weightsDown": weightsDown.weight[scoreIndex].detach(),
					 "biasDown": weightsDown.bias[scoreIndex].detach(),
					 "weightsUp": weightsUp.weight[:, scoreIndex].detach()}
				scores.append(d)
			# raise Exception("nose")
	scores.sort(key=lambda x: x["score"])
	# print("how ya like them apples")
	# for x in scores:
	# 	print(x["block"], x["score"])
	return scores


def appendWeightAndBiasFromNeuron(hid_sizes, weightVectorsDown, biasDown,
								  weightVectorsUp, neuron, layerIndex):
	hid_sizes[neuron["block"]][layerIndex] += 1
	weightVectorsDown[neuron["block"]][layerIndex].append(neuron["weightsDown"])
	biasDown[neuron["block"]][layerIndex].append(neuron["biasDown"])
	weightVectorsUp[neuron["block"]][layerIndex].append(neuron["weightsUp"])


def concatenateWeightsAndBiases(weightVectorsDown, biasDown, weightVectorsUp,
								block, layerIndex):
	conc = torch.stack(weightVectorsDown[block][layerIndex])
	weightVectorsDown[block][layerIndex] = conc

	conc = torch.stack(weightVectorsUp[block][layerIndex]).T
	weightVectorsUp[block][layerIndex] = conc

	conc = torch.stack(biasDown[block][layerIndex])
	biasDown[block][layerIndex] = conc


def setBiasUp(model, hid_sizes):
	depth = len(model.model.vit.encoder.layer)
	for layerIndex in range(depth):
		layer = model.model.vit.encoder.layer[layerIndex]
		for att in ["intermediate", "output"]:
			# biasUp = getattr(layer,att).adapter.block[2].bias
			with torch.no_grad():
				# print("prev",getattr(layer,att).adapter.block[2].bias[:5])
				ratio = (getattr(layer, att).adapter.block[2].in_features / hid_sizes[att][layerIndex])
				# print("rationes",ratio)
				getattr(layer, att).adapter.block[2].bias *= ratio
			# print("malone",getattr(layer,att).adapter.block[2].bias[:5])


def shrinkModel(opt, model, ro=0.75):
	scores = rank_hidden_values(opt, model)
	# scores.sort(key = lambda dic: dic["score"],reverse=True)
	# topNNeurons = scores[:int(len(scores)*ro)]

	toKeepSamples = int(len(scores) * ro)
	topNIndices = torch.multinomial(torch.Tensor([x["score"] for x in scores]), toKeepSamples)
	topNNeurons = [scores[i.item()] for i in topNIndices]

	depth = len(model.model.vit.encoder.layer)
	hid_sizes = {"intermediate": [0] * depth, "output": [0] * depth}
	weightVectorsDown = {"intermediate": [], "output": []}
	biasDown = {"intermediate": [], "output": []}
	weightVectorsUp = {"intermediate": [], "output": []}
	for layerIndex in range(depth):
		topNNeurons_layer = [x for x in topNNeurons if x["layerIndex"] == layerIndex]
		topNNeurons_layer.sort(key=lambda x: x["scoreIndex"])

		# print("blocks",layerIndex,set([x["block"] for x in topNNeurons_layer]))

		for block in ["intermediate", "output"]:
			weightVectorsDown[block].append([])
			biasDown[block].append([])
			weightVectorsUp[block].append([])

		if len(topNNeurons_layer) > 0:
			for neuron in topNNeurons_layer:
				appendWeightAndBiasFromNeuron(hid_sizes, weightVectorsDown,
											  biasDown, weightVectorsUp, neuron, layerIndex)
		else:
			print("No neurons left in layer..")
		for block in ["intermediate", "output"]:
			if len(weightVectorsDown[block][layerIndex]) == 0:
				neuron = sorted([x for x in scores if x["layerIndex"] == layerIndex
								 and x["block"] == block], key=lambda d: d["score"])[-1]
				# print("previously no values for",layerIndex,block)
				appendWeightAndBiasFromNeuron(hid_sizes, weightVectorsDown,
											  biasDown, weightVectorsUp, neuron, layerIndex)

			concatenateWeightsAndBiases(weightVectorsDown, biasDown, weightVectorsUp,
										block, layerIndex)

	setBiasUp(model, hid_sizes)
	newModel = ViT_TINA(hid_sizes=hid_sizes, n_classes=model.n_classes)
	newModel.reassignWeights(oldModel=model, weightVectorsDown=weightVectorsDown, biasDown=biasDown,
							 weightVectorsUp=weightVectorsUp, ro1=(ro == 1))

	# return model
	print("new hidden sizes", hid_sizes)

	return newModel, hid_sizes