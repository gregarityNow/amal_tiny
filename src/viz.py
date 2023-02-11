
from .basis_funcs import *

import matplotlib.patches as patches
import matplotlib.pyplot as plt


def visualize_loss_acc(allLosses, allAccs, compRates):
	fig, (accAx, lossAx) = plt.subplots(1, 2, dpi=400)

	trainAcc, trainLoss, testAcc, testLoss = [], [], [], []
	xPrev = 0

	xticks = []
	for compRateIndex, accs in enumerate(allAccs):

		epochLen = len(allAccs[compRateIndex]["train"]) / len(allAccs[0])
		xticks.append(xPrev + epochLen / 2)

		for ax in [accAx, lossAx]:
			rect = patches.Rectangle((xPrev, 0), epochLen, 1, alpha=0.4,
									 facecolor=("blue" if compRateIndex % 2 == 0 else "orange"))
			ax.add_patch(rect)
		xPrev += epochLen
		trainAcc.extend(allAccs[compRateIndex]["train"])
		trainLoss.extend(allLosses[compRateIndex]["train"])
		testAcc.extend(allAccs[compRateIndex]["test"])
		testLoss.extend(allLosses[compRateIndex]["test"])

	smidge = 0.5 / len(allAccs[0])
	lossAx.plot(smidge + np.arange(len(trainLoss)) / len(allAccs[0]), trainLoss, label="Train")
	lossAx.plot(smidge + np.arange(len(testLoss)) / len(allAccs[0]), testLoss, label="Test")
	accAx.plot(smidge + np.arange(len(trainAcc)) / len(allAccs[0]), trainAcc, label="Train")
	accAx.plot(smidge + np.arange(len(testAcc)) / len(allAccs[0]), testAcc, label="Test")
	accAx.legend()
	lossAx.legend()
	lossAx.set_title("Loss")
	accAx.set_title("Accuracy")
	for ax in [accAx, lossAx]:
		ax.set_xticks(xticks)
		ax.set_xticklabels([str(int(x)) for x in compRates])
		ax.set_xlabel(r"Compression rate $\sigma$")