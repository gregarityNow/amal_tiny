
from .basis_funcs import *

import matplotlib.patches as patches

from .load_data import saveFig


def visualize_loss_acc(opt, allLosses, allAccs, epochLengths, compRates = None):
	fig, (accAx, lossAx) = plt.subplots(2,1, dpi=400)

	trainAcc, trainLoss, testAcc, testLoss = [], [], [], []
	xPrev = 0

	for compRateIndex, accs in enumerate(allAccs):

		compressionSizeLen = epochLengths[compRateIndex]

		for axIndex, ax in enumerate([accAx, lossAx]):
			rect = patches.Rectangle((xPrev, 0), compressionSizeLen, 1, alpha=0.4,
									 facecolor=("blue" if compRateIndex % 2 == 0 else "orange"))
			ax.add_patch(rect)

			if compRates is not None:
				ax.text(xPrev + compressionSizeLen / 2, 0.1 if axIndex == 0 else 0.9, s = r"$\sigma=$"+str(compRates[compRateIndex]))


		xPrev += compressionSizeLen
		trainAcc.extend(allAccs[compRateIndex]["train"])
		trainLoss.extend(allLosses[compRateIndex]["train"])
		testAcc.extend(allAccs[compRateIndex]["test"])
		testLoss.extend(allLosses[compRateIndex]["test"])

	smidge = 0.5 / len(allAccs[0]["train"])

	lossAx.plot(smidge + np.arange(len(trainLoss)) / len(allAccs[0]["train"]), trainLoss, label="Train")
	lossAx.plot(smidge + np.arange(len(testLoss)) / len(allAccs[0]["test"]), testLoss, label="Test")
	accAx.plot(smidge + np.arange(len(trainAcc)) / len(allAccs[0]["train"]), trainAcc, label="Train")
	accAx.plot(smidge + np.arange(len(testAcc)) / len(allAccs[0]["test"]), testAcc, label="Test")
	accAx.legend()
	lossAx.legend()
	lossAx.set_title("Loss")
	accAx.set_title("Accuracy")
	for ax in [accAx, lossAx]:
		ax.set_xticks(np.arange(xPrev));
		ax.set_xticklabels(np.arange(xPrev))
		ax.set_xlabel("Epoch")

	saveFig(opt, "accLossThroughEpochs");