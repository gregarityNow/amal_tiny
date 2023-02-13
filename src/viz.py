
from .basis_funcs import *

import matplotlib.patches as patches

from .load_data import saveFig, loadRunDataFromPath

def getMaxLoss(allLosses):
	maxLoss = 0
	for l in allLosses:
		for v in l.values():
			maxLoss = max(np.max(v),maxLoss)
	return maxLoss

def visualize_loss_acc(opt, allLosses, allAccs, epochLengths, compRates = None):

	print("visualizing",epochLengths, compRates)

	fig, (accAx, lossAx) = plt.subplots(2,1, dpi=400)

	trainAcc, trainLoss, testAcc, testLoss = [], [], [], []
	xPrev = 0

	maxLoss = getMaxLoss(allLosses)

	for compRateIndex, accs in enumerate(allAccs):

		compressionSizeLen = epochLengths[compRateIndex]

		for axIndex, ax in enumerate([accAx, lossAx]):
			rect = patches.Rectangle((xPrev, 0), compressionSizeLen, 1.2 if axIndex == 0 else maxLoss*1.2, alpha=0.4,
									 facecolor=("blue" if compRateIndex % 2 == 0 else "orange"))
			ax.add_patch(rect)

			if compRates is not None:
				ax.text(xPrev + compressionSizeLen / 2.2,(1.1 if axIndex == 0 else maxLoss*1.1), s = r"$\sigma=$"+str(compRates[compRateIndex]))

		xPrev += compressionSizeLen
		trainAcc.extend(allAccs[compRateIndex]["train"])
		trainLoss.extend(allLosses[compRateIndex]["train"])
		testAcc.extend(allAccs[compRateIndex]["test"])
		testLoss.extend(allLosses[compRateIndex]["test"])

	smidge = 0.5 / len(allAccs[0]["train"])

	lossAx.plot(smidge + np.arange(len(trainLoss)) * xPrev / len(trainLoss), trainLoss, label="Train")
	lossAx.plot(smidge + np.arange(len(testLoss)) * xPrev / len(testLoss), testLoss, label="Test")
	accAx.plot(smidge + np.arange(len(trainAcc)) * xPrev / len(trainAcc), trainAcc, label="Train")
	accAx.plot(smidge + np.arange(len(testAcc)) * xPrev / len(testAcc), testAcc, label="Test")
	accAx.legend()
	lossAx.legend()
	lossAx.set_title("Loss")
	accAx.set_title("Accuracy")
	for ax in [accAx, lossAx]:
		ax.set_xticks(np.arange(xPrev));
		ax.set_xticklabels([str(x) if x % 10 == 0 else "" for x in np.arange(xPrev)])
		ax.set_xlabel("Epoch")

	fig.tight_layout(rect=[0, 0.03, 1, 0.95])
	plt.suptitle("Evolution of model performance for increasing compression rate (ro="+str(opt.ro)+")")

	saveFig(opt, "accLossThroughEpochs");

import os
def enumerate_results():
	runDataDir = "./../out/runData/"
	for run in os.listdir(runDataDir):
		path = runDataDir + "/" + run;
		if "quickie" in path: continue;
		if not "_v" in path or "anilla" in path or "mall" in path: continue
		if "anilla" in path:
			allLosses, allAccs = loadRunDataFromPath(path);
			finalTrainAcc, finalTestAcc = np.mean(allAccs[0]["train"][-1 * int(len(allAccs[0]["train"])*0.01)]), np.mean(allAccs[0]["test"][-1 * int(len(allAccs[0]["test"])*0.01)])
		else:
			allLosses, allAccs, compRates, epochLengths, allHiddenSizes = loadRunDataFromPath(path)
			epochSize = len(allAccs[-1]["train"])/epochLengths[-1]
			finalTrainAcc, finalTestAcc = np.mean(allAccs[0]["train"][-1*epochSize:]), np.mean(allAccs[0]["test"][-1*epochSize:])
		# print(run,allAccs[0]["train"][:5],len(allAccs[0]["train"]),epochLengths)
		print(run, finalTrainAcc, finalTestAcc)


