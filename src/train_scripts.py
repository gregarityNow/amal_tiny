
from .basis_funcs import *;
from .viz import visualize_loss_acc
from .load_data import *
from .models import ViT_TINA, initializeModel
from .tina_imp import shrinkModel


def train_ViT_vanilla(opt):
    ro, lr = opt.ro, opt.lr

    allLosses = []
    allAccs = []
    epochLengths = []
    torch.cuda.empty_cache()
    if opt.dsName == "mnist":
        train_loader, test_loader, numClasses = get_mnist_loaders(opt);
    elif opt.dsName == "cifar10":
        train_loader, test_loader, numClasses = get_cifar10_loaders(opt);
    else:
        raise Exception("Don't know dataset", opt.dsName)

    model = initializeModel(opt, numClasses);

    model, accByEpoch, lossByEpoch, numEpochs = finetune_ViT(train_loader, test_loader, model, n_epochs=opt.n_epochs, lr=lr,doAdapt=0)

    saveModel(model, opt);

    allLosses.append(lossByEpoch)
    allAccs.append(accByEpoch)
    epochLengths.append(numEpochs)

    saveRunData(opt, (allLosses, allAccs))

    visualize_loss_acc(opt, allLosses, allAccs,epochLengths,compRates = None)

momentum = 0.9
weight_decay = 1e-4
import time
def train_ViT(opt):

    ro, lr = opt.ro, opt.lr

    allLosses = []
    allAccs = []
    epochLengths = []
    torch.cuda.empty_cache()
    if opt.dsName == "mnist":
        train_loader, test_loader, numClasses = get_mnist_loaders(opt);
    elif opt.dsName == "cifar10":
        train_loader, test_loader, numClasses = get_cifar10_loaders(opt);
    else:
        raise Exception("Don't know dataset",opt.dsName)

    model = initializeModel(opt, numClasses);

    allHiddenSizes = [model.hid_sizes]

    model, accByEpoch, lossByEpoch, numEpochs = finetune_ViT(train_loader, test_loader ,model,n_epochs=opt.n_epochs, lr = lr)
    baselineAcc = np.mean(accByEpoch["train"][int(len(accByEpoch["train"])*0.99):])
    print("Established baseline",baselineAcc)

    saveModel(model, opt);

    allLosses.append(lossByEpoch)
    allAccs.append(accByEpoch)
    epochLengths.append(numEpochs)

    compRates = [1]

    for compRound in range(10):

        shrinkTime = time.time()
        newModel, new_hidden_sizes = shrinkModel(model, ro)
        allHiddenSizes.append(new_hidden_sizes)
        shrinkTime = time.time( ) -shrinkTime
        print("shrunk in" ,shrinkTime)
        currCompRate = 1/ np.power(ro, compRound + 1)
        compRates.append(currCompRate)

        if ro == 1:
            for layerIndex in range(12):
                assert (torch.all(model.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].weight ==
                                  newModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].weight))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].weight ==
                                  newModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].weight))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].output.adapter.block[0].weight ==
                                  newModel.model.vit.encoder.layer[layerIndex].output.adapter.block[0].weight))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].output.adapter.block[2].weight ==
                                  newModel.model.vit.encoder.layer[layerIndex].output.adapter.block[2].weight))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].bias ==
                                  newModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[0].bias))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].output.adapter.block[0].bias ==
                                  newModel.model.vit.encoder.layer[layerIndex].output.adapter.block[0].bias))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].bias ==
                                  newModel.model.vit.encoder.layer[layerIndex].intermediate.adapter.block[2].bias))

                assert (torch.all(model.model.vit.encoder.layer[layerIndex].output.adapter.block[2].bias ==
                                  newModel.model.vit.encoder.layer[layerIndex].output.adapter.block[2].bias))

        model = newModel
        model, accByEpoch, lossByEpoch, numEpochs = finetune_ViT(train_loader, test_loader, model, n_epochs=opt.comp_n_epochs,
                                                      baseline=baselineAcc,lr=lr)
        allLosses.append(lossByEpoch)
        allAccs.append(accByEpoch)
        epochLengths.append(numEpochs)

        saveModel(model, opt,compRate=int(currCompRate));

        if currCompRate >= opt.targetCompRate:
            print("dunzo")
            break
        else:
            print(currCompRate, opt.targetCompRate, "still hackin")

    saveRunData(opt, (allLosses, allAccs, compRates, epochLengths, allHiddenSizes))

    visualize_loss_acc(opt, allLosses, allAccs, epochLengths, compRates)

    return allLosses, allAccs, compRates, allHiddenSizes


# from tqdm.notebook import tqdm


def getNonFrozenParams(model, fullTrain = False):
    if fullTrain:
        return model.parameters();
    params = [x[1] for x in model.named_parameters() if "adapter" in x[0] or "classifier" in x[0]]
    print("non-frozen",len(params),len(list(model.parameters())))
    return params


from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")



def converged(accs, baseline, prevBestAcc):
    halfWay = np.mean(accs[int(len(accs)*0.48):int(len(accs)*0.52)])
    final = np.mean(accs[int(len(accs)*0.98):])
    print("testing convergence for halfway",halfWay,"final",final,"baseline",baseline);
    if halfWay > final:
        print("decreasing accuracy convergence")
        return True
    if prevBestAcc > final:
        print("convergence due to previous epoch's best being higher")
        return True
    if (final-halfWay)/final < 0.02:
        print("final/halfway convergence")
        return True
    if (baseline-final)/baseline < 0.03:
        print("final/baseline convergence")
        return True
    print("No convergence")
    return False




def finetune_ViT(train_loader, test_loader, model, n_epochs=20, lr=0.01, criterion=nn.CrossEntropyLoss(),
                 momentum=0.9, weight_decay=1e-4, baseline = 1, doAdapt = 1):
    model = model.to(device)
    model.train()
    nonFrozenParams = getNonFrozenParams(model)
    optimizer = torch.optim.SGD(nonFrozenParams, lr=lr, momentum=momentum, weight_decay=weight_decay)


    accByEpoch, lossByEpoch = {"train": [], "test": []}, {"train": [], "test": []}

    prevBestAcc = 0
    didConverge = 0

    for epoch in range(n_epochs):
        total_loss, total_correct, total_seen = 0.0, 0.0, 0
        accForEpoch, lossForEpoch = {"train": [], "test": []}, {"train": [], "test": []}
        for batchIndex, (images, labels) in tqdm(enumerate(train_loader)):
            labels = labels.to(device)

            images = image_processor([images[i] for i in range(len(images))], return_tensors="pt")
            images = images.to(device)

            optimizer.zero_grad()
            model.zero_grad()
            if doAdapt:
                output = model(images).logits
            else:
                output = model(**images).logits
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct = (output.argmax(-1) == labels).sum().item()
            total_seen += len(output)
            total_correct += correct
            loss.backward()
            optimizer.step()
            if total_seen >= 96:
                lossForEpoch["train"].append(total_loss / total_seen)
                accForEpoch["train"].append(total_correct / total_seen)
                print("Batch", batchIndex, "loss", total_loss / total_seen, "accuracy", total_correct / total_seen)
                total_loss, total_correct, total_seen = 0.0, 0.0, 0

        print("Epoch loss",np.mean(lossForEpoch["train"]),"Epoch acc",np.mean(accForEpoch["train"]))

        accTest, lossTest = evaluate(model, test_loader)
        model.zero_grad();
        lossByEpoch["test"].extend(lossTest)
        accByEpoch["test"].extend(accTest)

        lossByEpoch["train"].extend(lossForEpoch["train"])
        accByEpoch["train"].extend(accForEpoch["train"])

        if converged(accForEpoch["train"], baseline, prevBestAcc):
            didConverge += 1
        else:
            didConverge = 0
        if didConverge > 1:
            break;

    return model, accByEpoch, lossByEpoch, epoch+1


def evaluate(model, test_loader, criterion=nn.CrossEntropyLoss(), doAdapt = True):
    model.eval()

    with torch.no_grad():
        total_loss, total_correct, total_seen = 0.0, 0.0, 0
        lossByEpoch, accByEpoch = [],[]

        for batchIndex, (images, labels) in tqdm(enumerate(test_loader)):
            labels = labels.to(device)
            # print(labels)

            images = image_processor([images[i] for i in range(len(images))], return_tensors="pt")
            images = images.to(device)

            if doAdapt:
                output = model(images).logits
            else:
                output = model(**images).logits
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct = (output.argmax(-1) == labels).sum().item()
            total_seen += len(output)
            total_correct += correct
            if total_seen >= 96:
                lossByEpoch.append(total_loss / total_seen)
                accByEpoch.append(total_correct / total_seen)
                print("Test Batch", batchIndex, "loss", total_loss / total_seen, "accuracy", total_correct / total_seen)
                total_loss, total_correct, total_seen = 0.0, 0.0, 0
        print("Test loss", np.mean(lossByEpoch), "accuracy", np.mean(accByEpoch))
    return accByEpoch, lossByEpoch