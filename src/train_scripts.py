
from .basis_funcs import *;
from .viz import visualize_loss_acc
from .load_data import *
from .models import ViT_TINA
from .tina_imp import shrinkModel



momentum = 0.9
weight_decay = 1e-4
import time
def train_ViT(opt, dsName):

    ro, lr = opt.ro, opt.lr

    allLosses = []
    allAccs = []
    torch.cuda.empty_cache()
    if dsName == "mnist":
        train_loader, test_loader, numClasses = get_mnist_loaders(opt.batch_size, quickie=opt.quickie)
    elif dsName == "cifar10":
        train_loader, test_loader, numClasses = get_cifar10_loaders(opt.batch_size, quickie=opt.quickie)
    else:
        raise Exception("Don't know dataset",dsName)
    model = ViT_TINA(opt.hid_size ,n_classes=10)

    allHiddenSizes = [model.hid_sizes]

    model, accByEpoch, lossByEpoch = finetune_ViT(train_loader, test_loader ,model,n_epochs=opt.n_epochs, lr = lr)

    saveModel(model, opt);

    allLosses.append(lossByEpoch)
    allAccs.append(accByEpoch)

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
        model, accByEpoch, lossByEpoch = finetune_ViT(train_loader, test_loader, model, n_epochs=opt.comp_n_epochs,
                                                      lr=lr)
        allLosses.append(lossByEpoch)
        allAccs.append(accByEpoch)

        saveModel(model, opt,compRate=int(currCompRate));

        if currCompRate >= opt.targetCompRate:
            print("dunzo")
            break
        else:
            print(currCompRate, opt.targetCompRate, "still hackin")

    saveRunData(opt, (allLosses, allAccs, compRates, allHiddenSizes))

    visualize_loss_acc(opt, allLosses, allAccs, compRates)

    return allLosses, allAccs, compRates, allHiddenSizes


# from tqdm.notebook import tqdm


def getNonFrozenParams(model):
    params = [x[1] for x in model.named_parameters() if "adapter" in x[0] or "classifier" in x[0]]
    print("non-frozen",len(params),len(list(model.parameters())))
    return params


from transformers import AutoImageProcessor
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")





def finetune_ViT(train_loader, test_loader, model, n_epochs=20, lr=0.01, criterion=nn.CrossEntropyLoss(),
                 momentum=0.9, weight_decay=1e-4):
    model = model.to(device)
    model.train()
    nonFrozenParams = getNonFrozenParams(model)
    optimizer = torch.optim.SGD(nonFrozenParams, lr=lr, momentum=momentum, weight_decay=weight_decay)

    lossByEpoch = {"train": [], "test": []}
    accByEpoch = {"train": [], "test": []}

    for i in range(n_epochs):
        total_loss, total_correct, total_seen = 0.0, 0.0, 0
        for batchIndex, (images, labels) in tqdm(enumerate(train_loader)):
            labels = labels.to(device)

            images = image_processor([images[i] for i in range(len(images))], return_tensors="pt")
            images = images.to(device)
            # print("james",images.keys())

            optimizer.zero_grad()
            model.zero_grad()
            output = model(images).logits
            loss = criterion(output, labels)
            total_loss += loss.item()
            correct = (output.argmax(-1) == labels).sum().item()
            total_seen += len(output)
            total_correct += correct
            loss.backward()
            optimizer.step()
            if batchIndex % 10 == 0:
                print("Batch", batchIndex, "loss", total_loss / total_seen, "accuracy", total_correct / total_seen)
            lossByEpoch["train"].append(total_loss / total_seen)
            accByEpoch["train"].append(total_correct / total_seen)
        print(f"[Epoch {i + 1:2d}] loss: {total_loss / total_seen:.2E} accuracy_train: {total_correct / total_seen:.2%}")


        accTest, lossTest = evaluate(model, test_loader)
        lossByEpoch["test"].extend(lossTest)
        accByEpoch["test"].extend(accTest)


    return model, accByEpoch, lossByEpoch


def evaluate(model, test_loader, criterion=nn.CrossEntropyLoss()):
    model.eval()

    total_loss, total_correct, total_seen = 0.0, 0.0, 0
    lossByEpoch, accByEpoch = [],[]

    for batchIndex, (images, labels) in tqdm(enumerate(test_loader)):
        labels = labels.to(device)
        # print(labels)

        images = image_processor([images[i] for i in range(len(images))], return_tensors="pt")
        images = images.to(device)

        output = model(images).logits
        loss = criterion(output, labels)
        total_loss += loss.item()
        correct = (output.argmax(-1) == labels).sum().item()
        total_seen += len(output)
        total_correct += correct
        if batchIndex % 10 == 0:
            print("Test Batch", batchIndex, "loss", total_loss / total_seen, "accuracy", total_correct / total_seen)
        lossByEpoch.append(total_loss / total_seen)
        accByEpoch.append(total_correct / total_seen)
    print("Test loss", total_loss / total_seen, "accuracy", total_correct / total_seen)
    return accByEpoch, lossByEpoch