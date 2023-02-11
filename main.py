
from .src import train_ViT

allLosses, allAccs, compRates = train_ViT(quickie=100,batch_size=16,
                targetCompRate=8,n_epochs=5,ro=0.5,comp_n_epochs=5,hid_size=50)