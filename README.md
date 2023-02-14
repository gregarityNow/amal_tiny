To run an experiment, run:

python3 main.py -runType="train"

Specify various hyper-parameters:

-batch_size
-targetCompRate
-hid_size
-ro
-lr
-comp_n_epochs (maximal number of epochs of fine-tuning for fixed compression rate)
-n_epochs (number of epochs of initial fine-tuning for compression rate of 1)
-dsName ("mnist","cifar10", or "cifar100")
-stopEarly (number of epochs of no train accuracy amelioration before stopping training)
-randomScores (random neuron selection during pruning)
-normType (norm to use in scoring function)

To create accuracy/loss visualization (like figure 3), use runType="viz"

To create adapter size visualization (like figure 4), use runType="layerShrink"
