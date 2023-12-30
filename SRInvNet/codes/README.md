# SRInvNet: Simultaneous resolution enhancement and denoising of seismic data based on invertible neural network

# Training
First set a config file in options/train/, then run as following:

	python train.py -opt options/train/train_InvNet.yml


## Contents

**Config**: [`options/`](./options) Configure the options for data loader, network structure, model, training strategies and etc.

**Data**: [`data/`](./data) A data loader to provide data for training, validation and testing.

**Model**: [`models/`](./models) Construct models for training and testing.

**Network**: [`models/modules/`](./models/modules) Construct different network architectures.

