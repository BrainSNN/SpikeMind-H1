import torch
#snn parameters
VDECAY = 0.1
CDECAY = 0.1
VTH = 0.1
GRAD_WIN = 0.3
TH_AMP = 0.01
TH_DECAY = 0.1
BASE_TH = 0.01
PARAM_LIST = [CDECAY, VDECAY, VTH, GRAD_WIN, TH_AMP, TH_DECAY, BASE_TH]

#init parameters
time_window = 28 # time windows, we set T = 28 in our paper
w_decay = 0.85 # weight decay factor
cfg_fc = [512, 512] # Network structure
AMP = 0.3
num_nodes = 512
output_feature =2 # output feature

#dataset processing
noise_level = 1 # noise level
data_num_percent=0.8
batch_size = 32
test_batch_size = 32

#train parameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_updates = 1 # meta-parameter update epochs, not used in this demo
thresh = 0.35 # threshold
lens = 0.5 # hyper-parameter in the approximate firing functions
decay = 0.4  # the decay constant of membrane potentials
num_classes = output_feature

num_epochs = 100
tau_w = 40 # synaptic filtering constant
lp_learning_rate = 5e-3  # learning rate of meta-local parameters
gp_learning_rate = 1e-2 # learning rate of gp-based parameters

