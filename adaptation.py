from torch import nn
from torch import optim
import torch.nn.functional as F
import os
from data import *
from ode_model import Forecaster
from utils import create_logger, DataLoaderODE, write_image, batch_transform, batch_transform_inverse, \
    batch_transform_loss, save_numpy
from datetime import datetime
import getopt
import sys
import math
from itertools import product

# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

dataset = "lotka"
gpu = 0
gpu_id = 0
path_model = ""
regul = ""
home = './results'
model_home = "./results"
opts, args = getopt.getopt(sys.argv[1:], "c:d:g:m:h:e:")
for opt, arg in opts:
    if opt == "-c":
        l_c = float(arg)
    if opt == "-d":
        dataset = arg
    if opt == "-g":
        gpu = int(arg)
    if opt == "-m":
        model_home = arg
    if opt == "-h":
        home = arg
    if opt == "-e":
        model_exp = arg

now = datetime.now()
ts = now.strftime("%Y%m%d_%H%M%S")
cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')
filename = f"{str(ts)}"
path_results = os.path.join(home, dataset)
path_checkpoint = os.path.join(path_results, ts)
if model_home == 'pascal':
    path_model = os.path.join('/data', 'yiny', model_home, 'exp', 'mask_code_net', dataset, model_exp, 'model_ind.pt')
else:
    path_model = os.path.join('/net', model_home, 'yiny', 'exp', 'mask_code_net', dataset, model_exp, 'model_ind.pt')
logger = create_logger(path_checkpoint, os.path.join(path_checkpoint, 'log'))
os.makedirs(path_checkpoint, exist_ok=True)

# Dataset param
is_ode = any(name in dataset for name in ["lotka", "pendulum", "g_osci"])

if dataset == "lotka":
    beta  = [0.625, 0.625, 1.125, 1.125]
    delta = [0.625, 1.125, 0.625, 1.125]
    dataset_train_params = {"n_data_per_env": 1, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
                            "params": [{"alpha": 0.5, "beta": beta_i, "gamma": 0.5, "delta": delta_i} for beta_i, delta_i in zip(beta, delta)]}
    minibatch_size = 1
    n_env = len(beta)
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(**dataset_test_params)

elif dataset == "g_osci":
    k1 = [85, 95]
    K1 = [0.625, 0.875]
    dataset_train_params = {'n_data_per_env': 1, 't_horizon': 1, "dt": 0.05, 'method': 'RK45', 'group': 'train',
                            'params': [{'J0': 2.5, 'k1': k1_i, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1_i,
                                        'q': 4, 'N': 1, 'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1_i, K1_i in product(k1, K1)]}
    minibatch_size = 1
    n_env = len(dataset_train_params['params'])
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = GlycolyticOscillatorDataset(**dataset_train_params), GlycolyticOscillatorDataset(**dataset_test_params)

# elif dataset == "pendulum":
#     dataset_train_params = {"n_data_per_env": 1, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
#                             "params": [{"a": 0.6, "b": 0.65}]}   # {"a": 0.45, "b": 0.65}
#     dataset_test_params = dict()
#     dataset_test_params.update(dataset_train_params)
#     dataset_test_params["n_data_per_env"] = 32
#     dataset_test_params["group"] = "test"
#     dataset_train, dataset_test = PendulumDataset(**dataset_train_params), PendulumDataset(**dataset_test_params)
elif dataset == "gray":
    f = [0.033, 0.036]
    k = [0.059, 0.061]
    minibatch_size = 1
    dataset_train_params = {"n_data_per_env": 1, "t_horizon": 400, "dt": 40, "size": 32, "n_block": 3, "dx": 1, "method": "RK45", 
                            "buffer_file": f"{path_results}/gray_buffer_train_ada.shelve",
                            "group": "train", "params": [{"f": f_i, "k": k_i, "r_u": 0.2097, "r_v": 0.105} for f_i, k_i in product(f, k)]}
    dataset_test_params = dict()
    n_env = len(dataset_train_params['params'])
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["buffer_file"] = f"{path_results}/gray_buffer_test_ada.shelve"
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)

elif dataset == "navier":
    size = 32
    tt = torch.linspace(0, 1, size + 1)[0:-1]
    X, Y = torch.meshgrid(tt, tt)
    f = 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y)))
    viscs = [8.5e-4, 9.5e-4, 1.05e-3, 1.15e-3]
    minibatch_size = 1
    n_env = len(viscs)
    dataset_train_params = {"n_data_per_env": 1, "t_horizon": 10, "dt_eval": 1, "size": size, "method": "euler",
                            "buffer_file": f"{path_results}/ns_buffer_ref_train_ada.shelve", "group": "train",
                            "params": [{"f": f, "visc": visc} for visc in viscs] }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_test_params["buffer_file"] = f"{path_results}/ns_buffer_ref_test_ada.shelve"
    dataset_train, dataset_test = NavierStokesDataset(**dataset_train_params), NavierStokesDataset(**dataset_test_params)
else:
    raise Exception(f"{dataset} does not exist")
dataloader_train, dataloader_test = DataLoaderODE(dataset_train, minibatch_size, n_env), \
                                    DataLoaderODE(dataset_test, minibatch_size, n_env, is_train=False)

# Forecaster
epsilon = epsilon_t = 0.95
update_epsilon_every = 30
log_every = 10
n_epochs = 120000
lr = 1e-3  # 1e-2
test_type = "ind"
checkpoint = torch.load(f"{path_model}", map_location=device)
forecaster_params = checkpoint["forecaster_params"]
forecaster_params['n_env'] = n_env
net = Forecaster(**forecaster_params, logger=logger, device=device)
model_dict = net.state_dict()
pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if
                (k in model_dict and not ("ghost_structure" in k or "codes" in k))}
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
net = net.to(device)

for name, param in net.named_parameters():
    if param.requires_grad and ("net_root" in name or "net_hyper" in name or "mask" in name):
        param.requires_grad = False
    logger.info(f"{name}, {param.requires_grad}")

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()

# Logs
logger.info(f"lr: {lr}")
logger.info(f"run_id: {ts}")
logger.info(f"gpu_id: {gpu_id}")
logger.info(f"dataset: {dataset}")
logger.info(f"path_model: {path_model}")
logger.info(f"regul: {regul}")

# Train
loss_test_min, loss_relative_min = float('inf'), float('inf')
last_train_loss = float('inf')
loss_test_env_min, loss_relative_env_min, code_min = None, None, None
done = False
for epoch in range(n_epochs):
    # if done:
    #     break
    for i, data in enumerate(dataloader_train, 0):  # (n_data_per_env/minibatch_size, [n_env*minibatch_size, state_c, t_horizon/dt])
        state = data["state"].to(device)  # [n_env * minibatch_size, state_c, t_horizon / dt]
        t = data["t"].to(device)
        targets = state
        if epoch == 0 and i == 0:
            logger.info(f"state: {list(state.size())}")
            logger.info(f"t: {t[0]}")
        inputs = batch_transform(state, minibatch_size)
        net.derivative.net_leaf.update_ghost()
        outputs = batch_transform_inverse(net(inputs, t[0], epsilon_t), n_env)
        loss = criterion(outputs, targets)

        # Total
        tot_loss = loss
        tot_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        difference = abs(loss.item() - last_train_loss)
        # print(f'{difference:.5e}')
        if difference < 1e-12:
            done = True
        last_train_loss = loss.item()

        if (epoch * len(dataloader_train) + i) % log_every == 0:
            logger.info("Runid %s, Epoch %d, Iter %d, Loss Train: %.3e" %
                    (ts, epoch + 1, i + 1, loss.item()))

        if (epoch * (len(dataset_train) // (minibatch_size * n_env)) + (i + 1)) % update_epsilon_every == 0:
            epsilon_t *= epsilon
            logger.info(f"epsilon: {epsilon_t:.3}")
            # logger.info("--Code--")
            # logger.info(net.derivative.codes.data)
            # logger.info("--------")

            loss_test = 0.0
            loss_test_env = torch.zeros(n_env)
            loss_relative = 0.0
            loss_relative_env = torch.zeros(n_env)
            with torch.no_grad():
                for j, data_test in enumerate(dataloader_test, 0):
                    state = data_test["state"].to(device)
                    t = data_test["t"].to(device)
                    targets = state
                    inputs = batch_transform(state, minibatch_size)
                    net.derivative.net_leaf.update_ghost()
                    outputs = batch_transform_inverse(net(inputs, t[0], epsilon_t), n_env)
                    loss_test += criterion(outputs, targets)
                    raw_loss_relative = torch.abs(outputs - targets) / torch.abs(targets)
                    loss_relative += raw_loss_relative.nanmean()

                    outputs, targets, raw_loss_relative = batch_transform_loss(outputs, minibatch_size), batch_transform_loss(targets, minibatch_size), batch_transform_loss(raw_loss_relative, minibatch_size)
                    dim = list(range(outputs.dim()))
                    dim.remove(1)
                    loss_test_env += F.mse_loss(outputs, targets, reduction='none').mean(dim=dim).cpu()
                    loss_relative_env += raw_loss_relative.nanmean(dim=dim).cpu()
                loss_test /= j + 1
                loss_test_env /= j + 1
                loss_relative /= j + 1
                loss_relative_env /= j + 1

            if loss_test_min > loss_test:
                logger.info(f"Checkpoint created: min test loss was {loss_test_min}, new is {loss_test}")
                loss_test_min = loss_test
                loss_relative_min = loss_relative
                loss_test_env_min = loss_test_env
                loss_relative_env_min = loss_relative_env
                code_min = net.derivative.codes.data.detach()
                if not is_ode:
                    save_numpy(targets, outputs, os.path.join(path_checkpoint, f"numpy_viz.npy"))

            logger.info("Runid %s, Epoch %d, Iter %d, Loss Test: %.3e, Loss Relative: %.3e" % (ts, epoch + 1, i + 1, loss_test, loss_relative))
            logger.info("========")

loss_per_param = loss_test_env
loss_relative_per_param = loss_relative_env
codes_per_param = net.derivative.codes.data.detach()
if dataset == 'lotka':
    logger.info(f'beta: {beta}, delta: {delta}, loss: {loss_test}, relative loss: {loss_relative}')
if dataset == 'g_osci':
    logger.info(f'k1: {k1_ii}, K1: {K1_ii}, loss: {loss_test}, relative loss: {loss_relative}')


torch.save(loss_per_param, os.path.join(path_checkpoint, f"loss.pt"))
torch.save(loss_relative_per_param, os.path.join(path_checkpoint, f"loss_relative.pt"))
torch.save(codes_per_param, os.path.join(path_checkpoint, f"codes.pt"))


            
