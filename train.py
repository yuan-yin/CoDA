from torch import optim
from data import *
from ode_model import Forecaster
from utils import *
from datetime import datetime
import getopt
import sys, os
import math
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
# torch.backends.cudnn.benchmark = False
# torch.use_deterministic_algorithms(True)

log_every = 5
dataset = "lotka"
gpu = 0
gpu_id = 0
regul = ""
home = "./results"
is_layer = False
layers = [-1]
l_m = 1e-6
l_c = l_m * 100
l_t = l_m
seed = 1
options = {}
opts, args = getopt.getopt(sys.argv[1:], "c:d:g:h:l:m:r:s:t:x:")
code_c = 2
for opt, arg in opts:
    if opt == "-c":
        l_c = float(arg)
    if opt == "-d":
        dataset = arg
    if opt == "-g":
        gpu = int(arg)
    if opt == "-h":
        home = arg
    if opt == "-l":
        layers = [int(x) for x in arg.split(',')]
    if opt == "-m":
        l_m = float(arg)
    if opt == "-r":
        regul = arg
    if opt == "-s":
        seed = int(arg)
    if opt == "-t":
        l_t = float(arg)
    if opt == "-x":
        code_c = int(arg)

assert (code_c != None), 'Code dimension needed: add option -x [DIM_CODE]' 

freeze_code = "freeze" in regul
init_code_true = "true" in regul
is_layer = (layers[0] != -1)

dataset_test_ood = None
now = datetime.now()
ts = now.strftime("%Y%m%d_%H%M%S")
cuda = torch.cuda.is_available()
if cuda:
    gpu_id = gpu
    device = torch.device(f'cuda:{gpu_id}')
else:
    device = torch.device('cpu')

path_results = os.path.join(home, dataset)
path_checkpoint = os.path.join(path_results, ts)
logger = create_logger(path_checkpoint, os.path.join(path_checkpoint, "log"))
os.makedirs(path_checkpoint, exist_ok=True)
scaler = MinMaxScaler(feature_range=(-0.02, 0.02))  # StandardScaler()  #
is_ode = any(name in dataset for name in ["lotka", "pendulum", "g_osci"])
init_type = "default"
set_rdm_seed(seed)
codes_init = None

if dataset == "lotka":
    minibatch_size = 4
    factor = 1.0
    state_c = 2
    init_gain = 0.15
    method = "rk4"
    dataset_train_params = {
        "n_data_per_env": minibatch_size, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
        "params": [
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0}]}
    if init_code_true:
        true_params = torch.stack(
            [torch.tensor([list(params.values())])[0][1::2] for params in dataset_train_params["params"]])
        scaler.fit(true_params)
        transformed_params = scaler.transform(true_params)
        codes_init = nn.Parameter(torch.tensor(transformed_params, dtype=torch.float))
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(**dataset_test_params)
    # dataset_test_params_ood = dict()
    # dataset_test_params_ood.update(dataset_test_params)
    # dataset_test_params_ood["rdn_gen"] = 2
    # dataset_test_ood = LotkaVolterraDataset(**dataset_test_params_ood)
elif dataset == "lotka_50":
    minibatch_size = 4
    factor = 1.0
    state_c = 2
    init_gain = 0.15
    method = "rk4"
    params = np.random.normal(0.75, 0.15, 50 * 2)
    dataset_train_params = {
        "n_data_per_env": minibatch_size, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
        "params": [{"alpha": 0.5, "beta": params[n], "gamma": 0.5, "delta": params[50 + n]} for n in range(50)]}
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(**dataset_test_params)
elif dataset == "lotka_100":
    minibatch_size = 4
    factor = 1.0
    state_c = 2
    init_gain = 0.15
    method = "rk4"
    params = np.random.normal(0.75, 0.15, 100 * 2)
    dataset_train_params = {
        "n_data_per_env": minibatch_size, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
        "params": [{"alpha": 0.5, "beta": params[n], "gamma": 0.5, "delta": params[100 + n]} for n in range(100)]}
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(**dataset_test_params)
elif dataset == "lotka_4":
    minibatch_size = 4
    factor = 1.0
    state_c = 2
    init_gain = 0.15
    method = "rk4"
    dataset_train_params = {
        "n_data_per_env": minibatch_size, "t_horizon": 10, "dt": 0.5, "method": "RK45", "group": "train",
        "params": [
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.5},
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 0.5, "gamma": 0.5, "delta": 1.0},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 0.75, "gamma": 0.5, "delta": 1.0},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 0.75},
            {"alpha": 0.5, "beta": 1.0, "gamma": 0.5, "delta": 1.0},

            {"beta": 0.5, "alpha": 0.5, "delta": 0.5, "gamma": 0.5},
            {"beta": 0.5, "alpha": 0.75, "delta": 0.5, "gamma": 0.5},
            {"beta": 0.5, "alpha": 1.0, "delta": 0.5, "gamma": 0.5},
            {"beta": 0.5, "alpha": 0.5, "delta": 0.5, "gamma": 0.75},
            {"beta": 0.5, "alpha": 0.5, "delta": 0.5, "gamma": 1.0},
            {"beta": 0.5, "alpha": 0.75, "delta": 0.5, "gamma": 0.75},
            {"beta": 0.5, "alpha": 0.75, "delta": 0.5, "gamma": 1.0},
            {"beta": 0.5, "alpha": 1.0, "delta": 0.5, "gamma": 0.75},
            {"beta": 0.5, "alpha": 1.0, "delta": 0.5, "gamma": 1.0},
        ]
    }
    if init_code_true:
        true_params = torch.stack(
            [torch.tensor([list(params.values())])[0] for params in dataset_train_params["params"]])
        scaler.fit(true_params)
        transformed_params = scaler.transform(true_params)
        codes_init = nn.Parameter(torch.tensor(transformed_params, dtype=torch.float))
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(**dataset_test_params)
elif dataset == "lotka_17":
    minibatch_size = 4
    factor = 1.0
    state_c = 2
    init_gain = 0.15
    method = "rk4"
    circ_x = lambda centre, radius, radian: centre[0] + radius * math.cos(radian)
    circ_y = lambda centre, radius, radian: centre[1] + radius * math.sin(radian)

    centre = [0.75, 0.75]
    centre_2 = [1.25, 0.75]
    centre_3 = [1.25, 0.5]
    radius = 0.25
    code_c = 3

    dataset_train_params = {
        'n_data_per_env': 4, 't_horizon': 10,  "dt": 0.5, 'method': 'RK45', 'group': 'train',
        'params': [{'alpha': 0.5, 'beta': circ_x(centre, radius, 2 * i * math.pi / 9), 'gamma': 0.5,
              'delta': circ_y(centre, radius, 2 * i * math.pi / 9)} for i in range(9)] \
            + [{'alpha': 0.5, 'beta': circ_x(centre_2, radius, 2 * i * math.pi / 9 + math.pi), 'gamma': 0.5,
                'delta': circ_y(centre_2, radius, 2 * i * math.pi / 9 + math.pi)} for i in range(1,9)] # \
            # + [{'alpha': 0.5, 'beta': circ_x(centre_3, radius, 2 * i * math.pi / 9 + math.pi),
                 #'gamma': circ_y(centre_3, radius, 2 * i * math.pi / 9 + math.pi), 'delta': 0.75} for i in range(1,9)],
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = LotkaVolterraDataset(**dataset_train_params), LotkaVolterraDataset(**dataset_test_params)
elif dataset == "pendulum":
    minibatch_size = 8
    factor = 1.0
    state_c = 2
    init_gain = 0.1
    method = "rk4"
    dataset_train_params = {
        "n_data_per_env": 8, "t_horizon": 4, "dt": 0.2, "method": "RK45", "group": "train",
        "params": [
            {"b": 0.3, "m": 1, 'g':9.8, 'L':10.0},
            {"b": 0.4, "m": 1, 'g':9.8, 'L':10.0},
            {"b": 0.5, "m": 1, 'g':9.8, 'L':10.0},
            {"b": 0.3, "m": 2, 'g':9.8, 'L':10.0},
            {"b": 0.4, "m": 2, 'g':9.8, 'L':10.0},
            {"b": 0.5, "m": 2, 'g':9.8, 'L':10.0},
            {"b": 0.3, "m": 3, 'g':9.8, 'L':10.0},
            {"b": 0.4, "m": 3, 'g':9.8, 'L':10.0},
            {"b": 0.5, "m": 3, 'g':9.8, 'L':10.0}
        ]
    }
    if init_code_true:
        true_params = torch.stack(
            [torch.tensor([list(params.values())])[0][:2] for params in dataset_train_params["params"]])
        scaler.fit(true_params)
        transformed_params = scaler.transform(true_params)
        codes_init = nn.Parameter(torch.tensor(transformed_params, dtype=torch.float))
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = PendulumDataset(**dataset_train_params), PendulumDataset(**dataset_test_params)
elif "g_osci" in dataset:
    minibatch_size = 32
    factor = 1.0
    state_c = 7
    init_gain = 0.1
    method = "rk4"
    k1_range = [100, 90, 80]
    if "_1" in dataset:
        k1_range = [100, 97.5, 95]
    elif "_2" in dataset:
        k1_range = [100, 95, 90]
    elif "_3" in dataset:
        k1_range = [100, 99.5, 99]
    K1_range = [1, 0.75, 0.5]
    dataset_train_params = {
        'n_data_per_env': 32, 't_horizon': 1,  "dt": 0.05, 'method': 'RK45', 'group': 'train',
        'params': [{'J0': 2.5, 'k1': k1, 'k2': 6, 'k3': 16, 'k4': 100, 'k5': 1.28, 'k6': 12, 'K1': K1, 'q': 4, 'N': 1,
                    'A': 4, 'kappa': 13, 'psi': 0.1, 'k': 1.8} for k1 in k1_range for K1 in K1_range]}
    if init_code_true:
        unscaled_codes_init = [[list(params.values())[t] for t in [1, 7]] for params in dataset_train_params["params"]]
        # unscaled_codes_init = np.ravel(unscaled_codes_init).reshape(-1, 1)
        scaled_codes_init = scaler.fit_transform(unscaled_codes_init)
        # scaled_codes_init = np.reshape(scaled_codes_init, (len(dataset_train_params["params"]), 2))
        codes_init = nn.Parameter(torch.stack([torch.tensor(code).float() for code in scaled_codes_init]))
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = GlycolyticOscillatorDataset(**dataset_train_params), GlycolyticOscillatorDataset(**dataset_test_params)
elif dataset == "gray":
    minibatch_size = 1
    factor = 5e-4
    state_c = 2
    init_gain = 1
    # init_type = "orthogonal"
    method = "rk4"
    # options['step_size'] = 1
    dataset_train_params = {
        "n_data_per_env": 1, "t_horizon": 400, "dt": 40, "method": "RK45", "size": 32, "n_block": 3, "dx": 1, "group": "train",
        "buffer_file": f"{path_results}/gray_buffer_train.shelve",
        "params": [
            # {"f": 0.037, "k": 0.06, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.03, "k": 0.062, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.039, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.03, "k": 0.058, "r_u": 0.2097, "r_v": 0.105},
            {"f": 0.039, "k": 0.062, "r_u": 0.2097, "r_v": 0.105}  # ,
            # {"f": 0.0345, "k": 0.062, "r_u": 0.2097, "r_v": 0.105},
            # {"f": 0.0345, "k": 0.058, "r_u": 0.2097, "r_v": 0.105}
        ]
    }
    if init_code_true:
        with torch.no_grad():
            true_params = torch.stack(
                [torch.tensor([list(params.values())])[0][:2] for params in dataset_train_params["params"]])
            scaler.fit(true_params)
            transformed_params = scaler.transform(true_params)
            codes_init = nn.Parameter(torch.tensor(transformed_params, dtype=torch.float)).to(device)
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["buffer_file"] = f"{path_results}/gray_buffer_test.shelve"
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = GrayScottDataset(**dataset_train_params), GrayScottDataset(**dataset_test_params)
elif dataset == "wave":
    minibatch_size = 1
    factor = 1.e-3
    state_c = 2
    init_gain = 0.1
    method = "rk4"
    dataset_train_params = {
        "n_data_per_env": 1, "t_horizon": 400, "dt": 40, "method": "RK45", "size": 32, "dx": 1, "group": "train",
        "params": [
            {"c": 300, "k": 50},
            {"c": 350, "k": 50},
            {"c": 400, "k": 50}
        ]
    }
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_train, dataset_test = WaveDataset(**dataset_train_params), WaveDataset(**dataset_test_params)
elif dataset == "navier":
    minibatch_size = 16
    factor = 1
    size = 32
    state_c = 1
    # init_type = "normal"
    init_gain = 0.1
    method = "euler"
    tt = torch.linspace(0, 1, size + 1)[0:-1]
    X, Y = torch.meshgrid(tt, tt)
    # dataset_train_params = {
    #     "n_data_per_env": 16, "t_horizon": 10, "dt_eval": 1, "method": "euler", "size": size, "group": "train",
    #     "buffer_file": f"{path_results}/ns_buffer_train_3env_06-10_64.shelve",  # ns_buffer_train_30+10_1e-3.shelve
    #     "params": [
    #         ## {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.25e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.2e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.3e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.4e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.5e-3},
    #         {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 6e-4},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.65e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.7e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.75e-3},
    #         {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 8e-4},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.85e-3},
    #         #{"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.9e-3},
    #         # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 0.95e-3},
    #         {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.0e-3},
    #         ## {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 2.5e-3}
    #     ]
    # }
    dataset_train_params = {
        "n_data_per_env": 16, "t_horizon": 10, "dt_eval": 1, "method": "euler", "size": size, "group": "train",
        "buffer_file": f"{path_results}/ns_buffer_train_3env_08-12_32.shelve",  # ns_buffer_train_30+10_1e-3.shelve
        "params": [
            # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 6e-4},
            # {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 7e-4},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 8e-4},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 9e-4},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.0e-3},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.1e-3},
            {"f": 0.1 * (torch.sin(2 * math.pi * (X + Y)) + torch.cos(2 * math.pi * (X + Y))), "visc": 1.2e-3},
        ]
    }
    if init_code_true:
        with torch.no_grad():
            true_params = torch.stack(
                [torch.tensor([params['visc']]) for params in dataset_train_params["params"]])
            print(true_params)
            scaler.fit(true_params)
            transformed_params = scaler.transform(true_params)
            codes_init = nn.Parameter(torch.tensor(transformed_params, dtype=torch.float)).to(device)
    dataset_test_params = dict()
    dataset_test_params.update(dataset_train_params)
    dataset_test_params["n_data_per_env"] = 32
    dataset_test_params["group"] = "test"
    dataset_test_params["buffer_file"] = f"{path_results}/ns_buffer_test_3env_08-12_32.shelve"  # ns_buffer_test_30+10_1e-3.shelve
    dataset_train, dataset_test = NavierStokesDataset(**dataset_train_params), NavierStokesDataset(**dataset_test_params)
else:
    raise Exception(f"{dataset} does not exist")

n_env = len(dataset_train_params["params"])
dataloader_train, dataloader_test = DataLoaderODE(dataset_train, minibatch_size, n_env), \
                                    DataLoaderODE(dataset_test, minibatch_size, n_env, is_train=False)
if dataset_test_ood:
    dataloader_test_ood = DataLoaderODE(dataset_test_ood, minibatch_size, n_env, is_train=False)

# Forecaster
epsilon = epsilon_t = 0.99  # if is_ode else 0.98
update_epsilon_every = 30
if dataset == "navier":
    update_epsilon_every = 15
n_epochs = 120000
forecaster_params = {
    "dataset": dataset,
    "is_ode": is_ode,
    "state_c": state_c,
    "hidden_c": 64,
    "code_c": code_c,
    "n_env": n_env,
    "factor": factor,
    "method": method,
    "nl": "swish",
    "size": 0 if is_ode else dataset_train_params["size"],
    "is_layer": is_layer,
    "layers": layers,
    "htype": 'hyper',
    "options": options,
}
lr = 1e-3  # 1e-2
nl = forecaster_params["nl"]
net = Forecaster(**forecaster_params, logger=logger, codes_init=codes_init, device=device)
init_weights(net, init_type=init_type, init_gain=init_gain)
net = net.to(device)
net.logger = logger

# Optimizer
optimizer = optim.Adam(net.parameters(), lr=lr)
criterion = nn.MSELoss()

# Logs
logger.info(f"run_id: {ts}")
if cuda:
    logger.info(f"gpu_id: {gpu_id}")
logger.info(f"dataset: {dataset}")
logger.info(f"freeze_code: {freeze_code}")
logger.info(f"init_code_true: {init_code_true}")
logger.info(f"is_layer: {is_layer}")
logger.info(f"regul: {regul}")
logger.info(f"l_m: {l_m}")
logger.info(f"l_t: {l_t}")
logger.info(f"l_c: {l_c}")
logger.info(f"code_c: {code_c}")
logger.info(f"layers: {layers}")
logger.info(f"seed: {seed}")
logger.info(f"codes_init: {codes_init}")
logger.info(f"lr: {lr}")
logger.info(f"init_gain: {init_gain}")
logger.info(f"nl: {nl}")
logger.info(f"n_params forecaster: {count_parameters(net)}")
logger.info(f"n_params net_root.net: {count_parameters(net.derivative.net_root)}")

# Params Logs
if freeze_code:
    for name, param in net.named_parameters():
        if param.requires_grad and ("code" in name):
            param.requires_grad = False
        logger.info(f"{name}, {param.requires_grad}")
logger.info(f"net parameters: {dict(net.named_parameters()).keys()}")

# Train
loss_test_min_ind, loss_test_min_ood, loss_relative_min = float('inf'), float('inf'), float('inf')
for epoch in range(n_epochs):
    for i, data in enumerate(dataloader_train, 0):  # (n_data_per_env/minibatch_size, [n_env*minibatch_size, state_c, t_horizon/dt])
        state = data["state"].to(device)  # [n_env * minibatch_size, state_c, t_horizon / dt, h, w]
        t = data["t"].to(device)
        targets = state
        if epoch == 0 and i == 0:
            logger.info(f"state: {list(state.size())}")
            logger.info(f"t: {t[0]}")
        inputs = batch_transform(state, minibatch_size)
        net.derivative.net_leaf.update_ghost()
        outputs = batch_transform_inverse(net(inputs, t[0], epsilon_t), n_env)
        loss = criterion(outputs, targets)            

        # Regularization
        loss_reg_row = torch.zeros(1).to(device)
        loss_reg_col = torch.zeros(1).to(device)
        loss_reg_theta = torch.zeros(1).to(device)
        loss_reg_code = torch.zeros(1).to(device)
        if "l2t" in regul:
            for env_id in range(n_env):
                loss_reg_theta += torch.norm(net.derivative.net_hyper(net.derivative.codes[env_id])) ** 2
        if "l2c" in regul:
            # loss_reg_code += (torch.norm(net.derivative.codes, dim=1) ** 2).sum()
            loss_reg_code += (torch.norm(net.derivative.codes, dim=0) ** 2).sum()
        if "l12m" in regul:
            loss_reg_row += (torch.norm(net.derivative.net_hyper.weight, dim=1)).sum()
        if "l2m" in regul:
            loss_reg_row += torch.norm(net.derivative.net_hyper.weight) ** 2
        if "l12col" in regul:
            loss_reg_col += torch.norm(net.derivative.net_hyper.weight, dim=0).sum()
        if "lcos" in regul:
            weight = net.derivative.net_hyper.weight # n x n_xi
            norm_weight = torch.norm(weight, dim=1, keepdim=True)
            weight_normalized = weight / norm_weight
            codes = net.derivative.codes  # n_env x n_xi
            norm_codes = torch.norm(codes, dim=1, keepdim=True)
            codes_normalized = codes / norm_codes
            cosines = F.linear(codes, weight_normalized)
            # print(codes)
            loss_reg_row += torch.norm(cosines, dim=0).sum()

        # Total

        tot_loss = loss + l_m * (loss_reg_row + loss_reg_col) + l_t * loss_reg_theta + l_c * loss_reg_code
        tot_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()

        if (epoch * len(dataloader_train) + i) % log_every == 0:
            logger.info("Dataset %s, Runid %s, Epoch %d, Iter %d, Loss Train: %.2e, Loss RegRow: %.2e, Loss RegCol: %.2e, "
                        "Loss RegTheta: %.2e, Loss RegCode: %.2e" % (dataset, ts, epoch + 1, i + 1, loss.item(),
                                                                     loss_reg_row.item(), loss_reg_col.item(),
                                                                     loss_reg_theta.item(), loss_reg_code.item()))

        if (epoch * (len(dataset_train) // (minibatch_size * n_env)) + (i + 1)) % update_epsilon_every == 0:
            epsilon_t *= epsilon
            logger.info(f"epsilon: {epsilon_t:.3}")

            with torch.no_grad():
                print(net.derivative.codes)
                dataloader_test_list = [(dataloader_test, "ind"), (dataloader_test_ood, "ood")] if dataset_test_ood else [(dataloader_test, "ind")]
                for (dataloader_test_instance, test_type) in dataloader_test_list:
                    loss_test = 0.0
                    loss_relative = 0.0
                    # loss_env = [0.0 for _ in range(len(dataset_train_params["params"]))]
                    # loss_test_tot = 0.0
                    for j, data_test in enumerate(dataloader_test_instance, 0):
                        state = data_test["state"].to(device)
                        t = data_test["t"].to(device)
                        targets = state
                        inputs = batch_transform(state, minibatch_size)
                        net.derivative.net_leaf.update_ghost()
                        outputs = batch_transform_inverse(net(inputs, t[0]), n_env)
                        loss_test += criterion(outputs, targets)
                        raw_loss_relative = torch.abs(outputs - targets) / torch.abs(targets)
                        loss_relative += raw_loss_relative[~(torch.isnan(raw_loss_relative))].mean()
                        # loss_relative += (torch.abs(outputs - targets) / (torch.abs(targets) + 1e-6)).mean()
                    loss_test /= j + 1
                    loss_relative /= j + 1
                    # loss_env_norm = [x.numpy() / (j + 1) for x in loss_env]
                    # loss_test_tot = np.mean(loss_env_norm)
                    logger.info(f"loss_test: {loss_test}, loss_relative: {loss_relative}")
                    # logger.info(f"loss_env_norm: {loss_env_norm}")

                    loss_test_min = loss_test_min_ind if test_type == "ind" else loss_test_min_ood
                    if loss_test_min > loss_test:
                        logger.info(f"Checkpoint created: min {test_type} test loss was {loss_test_min}, new is {loss_test}")
                        if test_type == "ind":
                            loss_test_min_ind = loss_test
                        else:
                            loss_test_min_ood = loss_test
                        torch.save({
                            "epoch": epoch,
                            "mask": net.derivative.mask["mask"] if is_layer else None,
                            "model_state_dict": net.state_dict(),
                            "codes": [net.derivative.codes[e] for e in range(n_env)],
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss_ind": loss_test_min_ind,
                            "loss_ood": loss_test_min_ood,
                            "forecaster_params": forecaster_params,
                            "dataset_train_params": dataset_train_params}, os.path.join(path_checkpoint, f"model_{test_type}.pt"))
                        if not is_ode:
                            write_image(targets, outputs, os.path.join(path_checkpoint, f"img_{test_type}.png"), (dataset == "navier"))
                    if loss_relative_min > loss_relative:
                        logger.info(
                            f"Checkpoint created: min {test_type} relative loss was {loss_relative_min}, new is {loss_relative}")
                        loss_relative_min = loss_relative
                        torch.save({
                            "epoch": epoch,
                            "mask": net.derivative.mask["mask"] if is_layer else None,
                            "model_state_dict": net.state_dict(),
                            "codes": [net.derivative.codes[e] for e in range(n_env)],
                            "optimizer_state_dict": optimizer.state_dict(),
                            "loss_ind": loss_test_min_ind,
                            "loss_ood": loss_test_min_ood,
                            "forecaster_params": forecaster_params,
                            "dataset_train_params": dataset_train_params},
                            os.path.join(path_checkpoint, f"model_rel.pt"))
                        if not is_ode:
                            write_image(targets, outputs, os.path.join(path_checkpoint, f"img_rel.png"), (dataset == "navier"))

                    torch.save({
                        "epoch": epoch,
                        "mask": net.derivative.mask["mask"] if is_layer else None,
                        "model_state_dict": net.state_dict(),
                        "codes": [net.derivative.codes[e] for e in range(n_env)],
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss_ind": loss_test_min_ind,
                        "loss_ood": loss_test_min_ood,
                        "forecaster_params": forecaster_params,
                        "dataset_train_params": dataset_train_params},
                        os.path.join(path_checkpoint, f"model_train.pt"))

                    logger.info("Dataset %s, Runid %s, Epoch %d, Iter %d, Loss Test %s: %.2e" % (dataset, ts, epoch + 1, i + 1, test_type, loss_test))
                logger.info("========")
