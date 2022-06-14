import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.integrate import solve_ivp
from torch.utils.data import Dataset
from functools import partial
import math
import shelve


##################
# Lotka-Volterra #
##################

class ODEDataset(Dataset):
    def __init__(self, n_data_per_env, t_horizon, params, dt, random_influence=0.2, method='RK45', group='train',
                 rdn_gen=1.):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.t_horizon = float(t_horizon)
        self.dt = dt
        self.random_influence = random_influence
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.buffer = dict()
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.method = method
        self.rdn_gen = rdn_gen

    def _f(self, t, x, env=0):
        raise NotImplemented

    def _get_init_cond(self, index):
        raise NotImplemented

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt).float()
        out = {'t': t, 'env': env}
        if self.buffer.get(index) is None:
            y0 = self._get_init_cond(env_index)
            y = solve_ivp(partial(self._f, env=env), (0., self.t_horizon), y0=y0, method=self.method,
                          t_eval=np.arange(0., self.t_horizon, self.dt))
            y = torch.from_numpy(y.y).float()
            out['state'] = y
            self.buffer[index] = y.numpy()
        else:
            out['state'] = torch.from_numpy(self.buffer[index])
        
        out['index'] = index
        out['param'] = torch.tensor(list(self.params_eq[env].values()))
        return out

    def __len__(self):
        return self.len


class LotkaVolterraDataset(ODEDataset):
    def _f(self, t, x, env=0):
        alpha = self.params_eq[env]['alpha']
        beta = self.params_eq[env]['beta']
        gamma = self.params_eq[env]['gamma']
        delta = self.params_eq[env]['delta']
        d = np.zeros(2)
        d[0] = alpha * x[0] - beta * x[0] * x[1]
        d[1] = delta * x[0] * x[1] - gamma * x[1]
        return d

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        return np.random.random(2) + self.rdn_gen


############
# Pendulum #
############


class PendulumDataset(ODEDataset):
    def _f(self, t, x, env=0):
        b = self.params_eq[env]['b']
        m = self.params_eq[env]['m']
        g = self.params_eq[env]['g']
        L = self.params_eq[env]['L']
        d = np.zeros(2)
        d[0] = x[1]
        d[1] = -(b/m) * x[1] - (g/L) * np.sin(x[0])
        return d

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        return np.random.randn(2)

##############
# Gray-Scott #
##############


class GrayScottDataset(Dataset):
    def __init__(self, n_data_per_env, size, t_horizon, params, dt, n_block, dx=2., random_influence=0.2,
                 buffer_file=None, method='RK45', group='train'):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.size = int(size)  # size of the 2D grid
        self.dx = dx  # space step discretized domain [-1, 1]
        self.time_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt)  # number of iterations
        self.random_influence = random_influence
        self.dt_eval = dt
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.buffer = shelve.open(buffer_file)
        self.method = method
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.n_block = n_block

    def _laplacian2D(self, a):
        # a_nn | a_nz | a_np
        # a_zn | a    | a_zp
        # a_pn | a_pz | a_pp
        a_zz = a

        a_nz = np.roll(a_zz, (+1, 0), (0, 1))
        a_pz = np.roll(a_zz, (-1, 0), (0, 1))
        a_zn = np.roll(a_zz, (0, +1), (0, 1))
        a_zp = np.roll(a_zz, (0, -1), (0, 1))

        a_nn = np.roll(a_zz, (+1, +1), (0, 1))
        a_np = np.roll(a_zz, (+1, -1), (0, 1))
        a_pn = np.roll(a_zz, (-1, +1), (0, 1))
        a_pp = np.roll(a_zz, (-1, -1), (0, 1))

        return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (self.dx ** 2)

    def _vec_to_mat(self, vec_uv):
        UV = np.split(vec_uv, 2)
        U = np.reshape(UV[0], (self.size, self.size))
        V = np.reshape(UV[1], (self.size, self.size))
        return U, V

    def _mat_to_vec(self, mat_U, mat_V):
        dudt = np.reshape(mat_U, self.size * self.size)
        dvdt = np.reshape(mat_V, self.size * self.size)
        return np.concatenate((dudt, dvdt))

    def _f(self, t, uv, env=0):
        U, V = self._vec_to_mat(uv)
        deltaU = self._laplacian2D(U)
        deltaV = self._laplacian2D(V)
        dUdt = (self.params_eq[env]['r_u'] * deltaU - U * (V ** 2) + self.params_eq[env]['f'] * (1. - U))
        dVdt = (self.params_eq[env]['r_v'] * deltaV + U * (V ** 2) - (self.params_eq[env]['f'] + self.params_eq[env]['k']) * V)
        duvdt = self._mat_to_vec(dUdt, dVdt)
        return duvdt

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        size = (self.size, self.size)
        U = 0.95 * np.ones(size)
        V = 0.05 * np.ones(size)
        for _ in range(self.n_block):
            r = int(self.size / 10)
            N2 = np.random.randint(low=0, high=self.size - r, size=2)
            U[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 0.
            V[N2[0]:N2[0] + r, N2[1]:N2[1] + r] = 1.
        return U, V

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.time_horizon, self.dt_eval).float()
        if self.buffer.get(f'{env},{env_index}') is None:
            print(f'generating {env},{env_index}')
            uv_0 = self._mat_to_vec(*self._get_init_cond(env_index))
            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=uv_0, method=self.method,
                            t_eval=np.arange(0., self.time_horizon, self.dt_eval))
            res_uv = res.y
            u, v = [], []
            for i in range(self.n):
                res_U, res_V = self._vec_to_mat(res_uv[:, i])
                u.append(torch.from_numpy(res_U).unsqueeze(0))
                v.append(torch.from_numpy(res_V).unsqueeze(0))
            u = torch.stack(u, dim=1)
            v = torch.stack(v, dim=1)
            state = torch.cat([u, v], dim=0).float()
            self.buffer[f'{env},{env_index}'] = {'state': state.numpy()}
            return {'state': state, 't': t, 'env': env, 'index': index}
        else:
            buf = self.buffer[f'{env},{env_index}']
            return {'state': torch.from_numpy(buf['state']), 't': t, 'env': env, 'index': index}

    def __len__(self):
        return self.len

########
# Wave #
########


class WaveDataset(Dataset):
    def __init__(self, n_data_per_env, size, t_horizon, params, dt, n_block, dx=2., random_influence=0.2,
                 buffer=dict(), method='RK45', group='train'):
        super().__init__()
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.size = int(size)  # size of the 2D grid
        self.dx = dx  # space step discretized domain [-1, 1]
        self.time_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt)  # number of iterations
        self.random_influence = random_influence
        self.dt_eval = dt
        self.params_eq = params
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.buffer = buffer
        self.method = method
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]
        self.n_block = n_block

    def _laplacian2D(self, a):
        # a_nn | a_nz | a_np
        # a_zn | a    | a_zp
        # a_pn | a_pz | a_pp
        a_zz = a

        a_nz = np.roll(a_zz, (+1, 0), (0, 1))
        a_pz = np.roll(a_zz, (-1, 0), (0, 1))
        a_zn = np.roll(a_zz, (0, +1), (0, 1))
        a_zp = np.roll(a_zz, (0, -1), (0, 1))

        a_nn = np.roll(a_zz, (+1, +1), (0, 1))
        a_np = np.roll(a_zz, (+1, -1), (0, 1))
        a_pn = np.roll(a_zz, (-1, +1), (0, 1))
        a_pp = np.roll(a_zz, (-1, -1), (0, 1))

        return (- 3 * a + 0.5 * (a_nz + a_pz + a_zn + a_zp) + 0.25 * (a_nn + a_np + a_pn + a_pp)) / (self.dx ** 2)

    def _vec_to_mat(self, vec_uv):
        UV = np.split(vec_uv, 2)
        U = np.reshape(UV[0], (self.size, self.size))
        V = np.reshape(UV[1], (self.size, self.size))
        return U, V

    def _mat_to_vec(self, mat_U, mat_V):
        dudt = np.reshape(mat_U, self.size * self.size)
        dvdt = np.reshape(mat_V, self.size * self.size)
        return np.concatenate((dudt, dvdt))

    def _f(self, t, uv, env=0):
        U, V = self._vec_to_mat(uv)
        deltaU = self._laplacian2D(U)
        deltaV = self._laplacian2D(V)
        dUdt = (self.params_eq[env]['r_u'] * deltaU - U * (V ** 2) + self.params_eq[env]['f'] * (1. - U))
        dVdt = (self.params_eq[env]['r_v'] * deltaV + U * (V ** 2) - (self.params_eq[env]['f'] + self.params_eq[env]['k']) * V)
        duvdt = self._mat_to_vec(dUdt, dVdt)
        return duvdt

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        size = (self.size, self.size)
        w = np.ones(size)
        sigma = np.random.randint(low=10, high=100)
        return w

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.time_horizon, self.dt_eval).float()
        if self.buffer.get(index) is None:
            uv_0 = self._mat_to_vec(*self._get_init_cond(env_index))
            res = solve_ivp(partial(self._f, env=env), (0., self.time_horizon), y0=uv_0, method=self.method,
                            t_eval=np.arange(0., self.time_horizon, self.dt_eval))
            res_uv = res.y
            u, v = [], []
            for i in range(self.n):
                res_U, res_V = self._vec_to_mat(res_uv[:, i])
                u.append(torch.from_numpy(res_U).unsqueeze(0))
                v.append(torch.from_numpy(res_V).unsqueeze(0))
            u = torch.stack(u, dim=1)
            v = torch.stack(v, dim=1)
            state = torch.cat([u, v], dim=0).float()
            self.buffer[index] = {'state': state.numpy()}
            return {'state': state, 't': t, 'env': env, 'index': index}
        else:
            buf = self.buffer[index]
            return {'state': torch.from_numpy(buf['state']), 't': t, 'env': env, 'index': index}

    def __len__(self):
        return self.len

#################
# Navier Stokes #
#################


class GaussianRF(object):
    def __init__(self, dim, size, alpha=2, tau=3, sigma=None, boundary="periodic"):
        self.dim = dim
        if sigma is None:
            sigma = tau ** (0.5 * (2 * alpha - self.dim))
        k_max = size // 2
        if dim == 1:
            k = torch.cat((torch.arange(start=0, end=k_max, step=1), torch.arange(start=-k_max, end=0, step=1)), 0)
            self.sqrt_eig = size * math.sqrt(2.0) * sigma * ((4 * (math.pi ** 2) * (k ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0] = 0.
        elif dim == 2:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, 1)
            k_x = wavenumers.transpose(0, 1)
            k_y = wavenumers
            self.sqrt_eig = (size ** 2) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0] = 0.0
        elif dim == 3:
            wavenumers = torch.cat((torch.arange(start=0, end=k_max, step=1),
                                    torch.arange(start=-k_max, end=0, step=1)), 0).repeat(size, size, 1)
            k_x = wavenumers.transpose(1, 2)
            k_y = wavenumers
            k_z = wavenumers.transpose(0, 2)
            self.sqrt_eig = (size ** 3) * math.sqrt(2.0) * sigma * (
                        (4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2 + k_z ** 2) + tau ** 2) ** (-alpha / 2.0))
            self.sqrt_eig[0, 0, 0] = 0.0
        self.size = []
        for j in range(self.dim):
            self.size.append(size)
        self.size = tuple(self.size)

    def sample(self):
        coeff = torch.randn(*self.size, dtype=torch.cfloat)
        coeff = self.sqrt_eig * coeff
        u = torch.fft.ifftn(coeff)
        u = u.real
        return u


class NavierStokesDataset(Dataset):
    def __init__(self, n_data_per_env, size, t_horizon, params, dt_eval, dx=2., buffer_file=None, method='RK45', group='train'):
        super().__init__()
        self.size = int(size)  # size of the 2D grid
        self.params_eq = params
        self.forcing_zero = params[0]['f']
        self.n_data_per_env = n_data_per_env
        self.num_env = len(params)
        self.len = n_data_per_env * self.num_env
        self.dx = dx  # space step discretized domain [-1, 1]
        self.t_horizon = float(t_horizon)  # total time
        self.n = int(t_horizon / dt_eval)  # number of iterations
        self.sampler = GaussianRF(2, self.size, alpha=2.5, tau=7)
        self.dt_eval = dt_eval
        self.dt = 1e-3
        self.buffer = shelve.open(buffer_file)
        self.test = (group == 'test')
        self.max = np.iinfo(np.int32).max
        self.method = method
        self.indices = [list(range(env * n_data_per_env, (env + 1) * n_data_per_env)) for env in range(self.num_env)]

    def navier_stokes_2d(self, w0, f, visc, T, delta_t, record_steps):
        # Grid size - must be power of 2
        N = w0.size()[-1]
        # Maximum frequency
        k_max = math.floor(N / 2.0)
        # Number of steps to final time
        steps = math.ceil(T / delta_t)
        # Initial vorticity to Fourier space
        w_h = torch.fft.fftn(w0, (N, N))
        # Forcing to Fourier space
        f_h = torch.fft.fftn(f, (N, N))
        # If same forcing for the whole batch
        if len(f_h.size()) < len(w_h.size()):
            f_h = torch.unsqueeze(f_h, 0)
        # Record solution every this number of steps
        record_time = math.floor(steps / record_steps)
        # Wavenumbers in y-direction
        k_y = torch.cat((torch.arange(start=0, end=k_max, step=1, device=w0.device),
                         torch.arange(start=-k_max, end=0, step=1, device=w0.device)), 0).repeat(N, 1)
        # Wavenumbers in x-direction
        k_x = k_y.transpose(0, 1)
        # Negative Laplacian in Fourier space
        lap = 4 * (math.pi ** 2) * (k_x ** 2 + k_y ** 2)
        lap[0, 0] = 1.0
        # Dealiasing mask
        dealias = torch.unsqueeze(
            torch.logical_and(torch.abs(k_y) <= (2.0 / 3.0) * k_max, torch.abs(k_x) <= (2.0 / 3.0) * k_max).float(), 0)
        # Saving solution and time
        sol = torch.zeros(*w0.size(), record_steps, 1, device=w0.device, dtype=torch.float)
        sol_t = torch.zeros(record_steps, device=w0.device)
        # Record counter
        c = 0
        # Physical time
        t = 0.0
        for j in range(steps):
            if j % record_time == 0:
                # Solution in physical space
                w = torch.fft.ifftn(w_h, (N, N))
                # Record solution and time
                sol[..., c, 0] = w.real
                # sol[...,c,1] = w.imag
                sol_t[c] = t
                c += 1
            # Stream function in Fourier space: solve Poisson equation
            psi_h = w_h.clone()
            psi_h = psi_h / lap
            # Velocity field in x-direction = psi_y
            q = psi_h.clone()
            temp = q.real.clone()
            q.real = -2 * math.pi * k_y * q.imag
            q.imag = 2 * math.pi * k_y * temp
            q = torch.fft.ifftn(q, (N, N))
            # Velocity field in y-direction = -psi_x
            v = psi_h.clone()
            temp = v.real.clone()
            v.real = 2 * math.pi * k_x * v.imag
            v.imag = -2 * math.pi * k_x * temp
            v = torch.fft.ifftn(v, (N, N))
            # Partial x of vorticity
            w_x = w_h.clone()
            temp = w_x.real.clone()
            w_x.real = -2 * math.pi * k_x * w_x.imag
            w_x.imag = 2 * math.pi * k_x * temp
            w_x = torch.fft.ifftn(w_x, (N, N))
            # Partial y of vorticity
            w_y = w_h.clone()
            temp = w_y.real.clone()
            w_y.real = -2 * math.pi * k_y * w_y.imag
            w_y.imag = 2 * math.pi * k_y * temp
            w_y = torch.fft.ifftn(w_y, (N, N))
            # Non-linear term (u.grad(w)): compute in physical space then back to Fourier space
            F_h = torch.fft.fftn(q * w_x + v * w_y, (N, N))
            # Dealias
            F_h = dealias * F_h
            # Cranck-Nicholson update
            w_h = (-delta_t * F_h + delta_t * f_h + (1.0 - 0.5 * delta_t * visc * lap) * w_h) / \
                  (1.0 + 0.5 * delta_t * visc * lap)
            # Update real time (used only for recording)
            t += delta_t

        return sol, sol_t

    def _get_init_cond(self, index):
        torch.manual_seed(index if not self.test else self.max - index)
        if self.buffer.get(f'init_cond_{index}') is None:
            w0 = self.sampler.sample()
            state, _ = self.navier_stokes_2d(w0, f=self.forcing_zero, visc=8e-4, T=30.0,
                                             delta_t=self.dt, record_steps=20)
            init_cond = state[:, :, -1, 0]
            self.buffer[f'init_cond_{index}'] = init_cond.numpy()
        else:
            init_cond = torch.from_numpy(self.buffer[f'init_cond_{index}'])

        return init_cond

    def __getitem__(self, index):
        env = index // self.n_data_per_env
        env_index = index % self.n_data_per_env
        t = torch.arange(0, self.t_horizon, self.dt_eval).float()
        if self.buffer.get(f'{env},{env_index}') is None:
            print(f'calculating index {env_index} of env {env}')
            w0 = self._get_init_cond(env_index)
            
            # w0 = F.interpolate(w0.unsqueeze(0).unsqueeze(0), scale_factor=2).squeeze(0).squeeze(0)
            state, _ = self.navier_stokes_2d(w0, f=self.params_eq[env]['f'], visc=self.params_eq[env]['visc'],
                                             T=self.t_horizon, delta_t=self.dt, record_steps=self.n)
            # h, w, t, nc
            state = state.permute(3, 2, 0, 1)[:, :self.n]  # nc, t, h, w
            # state = F.avg_pool2d(state, kernel_size=2, stride=2)
            # print(state.shape)
            self.buffer[f'{env},{env_index}'] = {'state': state.numpy()}
            return {'state': state, 't': t, 'env': env}
        else:
            buf = self.buffer[f'{env},{env_index}']
            return {'state': torch.from_numpy(buf['state'][:, :self.n]), 't': t, 'env': env, 'index': index}

    def __len__(self):
        return self.len


class GlycolyticOscillatorDataset(ODEDataset):
    def _f(self, t, x, env=0):
        keys = ['J0', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'K1', 'q', 'N', 'A', 'kappa', 'psi', 'k']
        J0, k1, k2, k3, k4, k5, k6, K1, q, N, A, kappa, psi, k = [self.params_eq[env][k] for k in keys]

        d = np.zeros(7)
        k1s1s6 = k1 * x[0] * x[5] / (1 + (x[5]/K1) ** q)
        d[0] = J0 - k1s1s6
        d[1] = 2 * k1s1s6 - k2 * x[1] * (N - x[4]) - k6 * x[1] * x[4]
        d[2] = k2 * x[1] * (N - x[4]) - k3 * x[2] * (A - x[5])
        d[3] = k3 * x[2] * (A - x[5]) - k4 * x[3] * x[4] - kappa * (x[3] - x[6])
        d[4] = k2 * x[1] * (N - x[4]) - k4 * x[3] * x[4] - k6 * x[1] * x[4] 
        d[5] = -2 * k1s1s6 + 2 * k3 * x[2] * (A - x[5]) - k5 * x[5]
        d[6] = psi * kappa * (x[3] - x[6]) - k * x[6]
        return d

    def _get_init_cond(self, index):
        np.random.seed(index if not self.test else self.max - index)
        ic_range = [(0.15, 1.60), (0.19, 2.16), (0.04, 0.20), (0.10, 0.35), (0.08, 0.30), (0.14, 2.67), (0.05, 0.10)]
        return np.random.random(7) * np.array([b-a for a, b in ic_range]) + np.array([a for a, _ in ic_range])
