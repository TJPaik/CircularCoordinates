import numpy as np
import torch
from ripser import ripser
from tqdm import tqdm

from .w_cc import weighted_circular_coordinate


def appr(el):
    return f"{el:.5f}"


class CircCoordLn:
    def __init__(self, data: np.ndarray, prime=47):
        self.data = data

        self.prime = prime
        self.ripser_result = ripser(data, coeff=prime, do_cocycles=True)
        self.dgm = self.ripser_result['dgms'][1]
        self.argsort = np.argsort(self.dgm[:, 1] - self.dgm[:, 0])[::-1]

        self.f_value = torch.zeros(len(data))
        self.f_value.requires_grad = True

        self.delta, self.vertex_values, self.cocycle_mat = None, None, None
        self.delta_torch, self.cocycle_torch = None, None
        self.eps = None
        self.optimizer = None
        self.arg_eps = None

    def setup(self, index):
        # index==0 : max
        birth = self.dgm[:, 0][self.argsort[index]]
        death = self.dgm[:, 1][self.argsort[index]]
        self.eps = (birth + death) / 2
        # self.arg_eps = self.argsort[index]
        self.arg_eps = index
        return death - birth

    def cc_original(self):
        # circ = circular_coordinate(prime=self.prime)
        # self.delta, self.vertex_values, self.cocycle_mat = circ.circular_coordinate(self.ripser_result, self.prime,
        #                                                                             weight=None, eps=self.eps,
        #                                                                             arg_eps=self.arg_eps)
        # self.vertex_values = np.mod(self.vertex_values, 1.0)
        self.delta, self.vertex_values, self.cocycle_mat = weighted_circular_coordinate(self.ripser_result,
                                                                                        ripser_result=True,
                                                                                        prime=self.prime,
                                                                                        cocycle_n=self.arg_eps,
                                                                                        eps=self.eps, weight_ft=None,
                                                                                        return_aux=True)
        return self.vertex_values

    def f_reset(self):
        self.f_value = torch.zeros(len(self.data))
        self.f_value.requires_grad = True

    def f_reset_L2(self):
        self.f_value = torch.from_numpy(self.vertex_values.copy()).float()
        self.f_value.requires_grad = True

    def cc_Lp_setup(self, lr):
        self.delta_torch = torch.sparse_coo_tensor(np.asarray([self.delta.row, self.delta.col]),
                                                   self.delta.data).float()
        self.delta_torch.requires_grad = False
        # self.cocycle_torch = torch.as_tensor(self.cocycle_mat[0])[0]
        self.cocycle_torch = torch.as_tensor(self.cocycle_mat.copy())
        self.cocycle_torch.requires_grad = False

        self.optimizer = torch.optim.SGD([self.f_value], lr=lr)

    def cc_aux_soft(self, temp):
        alpha_bar_abs = torch.abs(self.delta_torch @ self.f_value - self.cocycle_torch)
        softmax_loss = torch.sum(torch.softmax(alpha_bar_abs * temp, dim=-1) * alpha_bar_abs)
        self.optimizer.zero_grad()
        softmax_loss.backward()
        self.optimizer.step()
        return alpha_bar_abs, softmax_loss

    def cc_aux_p(self, p):
        alpha_bar = self.delta_torch @ self.f_value - self.cocycle_torch
        alpha_bar_p_norm = torch.sum(torch.abs(alpha_bar ** p)) ** (1 / p)
        self.optimizer.zero_grad()
        alpha_bar_p_norm.backward()
        self.optimizer.step()
        return alpha_bar, alpha_bar_p_norm

    def cc_aux_inf(self):
        alpha_bar = self.delta_torch @ self.f_value - self.cocycle_torch
        alpha_bar_inf_norm = torch.max(torch.abs(alpha_bar))
        self.optimizer.zero_grad()
        alpha_bar_inf_norm.backward()
        self.optimizer.step()
        return alpha_bar, alpha_bar_inf_norm

    def cc_Lp(self, epoch: int, lr=0.0001, p_norm: int = 2, delta_thr=None, max_count=10):
        self.cc_Lp_setup(lr)
        pbar = tqdm(range(epoch))
        losses = []
        prev_norm = 1e5
        count = 0
        if p_norm == np.inf:
            for _ in pbar:
                alpha_bar, alpha_bar_inf_norm = self.cc_aux_inf()
                pbar.set_postfix_str(f'{appr(alpha_bar_inf_norm.item())}')
                losses.append(alpha_bar_inf_norm.item())
                if delta_thr is not None and abs(alpha_bar_inf_norm.item() - prev_norm) < delta_thr:
                    count += 1
                    if count >= max_count:
                        return self.f_value.detach().numpy().copy(), losses
                prev_norm = alpha_bar_inf_norm.item()
        else:
            assert p_norm >= 1
            for _ in pbar:
                alpha_bar, alpha_bar_p_norm = self.cc_aux_p(p_norm)
                pbar.set_postfix_str(f'{appr(alpha_bar_p_norm.item())} / {appr(torch.max(alpha_bar).item())}')
                losses.append(torch.max(alpha_bar).item())
                if delta_thr is not None and abs(alpha_bar_p_norm.item() - prev_norm) < delta_thr:
                    count += 1
                    if count >= max_count:
                        return self.f_value.detach().numpy().copy(), losses
                prev_norm = alpha_bar_p_norm.item()

        return self.f_value.detach().numpy().copy(), losses

    def cc_Linf_Lp(self, epoch: int, lr: float = 1e-2, delta_thr=1e-5, lower_p=10, upper_p=30):
        self.cc_Lp_setup(lr)
        pbar = tqdm(range(epoch))
        losses = []
        p_norm = lower_p
        prev_norm = 1e5
        for epo in pbar:
            if p_norm < upper_p:
                alpha_bar, alpha_bar_p_norm = self.cc_aux_p(p_norm)
                if abs(alpha_bar_p_norm.item() - prev_norm) < delta_thr:
                    p_norm += 1
                prev_norm = alpha_bar_p_norm.item()
                alpha_bar_inf_norm = torch.max(torch.abs(alpha_bar)).item()
                pbar.set_postfix_str(f'{alpha_bar_p_norm.item()} / {alpha_bar_inf_norm} / p = {p_norm}')
                if torch.any(torch.isnan(self.f_value)).item():
                    raise ValueError
                losses.append(alpha_bar_inf_norm)
            else:
                alpha_bar, alpha_bar_inf_norm = self.cc_aux_inf()
                prev_norm = alpha_bar_inf_norm.item()
                pbar.set_postfix_str(
                    f'{appr(alpha_bar_inf_norm.item())} / {appr(alpha_bar_inf_norm.item())} / p = infty')
                losses.append(alpha_bar_inf_norm.item())

        return self.f_value.detach().numpy().copy(), losses

    def cc_Linf_softmax(self, epoch: int, lr: float = 1e-2, delta_thr=1e-5, lower_temp=2):
        self.cc_Lp_setup(lr)
        pbar = tqdm(range(epoch))
        losses = []
        prev_norm = 1e5
        for epo in pbar:
            alpha_bar_abs, softmax_loss = self.cc_aux_soft(lower_temp)
            if abs(softmax_loss.item() - prev_norm) < delta_thr:
                lower_temp += 1
            prev_norm = softmax_loss.item()
            pbar.set_postfix_str(f'{softmax_loss.item()} / {torch.max(alpha_bar_abs).item()} / temp = {lower_temp}')
            if torch.any(torch.isnan(self.f_value)).item():
                raise ValueError
            losses.append(torch.max(alpha_bar_abs).item())
        return self.f_value.detach().numpy().copy(), losses
