import time
from typing import List, Optional, Tuple, Union, Dict
import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import utils.dist as dist
from models import VAR, VQVAE, VectorQuantizer2, build_vae_var_from_config
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

from functools import partial
from pprint import pformat

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

def filter_params(model, nowd_keys=()) -> Tuple[
    List[str], List[torch.nn.Parameter], List[Dict[str, Union[torch.nn.Parameter, float]]]
]:
    para_groups, para_groups_dbg = {}, {}
    names, paras = [], []
    names_no_grad = []
    count, numel = 0, 0
    for name, para in model.named_parameters():
        name = name.replace('_fsdp_wrapped_module.', '')
        if not para.requires_grad:
            names_no_grad.append(name)
            continue  # frozen weights
        count += 1
        numel += para.numel()
        names.append(name)
        paras.append(para)

        if para.ndim == 1 or name.endswith('bias') or any(k in name for k in nowd_keys):
            cur_wd_sc, group_name = 0., 'ND'
        else:
            cur_wd_sc, group_name = 1., 'D'
        cur_lr_sc = 1.
        if group_name not in para_groups:
            para_groups[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
            para_groups_dbg[group_name] = {'params': [], 'wd_sc': cur_wd_sc, 'lr_sc': cur_lr_sc}
        para_groups[group_name]['params'].append(para)
        para_groups_dbg[group_name]['params'].append(name)
    
    for g in para_groups_dbg.values():
        g['params'] = pformat(', '.join(g['params']), width=200)
    
    print(f'[get_param_groups] param_groups = \n{pformat(para_groups_dbg, indent=2, width=240)}\n')
    
    for rk in range(dist.get_world_size()):
        dist.barrier()
        if dist.get_rank() == rk:
            print(f'[get_param_groups][rank{dist.get_rank()}] {type(model).__name__=} {count=}, {numel=}', flush=True, force=True)
    print('')
    
    assert len(names_no_grad) == 0, f'[get_param_groups] names_no_grad = \n{pformat(names_no_grad, indent=2, width=240)}\n'
    return names, paras, list(para_groups.values())


class NullDDP(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super(NullDDP, self).__init__()
        self.module = module
        self.require_backward_grad_sync = False

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)


class BaseTrainer(object):
    def __init__(
        self,
        device,
        config,
        reward_func = None,
    ):
        self.device = device
        self.cfg = config

        # self.L = sum(pn * pn for pn in patch_nums)
        # self.last_l = patch_nums[-1] * patch_nums[-1]

        vae_local, var_wo_ddp = build_vae_var_from_config(device, config.model)
        quantize_local = vae_local.quantize

        vae_ckpt = 'vae_ch160v4096z32.pth'
        if dist.is_local_master():
            if not os.path.exists(vae_ckpt):
                os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
        dist.barrier()
        vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)

        vae_local: VQVAE = self.compile_model(vae_local, config.vfast)
        var_wo_ddp: VAR = self.compile_model(var_wo_ddp, config.tfast)
        var: DDP = (DDP if dist.initialized() else NullDDP)(var_wo_ddp, device_ids=[dist.get_local_rank()], find_unused_parameters=False, broadcast_buffers=False)

        print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
        count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
        print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
        print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')


        # build optimizer
        names, paras, para_groups = filter_params(var_wo_ddp, nowd_keys={
            'cls_token', 'start_token', 'task_token', 'cfg_uncond',
            'pos_embed', 'pos_1LC', 'pos_start', 'start_pos', 'lvl_embed',
            'gamma', 'beta',
            'ada_gss', 'moe_bias',
            'scale_mul',
        })

        opt_clz = {
            'adam':  partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=config.afuse),
            'adamw': partial(torch.optim.AdamW, betas=(0.9, 0.95), fused=config.afuse),
        }[config.opt.lower().strip()]
        opt_kw = dict(lr=config.tlr, weight_decay=0)

        var_optim = AmpOptimizer(
            mixed_precision=config.fp16, optimizer=opt_clz(params=para_groups, **opt_kw), names=names, paras=paras,
            grad_clip=config.tclip, n_gradient_accumulation=config.ac
        )
        del names, paras, para_groups

        self.var = var
        self.var_wo_ddp = var_wo_ddp
        self.var_opt = var_optim
        self.vae_local = vae_local

        self.reward_func = reward_func

    def train_step(self):

        raise NotImplementedError("train_step() is not implemented")

    def sample_step(self):

        raise NotImplementedError("sample_step() is not implemented")

    def compile_model(self, m, fast):
        if fast == 0 or self.cfg.local_debug:
            return m
        return torch.compile(m, mode={
            1: 'reduce-overhead',
            2: 'max-autotune',
            3: 'default',
        }[fast]) if hasattr(torch, 'compile') else m


# class VARTrainer(object):
#     def __init__(
#         self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
#         vae_local: VQVAE, var_wo_ddp: VAR, var: DDP,
#         var_opt: AmpOptimizer, label_smooth: float,
#     ):
#         super(VARTrainer, self).__init__()
        
#         self.var, self.vae_local, self.quantize_local = var, vae_local, vae_local.quantize
#         self.quantize_local: VectorQuantizer2
#         self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
#         self.var_opt = var_opt
        
#         del self.var_wo_ddp.rng
#         self.var_wo_ddp.rng = torch.Generator(device=device)
        
#         self.label_smooth = label_smooth
#         # self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
#         # self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
#         self.L = sum(pn * pn for pn in patch_nums)
#         self.last_l = patch_nums[-1] * patch_nums[-1]
#         self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
#         self.patch_nums, self.resos = patch_nums, resos
#         self.begin_ends = []
#         cur = 0
#         for i, pn in enumerate(patch_nums):
#             self.begin_ends.append((cur, cur + pn * pn))
#             cur += pn*pn
        
#         self.prog_it = 0
#         self.last_prog_si = -1
#         self.first_prog = True
    
#     @torch.no_grad()
#     def eval_ep(self, ld_val: DataLoader):
#         tot = 0
#         L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
#         stt = time.time()
#         training = self.var_wo_ddp.training
#         self.var_wo_ddp.eval()
#         for inp_B3HW, label_B in ld_val:
#             B, V = label_B.shape[0], self.vae_local.vocab_size
#             inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
#             label_B = label_B.to(dist.get_device(), non_blocking=True)
            
#             gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
#             gt_BL = torch.cat(gt_idx_Bl, dim=1)
#             x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
#             self.var_wo_ddp.forward
#             logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
#             L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
#             L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
#             acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
#             acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
#             tot += B
#         self.var_wo_ddp.train(training)
        
#         stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
#         dist.allreduce(stats)
#         tot = round(stats[-1].item())
#         stats /= tot
#         L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
#         return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
#     def get_config(self):
#         return {
#             'patch_nums':   self.patch_nums, 'resos': self.resos,
#             'label_smooth': self.label_smooth,
#             'prog_it':      self.prog_it, 'last_prog_si': self.last_prog_si, 'first_prog': self.first_prog,
#         }
    
#     def state_dict(self):
#         state = {'config': self.get_config()}
#         for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
#             m = getattr(self, k)
#             if m is not None:
#                 if hasattr(m, '_orig_mod'):
#                     m = m._orig_mod
#                 state[k] = m.state_dict()
#         return state
    
#     def load_state_dict(self, state, strict=True, skip_vae=False):
#         for k in ('var_wo_ddp', 'vae_local', 'var_opt'):
#             if skip_vae and 'vae' in k: continue
#             m = getattr(self, k)
#             if m is not None:
#                 if hasattr(m, '_orig_mod'):
#                     m = m._orig_mod
#                 ret = m.load_state_dict(state[k], strict=strict)
#                 if ret is not None:
#                     missing, unexpected = ret
#                     print(f'[VARTrainer.load_state_dict] {k} missing:  {missing}')
#                     print(f'[VARTrainer.load_state_dict] {k} unexpected:  {unexpected}')
        
#         config: dict = state.pop('config', None)
#         self.prog_it = config.get('prog_it', 0)
#         self.last_prog_si = config.get('last_prog_si', -1)
#         self.first_prog = config.get('first_prog', True)
#         if config is not None:
#             for k, v in self.get_config().items():
#                 if config.get(k, None) != v:
#                     err = f'[VAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
#                     if strict: raise AttributeError(err)
#                     else: print(err)
