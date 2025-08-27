import torch
from typing import Optional, Union
import utils.dist as dist
import time

from BaseTrainer import BaseTrainer

class GRPOTrainer(BaseTrainer):
    def __init__(
        self,
        device,
        config,
        reward_model = None,
    ):
        super().__init__(device, config, reward_model)

    @torch.no_grad()
    def sample_reference_model(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False
    ):
        images, f_hats, next_token_maps, logits_BlVs, logits_BlVs_sampled = self.var.rlhf_sample_infer_cfg(
            B, label_B, g_seed, cfg, top_k, top_p, more_smooth
        )

        rewards = self.RM(images)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        
        samples = {
            "f_hats": f_hats,
            "logits_BlVs": logits_BlVs,
            "next_token_maps": next_token_maps,
            "rewards": rewards,
        }

        return samples

    def train_step(
        self,
        loader,
        optimizer,
    ):
        total_loss = 0.0
        kl_total_loss = 0.0
        policy_total_loss = 0.0
        total_clip_frac = 0.0
        optimizer.zero_grad()
        (
            labels
        ) = next(loader)

        B = 25

        seed = 0

        samples = self.sample_reference_model(
            B=B*2, label_B=labels, g_seed=seed, cfg=4, top_k=900, top_p=0.95
        )

        prev_token_maps = samples["next_token_maps"]
        prev_f_hats = samples["f_hats"]

        logits_BlVs, logits_BlVs_sampled = self.var.grpo_step_infer_cfg(
            B=B*2, label_B=labels, g_seed=seed, prev_f_hats=prev_f_hats, prev_token_maps=prev_token_maps, cfg=4, top_k=900, top_p=0.95
        )



