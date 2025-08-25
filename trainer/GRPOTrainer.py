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
        reward_func = None,
    ):
        super().__init__(device, config, reward_func)

    def sample_step(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, grpo=False, init=False
    ):
        if init:
            rng, cond_BD, cur_L, f_hat_0, next_token_map_0 = self.rlhf_prestep(B, label_B)

    def sample_reference_model(
        self,B
    ):
        batch_size = 1
        batch_indices = torch.chunk(torch.arange(B), B // batch_size)

        all_log_probs = []
        all_rewards = []  
        all_multi_rewards = {}
        all_image_ids = []

        if dist.get_rank() == 0:
            sampling_time = 0
        for index, batch_idx in enumerate(batch_indices): # len(batch_indices)=12
            if dist.get_rank() == 0:
                meta_sampling_time = time.time()

        batch_caption = [caption[i] for i in batch_idx]

    def train_step(self):
        pass


