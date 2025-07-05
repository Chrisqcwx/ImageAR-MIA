################## 1. Download checkpoints and build models
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

image_net_path = "../imagenet/images"
ckpt_dir = ".cache/models--FoundationVision--var/snapshots/6d0ee6598a42f75079aa79e58b012b57b284c91b"

# if os.path.exists('/content/VAR'): os.chdir('/content/VAR')
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw

setattr(
    torch.nn.Linear, 'reset_parameters', lambda self: None
)  # disable default parameter init for faster speed
setattr(
    torch.nn.LayerNorm, 'reset_parameters', lambda self: None
)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var


from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from models.var import AdaLNSelfAttn

from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class VarPredCollection:

    cond: dict = field(default_factory=lambda: {'logits': [], 'gt': []})
    uncond: dict = field(default_factory=lambda: {'logits': [], 'gt': []})

    def __getitem__(self, item):
        if item == 'all':
            return VarPredCollection(
                {
                    'logits': torch.cat(self.cond['logits'], dim=1),
                    'gt': torch.cat(self.cond['gt'], dim=1),
                },
                {
                    'logits': torch.cat(self.uncond['logits'], dim=1),
                    'gt': torch.cat(self.uncond['gt'], dim=1),
                },
            )
        else:
            return VarPredCollection(
                {
                    'logits': self.cond['logits'][item],
                    'gt': self.cond['gt'][item],
                },
                {
                    'logits': self.uncond['logits'][item],
                    'gt': self.uncond['gt'][item],
                },
            )


from einops import rearrange


# def get_loss(logits, gt):
# return F.cross_entropy(logits, gt, reduction='none').mean(dim=-1)

EPS = 1e-8


def get_statistics_single_individual(logits_BlV, gt_Bl):
    B, L, V = logits_BlV.shape
    results = {}

    results['loss'] = F.cross_entropy(
        logits_BlV.permute(0, 2, 1), gt_Bl, reduction='none'
    ).mean(dim=-1)

    probs = torch.softmax(logits_BlV, dim=-1)
    log_probs = torch.log_softmax(logits_BlV, dim=-1)
    gt_probs_Bl = probs.gather(2, gt_Bl.unsqueeze(-1)).squeeze(-1)
    results['confidence'] = gt_probs_Bl.mean(dim=-1)

    entropy_Bl = (-probs * log_probs).sum(dim=-1)
    results['entropy'] = entropy_Bl.mean(dim=-1)
    results['rank'] = (
        (gt_probs_Bl.unsqueeze(-1) < probs).float().sum(dim=-1).mean(dim=-1)
    )

    gt_probs_sorted, _ = torch.sort(gt_probs_Bl, dim=-1)
    probs_sorted, _ = torch.sort(probs, dim=-1)

    if L >= 10:
        # mink
        mu_Bl = (probs * log_probs).sum(dim=-1)
        sigma_Bl = (probs * torch.square(log_probs)).sum(dim=-1) - torch.square(mu_Bl)
        mink_pp_Bl = (gt_probs_Bl - mu_Bl) / (sigma_Bl + EPS)
        for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            k_length = int(L * ratio)
            mink = gt_probs_sorted[:, :k_length].mean(dim=-1)
            mink_pp = torch.topk(
                mink_pp_Bl, k_length, dim=-1, largest=False
            ).values.mean(dim=-1)
            results[f'mink_{ratio}'] = mink
            results[f'mink++_{ratio}'] = mink_pp

    # renyi
    if L >= 2:
        prob_save = torch.clamp(probs, min=EPS, max=1 - EPS)

        max_gt_probs_Bl = probs_sorted[:, :, -1]
        gap_gt_probs_Bl = probs_sorted[:, :, -1] - probs_sorted[:, :, -2]
        max_log_gt_probs_Bl = torch.log(max_gt_probs_Bl + EPS)
        gap_log_gt_probs_Bl = max_log_gt_probs_Bl - torch.log(
            probs_sorted[:, :, -2] + EPS
        )

        modified_entropy_Bl = (
            -(1 - gt_probs_Bl) * torch.log(gt_probs_Bl + EPS)
            - ((probs * torch.log(1 - prob_save + EPS)).sum(dim=-1))
            + (gt_probs_Bl) * torch.log(1 - gt_probs_Bl + EPS)
        )
        assert modified_entropy_Bl.ndim == 2

        results['modified_entropy'] = modified_entropy_Bl.mean(dim=-1)
        results['max_gt_probs'] = max_gt_probs_Bl.mean(dim=-1)
        results['gap_gt_probs'] = -gap_gt_probs_Bl.mean(dim=-1)
        results['max_log_gt_probs'] = max_log_gt_probs_Bl.mean(dim=-1)
        results['gap_log_gt_probs'] = -gap_log_gt_probs_Bl.mean(dim=-1)

        all_renyi = [entropy_Bl, -max_log_gt_probs_Bl]

        for alpha in [0.5, 2]:
            renyi_Bl = (1 / (1 - alpha)) * torch.log(
                torch.sum(torch.pow(prob_save, alpha), dim=-1) + EPS
            )
            all_renyi.append(renyi_Bl)
            p_y = gt_probs_Bl
            modified_renyi_Bl = -(1 / abs(1 - alpha)) * (
                (1 - p_y) * p_y ** (abs(1 - alpha))
                + 2 * p_y
                - 1
                - p_y
                - (1 - p_y) ** (abs(1 - alpha))
                + (probs * (1 - prob_save) ** (abs(1 - alpha))).sum(dim=-1)
                - probs.sum(dim=-1)
            )
            assert modified_renyi_Bl.ndim == 2
            results[f'modified_renyi_{alpha}'] = modified_renyi_Bl.mean(dim=-1)

        if L >= 10:
            alphas = [1, 'inf', 0.5, 2]
            for i in range(len(alphas)):
                alpha = alphas[i]
                renyi_Bl = all_renyi[i]
                renyi_sort, _ = torch.sort(renyi_Bl, dim=-1)
                for ratio in [0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                    k_length = int(L * ratio)
                    if k_length == 0:
                        k_length = 1
                    results[f'mink_{ratio}_renyi_{alpha}'] = renyi_sort[
                        :, :k_length
                    ].mean(dim=-1)
                    results[f'maxk_{ratio}_renyi_{alpha}'] = renyi_sort[
                        :, -k_length:
                    ].mean(dim=-1)

    return {key: value.cpu() for key, value in results.items()}


def get_statistics_single(collect: VarPredCollection):
    results = {}

    cond_logits_BlV = collect.cond['logits']
    gt_Bl = collect.cond['gt']
    uncond_logits_BlV = collect.uncond['logits']

    B, L, V = cond_logits_BlV.shape

    for prefix, logits in zip(['cond', 'uncond'], [cond_logits_BlV, uncond_logits_BlV]):
        prefix_result = get_statistics_single_individual(logits, gt_Bl)
        for k, v in prefix_result.items():
            results[f'{prefix}_{k}'] = v

    return results


def get_statistics(collect: VarPredCollection):
    # loss
    results = {}

    for layer in ['all'] + list(range(10)):
        # results[layer] = get_statistics_single(collect[layer])
        layer_result = get_statistics_single(collect[layer])
        for key in layer_result.keys():
            results[f'{layer}_{key}'] = layer_result[key]

    return results


class DictAccumulation:
    def __init__(self):
        self.data = defaultdict(list)

    def update(self, data):
        for k, v in data.items():
            self.data[k].append(v)

    def gather(self):
        result = {}
        for k in list(self.data.keys()):
            result[k] = torch.cat(self.data[k], axis=0)
        return result


# overwrite the function of 'VAR::autoregressive_infer_cfg'
def autoregressive_infer_cfg(
    self,
    B: int,
    label_B: Optional[Union[int, torch.LongTensor]],
    g_seed: Optional[int] = None,
    input_img_tokens: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:  # returns reconstructed image (B, 3, H, W) in [0, 1]
    """
    only used for inference, on autoregressive mode
    :param B: batch size
    :param label_B: imagenet label; if None, randomly sampled
    :param g_seed: random seed
    :param cfg: classifier-free guidance ratio
    :param top_k: top-k sampling
    :param top_p: top-p sampling
    :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
    :param input_img_tokens: (optional, only for zero-shot edit tasks) tokens of the image to be edited
    :param edit_mask: (optional, only for zero-shot edit tasks) binary mask, 1 for keeping given tokens; 0 for generating new tokens
    :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
    """
    if g_seed is None:
        rng = None
    else:
        self.rng.manual_seed(g_seed)
        rng = self.rng

    if label_B is None:
        label_B = torch.multinomial(
            self.uniform_prob, num_samples=B, replacement=True, generator=rng
        ).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full(
            (B,),
            fill_value=self.num_classes if label_B < 0 else label_B,
            device=self.lvl_1L.device,
        )

    sos = cond_BD = self.class_emb(
        torch.cat(
            (label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0
        )
    )

    lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
    next_token_map = (
        sos.unsqueeze(1).expand(2 * B, self.first_l, -1)
        + self.pos_start.expand(2 * B, self.first_l, -1)
        + lvl_pos[:, : self.first_l]
    )
    sos = cond_BD = sos

    cur_L = 0
    f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

    for b in self.blocks:
        b.attn.kv_caching(True)

    var_pred_collection = VarPredCollection()

    for si, pn in enumerate(self.patch_nums):  # si: i-th segment
        ratio = si / self.num_stages_minus_1
        # last_L = cur_L
        cur_L += pn * pn
        # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        x = next_token_map
        AdaLNSelfAttn.forward
        for b in self.blocks:
            x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
        logits_BlV = self.get_logits(x, cond_BD)  # .numpy()

        gt_Bl = input_img_tokens[si]  # .numpy()

        var_pred_collection.cond['logits'].append(logits_BlV[:B])
        var_pred_collection.cond['gt'].append(gt_Bl[:B])

        var_pred_collection.uncond['logits'].append(logits_BlV[B:])
        var_pred_collection.uncond['gt'].append(gt_Bl[B:])


        # if edit_mask is not None:
        gt_BChw = (
            self.vae_quant_proxy[0]
            .embedding(input_img_tokens[si])
            .transpose_(1, 2)
            .reshape(B, self.Cvae, pn, pn)
        )
        h_BChw = gt_BChw

        f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(self.patch_nums), f_hat, h_BChw
        )
        if si != self.num_stages_minus_1:  # prepare for next stage
            next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
            next_token_map = (
                self.word_embed(next_token_map)
                + lvl_pos[:, cur_L : cur_L + self.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(
                2, 1, 1
            )  # double the batch sizes due to CFG

    for b in self.blocks:
        b.attn.kv_caching(False)
    # return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5), all_acc
    results = get_statistics(var_pred_collection)

    torch.cuda.empty_cache()
    return results



def test(MODEL_DEPTH, split, BS):
    # download checkpoint
    hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    
    vae_ckpt, var_ckpt = (
        f'{ckpt_dir}/vae_ch160v4096z32.pth',
        f'{ckpt_dir}/var_d{MODEL_DEPTH}.pth',
    )
    if not osp.exists(vae_ckpt):
        os.system(f'wget {hf_home}/{vae_ckpt}')
    if not osp.exists(var_ckpt):
        os.system(f'wget {hf_home}/{var_ckpt}')

    # build vae, var
    FOR_512_px = MODEL_DEPTH == 36
    if FOR_512_px:
        patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
        resolution = 512
    else:
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        resolution = 256
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,  # hard-coded VQVAE hyperparameters
        device=device,
        patch_nums=patch_nums,
        num_classes=1000,
        depth=MODEL_DEPTH,
        shared_aln=FOR_512_px,
    )

    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    print(f'preparation finished.')

    ################## 2. Define some helper functions for zero-shot editing

    ############################# 3. Sample with classifier-free guidance



    # class_label = 1  # @param {type:"raw"}
    seed = 0  # @param {type:"number"}

    # seed
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # run faster
    tf32 = True
    torch.backends.cudnn.allow_tf32 = bool(tf32)
    torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    torch.set_float32_matmul_precision('high' if tf32 else 'highest')

    # load the image to be edited
    from utils.data import pil_loader, normalize_01_into_pm1
    from torchvision.transforms import transforms
    from torchvision.transforms.functional import to_tensor


    from torchvision import transforms as T

    transform = T.Compose(
        [
            # T.Resize((256, 256), antialias=True),
            T.Resize(round(256 * 1.125), interpolation=T.InterpolationMode.LANCZOS),
            T.CenterCrop(256),
            T.ToTensor(),
            normalize_01_into_pm1,
        ]
    )

    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset

    from tqdm import tqdm

    def get_result(split):

        image_net_split = split  # @param ['train', 'val']


        image_net = ImageFolder(
            f'{image_net_path}/{image_net_split}', transform=transform
        )
        if image_net_split == 'train':
            subset_idx = torch.load('imagenet_idx_50000.pt')
            image_net = Subset(image_net, subset_idx)
        image_net.name = f'imagenet_{image_net_split}'

        print(len(image_net))

        dataloader = torch.utils.data.DataLoader(
            image_net, batch_size=BS, shuffle=False, num_workers=8
        )

        # results = {}
        accumulator = DictAccumulation()

        for i, (img, label) in enumerate(tqdm(dataloader)):

            # if i >= 5:
            #     break
            img = img.to(device=device)
            label_B = label.to(device=device)
            # label_B = torch.LongTensor([1000 for _ in range(len(label))]).to(
            #     device=device
            # )
            # print(label)

            input_img_tokens = vae.img_to_idxBl(img, var.patch_nums)

            B = len(label_B)
            with torch.inference_mode():
                with torch.autocast(
                    'cuda', enabled=True, dtype=torch.float16, cache_enabled=True
                ):  # using bfloat16 can be faster
                    result = autoregressive_infer_cfg(
                        var,
                        B=B,
                        label_B=label_B,
                        # cfg=3,
                        # top_k=900,
                        # top_p=0.95,
                        g_seed=0,
                        # more_smooth=True,
                        input_img_tokens=input_img_tokens,
                        # edit_mask=None,
                    )

                    accumulator.update(result)



        return accumulator.gather()

    return get_result(split)


for i, model_depth in enumerate([16, 30, 24, 20]):
    bs = [32, 8, 8, 16][i]
    for split in ['train', 'val']:
        save_dir = f'./results_baseline/{model_depth}'
        os.makedirs(save_dir, exist_ok=True)
        save_name = f'{save_dir}/imagenet_{split}.pt'
        if os.path.exists(save_name):
            continue
        result = test(model_depth, split=split, BS=bs)
        torch.save(result, save_name)
