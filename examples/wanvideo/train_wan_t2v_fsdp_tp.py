import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import random
import numpy as np
# from accelerate.utils import set_seed
import time

import torch.distributed as dist
import torch_npu
from torch_npu.contrib import transfer_to_npu
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False

def main_print(*args):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(*args)

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        main_print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        main_print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = 0
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        # print(f"ERRERR {data_id=}")
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch

from typing import List, Optional, Tuple, Union
import torch
import torch_npu
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.adamw import AdamW as TorchAdamW


def adamw(params: List[Tensor],
          grads: List[Tensor],
          exp_avgs: List[Tensor],
          exp_avg_sqs: List[Tensor],
          max_exp_avg_sqs: List[Tensor],
          step_tensor: Tensor,
          *,
          amsgrad: bool,
          beta1: float,
          beta2: float,
          lr: float,
          weight_decay: float,
          eps: float,
          maximize: bool):
    r"""Functional API that performs AdamW algorithm computation.
    See :class:`~torch.optim.AdamW` for details.
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        max_exp_avg_sq = max_exp_avg_sqs[i] if amsgrad else None

        torch._fused_adamw_(
            [param],
            [grad],
            [exp_avg],
            [exp_avg_sq],
            [max_exp_avg_sq] if amsgrad else [],
            [step_tensor],
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize
        )


class FusedTorchAdamW(TorchAdamW):
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        capturable: bool = False,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):
        super().__init__(params, 
                lr=lr,
                betas=betas,
                eps=eps,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
                foreach=False,
                maximize=maximize,
                capturable=False,
                differentiable=False,
                fused=True,)


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, *, maximize: bool = False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1, beta2 = group['betas']

            if 'step' in group:
                group['step'] += 1
                if group['step'].is_cpu:
                    group['step'] = group['step'].cuda()
            else:
                group['step'] = torch.tensor(1, dtype=torch.int64, device=torch.cuda.current_device())

            for p in group['params']:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])

                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

            adamw(params_with_grad,
                  grads,
                  exp_avgs,
                  exp_avg_sqs,
                  max_exp_avg_sqs,
                  group['step'],
                  amsgrad=amsgrad,
                  beta1=beta1,
                  beta2=beta2,
                  lr=group['lr'],
                  weight_decay=group['weight_decay'],
                  eps=group['eps'],
                  maximize=group['maximize'])

        return loss


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        lora_rank=4, lora_alpha=4, train_architecture="lora", lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming",
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,
        pretrained_lora_path=None
    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        self.freeze_parameters()
        if train_architecture == "lora":
            self.add_lora_to_model(
                self.pipe.denoising_model(),
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
                lora_target_modules=lora_target_modules,
                init_lora_weights=init_lora_weights,
                pretrained_lora_path=pretrained_lora_path,
            )
        else:
            self.pipe.denoising_model().requires_grad_(True)
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        
        
    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        
        
    def add_lora_to_model(self, model, lora_rank=4, lora_alpha=4, lora_target_modules="q,k,v,o,ffn.0,ffn.2", init_lora_weights="kaiming", pretrained_lora_path=None, state_dict_converter=None):
        # Add LoRA to UNet
        self.lora_alpha = lora_alpha
        if init_lora_weights == "kaiming":
            init_lora_weights = True
            
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights=init_lora_weights,
            target_modules=lora_target_modules.split(","),
        )
        model = inject_adapter_in_model(lora_config, model)
        for param in model.parameters():
            # Upcast LoRA parameters into fp32
            if param.requires_grad:
                param.data = param.to(torch.float32)
                
        # Lora pretrained lora weights
        if pretrained_lora_path is not None:
            state_dict = load_state_dict(pretrained_lora_path)
            if state_dict_converter is not None:
                state_dict = state_dict_converter(state_dict)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            all_keys = [i for i, _ in model.named_parameters()]
            num_updated_keys = len(all_keys) - len(missing_keys)
            num_unexpected_keys = len(unexpected_keys)
            main_print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")
    

    # def training_step(self, batch, batch_idx):
    #     # Data
    #     latents = batch["latents"].to(self.device)
    #     prompt_emb = batch["prompt_emb"]
    #     prompt_emb["context"] = prompt_emb["context"][0].to(self.device)
        
    #     # Loss
    #     self.pipe.device = self.device
    #     noise = torch.randn_like(latents)
    #     timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
    #     timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
    #     extra_input = self.pipe.prepare_extra_input(latents)
    #     noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
    #     training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

    #     # Compute loss
    #     noise_pred = self.pipe.denoising_model()(
    #         noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
    #         use_gradient_checkpointing=self.use_gradient_checkpointing,
    #         use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
    #     )
    #     loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    #     loss = loss * self.pipe.scheduler.training_weight(timestep)

    #     # Record log
    #     # self.log("train_loss", loss, prog_bar=True)
    #     # self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
    #     print(f"train log ------------------, setp: {self.global_step}, loss: {loss:.4f}")
    #     return loss


    def configure_optimizers(self):
        trainable_modules = filter(lambda p: p.requires_grad, self.pipe.denoising_model().parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate, foreach=False)
        # optimizer = torch_npu.optim.NpuFusedAdamW(trainable_modules, lr=self.learning_rate)
        # optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        # optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        state_dict = self.pipe.denoising_model().state_dict()
        lora_state_dict = {}
        for name, param in state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default=None,
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default=None,
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="train_batch_size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="train seed.",
    )

    parser.add_argument(
        "--fsdp2_size",
        type=int,
        default=0,
        help="fsdp2_size.",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=0,
        help="tp_size.",
    )
    args = parser.parse_args()
    return args

def use_deterministic():
    # return
    torch.use_deterministic_algorithms(True, warn_only=True)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(42)
    # os.environ['NCCL_DEBUG'] = 'INFO'
    # os.environ['NVIDIA_TF32_OVERRIDE'] = '0'  # 禁用TF32

# use_deterministic()

def set_seed(seed, deterministic=False):
    print(f"ERRERR {seed=}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    if deterministic==True:
        # 设置 cudnn 为确定性模式
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # 可选：强制使用确定性算法（PyTorch >= 1.8）
        torch.use_deterministic_algorithms(True)


def compute_grad_l2_norm(model):
    total_norm = 0.0
    has_grad = False  # 标记是否存在有梯度的参数
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # print(f"Param: {name}, Shape: {param.shape}, Grad Norm: {param.grad.norm().item():.10f} Forwa Norm: {param.norm().item():.10f}")
            has_grad = True
            # 计算当前参数梯度的L2范数并累加平方
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() **2
    
    if not has_grad:
        return 0.0
    
    # 计算总L2范数
    return total_norm** 0.5

def compute_tensor_grad_l2_norm(tensor):
    if tensor.grad is None:
        return 0.0
    return tensor.grad.data.norm(2).item()



from torch.distributed.tensor.parallel import parallelize_module, ParallelStyle
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torch.distributed.tensor import Replicate, Shard, Partial


def init_model_mesh(fsdp2_size, tp_size):
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_mesh_name = "fsdp_mesh"
    tp_mesh_name = "tp_mesh"

    if fsdp2_size == 0:
        tp_mesh = init_device_mesh(
            "cuda",
            (world_size,)
        )
        return None, tp_mesh

    if tp_size == 0:
        fsdp_mesh = init_device_mesh(
            "cuda",
            (world_size,)
        )
        return fsdp_mesh, None


    model_mesh = init_device_mesh(
            "cuda",
            (world_size // tp_size, tp_size),
            mesh_dim_names=[fsdp_mesh_name, tp_mesh_name]
        )
    
    fsdp_mesh = model_mesh[fsdp_mesh_name]
    tp_mesh = model_mesh[tp_mesh_name]
    main_print(f"ERRERR {model_mesh=} {fsdp_mesh=} {tp_mesh=}")
    return fsdp_mesh, tp_mesh


def apply_tp(model, tp_mesh):
    # ColwiseParallel shard(0)
    # RowwiseParallel shard(1)

    blocks_tp_plan = {
        # "self_attn": PrepareModuleInput(
        #         input_layouts=(Replicate(), None),
        #         desired_input_layouts=(Replicate(), None),
        #     ),
        # 这个replicate会影响qx=self.q(x)的shape
        # 默认use_local_output: bool = True，所以qx的结果是tensor，会有to_local
        # output_layouts=Replicate 等于把dtensor状态下的qx的shard(0)转换成replicate再to_local
        # 因为这个linear后面紧接了一个norm，norm的weight是1536的，
        # "self_attn.q": ColwiseParallel(), 
        "self_attn.q": ColwiseParallel(output_layouts=Replicate()),
        # "self_attn.norm_q": SequenceParallel(),
        "self_attn.k": ColwiseParallel(output_layouts=Replicate()), 
        # "self_attn.norm_k": SequenceParallel(),
        "self_attn.v": ColwiseParallel(output_layouts=Replicate()), 
        "self_attn.o": RowwiseParallel(input_layouts=Replicate()), 
        "cross_attn.q": ColwiseParallel(output_layouts=Replicate()), 
        "cross_attn.k": ColwiseParallel(output_layouts=Replicate()), 
        "cross_attn.v": ColwiseParallel(output_layouts=Replicate()), 
        "cross_attn.o": RowwiseParallel(input_layouts=Replicate()), 
        # "ffn.0": RowwiseParallel(input_layouts=Replicate()), # weight: (8960, 1536) shard(1)
        # "ffn.2": ColwiseParallel(output_layouts=Replicate()), # weight: (1536, 8960) shard(0)
    }

    for block in model.blocks:
        parallelize_module(
            block,
            device_mesh=tp_mesh,
            parallelize_plan=blocks_tp_plan,
        )

    model_tp_plan = {
        # "text_embedding.0": ColwiseParallel(output_layouts=Replicate()), # weight: (1536, 4096) shard(0)
        "text_embedding.2": ColwiseParallel(output_layouts=Replicate()), # weight: (1536, 1536) shard(0)? 第二个step的loss对不上
        # "text_embedding.2": RowwiseParallel(input_layouts=Replicate()), # weight: (1536, 1536) shard(1)? 第一个step的loss对不上
        # "time_embedding.0": ColwiseParallel(output_layouts=Replicate()), # weight: (1536, 256) shard(0)
        # "time_embedding.2": ColwiseParallel(output_layouts=Replicate()), # weight: (1536, 256) shard(0)
        # "time_projection.1": RowwiseParallel(input_layouts=Replicate()), # weight: (8960, 1536) shard(1)
        # "time_projection.1": ColwiseParallel(output_layouts=Replicate()), # weight: (8960, 1536) shard(1)

        # 有点奇怪
        # "head.head": RowwiseParallel(input_layouts=Replicate()), # weight: (64, 1536) shard(1) # NotImplementedError: RowwiseParallel currently only support nn.Linear and nn.Embedding! Now is Head
    }

    # parallelize_module(
    #     model,
    #     device_mesh=tp_mesh,
    #     parallelize_plan=model_tp_plan,
    # )
    enable_async_tp = False
    if enable_async_tp:
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        # torch._inductor.config._micro_pipeline_tp = True
        enable_symm_mem_for_group(tp_mesh.get_group().group_name)
        
    main_print(f"ERRERRERR apply tp {tp_mesh=} {blocks_tp_plan=}")

def apply_fsdp2(model, fsdp_mesh):
    fsdp_kwargs = {
        "mesh": fsdp_mesh,
        # "reshard_after_forward": False, # 精度无区别
        "reshard_after_forward": True,
        # cast_forward_inputs: 会在FSDPState._pre_forward()函数中提前将所有的入参dtype转换为param_dtype
        # 这里全是默认参数，没有开混精
        "mp_policy": MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=False),
        # "mp_policy": MixedPrecisionPolicy(
        #     param_dtype=torch.float32,  # 参数精度
        #     reduce_dtype=torch.float32,  # 梯度归约精度
        #     output_dtype =torch.float32,
        #     cast_forward_inputs=False,  # 是否转换前向输入
        # ),
        # "offload_policy": CPUOffloadPolicy(),
    }
    main_print(f"ERRERRERR apply fsdp2 {fsdp_mesh=} {fsdp_kwargs=}")


    # embedding只是实验，不用包裹
    # fully_shard(
    #     model.patch_embedding,
    #     **fsdp_kwargs,
    # )
    # fully_shard(
    #     model.text_embedding,
    #     **fsdp_kwargs,
    # )
    # fully_shard(
    #     model.time_embedding,
    #     **fsdp_kwargs,
    # )
    
    # 检查是否是同一个 tensor 引用
    # print(f"ERRERR {model.blocks[0].modulation.data_ptr() == model.blocks[1].modulation.data_ptr()=}")

    for block in model.blocks:
        # fully_shard(
        #     block.cross_attn,
        #     **fsdp_kwargs,
        # )
        fully_shard(
            block,
            **fsdp_kwargs,
        )

    # 只有单独包裹整个模型的时候精度能对齐
    fully_shard(
        model,
        **fsdp_kwargs,
    )

    def set_prefetch(blocks: torch.nn.ModuleList, num_to_prefetch: int) -> None:
        for i in range(len(blocks)):
            if i >= len(blocks) - num_to_prefetch:
                break
            layers_to_prefetch = [
                blocks[i + j] for j in range(1, num_to_prefetch + 1)
            ]
            if hasattr(blocks[i], "set_modules_to_forward_prefetch"):
                blocks[i].set_modules_to_forward_prefetch(layers_to_prefetch)
                blocks[i].set_modules_to_backward_prefetch(layers_to_prefetch)
                

    set_prefetch(model.blocks, num_to_prefetch=2)

    # fully_shard([model.patch_embedding, model.text_embedding, model.time_embedding], **fsdp_kwargs)
    # fully_shard([model.patch_embedding, model.text_embedding, model.time_embedding], **fsdp_kwargs)


    # model.patch_embedding.set_unshard_in_backward(False) # 默认是True，RuntimeError: setStorage: sizes [1536, 1536], strides [1536, 1], storage offset 0, and itemsize 2 requiring a storage size of 4718592 are out of bounds for storage of size 0
    # model.text_embedding.set_unshard_in_backward(False)
    # model.time_embedding.set_unshard_in_backward(False)
    # model.reshard_after_forward(False) # 精度没有差异
    # model.set_requires_gradient_sync(True) # FSDPSequential变成了Sequential # 精度没有差异
    # model.set_reduce_scatter_divide_factor(4) # 后面通信的时候报错，可能是传入了bf16，TypeError: PreMulSum Data type must be half, float, or double


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    ShardingStrategy,
    StateDictType,
)
def apply_fsdp1(model):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    fsdp_mesh = init_device_mesh("cuda", (world_size,))
    fsdp_kwargs = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD
    }

    for i in range(len(model.blocks)):
        model.blocks[i] = FSDP(model.blocks[i], **fsdp_kwargs)

    model = FSDP(model, **fsdp_kwargs)
    

def print_tuple(tensor_tuple):
    for i, item in enumerate(tensor_tuple):
        if hasattr(item, "shape"):
            try:
                mean = item.mean().item()
                max_val = item.max().item()
                min_val = item.min().item()
                grad_l2_norm = item.norm(2).item()
                main_print(f"Element {i} shape: {item.shape}, dtype: {item.dtype}, grad_l2_norm: {grad_l2_norm:.10f}, mean: {mean:.10f}, max: {max_val:.10f}, min: {min_val:.10f}")
            except Exception as e:
                main_print(f"Element {i} shape: {item.shape}, but failed to compute stats: {e}")
        else:
            main_print(f"Element {i} does not have a shape attribute.")


def full_backward_hook(module, grad_input, grad_output):
    module_name = module.__class__.__name__
    main_print(f"---------------------Backward hook triggered for module: {module_name}")
    print_tuple(grad_input)
    print_tuple(grad_output)

def full_forward_hook(module, input, output):
    module_name = module.__class__.__name__
    main_print(f"---------------------Forward hook triggered for module: {module_name}")
    print_tuple(input)
    print_tuple(output)


def full_backward_hook_with_name(module, grad_input, grad_output, name):
    main_print(f"--------------------- Backward hook for module: {name} ({module.__class__.__name__})")
    main_print("** Gradient of input:")
    print_tuple(grad_input)
    main_print("** Gradient of output:")
    print_tuple(grad_output)


def register_backward_hooks(model):
    for name, module in model.named_modules():
        # 跳过顶层模块本身
        if module == model:
            continue
        # 只对叶子模块注册 hook（避免重复）
        if len(list(module.children())) == 0:
            module.register_full_backward_hook(
                lambda mod, grad_in, grad_out, name=name: full_backward_hook_with_name(mod, grad_in, grad_out, name)
            )


def forward_hook_with_name(module, input, output, name):
    main_print(f"--------------------- Forward hook for module: {name} ({module.__class__.__name__})")
    main_print("** Input:")
    print_tuple(input)
    main_print("** Output:")
    print_tuple(output)


def register_forward_hooks(model):
    for name, module in model.named_modules():
        # 跳过顶层模块本身
        if module == model:
            continue
        # 只对叶子模块注册 hook（避免重复）
        if len(list(module.children())) == 0:
            module.register_forward_hook(
                lambda mod, inp, out, name=name: forward_hook_with_name(mod, inp, out, name)
            )

def fp32_allreduce_hook(param):
    def hook(grad):
        # 转为 FP32
        grad_fp32 = grad.float()
        # 通信（all_reduce）
        torch.distributed.all_reduce(grad_fp32, op=torch.distributed.ReduceOp.SUM)
        # 再转回 FP16
        return grad_fp32.to(grad.dtype)
    return hook



def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    # set_seed(args.seed + rank)
    set_seed(2025)
    # initialize_sequence_parallel_state(args.sp_size)

    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers
    )

    lightning_model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        train_architecture=args.train_architecture,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_modules=args.lora_target_modules,
        init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        pretrained_lora_path=args.pretrained_lora_path,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    # trainer = pl.Trainer(
    #     max_epochs=args.max_epochs,
    #     accelerator="gpu",
    #     devices="auto",
    #     precision="bf16",
    #     strategy=args.training_strategy,
    #     default_root_dir=args.output_path,
    #     accumulate_grad_batches=args.accumulate_grad_batches,
    #     callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
    #     logger=logger,
    # )
    # trainer.fit(model, dataloader)
    global_step = 0
    model = lightning_model.pipe.denoising_model().to(device)
    # model = lightning_model.pipe.denoising_model().to(device).to(torch.flo)

    main_print(f"ERRERR {args.fsdp2_size=} {args.tp_size=} {world_size=}")

    if args.fsdp2_size>0 or args.tp_size>0:
        if (args.fsdp2_size==0 and args.tp_size != world_size) or (args.tp_size==0 and args.fsdp2_size != world_size) or (args.fsdp2_size!=0 and args.tp_size!=0 and args.fsdp2_size*args.tp_size != world_size):
            raise RuntimeError(f"args.fsdp2_size and args.tp_size wrong!!!")
        fsdp_mesh, tp_mesh = init_model_mesh(fsdp2_size=args.fsdp2_size, tp_size=args.tp_size)

        if tp_mesh is not None:
            apply_tp(model, tp_mesh)
        if fsdp_mesh is not None:
            apply_fsdp2(model, fsdp_mesh)

    # apply_fsdp1(model)

    # main_print(model)
    optimizer = lightning_model.configure_optimizers()
    for epoch in range(args.max_epochs):
        experimental_config = torch_npu.profiler._ExperimentalConfig( 
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
            l2_cache=False
        )

        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU
                ],
            schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1, skip_first=1000),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./profile/npu_profiling_fsdp4_tp2_no_ffn"),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
            experimental_config=experimental_config) as prof:

            start_time = time.time()
            for step, batch in enumerate(dataloader):
                optimizer.zero_grad()  
                global_step += 1
                # latents = batch["latents"].to(device)
                latents_cpu = batch["latents"]
                latents = latents_cpu.to(device)

                prompt_emb = batch["prompt_emb"]
                prompt_emb["context"] = prompt_emb["context"][0].to(device)
                
                # Loss
                lightning_model.pipe.device = device
                noise = torch.randn_like(latents_cpu).npu()
                # main_print(f"ERRERR step: {global_step} {noise.flatten()[0]=}") # npu和gpu已对齐
                timestep_id = torch.randint(0, lightning_model.pipe.scheduler.num_train_timesteps, (1,))
                # noise = torch.ones_like(latents) * 2
                # timestep_id = torch.full((1,), 1, dtype=torch.int32)
                timestep = lightning_model.pipe.scheduler.timesteps[timestep_id].to(dtype=lightning_model.pipe.torch_dtype, device=lightning_model.pipe.device)
                extra_input = lightning_model.pipe.prepare_extra_input(latents)
                noisy_latents = lightning_model.pipe.scheduler.add_noise(latents, noise, timestep)
                training_target = lightning_model.pipe.scheduler.training_target(latents, noise, timestep)


                # 整个注册 full_backward_hook，仅叶子节点
                # register_forward_hooks(model)
                # register_backward_hooks(model)

                # 单个模块注册 反向hook
                # handle = model.patch_embedding.register_full_backward_hook(full_backward_hook)
                # handle = model.text_embedding.register_full_backward_hook(full_backward_hook)
                # handle = model.time_embedding.register_full_backward_hook(full_backward_hook)
                # handle = model.time_projection.register_full_backward_hook(full_backward_hook)
                # # blocks未注册
                # handle = model.head.register_full_backward_hook(full_backward_hook)


                # # 单个模块注册前向hook
                # handle = model.patch_embedding.register_forward_hook(full_forward_hook) #Conv3d
                # handle = model.text_embedding.register_forward_hook(full_forward_hook) #Sequential
                # handle = model.time_embedding.register_forward_hook(full_forward_hook) #Sequential
                # handle = model.time_projection.register_forward_hook(full_forward_hook) #Sequential
                # # blocks未注册
                # handle = model.head.register_forward_hook(full_forward_hook) #Head


                # 注册 hook 到需要通信的参数
                # for name, param in model.named_parameters():
                #     if param.requires_grad:
                #         param.register_hook(fp32_allreduce_hook(param))

                # Compute loss
                # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):

                # 转换张量类型为 float32
                def to_float32(x):
                    if isinstance(x, torch.Tensor):
                        return x.to(torch.float32)
                    return x  # 非张量保持原样

                # 转换主输入
                # noisy_latents = to_float32(noisy_latents)
                # timestep = to_float32(timestep)

                # # 转换 prompt_emb 字典中的值
                # prompt_emb = {k: to_float32(v) for k, v in prompt_emb.items()}

                # # 转换 extra_input 字典中的值
                # extra_input = {k: to_float32(v) for k, v in extra_input.items()}

                # print(f"ERRERR brfore {noisy_latents.dtype=} {timestep.dtype=}")

                noise_pred = model(
                    noisy_latents, timestep=timestep, **prompt_emb, **extra_input,
                    use_gradient_checkpointing=lightning_model.use_gradient_checkpointing,
                    use_gradient_checkpointing_offload=lightning_model.use_gradient_checkpointing_offload
                )

                
                rank = int(os.environ["LOCAL_RANK"])
                # print(f"ERRERR [RANK{rank}] {noise_pred=}")

                # print(f"ERRERR {noise_pred.shape=} {training_target.shape=}")
                loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
                # loss = torch.nn.functional.mse_loss(noise_pred, training_target)
                loss = loss * lightning_model.pipe.scheduler.training_weight(timestep)
                loss.backward()
                # torch.cuda.synchronize()
                # torch.distributed.barrier()
                # main_print(f" =========================================================================================================== ")
                # main_print(f" =========================================================================================================== ")
                # main_print(f" ========================== train_log, setp: {global_step}, loss: {loss:.10f}, grad: {compute_grad_l2_norm(model):.10f} training_weight:{lightning_model.pipe.scheduler.training_weight(timestep):.10f}========================== ")
                # main_print(f" =========================================================================================================== ")
                # main_print(f" =========================================================================================================== ")
                
                optimizer.step()
                # torch_npu.npu.synchronize()
                main_print(f"[RANK{rank}]train_log ------------------, step: {global_step}, step_time:{time.time() - start_time:.4f} loss: {loss:.10f}, grad: {compute_grad_l2_norm(model):.10f} training_weight:{lightning_model.pipe.scheduler.training_weight(timestep):.10f}")
                start_time = time.time()
                # prof.step()


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
