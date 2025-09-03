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
from torch.distributed.pipelining import ScheduleGPipe, Schedule1F1B, PipelineStage, pipe_split, pipeline
import torch.distributed as dist
from torch.distributed.tensor import Replicate, Shard, Partial
import time
from pipeline_model.model import PPModel, get_model_stage
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import (
    CPUOffloadPolicy,
    fully_shard,
    MixedPrecisionPolicy,
)
from typing import Optional
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
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = 0
        # print("data_id, index = ", data_id, index)
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch


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
            print(f"{num_updated_keys} parameters are loaded from {pretrained_lora_path}. {num_unexpected_keys} parameters are unexpected.")

    def configure_optimizers(self, model):
        trainable_modules = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.AdamW(trainable_modules, lr=self.learning_rate)
        return optimizer

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
        '--nums_layers',
        nargs='+',
        type=int,
        help='block layers for pipeline'
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=0,
        help="pipeline size",
    )
    parser.add_argument(
        "--fsdp_size",
        type=int,
        default=0,
        help="fsdp size",
    )
    args = parser.parse_args()
    return args


def use_deterministic():
    # return
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# use_deterministic()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def init_model_mesh(fsdp2_size, pp_size):
    world_size = int(os.environ["WORLD_SIZE"])
    fsdp_mesh_name = "fsdp_mesh"
    pp_mesh_name = "pp_mesh"
    if fsdp2_size == 0:
        pp_mesh = init_device_mesh(
            "npu",
            (world_size,)
        )
        return None, pp_mesh

    if pp_size == 0:
        fsdp_mesh = init_device_mesh(
            "npu",
            (world_size,)
        )
        return fsdp_mesh, None


    model_mesh = init_device_mesh(
            "npu",
            (world_size // pp_size, pp_size),
            mesh_dim_names=[fsdp_mesh_name, pp_mesh_name]
        )
    
    fsdp_mesh = model_mesh[fsdp_mesh_name]
    pp_mesh = model_mesh[pp_mesh_name]
    main_print(f"ERRERR {model_mesh=} {fsdp_mesh=} {pp_mesh=}")
    return fsdp_mesh, pp_mesh


def _shard_placement_fn(param: torch.nn.Parameter) -> Optional[Shard]:
    largest_dim = -1
    largest_dim_size = -1
    for dim, dim_size in enumerate(param.shape):
        if dim_size > largest_dim_size:
            largest_dim = dim
            largest_dim_size = dim_size
    assert largest_dim >= 0, f"{param.shape}"
    return Shard(largest_dim)


def apply_fsdp2(model, fsdp_mesh):
    fsdp_kwargs = {
        "mesh": fsdp_mesh,
        "reshard_after_forward": True,
        "mp_policy": MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=False),
        "shard_placement_fn": _shard_placement_fn,
    }

    for block in model.model.blocks:
        fully_shard(
            block,
            **fsdp_kwargs,
        )
    # fully_shard(
    #     model.model,
    #     **fsdp_kwargs,
    # )
    fully_shard(
        model,
        **fsdp_kwargs,
    )

def compute_grad_l2_norm(model):
    total_norm = 0.0
    has_grad = False  # 标记是否存在有梯度的参数
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            # 计算当前参数梯度的L2范数并累加平方
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() **2
            # print(f"name = {name}", param_norm)
    
    if not has_grad:
        return 0.0
    
    # 计算总L2范数
    return total_norm** 0.5


def fix_data(batch, device, pipe):
    latents = batch["latents"].to(device, non_blocking=True)
    prompt_emb = batch["prompt_emb"]
    prompt_emb["context"] = prompt_emb["context"].to(device, non_blocking=True)
    # Loss
    pipe.device = device
    noisy_latents = []
    training_target = []
    timestep = []
    for i in range(latents.shape[0]):
        # noise = torch.randn_like(latents[i])
        noise = torch.randn_like(latents[i].cpu()).cuda()
        timestep_id = torch.randint(0, pipe.scheduler.num_train_timesteps, (1,))
        # main_print("titimestep_id = ", timestep_id, "noise = ", noise.mean())
        # noise = torch.ones_like(latents[i]) * 2
        # timestep_id = torch.full((1,), 1, dtype=torch.int32)
        _timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
        timestep.append(_timestep)
        extra_input = pipe.prepare_extra_input(latents)
        noisy_latents.append(pipe.scheduler.add_noise(latents[i], noise, _timestep))
        training_target.append(pipe.scheduler.training_target(latents[i], noise, _timestep))
    noisy_latents = torch.stack(noisy_latents, dim=0)
    training_target = torch.stack(training_target, dim=0)
    timestep = torch.cat(timestep, dim=0)
    # print("Compute loss =================================")
    # # Compute loss
    # print(f"noisy_latents={noisy_latents.shape}, timestep={timestep.shape}, prompt_emb['context']={prompt_emb['context'].shape}, training_target = {training_target.shape}")
    # print("extra_input = ", extra_input)
    # print("======================")
    return noisy_latents, timestep, prompt_emb["context"], training_target


experimental_config = torch_npu.profiler._ExperimentalConfig( 
    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
    l2_cache=False
    )


def train(args):
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("hccl")
    torch.cuda.set_device(local_rank)
    device = torch.cuda.current_device()
    set_seed(args.seed)
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
        num_workers=args.dataloader_num_workers,
        drop_last=True,
        pin_memory=True,
        pin_memory_device='npu',
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

    global_step = 0
    model = lightning_model.pipe.denoising_model().to(device)
    # print(model)
    pp_model = PPModel(model)

    def loss_fn(predict, target):
        if not isinstance(predict, torch.Tensor):
            true_predict = predict[0]
            true_predict = true_predict + 0 * torch.sum(predict[2])
        loss = torch.nn.functional.mse_loss(true_predict.float(), target.float())
        # print("lightning_model.pipe.scheduler.training_weight(predict[2]) = ", lightning_model.pipe.scheduler.training_weight(predict[2]))
        loss = loss * lightning_model.pipe.scheduler.training_weight(predict[2])
        return loss

    if args.train_batch_size > 1:
        fsdp_mesh, pp_mesh = init_model_mesh(args.fsdp_size, args.train_batch_size)
        pp_rank = pp_mesh.get_local_rank()
        pp_size = pp_mesh.size()
        stage, model_split = get_model_stage(pp_model, pp_rank, device, pp_mesh, args.nums_layers)
        del pp_model, model
        if args.fsdp_size > 0:
            apply_fsdp2(model_split, fsdp_mesh)
            stage.submod = model_split
        # print(model_split)
        optimizer = lightning_model.configure_optimizers(model_split)
    else:
        optimizer = lightning_model.configure_optimizers(pp_model)

    for epoch in range(args.max_epochs):
        epoch_time = time.time()
        # with torch_npu.profiler.profile(
        #     activities=[
        #         torch_npu.profiler.ProfilerActivity.CPU,
        #         torch_npu.profiler.ProfilerActivity.NPU
        #         ],
        #     schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1, skip_first=1),
        #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./result_pp2_with_stack_new1F1B"),
        #     record_shapes=False,
        #     profile_memory=False,
        #     with_stack=True,
        #     experimental_config=experimental_config) as prof:
        # with torch_npu.profiler.profile(
        #     activities=[ torch_npu.profiler.ProfilerActivity.NPU],
        #     schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1, skip_first=1),
        #     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./npu-profling-nobig")
        # ) as prof:
        start_time = time.time()
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            global_step += 1
            if pp_rank == 0 or pp_size - 1 == pp_rank:
                noisy_latents, timestep, prompt_emb, training_target = fix_data(batch, device, lightning_model.pipe)
            if args.train_batch_size == 1:
                noise_pred = pp_model(noisy_latents, prompt_emb, timestep)[0]
                loss = loss_fn(noise_pred, training_target)
                loss.backward()
                # step_time = time.time() - start_time
                # main_print(f"train log ------------------, epoch: {epoch}, rank: {rank}, setp: {global_step}, loss: {loss:.10f}, grad: {compute_grad_l2_norm(pp_model):.10f}, time: {step_time:.3f}s")
            else:
                schedule = Schedule1F1B(stage, pp_size, loss_fn=loss_fn, scale_grads=False)
                # schedule = ScheduleGPipe(stage, pp_size, loss_fn=loss_fn, scale_grads=False)
                # Compute loss
                losses = []
                if pp_rank == 0:
                    schedule.step(noisy_latents, prompt_emb, timestep)
                elif pp_size - 1 == pp_rank:
                    output = schedule.step(target=training_target.float(), losses=losses)
                else:
                    schedule.step()
            optimizer.step()
            if pp_size - 1 == pp_rank:
                if (fsdp_mesh and fsdp_mesh.get_local_rank() == 0) or fsdp_mesh is None:
                    print(f"train log ------------------, epoch: {epoch}, pp_rank: {pp_rank}, setp: {global_step}, loss: {torch.mean(torch.stack(losses)):.10f}, grad: {compute_grad_l2_norm(model_split):.10f}, time: {time.time() - start_time:.3f}s")
                    start_time = time.time()
                # prof.step()
        epoch_end = time.time() - epoch_time
        main_print(f"epoch_log ------------------,  epoch: {epoch}, time: {epoch_end / 60:.3f}min")
    torch.npu.synchronize()
    dist.destroy_process_group()
            


if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)
