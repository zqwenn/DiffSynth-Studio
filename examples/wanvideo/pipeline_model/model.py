from typing import Optional
from diffsynth.models.wan_video_dit import WanModel, sinusoidal_embedding_1d
import torch
import torch.nn as nn
from torch.distributed.pipelining import ScheduleGPipe, Schedule1F1B, PipelineStage
import copy


class PPModel(torch.nn.Module):
    def __init__(
        self,
        model: WanModel,
        pre_process: bool = True,
        post_process: bool = True,
        use_block: bool = True,
        use_gradient_checkpointing: bool = True,
        use_gradient_checkpointing_offload: bool = False,
    ):
        super().__init__()
        self.model = model
        self.pre_process = pre_process
        self.post_process = post_process
        self.use_block = use_block
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.f = 21
        self.h = 30
        self.w = 52
    
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        timestep: torch.Tensor,
        t_mod: Optional[torch.Tensor] = None,
        # freqs: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,
    ):
        if self.pre_process:
            x = x.to(torch.bfloat16)
            t = self.model.time_embedding(
                sinusoidal_embedding_1d(self.model.freq_dim, timestep))
            t_mod = self.model.time_projection(t).unflatten(1, (6, self.model.dim))
            if context.ndim == 4:
                context = context.squeeze(0)
            context = self.model.text_embedding(context)
            # print("self.model.has_image_input ======================= ", self.model.has_image_input)
            if self.model.has_image_input:
                x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
                clip_embdding = self.model.img_emb(clip_feature)
                context = torch.cat([clip_embdding, context], dim=1)
            
            x, (self.f, self.h, self.w) = self.model.patchify(x)
            # print(self.f, self.h, self.w, "========================")
            
            
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        if self.use_block:
            freqs = torch.cat([
                self.model.freqs[0][:self.f].view(self.f, 1, 1, -1).expand(self.f, self.h, self.w, -1),
                self.model.freqs[1][:self.h].view(1, self.h, 1, -1).expand(self.f, self.h, self.w, -1),
                self.model.freqs[2][:self.w].view(1, 1, self.w, -1).expand(self.f, self.h, self.w, -1)
            ], dim=-1).reshape(self.f * self.h * self.w, 1, -1).to(x.device)
            # freqs = torch.view_as_complex(freqs) if not self.pre_process else freqs
            for block in self.model.blocks:
                
                if self.model.training and self.use_gradient_checkpointing:
                    if self.use_gradient_checkpointing_offload:
                        with torch.autograd.graph.save_on_cpu():
                            x = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(block),
                                x, context, t_mod, freqs,
                                use_reentrant=False,
                            )
                    else:
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = block(x, context, t_mod, freqs)
        
        if self.post_process:
            # print("t = ================", t)
            x = self.model.head(x, t)
            x = self.model.unpatchify(x, (self.f, self.h, self.w))
        # return x, context, timestep, t_mod, torch.view_as_real(freqs), t
        return x, context, timestep, t_mod, t

model_stage1 = {'blocks', 'head'}
pre_process_layer = {'patch_embedding', 'text_embedding', 'time_embedding', 'time_projection'}
post_process_layer = {'head'}


def get_model_stage(whole_model, stage_idx, device, pp_mesh, num_layers):
    num_stages = pp_mesh.size()
    print("pipeline size = ", num_stages, "num_layers = ", num_layers)
    assert num_stages == len(num_layers) and sum(num_layers) == 30
    sum_layers = [sum(num_layers[:i+1]) for i, _ in enumerate(num_layers)]
    sum_layers.insert(0, 0)
    print("sum_layers = ", sum_layers)
    wrap_model = copy.deepcopy(whole_model)
    model = wrap_model.model

    for module_name, module_value in model.named_children():
        if module_name == 'blocks':
            new_layers = nn.ModuleList(
                [
                    layer
                    for i, layer in enumerate(module_value)
                    if sum_layers[stage_idx] <= i < sum_layers[stage_idx+1]
                ]
            )
            setattr(model, module_name, new_layers)

        if module_name in pre_process_layer and stage_idx > 0:
            wrap_model.pre_process = False
            print(f"to_abandon stage_idx: {stage_idx}, module_name: {module_name}")
            setattr(model, module_name, None)

        if module_name in post_process_layer and stage_idx < num_stages - 1:
            wrap_model.post_process = False
            print(f"to_abandon stage_idx: {stage_idx}, module_name: {module_name}")
            setattr(model, module_name, None)

    print("stage_idx = ", stage_idx, wrap_model)

    stage = PipelineStage(
            wrap_model,
            stage_idx,
            num_stages,
            device,
            group=pp_mesh.get_group(),
        )
    return stage, wrap_model
