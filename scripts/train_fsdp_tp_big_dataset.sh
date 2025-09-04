# export ASCEND_RT_VISIBLE_DEVICES="4,5,6,7"
# pkill python
# sleep 5
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
# export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
# export MULTI_STREAM_MEMORY_REUSE=2
# export HCCL_OP_EXPANSION_MODE="AIV"
# export HCCL_BUFFSIZE=500
# export TORCH_HCCL_ZERO_COPY=1
# export HCCL_RDMA_SL=3


torchrun --nproc_per_node=8 --rdzv_endpoint "localhost:29501" examples/wanvideo/train_wan_t2v.py \
  --task train \
  --train_architecture full \
  --dataset_path /home/wgb/model/wandataset \
  --output_path ./log \
  --dit_path "/home/wgb/model/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch 10 \
  --max_epochs 1 \
  --learning_rate 1e-4 \
  --accumulate_grad_batches 1 \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 \
  --train_batch_size 1 \
  --fsdp2_size 4 \
  --tp_size 2
