STEPS=1233
EPOCHS=10
ACCUMULATE=1
FSDP_SIZE=2

LOG_NAME="max_epochs${EPOCHS}_steps${STEPS}_accumulate_grad_batches${ACCUMULATE}_fsdp${FSDP_SIZE}pp_npu_1F1B_nodeter.log"

torchrun --nproc_per_node=8 examples/wanvideo/train_wan_t2v_pp_manual.py \
  --task train \
  --train_architecture full \
  --dataset_path ../wandataset \
  --output_path ./log \
  --dit_path "../Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors" \
  --steps_per_epoch $STEPS \
  --max_epochs $EPOCHS \
  --learning_rate 1e-4 \
  --accumulate_grad_batches $ACCUMULATE \
  --use_gradient_checkpointing \
  --dataloader_num_workers 4 \
  --train_batch_size 4 \
  --nums_layers 7 8 8 7\
  --fsdp_size $FSDP_SIZE\
  > "./scripts/log/${LOG_NAME}" 2>&1
