#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#           2026 Junjie LI (junjie98.li@connect.polyu.hk)

# U^3-xi recipe: uncertainty-aware speaker embedding learning and extraction.
#   https://arxiv.org/abs/2601.15719
#
# This is a copy of run.sh adapted for the U^3-xi models, the only
# differences being:
#   * stage 4 calls local/extract_vox_uncertainty.sh, which dumps the
#     embedding covariance next to the embeddings;
#   * stage 5 additionally runs local/score_uncertainty.sh, the
#     uncertainty-aware cosine scoring, into ${exp_dir}/scores_uncertainty;
#   * stage 8 (TorchScript export) is not supported, because the
#     multi-view self-attention pooling cannot be scripted.
# All other stages are identical to run.sh, so the standard models are
# not affected. See README_u3xi.md for details.

. ./path.sh || exit 1

# multi-node + multi-gpus:
#   bash run_u3xi.sh --stage 3 --stop-stage 3 --HOST_NODE_ADDR "xxx.xxx.xxx.xxx:port" --num_nodes num_node

stage=3
stop_stage=5

HOST_NODE_ADDR="localhost:29400"
num_nodes=1
job_id=2024

data=data
data_type="shard"  # shard/raw

config=conf/u_cube_xi_redimnet2.yaml
exp_dir=exp/U_CUBE_XI_ReDimNet2B6-tfmel-emb192-fbank72-num_frms200-aug0.6-spTrue-saFalse-ArcMarginUncertainty-SGD-epoch120
gpus="[0,1]"
num_avg=10
checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/u_cube_xi_redimnet2_lm.yaml

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox2_dev vox1; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  echo "$0: num_nodes is $num_nodes, proc_per_node is $num_gpus"
  torchrun --nnodes=$num_nodes --nproc_per_node=$num_gpus \
           --rdzv_id=$job_id --rdzv_backend="c10d" --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox2_dev/${data_type}.list \
      --train_label ${data}/vox2_dev/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  model_path=$avg_model

  # Extract embeddings *and* their uncertainty. This writes both
  # xvector.scp and xvector_variance.scp under ${exp_dir}/embeddings.
  echo "Extract embeddings and uncertainty ..."
  local/extract_vox_uncertainty.sh \
    --exp_dir $exp_dir --model_path $model_path \
    --nj 4 --gpus $gpus --data_type $data_type --data ${data}
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"

  # Uncertainty-aware cosine scoring, using the covariance dumped in stage 4.
  # Results go to ${exp_dir}/scores_uncertainty.
  echo "Uncertainty-aware score ..."
  local/score_uncertainty.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Score calibration ..."
  local/score_calibration.sh \
    --stage 1 --stop-stage 5 \
    --score_norm_method $score_norm_method \
    --calibration_trial "vox2_cali.kaldi" \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Skip exporting the TorchScript model: the multi-view self-attention "
  echo "pooling used by U^3-xi is not compatible with torch.jit.script. "
  echo "Use the standard run.sh if you need a scripted model."
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run_u3xi.sh --stage 3 --stop-stage 8 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
