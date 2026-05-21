#!/bin/bash
set -e

export PYTHONPATH=/home/MOBODY/src/d4rl:/home/MOBODY/src/rl-utils
git config --global --add safe.directory /workspace || true

GPU=0
SEED=1
PROP_RUN_DIR="runs_friction_shift_sweep_seed1_proposed_v2"
WANDB=1

SHIFT_LEVELS=(0.1 0.5 2.0 5.0)
ENVS=("walker2d-friction" "hopper-friction" "halfcheetah-friction")

mkdir -p logs

for SHIFT_LEVEL in "${SHIFT_LEVELS[@]}"; do
  for ENV_NAME in "${ENVS[@]}"; do

    # LOG_FILE="logs/MOBODY_PROPOSED_V2_BC_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
    # echo "Running MOBODY_PROPOSED_V2_BC | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

    # CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody_proposed_v2.py \
    #   --policy MOBODY \
    #   --env ${ENV_NAME} \
    #   --shift_level ${SHIFT_LEVEL} \
    #   --seed ${SEED} \
    #   --dir ${PROP_RUN_DIR} \
    #   --train_dynamics 0 \
    #   --penalty_type dara \
    #   --env_penalty_coef 5 \
    #   --src_rollout_length 1 \
    #   --trg_rollout_length 1 \
    #   --bc_coef 1 \
    #   --wandb ${WANDB} \
    #   --use_latent_reliability 1 \
    #   --reliability_target bc \
    #   --reliability_effect_mode delta_state \
    #   --reliability_metric l2 \
    #   --reliability_tau 5.0 \
    #   --reliability_min 0.5 \
    #   --reliability_normalize_mean 1 \
    #   --reliability_debug_interval 1000 \
    #   --max_step 100000 \
    #   --eval_freq 5000 \
    #   --out_dir_remark "_v2_bc" \
    #   > ${LOG_FILE} 2>&1

    # echo "Done: ${LOG_FILE}"

    LOG_FILE="logs/MOBODY_PROPOSED_V2_BOTH_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
    echo "Running MOBODY_PROPOSED_V2_BOTH | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

    CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody_proposed_v2.py \
      --policy MOBODY \
      --env ${ENV_NAME} \
      --shift_level ${SHIFT_LEVEL} \
      --seed ${SEED} \
      --dir ${PROP_RUN_DIR} \
      --train_dynamics 1 \
      --penalty_type dara \
      --env_penalty_coef 5 \
      --src_rollout_length 1 \
      --trg_rollout_length 1 \
      --bc_coef 1 \
      --wandb ${WANDB} \
      --use_latent_reliability 1 \
      --reliability_target both \
      --reliability_effect_mode delta_state \
      --reliability_metric l2 \
      --reliability_tau 5.0 \
      --reliability_min 0.5 \
      --reliability_normalize_mean 1 \
      --reliability_debug_interval 1000 \
      --max_step 100000 \
      --eval_freq 5000 \
      --out_dir_remark "_v2_both" \
      > ${LOG_FILE} 2>&1

    echo "Done: ${LOG_FILE}"

  done
done

echo "V2 sanity experiments completed."
