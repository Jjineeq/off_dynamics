#!/bin/bash
set -e

# MOBODY / D4RL paths
export PYTHONPATH=/home/MOBODY/src/d4rl:/home/MOBODY/src/rl-utils

# Avoid wandb git safe.directory error inside Docker/root-owned workspace
# If your repo path is different, change /workspace accordingly.
git config --global --add safe.directory /workspace || true

GPU=0
SEED=1
BASE_RUN_DIR="runs_friction_shift_sweep_seed1"
PROP_RUN_DIR="runs_friction_shift_sweep_seed1_proposed"
WANDB=1

SHIFT_LEVELS=(
  0.1
  0.5
  2.0
  5.0
)

ENVS=(
  "walker2d-friction"
  "hopper-friction"
  "halfcheetah-friction"
)

mkdir -p logs

for SHIFT_LEVEL in "${SHIFT_LEVELS[@]}"; do
  for ENV_NAME in "${ENVS[@]}"; do

    ########################################
    # 1) Original baselines: IQL / DARA
    ########################################
    # for POLICY in IQL DARA; do
    #   LOG_FILE="logs/${POLICY}_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
    #   echo "Running ${POLICY} | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

    #   if [ "${POLICY}" = "DARA" ]; then
    #     CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
    #       --policy ${POLICY} \
    #       --env ${ENV_NAME} \
    #       --shift_level ${SHIFT_LEVEL} \
    #       --seed ${SEED} \
    #       --dir ${BASE_RUN_DIR} \
    #       --train_dynamics 0 \
    #       --penalty_type dara \
    #       --wandb ${WANDB} \
    #       > ${LOG_FILE} 2>&1
    #   else
    #     CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
    #       --policy ${POLICY} \
    #       --env ${ENV_NAME} \
    #       --shift_level ${SHIFT_LEVEL} \
    #       --seed ${SEED} \
    #       --dir ${BASE_RUN_DIR} \
    #       --train_dynamics 0 \
    #       --wandb ${WANDB} \
    #       > ${LOG_FILE} 2>&1
    #   fi

    #   echo "Done: ${LOG_FILE}"
    # done

    # ########################################
    # # 2) Original MOBODY
    # ########################################
    # LOG_FILE="logs/MOBODY_ORIGINAL_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
    # echo "Running MOBODY_ORIGINAL | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

    # CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
    #   --policy MOBODY \
    #   --env ${ENV_NAME} \
    #   --shift_level ${SHIFT_LEVEL} \
    #   --seed ${SEED} \
    #   --dir ${BASE_RUN_DIR} \
    #   --train_dynamics 1 \
    #   --penalty_type dara \
    #   --env_penalty_coef 5 \
    #   --src_rollout_length 1 \
    #   --trg_rollout_length 1 \
    #   --bc_coef 1 \
    #   --wandb ${WANDB} \
    #   > ${LOG_FILE} 2>&1

    # echo "Done: ${LOG_FILE}"

    ########################################
    # 3) Proposed MOBODY: BC reliability only
    ########################################
    LOG_FILE="logs/MOBODY_PROPOSED_BC_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
    echo "Running MOBODY_PROPOSED_BC | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

    CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody_proposed.py \
      --policy MOBODY \
      --env ${ENV_NAME} \
      --shift_level ${SHIFT_LEVEL} \
      --seed ${SEED} \
      --dir ${PROP_RUN_DIR} \
      --train_dynamics 0 \
      --penalty_type dara \
      --env_penalty_coef 5 \
      --src_rollout_length 1 \
      --trg_rollout_length 1 \
      --bc_coef 1 \
      --wandb ${WANDB} \
      --use_latent_reliability 1 \
      --reliability_target bc \
      --reliability_effect_mode delta_state \
      --reliability_metric l2 \
      --reliability_tau 1.0 \
      --out_dir_remark "_proposed_bc" \
      > ${LOG_FILE} 2>&1

    echo "Done: ${LOG_FILE}"

    ########################################
    # 4) Proposed MOBODY: BC + rollout reliability
    # Optional stronger variant. Comment out if too slow.
    ########################################
    LOG_FILE="logs/MOBODY_PROPOSED_BOTH_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
    echo "Running MOBODY_PROPOSED_BOTH | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

    CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody_proposed.py \
      --policy MOBODY \
      --env ${ENV_NAME} \
      --shift_level ${SHIFT_LEVEL} \
      --seed ${SEED} \
      --dir ${PROP_RUN_DIR} \
      --train_dynamics 0 \
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
      --reliability_tau 1.0 \
      --out_dir_remark "_proposed_both" \
      > ${LOG_FILE} 2>&1

    echo "Done: ${LOG_FILE}"

  done
done

echo "All friction shift comparison experiments completed."
