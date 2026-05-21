#!/bin/bash
set -e

export PYTHONPATH=/home/MOBODY/src/d4rl:/home/MOBODY/src/rl-utils

GPU=0
RUN_DIR="runs_friction_shift_sweep_seed1to10"
WANDB=1

# --------------------------------------------------
# Domain shift levels
# --------------------------------------------------
SHIFT_LEVELS=(
  0.1
  0.5
  2.0
  5.0
)

# --------------------------------------------------
# Environments
# --------------------------------------------------
ENVS=(
  "walker2d-friction"
  "hopper-friction"
  "halfcheetah-friction"
)

# --------------------------------------------------
# Policies
# --------------------------------------------------
POLICIES=(
  "IQL"
  "DARA"
  "MOBODY"
)

# --------------------------------------------------
# Seeds: 1 ~ 10
# --------------------------------------------------
SEEDS=$(seq 1 10)

mkdir -p logs

# --------------------------------------------------
# Main loop
# --------------------------------------------------
for SEED in ${SEEDS}; do
  for SHIFT_LEVEL in "${SHIFT_LEVELS[@]}"; do
    for ENV_NAME in "${ENVS[@]}"; do
      for POLICY in "${POLICIES[@]}"; do

        LOG_FILE="logs/${POLICY}_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"

        echo "============================================================"
        echo "Running ${POLICY} | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"
        echo "Log: ${LOG_FILE}"
        echo "============================================================"

        # --------------------------------------------------
        # DARA
        # --------------------------------------------------
        if [ "${POLICY}" = "DARA" ]; then
          CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
            --policy DARA \
            --env ${ENV_NAME} \
            --shift_level ${SHIFT_LEVEL} \
            --seed ${SEED} \
            --dir ${RUN_DIR} \
            --train_dynamics 0 \
            --penalty_type dara \
            --wandb ${WANDB} \
            > ${LOG_FILE} 2>&1

        # --------------------------------------------------
        # MOBODY (only method that trains dynamics model)
        # --------------------------------------------------
        elif [ "${POLICY}" = "MOBODY" ]; then
          CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
            --policy MOBODY \
            --env ${ENV_NAME} \
            --shift_level ${SHIFT_LEVEL} \
            --seed ${SEED} \
            --dir ${RUN_DIR} \
            --train_dynamics 1 \
            --penalty_type dara \
            --env_penalty_coef 5 \
            --src_rollout_length 1 \
            --trg_rollout_length 1 \
            --bc_coef 1 \
            --wandb ${WANDB} \
            > ${LOG_FILE} 2>&1

        # --------------------------------------------------
        # IQL
        # --------------------------------------------------
        else
          CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
            --policy ${POLICY} \
            --env ${ENV_NAME} \
            --shift_level ${SHIFT_LEVEL} \
            --seed ${SEED} \
            --dir ${RUN_DIR} \
            --train_dynamics 0 \
            --wandb ${WANDB} \
            > ${LOG_FILE} 2>&1
        fi

        echo "Done: ${LOG_FILE}"
      done
    done
  done
done

echo "All friction shift sweep experiments (seeds 1-10) completed."