#!/bin/bash
set -e
export PYTHONPATH=/home/MOBODY/src/d4rl:/home/MOBODY/src/rl-utils
GPU=0
SEED=1
RUN_DIR="runs_friction_shift_sweep_seed1"
WANDB=1
  # 0.1
  # 0.5
  
SHIFT_LEVELS=(
  2.0
  5.0
)

ENVS=(
  "walker2d-friction"
  "hopper-friction"
  "halfcheetah-friction"
)

POLICIES=(
  "IQL"
  "DARA"
  "MOBODY"
)

mkdir -p logs

for SHIFT_LEVEL in "${SHIFT_LEVELS[@]}"; do
  for ENV_NAME in "${ENVS[@]}"; do
    for POLICY in "${POLICIES[@]}"; do

      LOG_FILE="logs/${POLICY}_${ENV_NAME}_shift${SHIFT_LEVEL}_seed${SEED}.log"
      echo "Running ${POLICY} | ${ENV_NAME} | shift=${SHIFT_LEVEL} | seed=${SEED}"

      if [ "${POLICY}" = "DARA" ]; then
        CUDA_VISIBLE_DEVICES=${GPU} python -u train_mobody.py \
          --policy ${POLICY} \
          --env ${ENV_NAME} \
          --shift_level ${SHIFT_LEVEL} \
          --seed ${SEED} \
          --dir ${RUN_DIR} \
          --train_dynamics 0 \
          --penalty_type dara \
          --wandb ${WANDB} \
          > ${LOG_FILE} 2>&1

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

echo "All friction shift sweep experiments completed."