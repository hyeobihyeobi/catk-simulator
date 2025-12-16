#!/usr/bin/env sh
set -e

# Resolve repository root and make src importable
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
repo_root=$(CDPATH= cd -- "${script_dir}/.." && pwd -P)
if [ -z "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${repo_root}"
else
  export PYTHONPATH="${repo_root}:${PYTHONPATH}"
fi

# Default batch dims and remainder handling; override via env if needed.
BATCH_DIMS="${BATCH_DIMS:-[4,32]}"
DROP_REMAINDER="${DROP_REMAINDER:-True}"

# for validation
# python -m src.preprocess.preprocess_data \
#     ++batch_dims=${BATCH_DIMS} \
#     ++waymax_conf.path="${WOMD_VAL_PATH}" \
#     ++waymax_conf.max_num_rg_points=30000 \
#     ++waymax_conf.drop_remainder=${DROP_REMAINDER} \
#     ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
#     ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}"

# for training
# python -m src.preprocess.preprocess_data \
#     ++batch_dims=${BATCH_DIMS} \
#     ++waymax_conf.path="${WOMD_TRAIN_PATH}" \
#     ++waymax_conf.max_num_rg_points=30000 \
#     ++waymax_conf.drop_remainder=${DROP_REMAINDER} \
#     ++data_conf.path_to_processed_map_route="${PRE_PROCESS_TRAIN_PATH}" \
#     ++metric_conf.intention_label_path="${INTENTION_TRAIN_PATH}"
# #     tips: batch_dims=[1,2000] means batch size is 2000 and used GPU  is 1, you MUST set GPU=1 here to make 'drop_remainder=False' validated
# #      It may take some time to preprocess the data


python -m src.preprocess.preprocess_data \
    ++batch_dims="[1, 1]" \
    ++waymax_conf.path="${WOMD_TRAIN_PATH}" \
    ++waymax_conf.max_num_rg_points=30000 \
    ++waymax_conf.drop_remainder=${DROP_REMAINDER} \
    ++data_conf.path_to_processed_map_route="/workspace/catk-simulator/train_preprocessed_path_reference_lines" \
    ++metric_conf.intention_label_path="/workspace/catk-simulator/train_intention_label_reference_lines"
#     tips: batch_dims=[1,2000] means batch size is 2000 and used GPU  is 1, you MUST set GPU=1 here to make 'drop_remainder=False' validated
#      It may take some time to preprocess the data
