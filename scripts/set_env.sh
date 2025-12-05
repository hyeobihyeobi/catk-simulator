# set your path here
export WAYMO_DATASET_PATH="/workspace/WOMD"
export ROOT_PATH="/workspace/catk-simulator"
# export PRE_PROCESS_ROOT_PATH="/workspace/Git/LatentDriver"
export PRE_PROCESS_ROOT_PATH="/workspace/catk-simulator"
# export CHEX_NOGUARD=1
export WOMD_VAL_PATH="${WAYMO_DATASET_PATH}/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/validation/validation_tfexample.tfrecord@150"
export WOMD_TRAIN_PATH="${WAYMO_DATASET_PATH}/waymo_open_dataset_motion_v_1_3_1/uncompressed/tf_example/training/training_tfexample.tfrecord@1000"

if [ -z "${CUDA_HOME:-}" ]; then
  if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/nvcc" ]; then
    export CUDA_HOME="${CONDA_PREFIX}"
  elif [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME="/usr/local/cuda"
  fi
fi

if [ -n "${CUDA_HOME:-}" ]; then
  case ":${PATH:-}:" in
    *":${CUDA_HOME}/bin:"*) ;;
    *) export PATH="${CUDA_HOME}/bin:${PATH:-}" ;;
  esac

  for cuda_lib_dir in "${CUDA_HOME}/lib64" "${CUDA_HOME}/lib"; do
    if [ -d "${cuda_lib_dir}" ]; then
      case ":${LD_LIBRARY_PATH:-}:" in
        *":${cuda_lib_dir}:"*) ;;
        *)
          if [ -n "${LD_LIBRARY_PATH:-}" ]; then
            export LD_LIBRARY_PATH="${cuda_lib_dir}:${LD_LIBRARY_PATH}"
          else
            export LD_LIBRARY_PATH="${cuda_lib_dir}"
          fi
          ;;
      esac
    fi
  done
fi

export PRE_PROCESS_VAL_PATH="${PRE_PROCESS_ROOT_PATH}/val_preprocessed_path"
export PRE_PROCESS_TRAIN_PATH="${PRE_PROCESS_ROOT_PATH}/train_preprocessed_path"

export INTENTION_VAL_PATH="${PRE_PROCESS_ROOT_PATH}/val_intention_label"
export INTENTION_TRAIN_PATH="${PRE_PROCESS_ROOT_PATH}/train_intention_label"

export TRAINING_DATA_PATH="${PRE_PROCESS_ROOT_PATH}/train_data"
