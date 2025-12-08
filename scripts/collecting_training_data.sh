SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [ -n "${PYTHONPATH:-}" ]; then
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${ROOT_DIR}"
fi

# Align data collection batch with available accelerator count.
DEVICE_COUNT=$(python - <<'PY'
try:
    import jax
except Exception:
    print(0)
else:
    try:
        print(jax.local_device_count())
    except Exception:
        print(0)
PY
)

case "${DEVICE_COUNT}" in
  ''|*[!0-9]*|0)
    DEVICE_COUNT=1
    ;;
esac

if [ -n "${COLLECT_BATCH_DEVICES:-}" ]; then
  DEVICE_COUNT="${COLLECT_BATCH_DEVICES}"
  case "${DEVICE_COUNT}" in
    ''|*[!0-9]*|0)
      echo "COLLECT_BATCH_DEVICES must be a positive integer. Falling back to 1."
      DEVICE_COUNT=1
      ;;
  esac
fi

python src/preprocess/saving_training_data.py \
    ++batch_dims=[${DEVICE_COUNT},32] \
    ++waymax_conf.drop_remainder=True \
    ++waymax_conf.path="${WOMD_TRAIN_PATH}" \
    ++data_conf.path_to_processed_map_route="${PRE_PROCESS_TRAIN_PATH}" \
    ++metric_conf.intention_label_path="${INTENTION_TRAIN_PATH}" \
    ++save_path="${TRAINING_DATA_PATH}"


# python src/preprocess/saving_training_data.py \
#     ++batch_dims=[${DEVICE_COUNT},25] \
#     ++waymax_conf.drop_remainder=True \
#     ++waymax_conf.path="${WOMD_VAL_PATH}" \
#     ++data_conf.path_to_processed_map_route="${PRE_PROCESS_VAL_PATH}" \
#     ++metric_conf.intention_label_path="${INTENTION_VAL_PATH}" \
#     ++save_path="${TRAINING_DATA_PATH}"
