python train.py method=carplan \
    ++data_path=${TRAINING_DATA_PATH} \
    ++exp_name=pluto \
    ++version=carplan \
    ++method.max_epochs=30 \
    ++method.train_batch_size=32 \
    ++method.learning_rate=1.0e-3 \
    ++load_num_workers=32