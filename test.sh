python test.py --model_name=12+4+1_mask_model.bin --output_dir=./saved_models --do_test --train_data_file=./data/processed_train.csv --eval_data_file=./data/processed_val.csv --test_data_file=./data/processed_test.csv --epochs 20 --encoder_block_size 512 --train_batch_size 16 --eval_batch_size 64 --learning_rate 5e-5 --max_grad_norm 1.0 --evaluate_during_training --seed 123456