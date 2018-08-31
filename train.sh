echo "Staring params server..."

python models/inception_v2_bn_adam_reg.py --model_dir model_dir/adam_reg_d05_bx1 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/patients_full_age.csv --val_set data/csv/val/patients_full_age.csv --img_width 224 --img_height 224 --num_classes 7 --epochs 10 --max_steps 30000 --batch_size 8 --buffer_size 1000 --norm_2 1 --augment 0 --map_first 1 --num_parallel_calls 4 --throttle_secs 120 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"ps","index":0},"environment":"cloud"}' > /tmp/tf-rsna-.ps.log 2>&1 &

sleep 15
echo "Starting chief..."

python models/inception_v2_bn_adam_reg.py --model_dir model_dir/adam_reg_d05_bx1 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/patients_full_age.csv --val_set data/csv/val/patients_full_age.csv --img_width 224 --img_height 224 --num_classes 7 --epochs 10 --max_steps 30000 --batch_size 8 --buffer_size 1000 --norm_2 1 --augment 0 --map_first 1 --num_parallel_calls 4 --throttle_secs 120 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"chief","index":0},"environment":"cloud"}' > /tmp/tf-rsna.chief.log 2>&1 &

sleep 5
echo "Starting worker..."

python models/inception_v2_bn_adam_reg.py --model_dir model_dir/adam_reg_d05_bx1 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/patients_full_age.csv --val_set data/csv/val/patients_full_age.csv --img_width 224 --img_height 224 --num_classes 7 --epochs 10 --max_steps 30000 --batch_size 8 --buffer_size 1000 --norm_2 1 --augment 0 --map_first 1 --num_parallel_calls 4 --throttle_secs 120 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"worker","index":0},"environment":"cloud"}' > /tmp/tf-rsna.worker.log 2>&1 &

python models/inception_v2_bn_adam_reg.py --model_dir model_dir/adam_reg_d05_bx1 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/patients_full_age.csv --val_set data/csv/val/patients_full_age.csv --img_width 224 --img_height 224 --num_classes 7 --epochs 1 --max_steps 28000 --batch_size 8 --buffer_size 1000 --norm_2 1 --augment 0 --map_first 1 --num_parallel_calls 4 --throttle_secs 120 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"evaluator","index":0},"environment":"cloud"}' > /tmp/tf-rsna.evaluator.log 2>&1 &
