echo "Staring params server..."

python models/inception_v2_bn_adam.py --model_dir model_dir/imagenet_adam_d05_tf10_l001 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/all_224x224.csv --val_set data/csv/val/all_224x224.csv --img_width 224 --img_height 224 --num_classes 1001 --epochs 100 --max_steps 1550000 --batch_size 8 --buffer_size 10000 --norm_2 1 --map_first 1 --num_parallel_calls 4 --throttle_secs 600 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"ps","index":0},"environment":"cloud"}' > /tmp/tf-train.ps.log 2>&1 &

sleep 15
echo "Starting chief..."

python models/inception_v2_bn_adam.py --model_dir model_dir/imagenet_adam_d05_tf10_l001 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/all_224x224.csv --val_set data/csv/val/all_224x224.csv --img_width 224 --img_height 224 --num_classes 1001 --epochs 100 --max_steps 1550000 --batch_size 8 --buffer_size 10000 --norm_2 1 --map_first 1 --num_parallel_calls 4 --throttle_secs 600 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"chief","index":0},"environment":"cloud"}' > /tmp/tf-train.chief.log 2>&1 &

sleep 5
echo "Starting worker..."

python models/inception_v2_bn_adam.py --model_dir model_dir/imagenet_adam_d05_tf10_l001 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/all_224x224.csv --val_set data/csv/val/all_224x224.csv --img_width 224 --img_height 224 --num_classes 1001 --epochs 100 --max_steps 1550000 --batch_size 8 --buffer_size 10000 --norm_2 1 --map_first 1 --num_parallel_calls 4 --throttle_secs 600 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"worker","index":0},"environment":"cloud"}' > /tmp/tf-train.worker.log 2>&1 &

python models/inception_v2_bn_adam.py --model_dir model_dir/imagenet_adam_d05_tf10_l001 --learning_rate 0.001 --dropout_keep_prob 0.5 --train_set data/csv/train/all_224x224.csv --val_set data/csv/val/all_1k_224x224.csv --img_width 224 --img_height 224 --num_classes 1001 --epochs 1 --max_steps 5000000 --batch_size 8 --buffer_size 1000 --norm_2 1 --map_first 1 --num_parallel_calls 4 --throttle_secs 600 --cluster_spec '{"cluster":{"chief": ["localhost:2221"],"worker":["localhost:2224"],"ps":["localhost:2220"]},"task":{"type":"evaluator","index":0},"environment":"cloud"}' > /tmp/tf-train.evaluator.log 2>&1 &

