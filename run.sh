#python data/prepare_train_data.py --dataset_dir=/home/jwlim/Downloads/KITTI_odometry/dataset/ --dataset_name='kitti_odom' --dump_root =/home/jwlim/hdd2/formatted_odom/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
python train.py --dataset_dir=/home/jwlim/hdd2/formatted_odom/ --checkpoint_dir=/home/jwlim/hdd2/checkpoints_seq3/ --img_width=416 --img_height=128 --batch_size=4
