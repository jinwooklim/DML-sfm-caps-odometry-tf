# Using Odometry data
## Preparing training data
```bash
python data/prepare_train_data.py --dataset_dir=/home/jwlim/hdd2/KITTI_odometry/dataset/ --dataset_name='kitti_odom' --dump_root =/home/jwlim/hdd2/formatted_odom/ --seq_length=5 --img_width=416 --img_height=128 --num_threads=4
```

## Training
```bash
python train.py --dataset_dir=/home/jwlim/hdd2/formatted_odom/ --checkpoint_dir=/home/jwlim/hdd2/checkpoints/ --img_width=416 --img_height=128 --batch_size=4
```

You can then start a `tensorboard` session by
```bash
tensorboard --logdir=/path/to/tensorflow/log/files --port=8888
```
and visualize the training progress by opening [https://localhost:8888](https://localhost:8888) on your browser


# Test
## Pose
For testing sequence [9]
```bash
python test_kitti_pose.py --test_seq [sequence_id] --dataset-dir /home/jwlim/hdd2/poses/ --output_dir /home/jwlim/hdd2/output/ --ckpt_file /home/jwlim/hdd2/checkpoints/model_file
```

---
---
---
# Using Raw data
We need preprocess data folder 
Campus, city, residential, road -> 2011_09_26, 2011_09_28 ...
In Downloads/raw_total/
```bash
cp -r */*/*/ ~/hdd2/raw/
```
```bash
cp -r */2011_10_03*/*/ ~/hdd2/raw/
```
## Training
```bash
python data/prepare_train_data.py --dataset_dir=/home/jwlim/hdd2/raw/ --dataset_name='kitti_raw_eigtn' --dump_root=/home/jwlim/hdd2/formatted/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
```
Preprocessed data : Train data(color image) + calib_cam_to_cam.txt
	-> formatted data : (0000000001.jpg, 000000000_cam.txt) # cam.txt is intrinsic value
	-> 000000000_cam.txt -> (fx, 0, cx, 0, fy, cy, 0, 0, 1)

Test data : dataset_odometry(/sequences/)

Process:
	# The duplicate 'batch_size' flag makes an error
	# So, I change the 'batch_size' -> 'batch_size2' and we specify 'batch_size' in command

	# Use virtual environmnet (python 3.5)
	# In Smartcar folder
	# Preparing training data (KITTI data)
	```bash
	python data/prepare_train_data.py --dataset_dir=/home/jwlim/hdd2/raw/ --dataset_name='kitti_raw_eigen' --dump_root/home/jwlim/hdd2/formatted/ --seq_length=3 --img_width=416 --img_height=128 --num_threads=4
	```
	#
	# Training
	#
	```bash
	python train.py --dataset_dir=/home/jwlim/hdd2/formatted/ --checkpoint_dir=/home/jwlim/hdd2/checkpoints/ --img_width=416 --img_height=128 --batch_size=4
	```
	#
	# KITTI Testing code
	# You Have To Set the 'seq_length' equal to 'seq_length' in train.py
	```bash
	python test_kitti_pose.py --test_seq 9 --dataset_dir /home/jwlim/hdd2/dataset_odometry/ --output_dir /home/jwlim/hdd2/output/ --ckpt_file ./models/model-100280
	```
	```bash
	python test_kitti_pose.py --test_seq 9 --dataset_dir /home/jwlim/hdd2/dataset_odometry/ --output_dir /home/jwlim/hdd2/output/ --ckpt_file /home/jwlim/hdd2/checkpoints/model-191121 --batch_size=1
	```
