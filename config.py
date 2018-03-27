import tensorflow as tf

flags = tf.app.flags


##########################
#   Add by jwlim         #
##########################
flags.DEFINE_integer("num_of_class", 11, "Number of class")
flags.DEFINE_string("capsdata_dir", "/home/jwlim/hdd2/capsnet_data/", "Dataset for Capsnet")
flags.DEFINE_string("ckpt_file", "", "Checkpoint file")

###########################
#   SfMLearner parameters #
###########################
#flags = tf.app.flags
#flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("dataset_dir", "/home/jwlim/hdd2/formatted_odom/", "Dataset directory")
#flags.DEFINE_string('checkpoint_dir', "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string('checkpoint_dir', "/home/jwlim/hdd2/checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.5, "Weight for smoothness")
flags.DEFINE_float("explain_reg_weight", 0.0, "Weight for explanability regularization")
flags.DEFINE_integer("batch_size", 4, "The size of of a sample batch")
#flags.DEFINE_integer("batch_size", 8, "The size of of a sample batch")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
#flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("seq_length", 5, "Sequence length for each example")
#flags.DEFINE_integer("max_steps", 200000, "Maximum number of training iterations")
flags.DEFINE_integer("max_steps", 20000, "Maximum number of training iterations")
#flags.DEFINE_integer("max_steps", 10000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 1000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
############################################################
############################
#   test_kitti_pose        #
############################
flags.DEFINE_integer("test_seq", 9, "Sequence length for test")
flags.DEFINE_string("output_dir", None, "Output directory")

############################
#    hyper parameters      #
############################

# For separate margin loss
flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')

# for training
#flags.DEFINE_integer('batch_size', 128, 'batch size')
#flags.DEFINE_integer('epoch', 50, 'epoch')
flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')
#flags.DEFINE_boolean('mask_with_y', False, 'use the true label to mask out target capsule or not')

flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')


############################
#   environment setting    #
############################
#flags.DEFINE_string('dataset', 'mnist', 'The name of dataset [mnist, fashion-mnist')
flags.DEFINE_boolean('is_training', True, 'train or predict phase')
flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing examples')
#flags.DEFINE_string('logdir', 'logdir', 'logs directory')
#flags.DEFINE_integer('train_sum_freq', 100, 'the frequency of saving train summary(step)')
#flags.DEFINE_integer('val_sum_freq', 500, 'the frequency of saving valuation summary(step)')
#flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
#flags.DEFINE_string('results', 'results', 'path for saving results')

############################
#   distributed setting    #
############################
#flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
#flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
#flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

#cfg = tf.app.flags.FLAGS
cfg = flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
