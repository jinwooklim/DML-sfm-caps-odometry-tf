from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
from config import cfg
from euler_to_rotation import yaw_to_rotation

# Range of disparity/inverse depth values
DISP_SCALING = 10
MIN_DISP = 0.01

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_exp_net(tgt_image, src_image_stack, capsnet, caps_X, caps_label, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                ''' 
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                print("pose_pred : ", pose_pred.get_shape())
                print("pose_avg : ", pose_avg.get_shape())
                
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
                print("pose_final : ", pose_final.get_shape())
            
                prediction = 5
                exit()
                '''
                #
                # Version # 1
                #
                '''
                #last_cnv = slim.conv2d(cnv5, 6*num_source, [1, 1], scope='last_cnv', stride=1, normalizer_fn=None, activation_fn=None)
                #last_cnv = slim.conv2d(cnv5, 256, [1, 1], scope='last_cnv', stride=1, padding='VALID')
                #cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                #cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                #last_cnv = slim.conv2d(cnv7, 256, [1, 1], scope='last_cnv', stride=1, normalizer_fn=None, activation_fn=None)
                last_cnv = slim.conv2d(cnv2, 256, [3, 3], scope='last_cnv', stride=1, padding='VALID')
                
                with tf.variable_scope('capsnet'):
                    if is_training == True:
                        X = caps_X
                        labels = caps_label
                        Y = tf.one_hot(labels, depth=cfg.num_of_class, axis=1, dtype=tf.float32)
                        capsnet_model = capsnet.model(last_cnv)
                        v_length, prediction = capsnet.predict(capsnet_model)
                        decoded = capsnet.decoder(capsnet_model, prediction)
                        margin_loss, reconstruction_loss, capsnet_total_loss = capsnet.loss(X, Y, v_length, decoded)
                        capsnet.summary(decoded, margin_loss, reconstruction_loss, capsnet_total_loss)
                    else:
                        capsnet_model = capsnet.model(last_cnv)
                        _, prediction = capsnet.predict(capsnet_model)
                        decoded = capsnet.decoder(capsnet_model, prediction)
                    
                    # convert yaw to rotation
                    # index : [0,1,2,3,4,5,6,7]
                    # value : [yaw_rate, tx, ty, tz, yaw_rate2, tx2, ty2, tz2]
                    tx_ty_tz = decoded[:,1:4] # (4, 3)
                    tx_ty_tz2 = decoded[:,5:] # (4, 3)
                    
                    rx, ry, rz = yaw_to_rotation(decoded[:,0])
                    rx_ry_rz = tf.transpose(tf.stack([rx, ry, rz])) # (4, 3)

                    rx2, ry2, rz2 = yaw_to_rotation(decoded[:,4])
                    rx_ry_rz2 = tf.transpose(tf.stack([rx2, ry2, rz2])) # (4, 3)

                    converted_decoded = tf.concat([tx_ty_tz, rx_ry_rz], 1) # (4, 6)
                    converted_decoded2 = tf.concat([tx_ty_tz2, rx_ry_rz2], 1) # (4, 6)
                    converted_decoded = tf.reshape(converted_decoded, [cfg.batch_size, 1, 6]) # (4, 1, 6)
                    converted_decoded2 = tf.reshape(converted_decoded2, [cfg.batch_size, 1, 6]) # (4, 1, 6)

                    #
                    # Make pose_pred
                    #
                    stacked_converted_decoded = tf.stack([converted_decoded, converted_decoded2], axis=1) # (4, 2, 1, 6)

                    pose_pred = tf.reshape(stacked_converted_decoded, [cfg.batch_size, num_source, 6]) # (4, 2, 6)
                    
                    pose_final = 0.01 * pose_pred
                    '''

                #
                # Version 2
                #
                ''' 
                with tf.variable_scope('capsnet'):
                    if is_training == True:
                        X = caps_X
                        labels = caps_label
                        Y = tf.one_hot(labels, depth=cfg.num_of_class, axis=1, dtype=tf.float32)
                        capsnet_model = capsnet.model(cnv5, num_source)
                        v_length, prediction = capsnet.predict(capsnet_model)
                        decoded = capsnet.decoder(capsnet_model, prediction)
                        margin_loss, reconstruction_loss, capsnet_total_loss = capsnet.loss(X, Y, v_length, decoded)
                        capsnet.summary(decoded, margin_loss, reconstruction_loss, capsnet_total_loss)
                    else:
                        capsnet_model = capsnet.model(cnv5, num_source)
                        _, prediction = capsnet.predict(capsnet_model)
                        decoded = capsnet.decoder(capsnet_model, prediction)
                
                    pose_pred = decoded # [batch_size, 1, 4, num_source * 6]
                    pose_avg = tf.reduce_mean(pose_pred, 0) # [batch_size, num_source * 6]
                    print("pose_pred : ", DISP_SCALINGose_pred.get_shape())
                    print("pose_avg : ", pose_avg.get_shape())
                
                    # Empirically we found that scaling by a small constant 
                    # facilitates training.
                    pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
                    print("pose_final : ", pose_final.get_shape())
                '''
                #
                # Version 3
                #
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                #cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                with tf.variable_scope('capsnet'):
                    if is_training == True:
                        X = caps_X
                        labels = caps_label
                        Y = tf.one_hot(labels, depth=cfg.num_of_class, axis=1, dtype=tf.float32)
                        capsnet_model = capsnet.model(cnv6, num_source)
                        v_length, prediction = capsnet.predict(capsnet_model)
                        decoded = capsnet.decoder(capsnet_model, prediction)
                        margin_loss, reconstruction_loss, capsnet_total_loss = capsnet.loss(X, Y, v_length, decoded)
                        capsnet.summary(decoded, margin_loss, reconstruction_loss, capsnet_total_loss)
                    else:
                        capsnet_model = capsnet.model(cnv6, num_source)
                        _, prediction = capsnet.predict(capsnet_model)
                        decoded = capsnet.decoder(capsnet_model, prediction) # (4, 24)
                    #decoded = tf.reshape(decoded, shape=(cfg.batch_size, num_source, -1)) # (4,4,6)
                    #pose_final = 0.01 * decoded
                    
                    with tf.name_scope('weights'):
                        regression_w = tf.get_variable('regression_w', shape=[24, 24], dtype=tf.float32)

                    with tf.name_scope('biases'):
                        regression_b = tf.get_variable('regression_b', shape=[24], dtype=tf.float32)
                    
                    with tf.name_scope('Wx_plus_b'):
                        decoded = tf.nn.xw_plus_b(decoded, regression_w, regression_b) # (4,4,6)
                        pose_final = tf.reshape(decoded, shape=(cfg.batch_size, num_source, -1))   
                        print("pose_final : ", np.shape(pose_final))
                #exit()
            
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
                        normalizer_fn=None, activation_fn=None)
                    
                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            
            return prediction, pose_final, [mask1, mask2, mask3, mask4], end_points

def disp_net(tgt_image, is_training=True):
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4') + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3') + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2') + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1') + MIN_DISP
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], end_points

