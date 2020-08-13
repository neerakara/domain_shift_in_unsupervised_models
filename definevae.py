import tensorflow as tf
import numpy as np
 
# ==================================================
# ==================================================
def definevae(lat_dim = 60,
              patchsize = 28,
              batchsize = 50,
              rescaled = False,
              half = False,
              mode = [],
              chunks40 = False,
              Melmodels = False,
              use_normalizer = False):
                            
     if mode ==[]:
          mode = 'MRIunproc'
     
     SEED = 10
     ndims = patchsize
     useMixtureScale = True
     noisy = 50
     batch_size = batchsize #50#361*2 # 1428#1071#714#714#714 #1000 1785#      
     usebce = False
     nzsamp = 1
     kld_div = 1.
     std_init = 0.05               
     input_dim = ndims*ndims
     fcl_dim = 500      
     print(">>> lat_dim value: "+str(lat_dim))
     print(">>> mode is: " + mode)
     lat_dim_1 = max(1, np.floor(lat_dim/2))     
     num_inp_channels = 1
     
     # ======================================
     # make a simple fully connected network
     # ======================================     
     tf.reset_default_graph()
     
     # ======================================
     # define the activation function to use
     # ======================================
     def fact(x):
          #return tf.nn.tanh(x)
          return tf.nn.relu(x)
     
     # ======================================
     # define the input place holder
     # ======================================
     x_inp = tf.placeholder("float", shape = [None, input_dim])
     x_inp_ = tf.reshape(x_inp, [batch_size, ndims, ndims, 1])
     nsampl = 50
     
     # ======================================
     # define the network layer parameters
     # ======================================
     intl = tf.truncated_normal_initializer(stddev = std_init, seed = SEED)

     # ======================================
     # define normalization module here
     # ======================================
     with tf.variable_scope("NORM") as scope:
         norm_k = 1
         # ======================================
         # define weights
         # ======================================
         norm_conv1_weights = tf.get_variable("norm_conv1_weights", [norm_k, norm_k, num_inp_channels, 32], initializer=intl)          
         norm_conv2_weights = tf.get_variable("norm_conv2_weights", [norm_k, norm_k, 32, 32], initializer=intl)          
         norm_conv3_weights = tf.get_variable("norm_conv3_weights", [norm_k, norm_k, 32, num_inp_channels], initializer=intl)
                   
     if use_normalizer:     
        norm_conv1 = fact(tf.nn.conv2d(x_inp_, norm_conv1_weights, strides = [1, 1, 1, 1], padding='SAME'))     
        norm_conv2 = fact(tf.nn.conv2d(norm_conv1, norm_conv2_weights, strides = [1, 1, 1, 1], padding='SAME'))
        delta_x = fact(tf.nn.conv2d(norm_conv2, norm_conv3_weights, strides = [1, 1, 1, 1], padding='SAME'))
        x_normalized = x_inp_ + delta_x
                
     else:
         x_normalized = x_inp_
         
     # ======================================
     # define the network here
     # ======================================
     with tf.variable_scope("VAE") as scope:
          
          # ======================================
          # define weights
          # ======================================
          enc_conv1_weights = tf.get_variable("enc_conv1_weights", [3, 3, num_inp_channels, 32], initializer=intl)
          enc_conv1_biases = tf.get_variable("enc_conv1_biases", shape = [32], initializer=tf.constant_initializer(value=0))
          
          enc_conv2_weights = tf.get_variable("enc_conv2_weights", [3, 3, 32, 64], initializer=intl)
          enc_conv2_biases = tf.get_variable("enc_conv2_biases", shape = [64], initializer=tf.constant_initializer(value=0))
          
          enc_conv3_weights = tf.get_variable("enc_conv3_weights", [3, 3, 64, 64], initializer=intl)
          enc_conv3_biases = tf.get_variable("enc_conv3_biases", shape = [64], initializer=tf.constant_initializer(value=0))
              
          mu_weights = tf.get_variable(name="mu_weights", shape = [int(input_dim*64), lat_dim], initializer=intl)
          mu_biases = tf.get_variable("mu_biases", shape = [lat_dim], initializer=tf.constant_initializer(value=0))
         
          logVar_weights = tf.get_variable(name="logVar_weights", shape=[int(input_dim*64), lat_dim], initializer=intl)
          logVar_biases = tf.get_variable("logVar_biases", shape=[lat_dim], initializer=tf.constant_initializer(value=0))
                  
          if useMixtureScale:        
               dec_fc1_weights = tf.get_variable(name="dec_fc1_weights", shape=[int(lat_dim), int(input_dim*48)], initializer=intl)
               dec_fc1_biases = tf.get_variable("dec_fc1_biases", shape=[int(input_dim*48)], initializer=tf.constant_initializer(value=0))
              
               dec_conv1_weights = tf.get_variable("dec_conv1_weights", [3, 3, 48, 48], initializer=intl)
               dec_conv1_biases = tf.get_variable("dec_conv1_biases", shape=[48], initializer=tf.constant_initializer(value=0))
                    
               dec_conv2_weights = tf.get_variable("decc_conv2_weights", [3, 3, 48, 90], initializer=intl)
               dec_conv2_biases = tf.get_variable("dec_conv2_biases", shape=[90], initializer=tf.constant_initializer(value=0))
                    
               dec_conv3_weights = tf.get_variable("dec_conv3_weights", [3, 3, 90, 90], initializer=intl)
               dec_conv3_biases = tf.get_variable("dec_conv3_biases", shape=[90], initializer=tf.constant_initializer(value=0))
               
               dec_out_weights = tf.get_variable("dec_out_weights", [3, 3, 90, 1], initializer=intl)
               dec_out_biases = tf.get_variable("dec_out_biases", shape=[1], initializer=tf.constant_initializer(value=0))
               
               dec1_out_cov_weights = tf.get_variable("dec1_out_cov_weights", [3, 3, 90, 1], initializer=intl)
               dec1_out_cov_biases = tf.get_variable("dec1_out_cov_biases", shape=[1], initializer=tf.constant_initializer(value=0))
              
          else:        
               pass
          
     # ======================================
     # A. make encoder layers
     # ======================================
     enc_conv1 = tf.nn.conv2d(x_normalized, enc_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')         
     enc_relu1 = fact(tf.nn.bias_add(enc_conv1, enc_conv1_biases))     
     enc_conv2 = tf.nn.conv2d(enc_relu1, enc_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
     enc_relu2 = fact(tf.nn.bias_add(enc_conv2, enc_conv2_biases))     
     enc_conv3 = tf.nn.conv2d(enc_relu2, enc_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
     enc_relu3 = fact(tf.nn.bias_add(enc_conv3, enc_conv3_biases))           
     flat_relu3 = tf.contrib.layers.flatten(enc_relu3)
     
     # ======================================
     # B. get the values for drawing z
     # ======================================
     mu = tf.matmul(flat_relu3, mu_weights) + mu_biases
     mu = tf.tile(mu, (nzsamp, 1)) # replicate for number of z's you want to draw
     logVar = tf.matmul(flat_relu3, logVar_weights) + logVar_biases
     logVar = tf.tile(logVar,  (nzsamp, 1)) # replicate for number of z's you want to draw
     std = tf.exp(0.5 * logVar)
     
     # ======================================
     # C. draw an epsilon and get z
     # ======================================
     epsilon = tf.random_normal(tf.shape(logVar), name='epsilon')
     z = mu + tf.multiply(std, epsilon)
     
     if useMixtureScale:
          indices1 = tf.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
          indices2 = tf.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')
          z1 = tf.transpose(tf.gather(tf.transpose(z),indices1))
          z2 = tf.transpose(tf.gather(tf.transpose(z),indices2))          
          # ======================================
          # D. build the decoder layers from z1 for mu(z)
          # ======================================
          dec_L1 = fact(tf.matmul(z, dec_fc1_weights) + dec_fc1_biases)          
     else:
          pass
      
     # ======================================
     # further decoder layers
     # ======================================
     dec_L1_reshaped = tf.reshape(dec_L1 ,[batch_size,int(ndims),int(ndims),48])     
     dec_conv1 = tf.nn.conv2d(dec_L1_reshaped, dec_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
     dec_relu1 = fact(tf.nn.bias_add(dec_conv1, dec_conv1_biases))     
     dec_conv2 = tf.nn.conv2d(dec_relu1, dec_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
     dec_relu2 = fact(tf.nn.bias_add(dec_conv2, dec_conv2_biases))     
     dec_conv3 = tf.nn.conv2d(dec_relu2, dec_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
     dec_relu3 = fact(tf.nn.bias_add(dec_conv3, dec_conv3_biases))
     
     # ======================================
     # e. build the output layer w/out activation function
     # ======================================
     dec_out = tf.nn.conv2d(dec_relu3, dec_out_weights, strides=[1, 1, 1, 1], padding='SAME')
     y_out_ = tf.nn.bias_add(dec_out, dec_out_biases)     
     y_out = tf.contrib.layers.flatten(y_out_)
                      
     # ======================================
     # e.2 build the covariance at the output if using mixture of scales
     # ======================================
     if useMixtureScale:
          # e. build the output layer w/out activation function
          dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
          y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)          
          y_out_prec_ = tf.exp(y_out_prec_log)          
          y_out_prec = tf.contrib.layers.flatten(y_out_prec_)          
          # DBG # y_out_cov=tf.ones_like(y_out)
     
     # ==============================================================================
     # build the loss functions and the optimizer
     # ==============================================================================     
     # KLD loss per sample in the batch
     KLD = -0.5 * tf.reduce_sum(1 + logVar - tf.pow(mu, 2) - tf.exp(logVar), reduction_indices=1)     
     x_inp__ = tf.tile(x_inp, (nzsamp, 1))
     
     # L2 loss per sample in the batch
     if useMixtureScale:
          l2_loss_1 = tf.reduce_sum(tf.multiply(tf.pow((y_out - x_inp__), 2), y_out_prec), axis=1)
          l2_loss_2 = tf.reduce_sum(tf.log(y_out_prec), axis=1) # tf.reduce_sum(tf.log(y_out_cov),axis=1)
          l2_loss_ = l2_loss_1 - l2_loss_2
     else:
          l2_loss_ = tf.reduce_sum(tf.pow((y_out - x_inp__), 2), axis=1)
          if usebce:
               l2_loss_ = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_out, labels=x_inp__), reduction_indices=1)
          
     # take the total mean loss of this batch
     loss_tot = tf.reduce_mean(1/kld_div * KLD + l2_loss_)
     
     # get the optimizer
     if useMixtureScale:
          train_step = tf.train.AdamOptimizer(5e-4).minimize(loss_tot)
     else:
          train_step = tf.train.AdamOptimizer(5e-3).minimize(loss_tot)

     # ==============================================================================     
     # make two lists of VAE and NORM variables
     # ==============================================================================     
     vae_vars = []
     norm_vars = []
     for v in tf.trainable_variables():
         var_name = v.name
         if 'NORM' in var_name: norm_vars.append(v)
         elif 'VAE' in var_name: vae_vars.append(v)
                 
     # ==============================================================================
     # gradient stuff, gd recon etc...
     # ==============================================================================
     nsampl = batchsize # 361*2 # 1428 # 1071 # 714 # 714 1785#
     x_rec = tf.get_variable('x_rec', shape = [nsampl, ndims*ndims], initializer = tf.constant_initializer(value=0.0)) 
     z_std_multip = tf.placeholder_with_default(1.0, shape=[])
     
     # ==============================================================================
     # REWIRE THE GRAPH
     # you need to rerun all operations after this as well!!!!
     # ==============================================================================     
     # rewire the graph input
     x_inp_ = tf.reshape(x_rec, [nsampl, ndims, ndims, 1])
     
     if use_normalizer:     
        norm_conv1 = fact(tf.nn.conv2d(x_inp_, norm_conv1_weights, strides=[1, 1, 1, 1], padding='SAME'))     
        norm_conv2 = fact(tf.nn.conv2d(norm_conv1, norm_conv2_weights, strides=[1, 1, 1, 1], padding='SAME'))
        delta_x = fact(tf.nn.conv2d(norm_conv2, norm_conv3_weights, strides=[1, 1, 1, 1], padding='SAME'))
        x_normalized = x_inp_ + delta_x
                
     else:
         x_normalized = x_inp_
        
     enc_conv1 = tf.nn.conv2d(x_normalized, enc_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
     enc_relu1 = fact(tf.nn.bias_add(enc_conv1, enc_conv1_biases))
     enc_conv2 = tf.nn.conv2d(enc_relu1, enc_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
     enc_relu2 = fact(tf.nn.bias_add(enc_conv2, enc_conv2_biases))
     enc_conv3 = tf.nn.conv2d(enc_relu2, enc_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
     enc_relu3 = fact(tf.nn.bias_add(enc_conv3, enc_conv3_biases))           
     flat_relu3 = tf.contrib.layers.flatten(enc_relu3)
     
     # b. get the values for drawing z
     mu = tf.matmul(flat_relu3, mu_weights) + mu_biases
     mu = tf.tile(mu, (nzsamp, 1)) # replicate for number of z's you want to draw
     logVar = tf.matmul(flat_relu3, logVar_weights) + logVar_biases
     logVar = tf.tile(logVar,  (nzsamp, 1)) # replicate for number of z's you want to draw
     std = tf.exp(0.5 * logVar)
     
     # c. draw an epsilon and get z
     epsilon = tf.random_normal(tf.shape(logVar), name='epsilon')
     z = mu + z_std_multip * tf.multiply(std, epsilon) # z_std_multip*epsilon # # KCT!!!  
     
     if useMixtureScale:
          indices1 = tf.range(start=0, limit=lat_dim_1, delta=1, dtype='int32')
          indices2 = tf.range(start=lat_dim_1, limit=lat_dim, delta=1, dtype='int32')
          z1 = tf.transpose(tf.gather(tf.transpose(z),indices1))
          z2 = tf.transpose(tf.gather(tf.transpose(z),indices2))          
          # d. build the decoder layers from z1 for mu(z)
          dec_L1 = fact(tf.matmul(z, dec_fc1_weights) + dec_fc1_biases)     
     else:
          pass
         
     dec_L1_reshaped = tf.reshape(dec_L1, [batch_size, int(ndims), int(ndims), 48])
     dec_conv1 = tf.nn.conv2d(dec_L1_reshaped, dec_conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
     dec_relu1 = fact(tf.nn.bias_add(dec_conv1, dec_conv1_biases))     
     dec_conv2 = tf.nn.conv2d(dec_relu1, dec_conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
     dec_relu2 = fact(tf.nn.bias_add(dec_conv2, dec_conv2_biases))     
     dec_conv3 = tf.nn.conv2d(dec_relu2, dec_conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
     dec_relu3 = fact(tf.nn.bias_add(dec_conv3, dec_conv3_biases))
     
     # e. build the output layer w/out activation function
     dec_out = tf.nn.conv2d(dec_relu3, dec_out_weights, strides=[1, 1, 1, 1], padding='SAME')
     y_out_ = tf.nn.bias_add(dec_out, dec_out_biases)          
     y_out = tf.contrib.layers.flatten(y_out_)
                      
     # e.2 build the covariance at the output if using mixture of scales
     if useMixtureScale:
          # e. build the output layer w/out activation function
          dec_out_cov = tf.nn.conv2d(dec_relu3, dec1_out_cov_weights, strides=[1, 1, 1, 1], padding='SAME')
          y_out_prec_log = tf.nn.bias_add(dec_out_cov, dec1_out_cov_biases)     
          y_out_prec_ = tf.exp(y_out_prec_log)
          y_out_prec = tf.contrib.layers.flatten(y_out_prec_)
          
     # reshape x_normalized to match other flattened shapes
     x_normalized_flattened = tf.reshape(x_normalized, [nsampl, ndims*ndims])    
     
     # ============================================================================== 
     # VAE needs to reconstruct only x_normalized
     # ==============================================================================          
     op_p_x_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((y_out - x_normalized_flattened),2), y_out_prec), axis=1) \
                 + 0.5 * tf.reduce_sum(tf.log(y_out_prec), axis=1) - 0.5 * ndims * ndims * tf.log(2*np.pi) ) 
          
     op_q_z_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - mu), 2), tf.reciprocal(std)), axis=1) \
                 - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5 * lat_dim * tf.log(2*np.pi))
     
     z_pl = tf.get_variable('z_pl', shape = [nsampl, lat_dim], initializer = tf.constant_initializer(value=0.0))
     
     op_q_zpl_x = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z_pl - mu),2), tf.reciprocal(std)),axis=1) \
                   - 0.5 * tf.reduce_sum(tf.log(std), axis=1) -  0.5 * lat_dim * tf.log(2*np.pi))
     
     op_p_z = (- 0.5 * tf.reduce_sum(tf.multiply(tf.pow((z - tf.zeros_like(mu)),2), tf.reciprocal(tf.ones_like(std))),axis=1) \
               - 0.5 * tf.reduce_sum(tf.log(tf.ones_like(std)), axis=1) -  0.5 * lat_dim * tf.log(2*np.pi))
     
     funop = op_p_x_z + op_p_z - op_q_z_x
     
     # ==============================================================================          
     # the gradients are still computed wrt x_rec (the input to the normalization module).
     # these gradients are used to update the x_rec, in order to increase its ELBO.
     # ==============================================================================          
     grd = tf.gradients(op_p_x_z + op_p_z - op_q_z_x, x_rec) # 
     grd_p_x_z0 = tf.gradients(op_p_x_z, x_rec)[0]
     grd_p_z0 = tf.gradients(op_p_z, x_rec)[0]
     grd_q_z_x0 = tf.gradients(op_q_z_x, x_rec)[0]     
     grd_q_zpl_x_az0 = tf.gradients(op_q_zpl_x, z_pl)[0]
     grd2 = tf.gradients(grd[0], x_rec)
     
     grd0 = grd[0]
     grd20 = grd2[0]

     if use_normalizer:
         # ============================================================================== 
         # create an optimizer for the normalization nodule. This also tries to increase the ELBO of the normalized image,
         # but not by changing the image itself, but by changing the parameters of the normalization module.
         # ============================================================================== 
         # optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3) 
         # normalization_op = optimizer.minimize(funop, var_list = norm_vars)
         
         # create an instance of the required optimizer
         optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
            
         # initialize variable holding the accumlated gradients and create a zero-initialisation op
         normalizer_accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in norm_vars]
            
         # op to set the accumulated gradients to 0
         normalizer_accumulated_gradients_zero_op = [ac.assign(tf.zeros_like(ac)) for ac in normalizer_accumulated_gradients]
    
         # calculate gradients and define accumulation op
         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
         with tf.control_dependencies(update_ops):
             gradients = optimizer.compute_gradients(funop, var_list = norm_vars) # compute_gradients return a list of (gradient, variable) pairs.
         normalizer_accumulate_gradients_op = [ac.assign_add(gg[0]) for ac, gg in zip(normalizer_accumulated_gradients, gradients)]
    
         # define the gradient mean op
         num_accumulation_steps_pl = tf.placeholder(dtype=tf.float32, name = 'num_accumulation_steps')
         normalizer_accumulated_gradients_mean_op = [ag.assign(tf.divide(ag, num_accumulation_steps_pl)) for ag in normalizer_accumulated_gradients]
    
         # reassemble the gradients in the [value, var] format and do define train op
         final_gradients = [(ag, gg[1]) for ag, gg in zip(normalizer_accumulated_gradients, gradients)]
         normalizer_update_op = optimizer.apply_gradients(final_gradients)
         
     else:
         normalizer_accumulated_gradients_zero_op = 0
         normalizer_accumulate_gradients_op = 0
         normalizer_accumulated_gradients_mean_op = 0
         num_accumulation_steps_pl = 0
         normalizer_update_op = 0         
         
     # ================================================================
     # sequence of running optimization ops:
     # 1. at the start of each epoch, run normalizer_accumulated_gradients_zero_op (no need to provide values for any placeholders)
     # 2. in each training iteration, run normalizer_accumulate_gradients_op with regular feed dict of inputs and outputs
     # 3. at the end of the epoch (after all batches of the volume have been passed), run normalizer_accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl
     # 4. finally, run the normalizer_update_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
     # ================================================================
     
     # ==============================================================================     
     # start session
     # ==============================================================================
     sess = tf.InteractiveSession()
     sess.run(tf.global_variables_initializer())
     print("Initialized parameters")
     saver = tf.train.Saver(var_list = vae_vars)
     
     # ==============================================================================
     # do post-training predictions
     # ==============================================================================     
     modelpath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/domain_shift_unsupervised/code/v2.0/'
     modelpath = modelpath + 'trainedmodel/cvae_MSJhalf_40chunks_fcl' + str(fcl_dim)
     modelpath = modelpath +  '_lat' + str(lat_dim) + '_ns' + str(noisy) + '_ps' + str(patchsize) + '_step' + str(500000)
     saver.restore(sess, modelpath)
     print("Loaded the new model, trained patchwise on the 40 chunk dataset.")
                                                                     
     return (x_rec,
             x_inp,
             funop,
             grd0,
             sess,
             grd_p_x_z0,
             grd_p_z0,
             grd_q_z_x0,
             grd20,
             y_out,
             y_out_prec,
             z_std_multip,
             op_q_z_x,
             mu,
             std,
             grd_q_zpl_x_az0,
             op_q_zpl_x,
             z_pl,
             z,
             normalizer_accumulated_gradients_zero_op,
             normalizer_accumulate_gradients_op,
             normalizer_accumulated_gradients_mean_op,
             num_accumulation_steps_pl,
             normalizer_update_op,
             x_normalized)