from tensorflow.contrib.data.python.ops import sliding
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.framework import ops
from os import listdir
from os.path import isfile, join
from datetime import datetime


now = datetime.utcnow().strftime("%Y%m%d%H%M%S")    #for tensorboard
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)
root_logdir_m = "tf_models"
logdir_m = "{}/run-{}/".format(root_logdir_m, now)

def _parse_function(example_proto):
    features = {
                "image_y": tf.FixedLenFeature((), tf.string )
                }

    parsed_features = tf.parse_single_example(example_proto, features)

    image_y = tf.decode_raw(parsed_features["image_y"],  tf.float64)

    image_y = tf.reshape(image_y, [256,256,1])
    image_y = tf.cast(image_y,dtype=tf.float32)

    return image_y

def conv_blc_lst(pixel,kernel_size,filter_numbers,stride,nonlinearity,conv_t):
    with tf.name_scope("conv") as scope:
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]

        if conv_t== "conv":
            kernel_d = pixel.get_shape().as_list()[3]
            kernel_o = filter_numbers
        if conv_t == "dconv":
            kernel_d = filter_numbers
            kernel_o = pixel.get_shape().as_list()[3]

        W = tf.get_variable('Weights', (kernel_h, kernel_w, kernel_d, kernel_o),
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        if conv_t == "conv":
            conv_out = tf.nn.conv2d(pixel, W, strides=stride, padding="VALID", name="conv")
        if conv_t == "dconv":
            out_shape_list = pixel.get_shape().as_list()
            out_shape_list[1] = ((pixel.get_shape().as_list()[1] + 1) * 2)
            out_shape_list[2] = ((pixel.get_shape().as_list()[2] + 1) * 2)
            out_shape_list[3] = filter_numbers
            out_shape = tf.constant(out_shape_list)
            conv_out = tf.nn.conv2d_transpose(pixel,W,out_shape,stride,padding="VALID",name="dconv")

        B = tf.get_variable('Biases',(1,1,1,conv_out.get_shape()[3]),
                            initializer=tf.constant_initializer(.01))

        normalized_out = tf.add(conv_out,B)

        if nonlinearity=="relu":
            up_pixel = tf.nn.relu(normalized_out, name="relu")
        elif nonlinearity=="leaky_relu":
            up_pixel = tf.nn.leaky_relu(normalized_out, name="leaky_relu")
        elif nonlinearity=="none":
            up_pixel = normalized_out

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("activations", up_pixel)

        return up_pixel



def conv_block(pixel,kernel_size,filter_numbers,stride,nonlinearity,conv_t):
    with tf.name_scope("conv") as scope:
        kernel_h = kernel_size[0]
        kernel_w = kernel_size[1]

        if conv_t== "conv":
            kernel_d = pixel.get_shape().as_list()[3]
            kernel_o = filter_numbers
        if conv_t == "dconv":
            kernel_d = filter_numbers
            kernel_o = pixel.get_shape().as_list()[3]

        W = tf.get_variable('Weights', (kernel_h, kernel_w, kernel_d, kernel_o),
                            initializer=tf.contrib.layers.variance_scaling_initializer())
        if conv_t == "conv":
            conv_out = tf.nn.conv2d(pixel, W, strides=stride, padding="VALID", name="conv")
        if conv_t == "dconv":
            out_shape_list = pixel.get_shape().as_list()
            out_shape_list[1] = ((pixel.get_shape().as_list()[1] + 1) * 2)-1
            out_shape_list[2] = ((pixel.get_shape().as_list()[2] + 1) * 2)-1
            out_shape_list[3] = filter_numbers
            out_shape = tf.constant(out_shape_list)
            conv_out = tf.nn.conv2d_transpose(pixel,W,out_shape,stride,padding="VALID",name="dconv")

        B = tf.get_variable('Biases',(1,1,1,conv_out.get_shape()[3]),
                            initializer=tf.constant_initializer(.01))

        normalized_out = tf.add(conv_out,B)

        if nonlinearity=="relu":
            up_pixel = tf.nn.relu(normalized_out, name="relu")
        elif nonlinearity=="leaky_relu":
            up_pixel = tf.nn.leaky_relu(normalized_out, name="leaky_relu")
        elif nonlinearity=="none":
            up_pixel = tf.nn.sigmoid(normalized_out, name="sigmoid")

        tf.summary.histogram("weights", W)
        tf.summary.histogram("biases", B)
        tf.summary.histogram("activations", up_pixel)

        return up_pixel



def encoder(pix,skip):
    with tf.variable_scope("encode",reuse=tf.AUTO_REUSE) as scope:

        non_lin = "relu"

        with tf.variable_scope("conv1") as scope:
            pixel_1 = conv_block(pix,[3,3],filter_numbers=4,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

        with tf.variable_scope("conv2") as scope:
            pixel_2 = conv_block(pixel_1,[3,3],filter_numbers=8,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

        with tf.variable_scope("conv3") as scope:
            pixel_3 = conv_block(pixel_2,[3,3],filter_numbers=16,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

        with tf.variable_scope("conv4") as scope:
            pixel_4 = conv_block(pixel_3,[3,3],filter_numbers=32,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

        with tf.variable_scope("conv5") as scope:
            pixel_5 = conv_block(pixel_4,[3,3],filter_numbers=48,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="conv")

        if skip=="okay":
            return pix,pixel_1,pixel_2,pixel_3,pixel_4,pixel_5
        elif skip=="nope" :
            return pixel_5

def concat(encode_pix,decode_pix):
    with tf.name_scope("concat") as scope:
        up_pixel = tf.concat([encode_pix, decode_pix], axis=3)
        return up_pixel

def decoder_skip(pix_lstm,pix_5,pix_4,pix_3,pix_2,pix_1,pix,skip = "okay"):
    with tf.variable_scope("decode_skip",reuse=tf.AUTO_REUSE) as scope:

        non_lin = "leaky_relu"

        if skip=="okay":
            pix_lstm = concat(pix_5,pix_lstm)
        elif skip=="nope":
            pix_lstm = pix_lstm

        with tf.variable_scope("dconv1") as scope:
            pixel_1 = conv_block(pix_lstm,[3,3],filter_numbers=64,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        #if skip=="okay":
        #    pixel_1 = concat(pix_4,pixel_1)

        with tf.variable_scope("dconv2") as scope:
            pixel_2 = conv_block(pixel_1,[3,3],filter_numbers=32,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        if skip=="okay":
            pixel_2 = concat(pix_3,pixel_2)
        elif skip=="nope":
            pixel_2 = pixel_2

        with tf.variable_scope("dconv3") as scope:
            pixel_3 = conv_block(pixel_2,[3,3],filter_numbers=16,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        #if skip=="okay":
        #    pixel_3 = concat(pix_2,pixel_3)

        with tf.variable_scope("dconv4") as scope:
            pixel_4 = conv_block(pixel_3,[3,3],filter_numbers=8,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        if skip=="okay":
            pixel_4 = concat(pix_1,pixel_4)
        elif skip=="nope":
            pixel_4 = pixel_4

        with tf.variable_scope("dconv5",reuse=tf.AUTO_REUSE) as scope:
            pixel_5 = conv_blc_lst(pixel_4,[3,3],filter_numbers=1,stride=[1,2,2,1],nonlinearity="none",conv_t="dconv")




        return pixel_5


def decoder_norm(pix_lstm):
    with tf.variable_scope("decode_norm",reuse=tf.AUTO_REUSE) as scope:

        non_lin = "leaky_relu"

        with tf.variable_scope("dconv1") as scope:
            pixel_1 = conv_block(pix_lstm,[3,3],filter_numbers=32,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        with tf.variable_scope("dconv2") as scope:
            pixel_2 = conv_block(pixel_1,[3,3],filter_numbers=16,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        with tf.variable_scope("dconv3") as scope:
            pixel_3 = conv_block(pixel_2,[3,3],filter_numbers=8,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        with tf.variable_scope("dconv4") as scope:
            pixel_4 = conv_block(pixel_3,[3,3],filter_numbers=4,stride=[1,2,2,1],nonlinearity=non_lin,conv_t="dconv")
        with tf.variable_scope("dconv5") as scope:
            pixel_5 = conv_blc_lst(pixel_4,[3,3],filter_numbers=1,stride=[1,2,2,1],nonlinearity="none",conv_t="dconv")
            #pixel_5 = conv_block(pixel_4,[3,3],filter_numbers=1,stride=[1,2,2,1],nonlinearity="none",conv_t="dconv")


        return pixel_5

def encoder_block(pix_1,pix_2,pix_3,pix_4):

    with tf.name_scope("encode1") as scope:
        out_1 = encoder(pix_1,skip="nope")

    with tf.name_scope("encode2") as scope:
        out_2 = encoder(pix_2,skip="nope")

    with tf.name_scope("encode3") as scope:
        out_3 = encoder(pix_3,skip="nope")

    #with tf.name_scope("encode_1") as scope:
    #    out_1 = encoder(pix_1,skip="nope")

    with tf.name_scope("encode4") as scope:
        two_pix,two_pixel_1,two_pixel_2,two_pixel_3,two_pixel_4,two_pixel_5 = encoder(pix_4,skip="okay")
        out_4 = two_pixel_5

    return out_1,out_2,out_3,out_4,two_pix,two_pixel_1,two_pixel_2,two_pixel_3,two_pixel_4,two_pixel_5

def decoder_block(pix_lstm,pix_lstm1,pix_5,pix_4,pix_3,pix_2,pix_1,pix,skip_try):

    if skip_try=="nope":
        with tf.name_scope("decode_1") as scope:
            out_1 = decoder_norm(pix_lstm)
        with tf.name_scope("decode_2") as scope:
            out_2 = decoder_norm(pix_lstm1)

    if skip_try=="oka":
        with tf.name_scope("decode_1") as scope:
            #out_1 = decoder_skip(pix_lstm,pix_5,pix_4,pix_3,pix_2,pix_1,pix,skip = "okay")
            out_1 = decoder_norm(pix_lstm)
        with tf.name_scope("decode_2") as scope:
            #out_2 = decoder_skip(pix_lstm1,pix_5,pix_4,pix_3,pix_2,pix_1,pix,skip = "okay")
            out_2 = decoder_norm(pix_lstm1)
    return out_1,out_2


'''def decoder_block(pix_lstm,pix_5,pix_4,pix_3,pix_2,pix_1,pix,skip_try,skip = "okay"):

    if skip_try=="nope":
        with tf.name_scope("decode_1") as scope:
            out_1 = decoder_norm(pix_lstm)


    if skip_try=="oka":
        with tf.name_scope("decode_1") as scope:
            out_1 = decoder_skip(pix_lstm,pix_5,pix_4,pix_3,pix_2,pix_1,pix,skip = "okay")

    return out_1'''

def conv_lstm(pixel,state,memory):
    with tf.variable_scope("conv_lstm_func",reuse=tf.AUTO_REUSE) as scope:
        kernel_h_x = 3 #
        kernel_w_x = 3
        kernel_d_x = pixel.get_shape().as_list()[3]
        kernel_o_x = pixel.get_shape().as_list()[3]

        kernel_h_h = 3
        kernel_w_h = 3
        kernel_d_h = pixel.get_shape().as_list()[3]
        kernel_o_h = pixel.get_shape().as_list()[3]

        kernel_h_c = memory.get_shape().as_list()[0]
        kernel_w_c = memory.get_shape().as_list()[1]
        kernel_d_c = memory.get_shape().as_list()[2]
        kernel_o_c = memory.get_shape().as_list()[3]


        with tf.name_scope("input_gate") as scope:

            W_ix = tf.get_variable('Weights_ix',(kernel_h_x, kernel_w_x, kernel_d_x, kernel_o_x),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_ih = tf.get_variable('Weights_ih', (kernel_h_h, kernel_w_h, kernel_d_h, kernel_o_h),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_ic = tf.get_variable('Weights_ic', (kernel_h_c, kernel_w_c, kernel_d_c, kernel_o_c),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())

            B_i = tf.get_variable('Biases_i',(1,1,1,pixel.get_shape()[3]),
                                    initializer=tf.constant_initializer(.01))
            one_i = tf.nn.conv2d(pixel,W_ix,strides=[1,1,1,1],padding="SAME",name="convi_1")
            two_i = tf.nn.conv2d(state,W_ih,strides=[1,1,1,1],padding="SAME",name="convi_2")

            input_gate = tf.sigmoid(tf.add(tf.add(tf.add(one_i,two_i),B_i),tf.multiply(W_ic,memory)))

        with tf.name_scope("forget_gate") as scope:

            W_fx = tf.get_variable('Weights_fx',(kernel_h_x, kernel_w_x, kernel_d_x, kernel_o_x),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_fh = tf.get_variable('Weights_fh', (kernel_h_h, kernel_w_h, kernel_d_h, kernel_o_h),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_fc = tf.get_variable('Weights_fc', (kernel_h_c, kernel_w_c, kernel_d_c, kernel_o_c),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())

            B_f = tf.get_variable('Biases_f',(1,1,1,pixel.get_shape()[3]),
                                    initializer=tf.constant_initializer(.01))
            one_f = tf.nn.conv2d(pixel,W_fx,strides=[1,1,1,1],padding="SAME",name="convf_1")
            two_f = tf.nn.conv2d(state,W_fh,strides=[1,1,1,1],padding="SAME",name="convf_2")

            forget_gate = tf.sigmoid(tf.add(tf.add(tf.add(one_f,two_f),B_f),tf.multiply(W_fc,memory)))



        with tf.name_scope("memory_cand") as scope:

            W_cx = tf.get_variable('Weights_cx',(kernel_h_x, kernel_w_x, kernel_d_x, kernel_o_x),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_ch = tf.get_variable('Weights_ch', (kernel_h_h, kernel_w_h, kernel_d_h, kernel_o_h),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())

            B_c = tf.get_variable('Biases_c',(1,1,1,pixel.get_shape()[3]),
                                    initializer=tf.constant_initializer(.01))
            one_c = tf.nn.conv2d(pixel,W_cx,strides=[1,1,1,1],padding="SAME",name="convc_1")
            two_c = tf.nn.conv2d(state,W_ch,strides=[1,1,1,1],padding="SAME",name="convc_2")

            memory_cand = tf.nn.tanh(tf.add(tf.add(one_c,two_c),B_c))

        memory_curr = tf.add(tf.multiply(forget_gate,memory),tf.multiply(input_gate,memory_cand))

        with tf.name_scope("output_gate") as scope:

            W_ox = tf.get_variable('Weights_ox',(kernel_h_x, kernel_w_x, kernel_d_x, kernel_o_x),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_oh = tf.get_variable('Weights_oh', (kernel_h_h, kernel_w_h, kernel_d_h, kernel_o_h),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())
            W_oc = tf.get_variable('Weights_oc', (kernel_h_c, kernel_w_c, kernel_d_c, kernel_o_c),
                                    initializer=tf.contrib.layers.variance_scaling_initializer())

            B_o = tf.get_variable('Biases_o',(1,1,1,pixel.get_shape()[3]),
                                    initializer=tf.constant_initializer(.01))
            one_o = tf.nn.conv2d(pixel,W_ox,strides=[1,1,1,1],padding="SAME",name="convo_1")
            two_o = tf.nn.conv2d(state,W_oh,strides=[1,1,1,1],padding="SAME",name="convo_2")

            output_gate = tf.sigmoid(tf.add(tf.add(tf.add(one_o,two_o),B_o),
                          tf.multiply(W_oc,memory_curr)))


        state_curr = tf.multiply(output_gate,tf.nn.tanh(memory_curr))
        #pixel_out = tf.sigmoid(state_curr)

        tf.summary.histogram("weights_input_gate_input", W_ix)
        tf.summary.histogram("weights_input_gate_state", W_ih)
        tf.summary.histogram("weights_memory_input", W_ic)
        tf.summary.histogram("biases_input_gate", B_i)
        #tf.summary.histogram("activations_input_gate", input_gate)

        tf.summary.histogram("weights_forget_gate_input", W_fx)
        tf.summary.histogram("weights_forget_gate_state", W_fh)
        tf.summary.histogram("weights_memory_forget", W_fc)
        tf.summary.histogram("biases_forget_gate", B_f)
        #tf.summary.histogram("activations_forget_gate", forget_gate)

        tf.summary.histogram("weights_output_gate_input", W_ox)
        tf.summary.histogram("weights_output_gate_state", W_oh)
        tf.summary.histogram("weights_memory_output", W_oc)
        tf.summary.histogram("biases_output_gate", B_o)
        #tf.summary.histogram("activations_output_gate", output_gate)

        tf.summary.histogram("weights_memory_cand_input", W_cx)
        tf.summary.histogram("weights_memory_cand_state", W_ch)
        tf.summary.histogram("biases_memory_cand", B_c)

        return state_curr,memory_curr


def lstm_block(pixel_0,pixel_1,pixel_2,pixel_3,pixel_4):

    gt_5 = encoder(pixel_4,skip="nope")

    #with tf.variable_scope("conv_lstm") as scope: #excluding reuse

    with tf.name_scope("convlstm_1") as scope:
        null_state = tf.zeros([pixel_0.get_shape()[0],pixel_0.get_shape()[1],
                     pixel_0.get_shape()[2],pixel_0.get_shape()[3]])
        null_mem = tf.zeros([pixel_0.get_shape()[0],pixel_0.get_shape()[1],
                   pixel_0.get_shape()[2],pixel_0.get_shape()[3]])

        state_1,memory_1 = conv_lstm(pixel_0,null_state,null_mem)

    with tf.name_scope("convlstm_2") as scope:

        state_2,memory_2 = conv_lstm(pixel_1,state_1,memory_1)

    with tf.name_scope("convlstm_3") as scope:

        state_3,memory_3 = conv_lstm(pixel_2,state_2,memory_2)

    with tf.name_scope("convlstm_4") as scope:

        state_4,memory_4 = conv_lstm(pixel_3,state_3,memory_3)

    with tf.name_scope("convlstm_5") as scope:
        #null_pixel = tf.zeros([pixel_0.get_shape()[0],pixel_0.get_shape()[1],pixel_0.get_shape()[2],pixel_0.get_shape()[3]])
        state_5,memory_5 = conv_lstm(state_4,state_4,memory_4)

    with tf.name_scope("convlstm_6") as scope:

        state_6,memory_6 = conv_lstm(gt_5,state_5,memory_5)

    return state_5,state_6

'''def cost(pixel_pre,pixel_gt):
    with tf.name_scope("cost") as scope:
        loss = tf.losses.absolute_difference(pixel_gt,pixel_pre,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        tf.summary.scalar('loss',loss)


    return loss
'''

def cost(pixel_pre1,pixel_pre2,pixel_gt1,pixel_gt2,pera_1,pera_2):
    with tf.name_scope("cost") as scope:
        loss1 = tf.losses.absolute_difference(pixel_gt1,pixel_pre1,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        loss2 = tf.losses.absolute_difference(pixel_gt2,pixel_pre2,reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        #loss1 = loss1**2
        #loss2 = loss2**2
        #loss1 = tf.losses.mean_squared_error(pixel_gt1,pixel_pre1)
        #loss2 = tf.losses.mean_squared_error(pixel_gt2,pixel_pre2)

        total_loss = (pera_1*loss1 + pera_2*loss2)/(pera_1+pera_2)

        tf.summary.scalar('loss',total_loss)


        return total_loss

def model(learning_rate,num_epochs,mini_size,pt_out,break_t,fil_conv,kernel_ls,decode_l,
            pera_1,pera_2,imp_skip,batch):

    tf.summary.scalar('learning_rate',learning_rate)
    tf.summary.scalar('batch_size',mini_size)
    tf.summary.scalar('epoch_num',num_epochs)
    tf.summary.scalar('out_step',pt_out)
    tf.summary.scalar('training_break',break_t)
    tf.summary.scalar('conv_filter_numbers',fil_conv)
    tf.summary.scalar('kernel_sizes_lstm',kernel_ls)
    tf.summary.scalar('number_of_prediction',decode_l)
    tf.summary.scalar('batch',batch)



    filenames = tf.placeholder(tf.string)

    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(_parse_function)
    window = mini_size
    #stride = 1
    dataset = dataset.apply(sliding.sliding_window_batch(window,stride=6))
    dataset = dataset.batch(batch,drop_remainder=True)
    dataset = dataset.shuffle(500)
    dataset = dataset.repeat(num_epochs)
    #dataset = dataset.shuffle(500)
    #iterator =  tf.data.Iterator.from_structure(dataset.output_types,dataset.output_shapes)
    iterator = dataset.make_initializable_iterator(shared_name="iter")

    pix_gt = iterator.get_next()
    #print(pix_gt.get_shape().as_list())
    #print(tf.shape(pix_gt))
    spli0, spli1, spli2, spli3, spli4, spli5 = tf.split(pix_gt,num_or_size_splits=batch,axis=0)

    spli0 = tf.reshape(spli0,[mini_size,256,256,1])
    spli1 = tf.reshape(spli1,[mini_size,256,256,1])
    spli2 = tf.reshape(spli2,[mini_size,256,256,1])
    spli3 = tf.reshape(spli3,[mini_size,256,256,1])
    spli4 = tf.reshape(spli4,[mini_size,256,256,1])
    spli5 = tf.reshape(spli5,[mini_size,256,256,1])

    pix_gt1 = tf.stack([spli0, spli1, spli2, spli3, spli4, spli5], axis=1)
    split0, split1, split2, split3, split4, split5 = tf.split(pix_gt1,
                                        num_or_size_splits=mini_size, axis=0)

    pix_0 = tf.reshape(split0,[batch,256,256,1])
    pix_1 = tf.reshape(split1,[batch,256,256,1])
    pix_2 = tf.reshape(split2,[batch,256,256,1])
    pix_3 = tf.reshape(split3,[batch,256,256,1])
    pix_4 = tf.reshape(split4,[batch,256,256,1])
    pix_5 = tf.reshape(split5,[batch,256,256,1])

    tf.summary.image("input_1",pix_0,3)
    tf.summary.image("input_2",pix_1,3)
    tf.summary.image("input_3",pix_2,3)
    tf.summary.image("input_4",pix_3,3)
    tf.summary.image("input_5",pix_4,3)
    tf.summary.image("input_6",pix_5,3)

    out_1,out_2,out_3,out_4,two_pix,two_pixel_1,two_pixel_2,two_pixel_3,two_pixel_4,two_pixel_5 = encoder_block(pix_0,pix_1,pix_2,pix_3)

    #out_5 = lstm_block(out_1,out_2,out_3,out_4,)
    out_5,out_6 = lstm_block(out_1,out_2,out_3,out_4,pix_4)

    out_pre,out_pre1 = decoder_block(out_5,out_6,two_pixel_5,two_pixel_4,two_pixel_3,two_pixel_2,two_pixel_1,two_pix,
                          skip_try="oka")

    tf.summary.image("prediction1",out_pre,3)
    tf.summary.image("prediction2",out_pre1,3)

    #loss = cost(pixel_pre = out_pre , pixel_gt = pix_4)

    loss = cost(out_pre,out_pre1,pix_4,pix_5,pera_1,pera_2)

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name="adam").minimize(loss)

    merge_sum = tf.summary.merge_all()
    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    saver = tf.train.Saver()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))


    sess.run(init)
    #saver.restore(sess,('/media/antor/Files/main_projects/gitlab/unet_check/tf_models/run-20181104123103/my_model.ckpt'))

    sess.run(iterator.initializer,feed_dict={filenames:"/media/antor/Files/ML/tfrecord/sir_demo/train.tfrecords"})

    mini_cost = 0.0
    counter = 1
    coun = 1
    epoch_cost = 0.0
    epoch = 0
    '''

    while True:
        try:

            if coun in range(imp_skip,31):
                dump_loss = sess.run(loss)
                #print(coun)
                if coun==30:
                    coun=0
                coun+=1

            else:
                _ , temp_cost = sess.run([optimizer,loss])
                mini_cost += temp_cost/pt_out
                epoch_cost += temp_cost/8200

                if counter%pt_out==0:
                    print("mini batch cost of batch " + str(counter) + " is : " + str(mini_cost))
                    mini_cost =0.0



                if counter%100 == 0:
                    s = sess.run(merge_sum)
                    file_writer.add_summary(s,counter)

                #if counter*mini_size>=break_t:
                 #   break

                if counter%10000==0:
                    print("cost after epoch " + str(epoch) + ": " + str(epoch_cost))
                    saver.save(sess,logdir_m+"my_model.ckpt")
                    epoch_cost =0.0
                    epoch+=1

                counter+=1
                #print(coun)
                coun+=1

        except tf.errors.OutOfRangeError:
            print(counter)
            saver.save(sess,logdir_m+"my_model.ckpt")
            break

     '''

    while True:
        try:
            '''
            #if coun in range(imp_skip,31):
            #if coun==5:

                dump_loss = sess.run(loss)
                #print(coun)
                #if coun==30:
                coun=0
                coun+=1

            else:
            '''
            _ , temp_cost = sess.run([optimizer,loss])
            mini_cost += temp_cost/pt_out
            epoch_cost += temp_cost/288

            # if counter%pt_out==0:
            #     print("mini batch cost of batch " + str(counter) + " is : " + str(mini_cost))
            #     mini_cost =0.0



            if counter%288== 0:
                s = sess.run(merge_sum)
                file_writer.add_summary(s,counter)

            #if counter*mini_size>=break_t:
             #   break

            if counter%288==0:
                print("cost after epoch " + str(epoch) + ": " + str(epoch_cost))
                #saver.save(sess,logdir_m+"my_model.ckpt")
                epoch_cost =0.0
                epoch+=1

            counter+=1
            #print(coun)
            coun+=1




        except tf.errors.OutOfRangeError:
            #print(counter)
            #saver.save(sess,logdir_m+"my_model.ckpt")
            break
    sess.close()



#import numpy as np
'''from tensorflow.python.framework import ops
f = np.random.uniform(np.log10(.001),np.log10(.00001),6)
print(f)
i = 10**f

print(i)

for l in i:

    #print(k)
    print(l)

    model(learning_rate=l,num_epochs=3,mini_size=6,pt_out=200,break_t=1000,fil_conv=32,kernel_ls=3,decode_l=2,
       pera_1=1,pera_2=1,imp_skip=26,batch=6)
    ops.reset_default_graph()
'''

model(learning_rate=.0005,num_epochs=5,mini_size=6,pt_out=100,break_t=1000,fil_conv=48,kernel_ls=3,decode_l=2,
       pera_1=1,pera_2=1,imp_skip=26,batch=6)
