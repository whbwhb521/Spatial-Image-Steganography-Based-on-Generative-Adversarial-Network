import pickle
import scipy.misc
import matplotlib
import matplotlib.image
from SomeAction import *
from scipy import ndimage
import random
import math


# 卷积
def conv2d(x, W, strides=2, padding='SAME', name=None):
    return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding, name=name)


# 反卷积
def deconv2d(x, W, output_shape, strides=2, padding='SAME', name=None):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=[1, strides, strides, 1], padding=padding,
                                  name=name)


def leakrelu(x, leak):
    f1 = 0.5 * (1 + leak)
    f2 = 0.5 * (1 - leak)
    return f1 * x + f2 * abs(x)


def generate(input, isTrain):
    with tf.variable_scope('gen') as scope:
        x_image = tf.reshape(input, [-1, 512, 512, 1])
        # 1
        Kernel = tf.Variable(tf.random_uniform([3, 3, 1, 16], -0.1, 0.1))
        con_image1 = conv2d(x_image, Kernel)
        bn_1 = tf.layers.batch_normalization(con_image1, training=isTrain)
        RL_1 = leakrelu(bn_1, 0.2)
        # 2
        Kerne2 = tf.Variable(tf.random_uniform([3, 3, 16, 32], -0.1, 0.1))
        con_image2 = conv2d(RL_1, Kerne2)
        bn_2 = tf.layers.batch_normalization(con_image2, training=isTrain)
        RL_2 = leakrelu(bn_2, 0.2)
        # 3
        Kerne3 = tf.Variable(tf.random_uniform([3, 3, 32, 64], -0.1, 0.1))
        con_image3 = conv2d(RL_2, Kerne3)
        bn_3 = tf.layers.batch_normalization(con_image3, training=isTrain)
        RL_3 = leakrelu(bn_3, 0.2)
        # 4
        Kerne4 = tf.Variable(tf.random_uniform([3, 3, 64, 128], -0.1, 0.1))
        con_image4 = conv2d(RL_3, Kerne4)
        bn_4 = tf.layers.batch_normalization(con_image4, training=isTrain)
        RL_4 = leakrelu(bn_4, 0.2)
        # 5
        Kerne5 = tf.Variable(tf.random_uniform([3, 3, 128, 128], -0.1, 0.1))
        con_image5 = conv2d(RL_4, Kerne5)
        bn_5 = tf.layers.batch_normalization(con_image5, training=isTrain)
        RL_5 = leakrelu(bn_5, 0.2)
        # 6
        Kerne6 = tf.Variable(tf.random_uniform([3, 3, 128, 128], -0.1, 0.1))
        con_image6 = conv2d(RL_5, Kerne6)
        bn_6 = tf.layers.batch_normalization(con_image6, training=isTrain)
        RL_6 = leakrelu(bn_6, 0.2)
        # 7
        Kerne7 = tf.Variable(tf.random_uniform([3, 3, 128, 128], -0.1, 0.1))
        con_image7 = conv2d(RL_6, Kerne7)
        bn_7 = tf.layers.batch_normalization(con_image7, training=isTrain)
        RL_7 = leakrelu(bn_7, 0.2)
        # 8
        Kerne8 = tf.Variable(tf.random_uniform([3, 3, 128, 128], -0.1, 0.1))
        con_image8 = conv2d(RL_7, Kerne8)
        bn_8 = tf.layers.batch_normalization(con_image8, training=isTrain)
        RL_8 = leakrelu(bn_8, 0.2)
        # 9
        Kerne9 = tf.Variable(tf.random_uniform([5, 5, 128, 128], -0.1, 0.1))
        con_image9 = deconv2d(RL_8, Kerne9, [BATCHSIZE, 4, 4, 128])
        bn_9 = tf.layers.batch_normalization(con_image9, training=isTrain)
        RL_9 = tf.nn.relu(bn_9)
        Con_9 = tf.concat([RL_7, RL_9], axis=-1)
        # 10
        Kerne10 = tf.Variable(tf.random_uniform([5, 5, 128, 256], -0.1, 0.1))
        con_image10 = deconv2d(Con_9, Kerne10, [BATCHSIZE, 8, 8, 128])
        bn_10 = tf.layers.batch_normalization(con_image10, training=isTrain)
        RL_10 = tf.nn.relu(bn_10)
        Con_10 = tf.concat([RL_6, RL_10], axis=-1)
        # 11
        Kerne11 = tf.Variable(tf.random_uniform([5, 5, 128, 256], -0.1, 0.1))
        con_image11 = deconv2d(Con_10, Kerne11, [BATCHSIZE, 16, 16, 128])
        bn_11 = tf.layers.batch_normalization(con_image11, training=isTrain)
        RL_11 = tf.nn.relu(bn_11)
        Con_11 = tf.concat([RL_5, RL_11], axis=-1)
        # 12
        Kerne12 = tf.Variable(tf.random_uniform([5, 5, 128, 256], -0.1, 0.1))
        con_image12 = deconv2d(Con_11, Kerne12, [BATCHSIZE, 32, 32, 128])
        bn_12 = tf.layers.batch_normalization(con_image12, training=isTrain)
        RL_12 = tf.nn.relu(bn_12)
        Con_12 = tf.concat([RL_4, RL_12], axis=-1)
        # 13
        Kerne13 = tf.Variable(tf.random_uniform([5, 5, 64, 256], -0.1, 0.1))
        con_image13 = deconv2d(Con_12, Kerne13, [BATCHSIZE, 64, 64, 64])
        bn_13 = tf.layers.batch_normalization(con_image13, training=isTrain)
        RL_13 = tf.nn.relu(bn_13)
        Con_13 = tf.concat([RL_3, RL_13], axis=-1)
        # 14
        Kerne14 = tf.Variable(tf.random_uniform([5, 5, 32, 128], -0.1, 0.1))
        con_image14 = deconv2d(Con_13, Kerne14, [BATCHSIZE, 128, 128, 32])
        bn_14 = tf.layers.batch_normalization(con_image14, training=isTrain)
        RL_14 = tf.nn.relu(bn_14)
        Con_14 = tf.concat([RL_2, RL_14], axis=-1)
        # 15
        Kerne15 = tf.Variable(tf.random_uniform([5, 5, 16, 64], -0.1, 0.1))
        con_image15 = deconv2d(Con_14, Kerne15, [BATCHSIZE, 256, 256, 16])
        bn_15 = tf.layers.batch_normalization(con_image15, training=isTrain)
        RL_15 = tf.nn.relu(bn_15)
        Con_15 = tf.concat([RL_1, RL_15], axis=-1)
        # 16
        Kerne16 = tf.Variable(tf.random_uniform([5, 5, 1, 32], -0.1, 0.1))
        con_image16 = deconv2d(Con_15, Kerne16, [BATCHSIZE, 512, 512, 1])
        bn_16 = tf.layers.batch_normalization(con_image16, training=isTrain)
        RL_16 = tf.nn.relu(bn_16)
        # 17
        RL_17 = tf.nn.relu(tf.nn.sigmoid(RL_16) - 0.5)

        return RL_17


def Tanh(input, BATCHSIZE):
    with tf.variable_scope('tanh') as scope:
        lemda = 1000
        N = tf.random_uniform(shape=[BATCHSIZE, 512, 512, 1], minval=0, maxval=1, dtype=tf.float32)
        M = -0.5 * tf.nn.tanh(lemda * (input - 2 * N)) + 0.5 * tf.nn.tanh(lemda * (input - 2 * (1 - N)))
        return M


def discrimation(input, prob, isTrain):
    x_image = tf.reshape(input, [-1, 512, 512, 1])
    #prob_image=tf.reshape(prob,[-1,512,512,1])
    hpf = np.zeros([5, 5, 1, 10], dtype=np.float32)  # [height,width,input,output]
    hpf[:, :, 0, 0] = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]],
        dtype=np.float32) / (12 * 255)
    hpf[:, :, 0, 1] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=np.float32) / (255)
    hpf[:, :, 0, 2] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=np.float32) / (255)
    hpf[:, :, 0, 3] = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=np.float32) / (2 * 255)
    hpf[:, :, 0, 4] = np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, -2, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=np.float32) / (2 * 255)
    hpf[:, :, 0, 5] = np.array(
        [[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0], [0, 0, 0, 0, 0]],
        dtype=np.float32) / (4 * 255)
    hpf[:, :, 0, 6] = np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=np.float32) / (4 * 255)
    hpf[:, :, 0, 7] = np.array([[0, 0, 0, 0, 0], [0, -1, 2, 0, 0], [0, 2, -4, 0, 0], [0, -1, 2, 0, 0], [0, 0, 0, 0, 0]],
                               dtype=np.float32) / (4 * 255)
    hpf[:, :, 0, 8] = np.array(
        [[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
        dtype=np.float32) / (12 * 255)
    hpf[:, :, 0, 9] = np.array(
        [[-1, 2, -2, 0, 0], [2, -6, 8, 0, 0], [-2, 8, -12, 0, 0], [2, -6, 8, 0, 0], [-1, 2, -2, 0, 0]],
        dtype=np.float32) / (12 * 255)

    kernel0 = tf.Variable(hpf, name="kernel0")
    #kernel_pro=tf.Variable(abs(hpf),name='kernel_pro')
    con_image0 = conv2d(x_image, kernel0, strides=1, padding='SAME')
    #con_prob=conv2d(prob_image,kernel_pro,strides=1,padding='SAME')
    #cpn_start=tf.concat([con_image0, con_prob], axis=-1)
    with tf.variable_scope('dis') as scope:
        kernel1 = tf.Variable(tf.random_normal([5, 5, 10, 8], mean=0.0, stddev=0.01))
        con_image1 = tf.abs(conv2d(con_image0, kernel1, strides=1, padding='SAME'))
        bn1 = tf.layers.batch_normalization(con_image1, training=isTrain)
        th_act1 = tf.nn.tanh(bn1)
        h_pool1 = tf.nn.avg_pool(th_act1, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel2 = tf.Variable(tf.random_normal([5, 5, 8, 16], mean=0.0, stddev=0.01))
        con_image2 = conv2d(h_pool1, kernel2, strides=1, padding='SAME')
        bn2 = tf.layers.batch_normalization(con_image2, training=isTrain)
        th_act2 = tf.nn.tanh(bn2)
        h_pool2 = tf.nn.avg_pool(th_act2, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel3 = tf.Variable(tf.random_normal([1, 1, 16, 32], mean=0.0, stddev=0.01))
        con_image3 = conv2d(h_pool2, kernel3, strides=1, padding='SAME')
        bn3 = tf.layers.batch_normalization(con_image3, training=isTrain)
        th_act3 = tf.nn.tanh(bn3)
        h_pool3 = tf.nn.avg_pool(th_act3, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel4 = tf.Variable(tf.random_normal([1, 1, 32, 64], mean=0.0, stddev=0.01))
        con_image4 = conv2d(h_pool3, kernel4, strides=1, padding='SAME')
        bn4 = tf.layers.batch_normalization(con_image4, training=isTrain)
        th_act4 = tf.nn.tanh(bn4)
        h_pool4 = tf.nn.avg_pool(th_act4, ksize=[1, 5, 5, 1], strides=[1, 2, 2, 1], padding='SAME')

        kernel5 = tf.Variable(tf.random_normal([1, 1, 64, 128], mean=0.0, stddev=0.01))
        con_image5 = conv2d(h_pool4, kernel5, strides=1, padding='SAME')
        bn5 = tf.layers.batch_normalization(con_image5, training=isTrain)
        th_act5 = tf.nn.tanh(bn5)
        h_pool5 = tf.nn.avg_pool(th_act5, ksize=[1, 32, 32, 1], strides=[1, 1, 1, 1], padding='VALID')
        h_pool5_flat = tf.reshape(h_pool5, [-1, 128])
        weights = tf.Variable(tf.random_normal([128, 2], mean=0.0, stddev=0.01), name="weights")
        bias = tf.Variable(tf.random_normal([2], mean=0.0, stddev=0.01), name="bias")
        y_ = tf.matmul(h_pool5_flat, weights) + bias
        return y_


def read_image_shufft(PATH):
    original_pic = []
    for i in os.listdir(PATH):
        original_pic.append(PATH + '\\' + i)
    random.seed(1234)
    random.shuffle(original_pic)
    return original_pic


def read_image(PATH):
    original_pic = []
    name=[]
    for i in os.listdir(PATH):
        name.append(i)
        original_pic.append(PATH + '\\' + i)
    return original_pic,name


Path1 = r'D:\GoogleDownload\val_large\cover'
Path2 = r'D:\GoogleDownload\BOSSbase_1.01\test\original'

BATCHSIZE = 5
isTrain = False
epoches = 92

input_image = tf.placeholder(dtype=tf.float32, shape=[None, 512, 512, 1])
isTraining = tf.placeholder(dtype=tf.bool)
probab_map = generate(input_image, isTraining)
modifi_map = Tanh(probab_map, BATCHSIZE)
modifi_map=tf.round(modifi_map)
stego_image = tf.clip_by_value(input_image + modifi_map,0,255)
if isTrain:

    x = tf.concat([input_image, stego_image], 0)
    y_array = np.zeros([BATCHSIZE * 2, 2], dtype=np.float32)
    for i in range(0, BATCHSIZE):
        y_array[i, 1] = 1
    for i in range(BATCHSIZE, BATCHSIZE * 2):
        y_array[i, 0] = 1
    y = tf.constant(y_array)
    y_ = discrimation(x,probab_map,isTraining)

    proChangeP = probab_map / 2.0 + 0.00000001
    proChangeM = probab_map / 2.0 + 0.00000001
    proUnchange = 1 - probab_map + 0.00000001
    entropy = tf.reduce_sum(
        -(proChangeP) * tf.log(proChangeP) / tf.log(2.0) - (proChangeM) * tf.log(proChangeM) / tf.log(
            2.0) - proUnchange * tf.log(proUnchange) / tf.log(2.0), reduction_indices=[1, 2, 3])
    d_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
    gloss = tf.reduce_mean(tf.pow(entropy - 512 * 512 * 0.4, 2))
    LossGen = -d_loss + 0.0000001 * gloss

    vars = tf.global_variables()
    d_vars = [var for var in vars if 'dis' in var.name]
    g_vars = [var for var in vars if 'gen' in var.name]
    tes_var = [var for var in vars if 'tanh' in var.name]
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optGen = tf.train.AdamOptimizer(0.0001).minimize(LossGen, var_list=g_vars + tes_var)
        optDis = tf.train.AdamOptimizer(0.0001).minimize(d_loss, var_list=d_vars)

    image_train_road = read_image_shufft(Path1)
    image_test_road,image_test_name = read_image(Path2)
    data_x = np.zeros([BATCHSIZE, 512, 512, 1])
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        tf.summary.scalar('loss_dis', d_loss)
        tf.summary.scalar('loss_gen', LossGen)
        saver = tf.train.Saver()
        writer = tf.summary.FileWriter('./my_graph/1', sess.graph)
        summary_op = tf.summary.merge_all()

        for epoch in range(epoches):
            for i in range(int(len(image_train_road) / BATCHSIZE)):
                for j in range(BATCHSIZE):
                    imc = ndimage.imread(image_train_road[i * BATCHSIZE + j])
                    data_x[j, :, :, 0] = imc

                _, d_loss_p, summary_op_p = sess.run([optDis, d_loss, summary_op],
                                                     feed_dict={input_image: data_x, isTraining: True})
                _, LossGen_p, gloss_p = sess.run([optGen, LossGen, gloss],
                                                 feed_dict={input_image: data_x, isTraining: True})

                print('train:[%d],step:%d,d_loss:%f,g_loss:%f,gloss:%f' % (epoch, i, d_loss_p, LossGen_p, gloss_p))
                if i % 10 == 0:
                    writer.add_summary(summary_op_p, epoch * int(len(image_train_road) / BATCHSIZE) + i)

            if epoch % 10 == 1:
                saver.save(sess, './myModel/ASDLmodel' + str(epoch) + '.cptk')

            rand = np.random.randint(low=0, high=int(len(image_test_road) / BATCHSIZE))
            for j in range(BATCHSIZE):
                imc = ndimage.imread(image_test_road[rand * BATCHSIZE + j])
                data_x[j, :, :, 0] = imc

            d_loss_p, LossGen_p, gloss_p, probab_map_p, modifi_map_p = sess.run(
                [d_loss, LossGen, gloss, probab_map, modifi_map], feed_dict={input_image: data_x, isTraining: False})
            print('test:[%d],step:#,d_loss:%f,g_loss:%f,gloss:%f' % (epoch, d_loss_p, LossGen_p, gloss_p))
            probab_map_p = np.asarray(probab_map_p)
            probab_map_p = np.reshape(probab_map_p, newshape=[-1, 512, 512])
            modifi_map_p = np.reshape(modifi_map_p, newshape=[-1, 512, 512])
            modifi_map_p = np.asarray(abs(modifi_map_p))

            for k in range(BATCHSIZE):
                scipy.misc.toimage(probab_map_p[k]).save(
                    'probab' + str(epoch) + '_' + str(k) + '_' + str(rand) + '.png')
                scipy.misc.toimage(modifi_map_p[k]).save(
                    'modifi' + str(epoch) + '_' + str(k) + '_' + str(rand) + '.png')

else:
    image_test_road,image_test_name = read_image(Path2)
    sess = tf.Session()
    saver = tf.train.Saver()
    ckpt = tf.train.latest_checkpoint('mymodel')
    saver.restore(sess, ckpt)

    data_x = np.zeros([BATCHSIZE, 512, 512, 1])
    for i in range(int(len(image_test_road) / BATCHSIZE)):
        for j in range(BATCHSIZE):
            imc = ndimage.imread(image_test_road[i * BATCHSIZE + j])
            data_x[j, :, :, 0] = imc

        probab_map_p, modifi_map_p, stego_image_p = sess.run(
            [probab_map, modifi_map, stego_image], feed_dict={input_image: data_x, isTraining: False})

        #probab_map_p = np.asarray(probab_map_p)
        #probab_map_p = np.reshape(probab_map_p, newshape=[-1, 512, 512])
        #modifi_map_p = np.reshape(modifi_map_p, newshape=[-1, 512, 512])
        #modifi_map_p = np.asarray(abs(modifi_map_p))
        stego_image_p = np.asarray(stego_image_p)
        stego_image_p = np.reshape(stego_image_p, newshape=[-1, 512, 512])

        for k in range(BATCHSIZE):
            #scipy.misc.toimage(probab_map_p[k]).save(
            #    'probab_' + str(i * 5 + 9000 + k) + '.png')
            #scipy.misc.toimage(modifi_map_p[k]).save(
            #    'modifi_' + str(i * 5 + 9000 + k) + '.png')
            scipy.misc.toimage(stego_image_p[k], cmin=0, cmax=255).save(image_test_name[i * BATCHSIZE + k])
