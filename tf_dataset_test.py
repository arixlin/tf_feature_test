import tensorflow as tf
import numpy as np
import os
import math
import glob

def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [100, 100])
    label = tf.one_hot(label, 3)
    return image_resized, label

def get_path_files(file_dir, ratio):
    """
    Args:
        file_dir: file directory
    Returns:
        list of images and labels
    """
    image_filenames = glob.glob(file_dir + '/*/*')
    image_labels = glob.glob(file_dir + '/*')

    labelset = []
    for labels in image_labels:
        label = labels.split('\\')[1]
        labelset.append(label)

    # print(labelset)
    # save_lable(labelset)
    image_list = []
    label_list = []
    for file in image_filenames:
        label = np.argwhere(np.array(labelset) == file.split('\\')[1])[0, 0]
        image_list.append(file)
        label_list.append(label)
    temp = np.array([image_list, label_list])
    temp = temp.transpose()

    np.random.shuffle(temp)

    # print (temp)

    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])

    n_sample = len(label_list)
    n_val = math.ceil(n_sample * ratio)  # number of validation samples
    n_train = n_sample - n_val  # number of trainning samples

    tra_images = image_list[0:n_train]
    tra_labels = label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = image_list[n_train:-1]
    val_labels = label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# Create model
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):
        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 100, 100, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 32 filters and a kernel size of 5
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)
        # Because 'softmax_cross_entropy_with_logits' already apply softmax,
        # we only apply softmax to testing network
        out = tf.nn.softmax(out) if not is_training else out

    return out



# Parameters
learning_rate = 0.0001
num_steps = 2000
batch_size = 32
display_step = 100

# Network Parameters
n_input = 10000 # MNIST data input (img shape: 28*28)
n_classes = 3 # MNIST total classes (0-9 digits)
dropout = 0.9 # Dropout, probability to keep units




tra_images, tra_labels, val_images, val_labels = get_path_files('D:/github/tensorflow_test/data/data', 0)

filenames = tf.constant(np.array(tra_images))
labels = tf.constant(np.array(tra_labels))
dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
dataset = dataset.map(_parse_function)

iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                           dataset.output_shapes)
next_element = iterator.get_next()

# dataset = dataset.shuffle(buffer_size=10000).batch(32).repeat(1)
dataset = dataset.batch(32)


# iterator = dataset.make_one_shot_iterator()
iterator = dataset.make_initializable_iterator()

_data = tf.placeholder(tf.float32, [None, 100, 100, 3])
_labels = tf.placeholder(tf.float32, [None, n_classes])
X, Y = iterator.get_next()

# train_set = iterator.make_initializer(dataset)

# Create a graph for training
logits_train = conv_net(X, n_classes, dropout, reuse=False, is_training=True)
# Create another graph for testing that reuse the same weights, but has
# different behavior for 'dropout' (not applied).
# logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)
# Define loss and optimizer (with train logits, for dropout to take effect)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits_train, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(iterator.initializer)
    # get each element of the training dataset until the end is reached
    while True:
        try:
            # image, label = sess.run([X, Y])
            # print(image, label)
            sess.run(train_op)
            loss, acc = sess.run([loss_op, accuracy])
            print(loss, acc)
        except tf.errors.OutOfRangeError:
            print("End of training dataset.")
            sess.run(iterator.initializer)
            # break

# # Training cycle
# for step in range(1, num_steps + 1):
#
#     try:
#         # Run optimization
#         sess.run(train_op)
#     except tf.errors.OutOfRangeError:
#         # Reload the iterator when it reaches the end of the dataset
#         sess.run(iterator.initializer,
#                  feed_dict={_data: X,
#                             _labels: Y})
#         sess.run(train_op)
#
#     if step % display_step == 0 or step == 1:
#         # Calculate batch loss and accuracy
#         # (note that this consume a new batch of data)
#         loss, acc = sess.run([loss_op, accuracy])
#         print("Step " + str(step) + ", Minibatch Loss= " + \
#               "{:.4f}".format(loss) + ", Training Accuracy= " + \
#               "{:.3f}".format(acc))
#
# print("Optimization Finished!")