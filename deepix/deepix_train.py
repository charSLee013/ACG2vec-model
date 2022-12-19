import argparse
import json
import os

import redis
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision

from dataset_generator import DataSetGenerator
from deepix import create_acg2vec_pixiv_predict_model
from deepix_resnet_swin_transformer import create_swin_deepix, pure_swin, pure_resnet

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


def batch_metric_to_tensorboard(batch, logs):
    tf.summary.scalar('batch_loss', data=logs['loss'], step=batch)
    tf.summary.scalar('bookmark_predict_loss', data=logs['bookmark_predict_loss'], step=batch)
    tf.summary.scalar('view_predict_loss', data=logs['view_predict_loss'], step=batch)
    tf.summary.scalar('sanity_predict_loss', data=logs['sanity_predict_loss'], step=batch)
    tf.summary.scalar('restrict_predict_loss', data=logs['restrict_predict_loss'], step=batch)
    tf.summary.scalar('x_restrict_predict_loss', data=logs['x_restrict_predict_loss'], step=batch)
    # tf.summary.scalar('tag_predict_loss', data=logs['tag_predict_loss'], step=batch)
    tf.summary.scalar('bookmark_predict_acc', data=logs['bookmark_predict_acc'], step=batch)
    tf.summary.scalar('view_predict_acc', data=logs['view_predict_acc'], step=batch)
    tf.summary.scalar('sanity_predict_acc', data=logs['sanity_predict_acc'], step=batch)
    tf.summary.scalar('restrict_predict_acc', data=logs['restrict_predict_acc'], step=batch)
    tf.summary.scalar('x_restrict_predict_acc', data=logs['x_restrict_predict_acc'], step=batch)
    # tf.summary.scalar('tag_predict_acc', data=logs['tag_predict_acc'], step=batch)
    # tf.summary.scalar('tag_predict_recall', data=logs['tag_predict_recall'], step=batch)
    # tf.summary.scalar('tag_predict_precision', data=logs['tag_predict_precision'], step=batch)
    # tf.summary.scalar('tag_predict_f1_score', data=logs['tag_predict_f1_score'], step=batch)
    return batch


def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)


def get_layer_index_by_name(model, name_patten):
    index_list = []
    for i, layer in enumerate(model.layers):
        if name_patten in layer.name:
            index_list.append(i)
    return index_list


def build_model(model_config):
    multi_loss = {'bookmark_predict': tf.keras.losses.CategoricalCrossentropy(),
                  'view_predict': tf.keras.losses.CategoricalCrossentropy(),
                  'sanity_predict': tf.keras.losses.CategoricalCrossentropy(),
                  'restrict_predict': tf.keras.losses.CategoricalCrossentropy(),
                  'x_restrict_predict': tf.keras.losses.CategoricalCrossentropy(),
                  # 'tag_predict':model_config['tag_predict_loss_function']
                  # 'binary_focal_crossentropy'
                  #    'binary_crossentropy'
                  # tfa.losses.SigmoidFocalCrossEntropy()
                  }
    multi_metrics = {'bookmark_predict': ['acc', tfa.metrics.F1Score(name='f1', num_classes=10)],
                     'view_predict': ['acc', tfa.metrics.F1Score(name='f1', num_classes=10)],
                     'sanity_predict': ['acc', tfa.metrics.F1Score(name='f1', num_classes=10)],
                     'restrict_predict': [tfa.metrics.F1Score(name='f1', num_classes=3)],
                     'x_restrict_predict': [tfa.metrics.F1Score(name='f1', num_classes=3)],
                     # 'tag_predict': [tf.keras.metrics.AUC(num_labels=10240,multi_label=True,name='auc'), tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.Precision(name='precision')]
                     }
    if model_config['model_name'] == 'deepix_v1':
        model = create_acg2vec_pixiv_predict_model(model_config['pretrained_model_path'])
    if model_config['model_name'] == 'deepix_v2':
        model = create_swin_deepix(model_config['pretrained_model_path'])
    if model_config['model_name'] == 'pure_swin':
        model = pure_swin(model_config['input_size'])
    if model_config['model_name'] == 'pure_resnet':
        model = pure_resnet(model_config['input_size'])
    if model_config['optimizer_type'] == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=model_config['learning_rate'])
    elif model_config['optimizer_type'] == 'sgd':
        optimizer = tf.optimizers.SGD(model_config['learning_rate'], momentum=0.9, nesterov=True)
    elif model_config['optimizer_type'] == 'adamW':
        step = tf.Variable(0, trainable=False)
        schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
            [10000, 15000], [1e-0, 1e-1, 1e-2])
        # lr and wd can be a function or a tensor
        lr = model_config['learning_rate'] * schedule(step)
        wd = lambda: 1e-4 * schedule(step)
        # optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
        optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
    else:
        multi_optimizer_config = model_config['multi_optimizer_config']
        custom_layers_index = []
        optimizers_and_layers = []
        other_layers_opt = None
        for config in multi_optimizer_config:
            if config['layer_keyword'] == 'other':
                if config['optimizer'] == 'sgd':
                    other_layers_opt = tf.optimizers.SGD(config['learning_rate'], momentum=0.9, nesterov=True)
                if config['optimizer'] == 'adam':
                    other_layers_opt = tf.keras.optimizers.Adam(config['learning_rate'])
                # other_layers_opt = tf.keras.optimizers.Adam(config['learning_rate'])
                if config['optimizer'] == 'adamW':
                    step = tf.Variable(0, trainable=False)
                    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
                        [10000, 15000], [1e-0, 1e-1, 1e-2])
                    # lr and wd can be a function or a tensor
                    lr = config['learning_rate'] * schedule(step)
                    wd = lambda: 1e-4 * schedule(step)
                    # optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
                    optimizers_and_layers.append(
                        (tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd), layers))
            else:
                layers_index = get_layer_index_by_name(model, config['layer_keyword'])
                layers = [layer for i, layer in enumerate(model.layers) if
                          i in layers_index]
                if config['optimizer'] == 'sgd':
                    optimizers_and_layers.append(
                        (tf.optimizers.SGD(config['learning_rate'], momentum=0.9, nesterov=True), layers))
                if config['optimizer'] == 'adam':
                    optimizers_and_layers.append(
                        (tf.keras.optimizers.Adam(config['learning_rate']), layers))
                if config['optimizer'] == 'adamW':
                    step = tf.Variable(0, trainable=False)
                    schedule = tf.optimizers.schedules.PiecewiseConstantDecay(
                        [10000, 15000], [1e-0, 1e-1, 1e-2])
                    # lr and wd can be a function or a tensor
                    lr = config['learning_rate'] * schedule(step)
                    wd = lambda: 1e-4 * schedule(step)
                    # optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd)
                    optimizers_and_layers.append(
                        (tfa.optimizers.AdamW(learning_rate=lr, weight_decay=wd), layers))

                custom_layers_index += layers_index

        optimizers_and_layers.append((other_layers_opt, [layer for i, layer in enumerate(model.layers) if
                                                         i not in custom_layers_index]))
        optimizer = tfa.optimizers.MultiOptimizer(optimizers_and_layers)

    model.compile(
        optimizer=optimizer,
        loss=multi_loss,
        # 权重需要在调整
        loss_weights=model_config['loss_weights'],
        metrics=multi_metrics,
    )

    return model


# 命令行参数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument("--load_ck", type=str2bool, default=True)
parser.add_argument("--test", type=str2bool, default=False)
parser.add_argument("--config_name", type=str, default='deepix_v1')
parser.add_argument("--ck", type=str2bool, default=False)
parser.add_argument("--h5", type=str2bool, default=False)
parser.add_argument("--epoch", type=int, default=100)

args = parser.parse_args()
print(args.load_ck)
# 初始化redis连接
redis_conn = redis.Redis(host='local.ipv4.host', port=6379, password='', db=0)

# 配置文件名
config_name = args.config_name
config_path = 'config/' + config_name + '.json'
with open(config_path, 'r') as load_f:
    model_config = json.load(load_f)

# 参数设置
save_weight_history_path = 'model_weight_history/' + config_name
checkpoint_path = "ck/" + config_name + "/{epoch:04d}.ckpt"
log_dir = "logs/fit/" + config_name
batch_size = model_config['batch_size']
input_size = model_config['input_size']
tensorBoard_update_freq = 'batch'
# epoch = 100
# resume_flag = True
# epoch数目
redis_epoch_key = 'deepix_epoch_index'
epoch_index = 0 if redis_conn.get(redis_epoch_key) is None else int(redis_conn.get(redis_epoch_key))

model = build_model(model_config)
# ouput_model_arch_to_image(model,'deepix.jpg')
# model.save('/Volumes/Data/oysterqaq/Desktop/deepix.h5')


# 从check_point加载参数
if args.load_ck:
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    print('加载历史权重完成' + latest)

dataset_generator = DataSetGenerator(batch_size, args.test, [input_size, input_size])
dataset = dataset_generator.build_dataset()

callbacks = []
if args.test:
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=tensorBoard_update_freq))
    callbacks.append(tf.keras.callbacks.LambdaCallback(on_batch_end=batch_metric_to_tensorboard))
if args.ck:
    callbacks.append([tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=0, save_weights_only=True,
                                                         save_freq=100 * batch_size),

                      # ,
                      # tf.keras.callbacks.TensorBoard(log_dir=log_dir),
                      # tf.keras.callbacks.LambdaCallback(on_epoch_end=save_h5model_each_epoch),
                      ])
if args.h5:
    callbacks.append([tf.keras.callbacks.ModelCheckpoint(save_weight_history_path + '/{epoch:08d}.h5',
                                                         period=1, save_freq='epoch', save_weights_only=True)])
# model.summary()
model.fit(dataset, epochs=args.epoch, steps_per_epoch=None, callbacks=callbacks, initial_epoch=epoch_index if (
    not args.test) else 0)
