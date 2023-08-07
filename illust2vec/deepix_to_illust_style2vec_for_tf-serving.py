import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import GlobalMaxPooling1D
from tensorflow import keras
from tensorflow.keras import mixed_precision

# 创建 mixed_precision 策略，使用 float32 数据类型
policy = mixed_precision.Policy('float32')
# 设置全局 mixed_precision 策略
mixed_precision.set_global_policy(policy)



class Base64DecoderLayer(tf.keras.layers.Layer):
  """
  # 创建一个自定义的图层 Base64DecoderLayer，用于将 base64 字符串解码为 RGB 图像
  # 它接收一个参数 target_size，表示目标图像的大小（宽度和高度）
  # byte_to_img 方法用于将输入的 base64 字符串解码为 RGB 图像张量
  # 这个方法首先对输入的 base64 字符串进行解码，并根据 channels=3 将其解码为 RGB 图像
  # 之后调整图像的大小为目标大小，并归一化到 [0, 1] 的范围内
  # 最后，通过 tf.map_fn 方法将 byte_to_img 方法应用到输入的每个 base64 字符串上，并返回解码后的图像张量
  """

  def __init__(self, target_size):
    self.target_size = target_size
    super(Base64DecoderLayer, self).__init__()

  def byte_to_img(self, byte_tensor):
    # 当使用 b64 JSON 时,base64 解码由 tensorflow serve 完成
    byte_tensor = tf.io.decode_base64(byte_tensor)
    imgs_map = tf.io.decode_image(byte_tensor,channels=3)
    imgs_map.set_shape((None, None, 3))
    img = tf.image.resize(imgs_map, self.target_size)
    # 将图像转换为浮点数，并归一化到 [0, 1] 范围内
    img = tf.cast(img, dtype=tf.float32) / 255
    return img

  def call(self, input, **kwargs):
    with tf.device("/cpu:0"):
      imgs_map = tf.map_fn(self.byte_to_img, input, dtype=tf.float32)
    return imgs_map
  
# 定义一个函数 gram_matrix，用于计算输入张量的 Gram 矩阵
# 输入张量的维度为 (batch_size, height, width, channels)
# 首先，使用 tf.linalg.einsum 方法计算输入张量与其自身转置的乘积，得到一个四维张量
# 之后，对输入张量的形状进行处理，计算图像中像素位置的总数
# 最后，将计算得到的乘积结果除以像素位置总数，得到最终的 Gram 矩阵
def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)


model = keras.models.load_model('/Volumes/Data/oysterqaq/PycharmProjects/ACG2vec-model/pix2score/model-resnet_custom_v3.h5', compile=False)
# 创建 Lambda 层 gram_matrix_layer，用于计算 Gram 矩阵
# 使用 model.get_layer 方法获取模型中的某一层输出，并将其作为输入传递给 gram_matrix_layer
# 对三个不同层的输出分别计算 Gram 矩阵，并进行全局最大池化操作
# 将三个结果堆叠起来，然后再次进行全局最大池化操作得到最终输出

gram_matrix_layer = tf.keras.layers.Lambda(gram_matrix)       # 创建 Lambda 层，用于计算 Gram 矩阵
add = gram_matrix_layer(model.get_layer('add').output)        # 对模型中的某一层输出计算 Gram 矩阵
add_1 = gram_matrix_layer(model.get_layer('add_1').output)
add_2 = gram_matrix_layer(model.get_layer('add_2').output)

add = GlobalMaxPooling1D()(add)                              # 全局最大池化操作
add_1 = GlobalMaxPooling1D()(add_1)
add_2 = GlobalMaxPooling1D()(add_2)

stack = tf.stack([add, add_1, add_2], 1)                      # 将结果堆叠起来
output = GlobalMaxPooling1D()(stack)                          # 再次进行全局最大池化操作得到最终输出


#归一化
#output=tf.math.l2_normalize(output)

# 创建一个新的模型对象 style_feature_extract_model
# 这个模型接受预训练模型的输入，并输出经过特征提取和池化操作后的特征张量
# 使用 keras.Model 函数指定输入和输出，并打印模型结构信息
style_feature_extract_model = keras.Model(inputs=model.input, outputs=output)
style_feature_extract_model.summary()

# 创建输入层 inputs，指定形状、数据类型和名称
# 调用自定义的 Base64DecoderLayer 处理输入，将 base64 格式的图像数据解码为特征张量
# 使用 style_feature_extract_model 进行特征提取，得到最终的特征张量
# 创建一个新的模型对象 base64_input_style_feature_extract_model，该模型接受 base64 格式的图像数据作为输入，并进行特征提取
inputs = tf.keras.layers.Input(shape=(), dtype=tf.string, name='b64_input_bytes')    # 创建输入层，接受 base64 格式的图像数据
x = Base64DecoderLayer([512, 512])(inputs)                  # 调用自定义的 Base64DecoderLayer 处理输入
x = style_feature_extract_model(x)                           # 特征提取
base64_input_style_feature_extract_model = keras.Model(inputs=inputs, outputs=x)   # 创建基于输入进行特征提取的模型


import base64
# pic = open("/Volumes/Data/oysterqaq/Desktop/004538_188768.jpg", "rb")
# pic_base64 = base64.urlsafe_b64encode(pic.read())

#print(pic_base64)
##转成base64后的字符串格式为 b'图片base64字符串'，前面多了 b'，末尾多了 '，所以需要截取一下

#print(base64_input_style_feature_extract_model(tf.stack([tf.convert_to_tensor(pic_base64)])))
base64_input_style_feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model_base64_input")

#style_feature_extract_model.save("/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model")

# #
# style_model = keras.models.load_model('/Volumes/Data/oysterqaq/Desktop/style_feature_extract_model', compile=False)
# #
# # style_model.summary()
# image = tf.io.decode_image(tf.io.read_file('/Volumes/Data/oysterqaq/Desktop/004538_188768.jpg'),
#                                channels=3)
# image = tf.image.resize(image, [512, 512])
# image /= 255.0
# image = tf.expand_dims(image, axis=0)
# p = style_feature_extract_model.predict(image)
# print(p)

