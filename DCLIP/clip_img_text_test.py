
import requests
import torch
import urllib
import clip
from io import BytesIO
import base64
from PIL import Image
from clip.model import build_model
import  tensorflow as tf
import  numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
# 加载Keras图像模型
keras_img_model = tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/clip_img_base64input', compile=False)
# 加载Keras文本模型
keras_text_model=tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/clip_text', compile=False)


# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)


image = preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg")).unsqueeze(0).to(device)
# 定义文本列表
text=["girl","men","cat","school"]
# 打印文本张量的形状
text = clip.tokenize(text)
print(text.shape)
# 使用CLIP模型编码图像和文本
with torch.no_grad():
    a = clip_model.encode_image(image)
    logits_per_image, logits_per_text = clip_model(
        image.to(device),
        text.to(device)
    )
    # 计算图像和文本的概率
    torch_probs = logits_per_image.softmax(dim=-1).cpu().numpy()
# 打印文本各自的概率
print(torch_probs)
# 打开并读取图片文件，并进行base64编码
pic = open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg", "rb")
pic_base64 = base64.urlsafe_b64encode(pic.read())
# 使用Keras图像模型处理图像
keras_img_model_feature=keras_img_model(tf.stack([tf.convert_to_tensor(pic_base64)]))
# 使用Keras文本模型处理文本
keras_text_model_feature=keras_text_model(tf.convert_to_tensor(text))


print(clip_model.encode_text(text))
print(keras_text_model_feature)
# 归一化图像特征向量和文本特征向量
"""
tf.norm() 函数计算了 keras_img_model_feature 的 L2 归一化值
L2 归一化将特征向量的每个元素除以其平方和的平方根，使得特征向量的 L2 范数等于 1。这样可以确保特征向量在单位长度的范围内。

"""
keras_img_model_feature = keras_img_model_feature / tf.norm(keras_img_model_feature, axis=-1, keepdims=True)
# 进行了与上一行相同的 L2 归一化操作
keras_text_model_feature = keras_text_model_feature / tf.norm(keras_text_model_feature, axis=-1, keepdims=True)
"""
这行代码定义了一个 TensorFlow 变量 logit_scale，它是一个标量（只有一个元素）。该变量初始化为 np.log(1 / 0.07)，即对数尺度的初始值。 logit_scale 在后续的计算中可能用于调整概率分布的范围。
"""
logit_scale = tf.Variable(np.ones([]) * np.log(1 / 0.07), dtype=tf.float32, name="logit_scale")

# cosine similarity as logits
# 将之前定义的 logit_scale 变量应用指数函数，即计算 e 的指数幂。这样做是为了将 logit_scale 转换为线性尺度。
logit_scale = tf.exp(logit_scale)
"""
这行代码执行矩阵乘法操作。keras_img_model_feature 是通过图像模型提取的特征向量，keras_text_model_feature 是通过文本模型提取的特征向量。通过将其转置并相乘，得到了一个矩阵 keras_logits_per_image，表示每个图像与文本之间的相关性得分。
"""
keras_logits_per_image = logit_scale * keras_img_model_feature @ tf.transpose(keras_text_model_feature)

"""
同样地，这行代码执行矩阵乘法操作，但是交换了 keras_img_model_feature 和 keras_text_model_feature 的位置。得到的矩阵 keras_logits_per_text 表示每个文本与图像之间的相关性得分。
"""
keras_logits_per_text = logit_scale * keras_text_model_feature @ tf.transpose(keras_img_model_feature)
"""
这行代码使用 TensorFlow 的 softmax 函数对 keras_logits_per_image 进行计算，将得分转换为概率。axis=1 表示沿着第一个维度进行 softmax 计算，即对每个图像得分进行归一化概率计算。
"""
tf_probs = tf.nn.softmax(keras_logits_per_image, axis=1)
# 将 TensorFlow 的张量 tf_probs 转换为 NumPy 数组类型，以便后续处理或打印。
tf_probs = np.array(tf_probs)
# 这行代码打印输出了概率值（经过 softmax 处理后的结果），表示每个图像对应的文本类别的概率分布。
print(tf_probs)

