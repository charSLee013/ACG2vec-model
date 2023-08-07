import requests
import torch
import urllib
import clip
from io import BytesIO
import  numpy as np

# 设置打印数组时的显示方式，禁止使用科学计数法
# 以小数形式显示数组的元素,而不是类似于1e+03表示1000
np.set_printoptions(suppress=True)

from PIL import Image
from clip.model import build_model
import  tensorflow as tf
import  numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import matplotlib.pyplot as plt
import base64
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-L/14", device=device)

keras_model = tf.keras.models.load_model(
        '/Volumes/Data/oysterqaq/Desktop/clip_img_base64input', compile=False)

image = preprocess(Image.open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg")).unsqueeze(0).to(device)


pic = open("/Volumes/Data/oysterqaq/Desktop/f36bf9532c6f4fd392ac98f113c40c6b.jpeg", "rb")
# 对图片进行base64编码
pic_base64 = base64.urlsafe_b64encode(pic.read())

with torch.no_grad():
    # 提取图像的特征向量
    image_features = clip_model.encode_image(image) 
    a=image_features

# 使用TensorFlow的convert_to_tensor函数将base64编码的图片数据转换为张量形式。这个张量可以作为输入传递给Keras模型。
pic_tensor = tf.convert_to_tensor(pic_base64)
# 由于keras_model的输入是一个张量列表，所以我们将单个张量包装在列表中，以符合模型的要求
pic_tensor_list = [pic_tensor]
#使用TensorFlow的stack函数将张量列表堆叠起来，生成一个包含单个元素的新张量。这个新张量是输入到Keras模型的张量形式的图片数据
pic_tensor_stack = tf.stack(pic_tensor_list)
#调用Keras模型（keras_model）并将处理好的图片数据输入模型进行预测
b=keras_model(pic_tensor_stack)

"""a是一个PyTorch张量，通过.detach()方法可以将其从计算图中分离出来，然后使用.numpy()方法将其转换为NumPy数组。最后，使用print()函数将a的值打印出来。这行代码的作用是输出特征向量a的值。
"""
print(a.detach().numpy())
print(b.numpy())

"""
np.isclose()函数是NumPy提供的用于比较两个数组（a和b.numpy()）是否接近的函数。其中，atol=1e-1参数指定了允许的最大绝对误差（公差）。函数会返回一个布尔数组，表示数组中的元素是否满足相似性条件。这行代码的作用是判断a和b两个特征向量是否相似。
"""
print(np.isclose(a,
                    b.numpy(),
                     atol=1e-1))
