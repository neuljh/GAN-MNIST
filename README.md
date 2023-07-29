# GAN-MNIST
# 普通的GAN来生成MNIST图片 ， 如果对您有帮助欢迎一件三连！！！
(1)初步完成
使用普通的GAN模型来生成MNIST图片。这里生成的图片是数字0-9。

①　导入对应的类库

```python
# from tensorflow import keras
import sys
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import adam_v2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
```

②　关键代码
```python
# Load the dataset
def load_data():
  (x_train, y_train), (_, _) = mnist.load_data()
  x_train = (x_train.astype(np.float32) - 127.5)/127.5
  x_train = x_train.reshape(60000, 784)
  return (x_train, y_train)

X_train, y_train = load_data()
print(X_train.shape, y_train.shape)
```

该代码的目的是加载MNIST数据集并将其准备为训练用的形式。
第一部分 load_data() 函数从Keras中的mnist数据集中加载训练数据。该函数返回一个元组 (x_train, y_train)，其中 x_train 是形状为 (60000, 28, 28) 的训练图像数组，y_train 是形状为 (60000,) 的训练标签数组。由于测试集不会被使用，因此用 _ 占位符表示。
接下来，将 x_train 的数据类型转换为 np.float32，并将所有像素值缩放到[-1,1]之间。这可以通过将每个像素值减去127.5并除以127.5来实现。这将导致所有像素值的范围从 [0, 255] 变为 [-1, 1]。
最后，将图像数据 x_train 从 (60000, 28, 28) 的形状重塑为 (60000, 784)，其中每个图像都被平铺成长度为784的一维向量。这是因为许多机器学习算法都期望输入数据的形状是二维的，即 (样本数量，特征数量)，其中每个特征都是一维的。
在最后一行，load_data() 函数被调用，返回 X_train 和 y_train。这些变量用于输出训练数据的形状，即 (60000, 784) 和 (60000,)。前者表示有60000个样本，每个样本由784个特征组成，后者表示有60000个标签，每个标签对应一个样本。
输出结果：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/f332a5d3-a3bc-44d6-854a-e804b07e93d5)

```python
def build_generator():
    model = Sequential()
    model.add(Dense(units=256, input_dim=100))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(units=784, activation='tanh'))
    model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(0.0002, 0.5))
    return model
generator = build_generator()
generator.summary()
```

该代码定义了一个生成器模型，其目的是生成与MNIST数据集中的手写数字类似的假图像。以下是对这段代码的详细解释：
build_generator() 函数定义了一个名为 model 的顺序模型。该函数开始定义了一个由四个全连接层组成的神经网络，其中每个层都使用了 LeakyReLU 激活函数。
第一个全连接层使用 256 个神经元，其输入维度为 100，这表示输入给生成器的是长度为 100 的一维噪声向量。
接下来，增加了两个全连接层，分别使用了 512 和 1024 个神经元。这些层逐渐将噪声向量转换为与MNIST手写数字类似的图像。
最后，输出层使用具有 tanh 激活函数的 784 个神经元。这是因为 MNIST 图像的大小是 28 x 28 = 784，每个像素的值在[-1, 1]之间。
最后一行中，使用 compile() 函数编译生成器模型。我们使用 binary_crossentropy 作为损失函数，这是因为我们希望生成器输出的图像能够与MNIST手写数字图像尽可能地接近。
在优化器方面，使用了 adam_v2.Adam() 优化器，其学习率为 0.0002，beta1 参数为 0.5。
最后，build_generator() 函数返回构建的生成器模型。在最后一行中，将该模型赋值给 generator 变量，并使用 summary() 函数输出了该模型的详细结构。这个函数打印出了每一层的大小、参数数量和总体参数数量，以及整个模型的参数数量。
输出结果：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/f29c5835-fa45-4944-93f3-c700d9920904)

```python
def build_discriminator():
    model = Sequential()
    model.add(Dense(units=1024, input_dim=784))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(units=512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(units=256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.3))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=adam_v2.Adam(0.0002, 0.5))
    return model
discriminator = build_discriminator()
discriminator.summary()
```

该代码定义了一个判别器模型，其目的是判别输入的图像是否来自于MNIST数据集中的真实图像。以下是对这段代码的详细解释：
build_discriminator() 函数定义了一个名为 model 的顺序模型。该函数开始定义了一个由四个全连接层组成的神经网络，其中每个层都使用了 LeakyReLU 激活函数和 Dropout 正则化。
第一个全连接层使用 1024 个神经元，其输入维度为 784，这表示输入给判别器的是一维的 28 x 28 = 784 的图像向量。
接下来，增加了两个全连接层，分别使用了 512 和 256 个神经元。这些层逐渐将输入图像向量转换为一个输出值，表示输入图像的真实性。
最后，输出层使用具有 sigmoid 激活函数的 1 个神经元，输出的值表示输入图像是否为真实的 MNIST 手写数字图像。
最后一行中，使用 compile() 函数编译判别器模型。我们使用 binary_crossentropy 作为损失函数，这是因为我们希望判别器能够尽可能地准确地区分真实和假的图像。
在优化器方面，同样使用了 adam_v2.Adam() 优化器，其学习率为 0.0002，beta1 参数为 0.5。
最后，build_discriminator() 函数返回构建的判别器模型。在最后一行中，将该模型赋值给 discriminator 变量，并使用 summary() 函数输出了该模型的详细结构。这个函数打印出了每一层的大小、参数数量和总体参数数量，以及整个模型的参数数量。
输出结果：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/bbc75318-95ab-4a7d-8be5-07da6a0870e8)

```python
def draw_images(generator, epoch, examples=25, dim=(5,5), figsize=(10,10)):
    noise= np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(25,28,28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i+1)
        plt.imshow(generated_images[i], interpolation='nearest', cmap='Greys')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Generated_images %d.png' %epoch)
```
这段代码定义了一个函数 draw_images()，该函数接受一个生成器模型、当前的 epoch 数量、要生成的示例数量、图像尺寸和绘图参数等参数。该函数的作用是使用给定的生成器模型生成一些假图像，并将它们可视化到一张图像中。
首先，该函数使用 np.random.normal() 函数从正态分布中随机生成一些噪声向量。这些噪声向量是输入到生成器模型中的，生成器将其转换为与 MNIST 数据集中的真实图像类似的假图像。
接下来，使用给定的生成器模型对噪声向量进行预测，以生成一些假图像。生成的图像是一个多维数组，需要进行形状转换，使其具有 28 x 28 的大小，与 MNIST 数据集中的真实图像相同。
然后，使用 Matplotlib 库中的 plt.subplot() 函数在一个新的图像上创建子图，并使用 plt.imshow() 函数显示生成的假图像。在显示图像之前，需要调用 plt.axis('off') 函数将坐标轴关闭。
最后，使用 plt.tight_layout() 函数来确保图像紧密地包装在一起，然后使用 plt.savefig() 函数将图像保存到本地磁盘。保存的图像名称包含当前的 epoch 数量，以便将来能够跟踪生成的图像是在哪个 epoch 中生成的。
输出结果：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/f6585537-86b1-47d1-a29f-5de1de224478)

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/0d56d56e-4857-4526-9d83-ebf0a0dc9a60)

```python
def train_GAN(epochs=1, batch_size=128):
    # Loading the data
    X_train, y_train = load_data()
    # Creating GAN
    generator = build_generator()
    discriminator = build_discriminator()
    GAN = build_GAN(discriminator, generator)
    for i in range(1, epochs + 1):
        print("Epoch %d" % i)
        for _ in tqdm(range(batch_size),file=sys.stdout):
            # Generate fake images from random noiset
            noise = np.random.normal(0, 1, (batch_size, 100))
            fake_images = generator.predict(noise)
            # Select a random batch of real images from MNIST
            real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
            # Labels for fake and real images
            label_fake = np.zeros(batch_size)
            label_real = np.ones(batch_size)
            # Concatenate fake and real images
            X = np.concatenate([fake_images, real_images])
            y = np.concatenate([label_fake, label_real])
            # Train the discriminator
            discriminator.trainable = True
            discriminator.train_on_batch(X, y)
            # Train the generator/chained GAN model (with frozen weights in discriminator)
            discriminator.trainable = False
            GAN.train_on_batch(noise, label_real)
        # Draw generated images every 15 epoches
        if i == 1 or i % 10 == 0:
            draw_images(generator, i)
train_GAN(epochs=100, batch_size=256)
```

该代码定义并训练了一个 GAN（生成对抗网络）模型，以从 MNIST 数据集中生成类似于手写数字的图像。 以下是对代码的逐步解释：
load_data() 函数加载 MNIST 数据集并将预处理后的图像作为 numpy 数组 x_train 返回，并将它们对应的标签作为 y_train 返回。 图像在 -1 和 1 之间归一化。
build_generator() 函数为生成器创建一个顺序模型。 它以形状为 (batch_size, 100) 的随机噪声向量作为输入，并生成形状为 (batch_size, 28, 28, 1) 的假图像作为输出。 生成器有 3 个具有 leaky ReLU 激活的致密层和一个具有双曲正切激活的最终致密层。 生成器使用二元交叉熵损失和学习率为 0.0002 且 beta_1=0.5 的 Adam 优化器编译。
build_discriminator() 函数为判别器创建一个顺序模型。 它以形状为 (batch_size, 28, 28, 1) 的图像作为输入，并输出标量分数，指示图像是真实的还是假的。 鉴别器有 3 个具有泄漏 ReLU 激活和丢失的致密层，以及一个具有 sigmoid 激活的最终致密层。 判别器使用二元交叉熵损失和学习率为 0.0002 且 beta_1=0.5 的 Adam 优化器编译。
draw_images() 函数采用生成器模型、纪元号和一些参数来生成和绘制假图像的网格。 它使用随机噪声生成示例数量的假图像，并将它们绘制在形状为 dim 且图形大小为 figsize 的网格中。 生成的图像被重塑为 (25, 28, 28) 以匹配原始 MNIST 图像形状。
train_GAN() 函数训练 GAN 模型。 它以 epoch 数和批量大小作为输入。 它首先使用 load_data() 加载 MNIST 数据集。 然后它通过分别调用 build_generator()、build_discriminator() 和 build_GAN() 函数来构建生成器、鉴别器和链式 GAN 模型。
该函数循环遍历 epoch 的数量，对于每个 epoch，它使用成批的真假图像训练生成器和鉴别器模型。 对于每个批次，它使用生成器生成 batch_size 数量的假图像，并从 MNIST 数据集中随机选择 batch_size 数量的真实图像。 然后它将真实和虚假图像连同它们的标签连接起来并训练鉴别器。 之后，它使用随机噪声和标签训练具有冻结鉴别器权重的链式 GAN 模型。
最后，函数调用 draw_images() 函数每 10 个 epochs 生成并保存假图像。
输出结果：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/63f2b438-d71e-44eb-81ce-b568f0322af8)

③　实验结果
实验结果(epochs=50,learning_rate=0.0002,batch_size=128)：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/a3458f10-5612-441a-8695-c90aa229d69d)

实验结果(epochs=100,learning_rate=0.0001,batch_size=32)：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/30b40a51-e770-4280-98d1-fa5627133372)

实验结果(epochs=100,learning_rate=0.0001,batch_size=128)：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/a23a2b79-9bb3-4d25-b1b6-d114f9f15fe8)

实验结果(epochs=100,learning_rate=0.0001,batch_size=256)：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/d1a87c2c-911e-42db-860d-ad2925068194)

实验结果(epochs=100,learning_rate=0.001,batch_size=32)：

![image](https://github.com/neuljh/GAN-MNIST/assets/132900799/11f9af12-06e9-4303-9f19-6763d045c06a)
