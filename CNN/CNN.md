卷积神经网络（Convolutional neural network， ConvNets or CNNs），是一种深度学习算法，可以输入图像，为图像中的各个方面/对象分配重要性（可学习的权重和偏差），并能够区分彼此。主要用于图像识别、图像分类任务，可以应用于目标检测、人脸识别等领域。

# 图像处理

对于计算机来说，一张输入的图片是以像素数组（array of pixels）的形式呈现，数组大小由图片的分辨率决定。对于一张图片来说，像素数组通常有3个维度：
$$
(h\times w\times d)\\
\begin{align}
&h:Height\quad高度\\
&w:Width\quad宽度\\
&d:Dimension\quad维度
\end{align}
$$
对于RGB标准的图片来说，通过对红(R)、绿(G)、蓝(B)三个颜色通道的变化以及它们相互之间的叠加来得到各式各样的颜色，所以维度有三层，分别对应三个颜色的通道。而对于一张灰度图像来说，维度通常只有一层。

![4x4x3 RGB Image](http://note.lizhihao999.cn/notes/20201211233045.png)

# 概述

卷积神经网络（CNN或ConvNet）是一种用于深度学习的网络体系结构，可直接从数据中学习，而无需手动提取特征。ConvNet的架构类似于人脑中神经元的连接模式，其灵感来自于视觉皮层的组织。 单个神经元仅在被称为感受野的视野受限区域内对刺激做出反应。 这些字段的集合重叠以覆盖整个可视区域。

从技术上讲，深度学习CNN模型可以用于训练和测试，与其他神经网络一样，CNN也由输入层、输出层和之间的很多隐藏层组成。这些隐藏层执行更改数据的操作，目的是学习数据的特征。最常见的三个层是：卷积、激活函数或ReLU和池化。每个输入图像将通过一系列带有卷积核Filters（内核Kernels）、池化Pooling、全连接层（FC）的卷积层传递，并应用Softmax函数对概率值为$(0,1)$之间的对象进行分类。

- 卷积将输入图像通过一组卷积核，每个卷积核都会激活图像中的某些特征；

- 线性整流单元（ReLU）通过将负值映射为零并保持正值，可以更快，更有效地进行训练。有时将其称为激活，因为只有激活的特征才被带入下一层；
- 池化通过执行非线性下降采样来简化输出，从而减少了网络需要学习的参数数量；

下图是一个具有许多卷积层的CNN示例，将卷积核以不同的分辨率（尺寸）应用于每个训练图像，并将每个卷积图像的输出用作下一层的输入。

![Example of a network with many convolutional layers. Filters are applied to each training image at different resolutions, and the output of each convolved image is used as the input to the next layer.](http://note.lizhihao999.cn/notes/20201211233101.jpg)



# 隐藏层：特征学习

卷积神经网络中通过隐藏层进行特征学习，单个隐藏层通常包含的三个层是：卷积层、激活函数或ReLU和池化层。

## 卷积层 Convolution Layer

卷积是从输入图像的第一层，以提取特征。 卷积通过使用输入数据的小方块学习图像特征来保留像素之间的关系。 这是一项数学运算，需要两个输入，例如图像矩阵和卷积核（内核）。

在每个卷积层，数据都是以三维形式存在的，可以把它看成许多个二维图片叠在一起，其中每一个称为一个特征图feature map。如果是灰度图片，那就只有一个feature map；如果是彩色图片，一般就是3个feature map（RGB）。层与层之间会有若干个卷积核，上一层和每个feature map跟每个卷积核做卷积，都会产生下一层的一个feature map。

![Image matrix multiplies kernel or filter matrix](http://note.lizhihao999.cn/notes/20201211233153.png)

### 步长 Stribes

卷积核以某个“步长”（Stride Value）向右移动，直到解析完整宽度为止，它将跳至具有相同“步长”的图像的开始（左侧），并重复该过程，直到遍历整个图像为止。 当步长为1时，我们一次将卷积核移动1个像素。 当步长为2时，我们一次将卷积核移动2个像素，依此类推。 

![Movement of the Kernel](http://note.lizhihao999.cn/notes/20201211233201.png)

在下面的演示中，将5 x 5 x 1图像矩阵的卷积与3 x 3 x 1卷积核矩阵相乘，称为“特征图”（feature map）。绿色部分类似于我们的5x5x1输入图像I。在卷积层的第一部分中执行卷积运算所涉及的元素称为内核/卷积核K，以黄色表示，我们选择K作为3x3x1矩阵。

![ Image matrix multiplies kernel or filter matrix](http://note.lizhihao999.cn/notes/20201211233249.png)

由于步长= 1（不跨步），内核每次移位9次，每次在K与内核所徘徊的图像I中部分P之间执行矩阵乘法运算。

![Convoluting a 5x5x1 image with a 3x3x1 kernel to get a 3x3x1 convolved feature](http://note.lizhihao999.cn/notes/20201211233308.gif)

对于具有多个通道的图像（例如RGB），内核的深度与输入图像的深度相同。 在Kn和堆栈内（[K1，I1]； [K2，I2]； [K3，I3]）之间执行矩阵乘法，所有结果与偏差相加，得到一个压缩的单深度通道卷积特征输出。

![Convolution operation on a MxNx3 image matrix with a 3x3x3 Kernel](http://note.lizhihao999.cn/notes/20201211233319.gif)

### 填充 Padding

有时执行操作时，卷积核无法完全适合输入图像，有两种填充方法：

1. **Same Padding**：用零填充图片（零填充），使其适合；
2. **Valid Padding**：删除图像中不适合卷积核的部分，这称为有效填充，仅保留图像的有效部分。

第一种方法下通常会让输出维数增加或保持不变，例如下图所示，当我们将5x5x1图像扩充为6x6x1图像，然后在其上应用3x3x1内核时，我们发现卷积矩阵的尺寸为5x5x1。 因此，名称为“相同填充”（Same Padding）。第二种方法，会保留和卷积核相同维度的一个矩阵，称为“有效填充”（Valid Padding）。

![**SAME padding:** 5x5x1 image is padded with 0s to create a 6x6x1 image](http://note.lizhihao999.cn/notes/20201211233327.gif)



## 线性整流单元 ReLU

线性整流单元（Rectified Linear Unit，ReLU），用于非线性操作。通常意义下，线性整流函数指代数学中的斜坡函数，即函数输出结果为：
$$
f(x)=\max{(0,x)}
$$
![ReLU](http://note.lizhihao999.cn/notes/20201212005722.png)

而在神经网络中，线性整流作为神经元的激活函数，定义了该神经元在线性变换$\bold{w^{T}x}+b$之后的非线性输出结果，即对于进入神经元的来自上一层神经网络的输入向量，使用线性整流激活函数的神经元会输出：
$$
\max{(0,\bold{w^{T}x}+b)}
$$
至下一层神经元或作为整个神经网络的输出（取决现神经元在网络结构中所处位置）。

![ReLU operation](http://note.lizhihao999.cn/notes/20201211233403.png)

为什么ReLU很重要：ReLU的目的是在我们的ConvNet中引入非线性。 因为，现实世界中的数据希望我们的ConvNet学习的是非负线性值。 还有其他一些非线性函数（例如tanh或Sigmoid）也可以代替ReLU使用。 大多数数据科学家都使用ReLU，因为在性能方面ReLU比其他两个要好。

## 池化层 Pooling Layer

与卷积层相似，池化层负责减小卷积特征的空间大小，即通过降维来减少处理数据所需的计算能力的同时保留重要信息。 此外，它对于提取旋转和位置不变的主要特征很有用，从而保持有效训练模型的过程。 池化有不同的方法，最基本、最常见的方法有两类：

- 最大池化（Max Pooling）
- 平均池化（Average Pooling）

“最大池化”从内核覆盖的图像部分返回最大值，“平均池化”从内核覆盖的图像部分返回所有值的平均值。

![Types of Pooling](http://note.lizhihao999.cn/notes/20201211233410.png)

Max Pooling还可以充当噪声抑制器，它完全放弃激活噪声（只保留最大值），并且还执行了降噪以及降维。 而平均池化仅执行降维作为噪声抑制机制。 因此，我们可以说“最大池”的性能要比“平均池”好得多。下图为3x3的卷积核在5x5卷积上执行池化。

![3x3 pooling over 5x5 convolved feature](http://note.lizhihao999.cn/notes/20201211233419.gif)

卷积层和池化层一起形成了卷积神经网络的第i层（隐藏层）。根据图像的复杂性，可以增加这种层的数量以进一步捕获更低粒度的特征，但是以更大的计算能力为代价。



# 分类功能

我们需要将得到的矩阵扁平化为向量，再将其输入一个全连接层（例如要一个神经网络）。添加全连接层是学习卷积层输出所表示的高级特征的非线性组合的（通常）方便快捷的方法。 全连接层可以学习该空间中可能存在的非线性函数。

![Classification — Fully Connected Layer](http://note.lizhihao999.cn/notes/20201211233428.jpeg)

经过一系列卷积（隐藏层），我们已经将输入图像转换为适合多层次感知器的形式，我们将把图像展平为列向量 (x1, x2, x3, …)。 扁平化后的输出作为前馈神经网络的输入，将反向传播应用于训练的每次迭代。 经过一系列epoch后，该模型能够区分图像中的主要特征和某些低级特征，并使用Softmax分类技术对其进行分类（输出层）。



# 总结

![完整的卷积神经网络图](E:\Recent\DLModel\CNN\CNN.assets\1606235181834.jpg)

学习完之后，再回顾一下这副卷积神经网络进行图像分类的图例，应该对卷积神经网络的结构和进行图像分类的流程比较了解了。流程大致如下：

1. 提供输入图像到卷积层；
2. 选择参数，根据“步长”应用卷积核，如果需要，使用填充方法；
3. 在图像上执行卷积，并将ReLU激活应用于矩阵；
4. 执行池化以减小维数大小；
5. 添加尽可能多的卷积层直到满意；
6. 展平输出并馈入完全连接的层（FC层）；
7. 使用激活函数输出并分类图像。

一般来说，可以认为一个卷积神经网络的前半部分为卷积（包括池化）进行特征学习，后半部分为一个分类器（一个或多个全连接层）。关于如何确定CNN中的卷积核大小、卷积层数以及每层feature map个数，目前还没有一个理论性很强的解释，更多的是在根据已有的经验设计，或者利用自动搜索的方法搜索出较为合适的取值。



# 参考资料

[A Comprehensive Guide to Convolutional Neural Networks — the ELI5 way](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53)

[Understanding of Convolutional Neural Network (CNN) — Deep Learning](https://medium.com/@RaghavPrabhu/understanding-of-convolutional-neural-network-cnn-deep-learning-99760835f148)

[Convolutional Neural Network](https://www.mathworks.com/discovery/convolutional-neural-network-matlab.html)

[百度百科：ReLU函数](https://baike.baidu.com/item/ReLU%20%E5%87%BD%E6%95%B0/22689567?fr=aladdin)

[理解卷积神经网络CNN中的特征图 feature map](https://blog.csdn.net/boon_228/article/details/81238091)