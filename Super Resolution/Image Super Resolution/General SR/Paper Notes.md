# General SR

For image SR, there are many branches like face hallucination and so on. For those specific fields, we can utilize more priors than others. 

This file will contain selected papers for general purpose in image SR.

**谈笑有鸿儒，往来无白丁。**

# Table

[Preface](#preface)

[Main Body](#main-body)
+ [Image Super-Resolution via Sparse Representation](#image-super-resolution-via-sparse-representation)
+ [Image Super-Resolution Using Deep Convolutional Networks](#image-super-resolution-using-deep-convolutional-networks)
+ [Deep Residual Learning for Image Recognition](#deep-residual-learning-for-image-recognition)
+ [Accurate Image Super-Resolution Using Very Deep Convolutional Networks](#accurate-image-super-resolution-using-very-deep-convolutional-networks)
+ [Deeply-Recursive Convolutional Network for Image Super-Resolution](deeply-recursive-convolutional-network-for-image-super-resolution)
+ [Image Super-Resolution Using Dense Skip Connections](image-super-resolution-using-dense-skip-connections)
+ [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](#deep-laplacian-pyramid-networks-for-fast-and-accurate-super-resolution)
+ [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](url)
+ [Enhanced Deep Residual Networks for Single Image Super-Resolution](url)
+ [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](#image-super-resolution-using-very-deep-residual-channel-attention-networks)
+ [Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model](#toward-real-world-single-image-super-resolution-a-new-benchmark-and-a-new-model)


# Preface

The paper listed in this file will be in chronological order. I will provide you with hyperlink related to authors, their background and their labs.

Some abbreviations you may need to know before next part:

`NAE = National Academy of Engineering` `NAS = National Academy of Science` `CAS = Chinese Academy of Science` `CAE = Chinese Academy of Engineering` `GT = Ground Truth`

Notice: Chinese(Simplified) and English will be used in this file.

# Main Body

#### Image Super-Resolution via Sparse Representation

Jianchao Yang, John Wright, Thomas Huang, Yi Ma. Image Super-Resolution via Sparse Representation. IEEE TIP 2010.

[Paper](https://ieeexplore.ieee.org/document/5466111)

[Jianchao Yang](https://www.linkedin.com/in/jianchao-yang-80aa796/) Director, Bytedance AI Lab US, PhD UIUC

[John Wright](http://www.columbia.edu/~jw2966/)  Associate Professor, Columbia University EE, PhD UIUC

UIUC [Image Formation and Processing Research Group](http://ifp-uiuc.github.io/) (IFP Group)

The IFP Group was founded by Professor **Thomas S. Huang** (1936 - 2020). (NAE Member, CAS & CAE Foreign Member, Life Fellow of IEEE, FIAPR, FSPIE, FOSA)

[Analysis of this paper(cnblog)](https://www.cnblogs.com/qizhou/p/14462540.html)

本文使用了稀疏表达的方法，实现了SR。基本思想如下图所示：

![image](https://user-images.githubusercontent.com/36061421/119659158-7e4c4e80-be60-11eb-8124-ca88ad2d1795.png)

之后进行训练，训练的过程涉及凸优化的求解，这里需要一些数学技巧。在上面的博文中，由于博主使用的是梯度下降的方法，所以得不到好结果。文章中一些trick也很关键，仅凭稀疏表达是不够的。

既然这里涉及到了sparse representation，那就可以多说一些：（关于字典学习和稀疏表示的概念）

**稀疏表示就是用有限个特征向量(基)来表示信号，字典学习就是根据信号自身的结构特征来学习得到这些特征向量(基)，这是两个相对独立又有关联的操作**。稀疏表示也可以用固定字典，比如DCT字典，小波基等等，字典学习就是可以通过对信号的学习，来得到更能准确描述信号特征的基(非正交)，以便更稀疏的表示信号。

文章中，还涉及了一些convex optimization的知识，下面对范数进行复习：
+ **0范数，向量中非零元素的个数**；
+ 1范数，为绝对值之和；
+ 2范数，就是通常意义上的模。

之所以对`0范数`比较上心，是因为sparse representation中，有对于表达更加稀疏的要求，所以对`0范数`比较在意。

至于为什么需要更加稀疏，我认为应该有至少两个原因：
1. 稀疏的数据可以降低运算复杂度；
2. 稀疏的数据可以防止过拟合。

When I am searching for sth related to L0-Norm, I find a concept named as `Compressed Sensing`(CS). 这种方法和稀疏表达比较类似，想要突破传统的等间隔采样与奈奎斯特采样定理的限制，aiming to sample less points but also recover original signal successfully. 针对这种思路，可以参考知乎文章[形象易懂讲解算法II——压缩感知](https://zhuanlan.zhihu.com/p/22445302)。

[Table](#Table)

#### Image Super-Resolution Using Deep Convolutional Networks

Chao Dong, Chen Change Loy, Kaiming He, Xiaoou Tang. Image Super-Resolution Using Deep Convolutional Networks. IEEE TPAMI 2016.

[Paper and Code](http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html)

[MMLab](http://mmlab.ie.cuhk.edu.hk/) CUHK  

[SIAT](http://www.siat.ac.cn/) Shenzhen Institute of Advanced Technology, Chinese Academy of Sciences

Director of MMLab: [Xiaoou Tang](https://www.ie.cuhk.edu.hk/people/xotang.shtml) FIEEE(Class 2009), PhD MIT, MS University of Rochester, BS USTC

Chao Dong Member of the SIAT, PhD CUHK MMLab, BEng BIT

The idea of SRCNN was firstly proposed in `Learning a Deep Convolutional Network for Image Super-Resolution`, which was accepted by ECCV 2014. At that time, ResNet was not proposed. Consequently, SRCNN has shallow layers and you will find that it seems difficult to train SRCNN though it only has TWO layers. 一些关于网络层数对于SR影响的分析，在SRCNN提出的时候还是比较片面的。实际上，在后面的`VDSR`中，还是可以很明显地看到：深层网络对于SR很有帮助。

在本篇文章中，有讨论到一些很实用的设定，到后面的几篇SR的文章都在使用：
+ 一个是**不同channel的超分** 。以前的文章会谈论到gray-scale或者single-channel的超分，这些超分的关注点在于luminance channel。当然了，除了single-channel的内容，还有团队将RGB的各个channel进行超分，最后综合在一起。不管是single-channel还是各个channel的综合，前人并没有分析`不同channel带来的影响`和`恢复全部channel的必要性`。
+ 另一个是CNN的引入，到后来深度网络DNN的引入。

SRCNN is inspired by `Image Super-Resolution via Sparse Representation`. The setting of CNN layers and 概念上面的迁移 都是和`Image Super-Resolution via Sparse Representation`对应的。

下面是SRCNN的结构图：

![image](https://user-images.githubusercontent.com/36061421/120927286-fd6c3d00-c712-11eb-8604-8863d929edbb.png)

下面是CNN和sparse coding的对应：

![image](https://user-images.githubusercontent.com/36061421/120961350-457d7500-c790-11eb-89bc-8cb527346bd4.png)


**主要处理步骤**
+ 首先，将LR图像upscale到desired size，upscale的方法是`bicubic interpolation`。interpolated image用`Y`表示，这时候`Y`和ground truth（GT） image `X`是同样的size。为了表达的方便，称`Y`为LR图像。
+ 完成上面的预处理之后，网络主要完成三个步骤：
     + `Patch extraction and representation`：这一步是从LR图像中提取一个f1 * f1大小的patch，使用了n1个filters。因此，一个patch对应一个维度为n1的vector。
     + `Non-linear mapping`：这一步是非线性映射，将n1的vector映射到n2的vector。n1可以比n2更大，使得映射结果的维度更小，更加sparse。
     + `Reconstruction`：这一步将上面得到的各个patch的结果，进行aggregate，生成最终的HR图像。

**数学表达式**

第一个步骤：
![image](https://user-images.githubusercontent.com/36061421/120954890-c2095700-c782-11eb-8868-2a05835e084b.png)

之所以是这种形式，和激活函数ReLU有关。里面的`W1 * Y + B1`就是卷积操作和bias。

第二个步骤：
![image](https://user-images.githubusercontent.com/36061421/120954914-c9306500-c782-11eb-8fff-22bb7a10887e.png)

形式类比于第一步。

第三个步骤：
![image](https://user-images.githubusercontent.com/36061421/120954932-d2213680-c782-11eb-9d8e-01edfca4e3c9.png)

对上面得到的结果进行aggregate，对结果进行averaging处理。这里的平均处理就是线性滤波的过程。

损失函数：

![image](https://user-images.githubusercontent.com/36061421/120955184-67bcc600-c783-11eb-8db3-284967fdc097.png)

很直观的MSE方法，这种方法有助于PSNR的提升。实际上，不只是PSNR表现好，SSIM等indicator也不错。上式中的n是训练样本的数量。

如果不使用MSE的方法也可以，但需要保证损失函数是可导的。

**超参数设定**

f1 = 9， f2 = 1， f3 = 5， n1 = 64， n2 = 32。特别说一下f2，这里f2设成1或者3、9都是可以的，但是如果设成3或9，那么对于原本patch的解释就要更加“广义”一些。设置成1，就是对原本的patch对应的vector进行非线性映射，不涉及太多解释的成分。这一点回想一下卷积的计算过程便可以明白。

**和稀疏编码的比较**

上面提到过，本篇文章CNN的思想来源于传统的稀疏编码的方法。主要的思想来源是[Image Super-Resolution via Sparse Representation](#image-super-resolution-via-sparse-representation)。

这方面的比较，可以看上面的CNN与sparse coding的对应图。

具体的细节这里不提及（因为我也没有深入了解这些数学原理），大概谈谈自己的想法，可能不对。

作者将CNN和稀疏编码进行类比，先使用滤波器提取n1个维度的向量，这个向量就相当于对应一个patch的编码；这些滤波器相当于一个字典。中间的非线性映射代表着稀疏编码中的一些处理的迭代过程。最后得到HR对应的coding，再使用HR对应的字典，可以得到最终HR的输出。这样的过程其实是和[Image Super-Resolution via Sparse Representation](#image-super-resolution-via-sparse-representation)中的核心思想对应的。

与传统方法相比：
+ CNN的方法利用了更多的信息，比如上述SRCNN，information包含了（9+5-1）^2 = 169 pixels，而传统方法只有81pixels。这里81是怎么得到的，需要看论文中对应的引文；
+ CNN方法的处理是端对端的映射，这其中包含了各种operation，好处在于可以一并optimize，更全面。

**训练**

损失函数在上面已经展示了。

梯度下降的方法是SGD；learning rate在不同层设置不同数值，比如前两层：10^-4，最后一层是10^-5。因为作者发现这样更好converge。

在训练数据的预处理上，需要将GT图片准备成f(sub) * f(sub) * c-pixel sub-images。这里需要强调：sub-images和patches在这里意义不同。patches需要averaging作为post-processing，因为它们overlapping，但是sub-images没有这个需求，而且被当做small images处理，是被randomly cropped from the training images。

LR图像的合成来自于HR图像，使用高斯kernal，n倍降采样。之后对着模糊后的图片，进行n倍bicubic插值上采样，得到desired size，之后进行CNN处理。（这里对于低分辨率图像的处理还比较传统。其实使用bicubic或者其他降采样的手段，并不能够很好地模拟HR和LR的对应关系，这会导致SR的泛化能力较差。真实的SR更加复杂，这里的设定会导致网络最终学会的是怎么应对bicubic等降采样算法的方法，而不是真实地解决SR问题。）

为了防止边缘影响（border effects），训练的时候，所有卷积层*都没有padding*。这样一来，最终的输出会smaller。但是：The MSE loss function is evaluated **only** by the difference between the central
pixels of Xi and the network output.

最后特别说明：尽管使用fixed size image进行training，但可以使用各种不同size的image进行testing。

**实验部分**

主要进行以下实验：
1. the impact of using different datasets on the model performance
2. explore different architecture designs of the network, and study the relations between super-resolution performance and factors like depth, number of filters, and filter sizes
3. extend the network to cope with color images and evaluate the performance on
different channels

针对1，发现使用larger dataset会有提升，但是没有想象中的多。进行对比的是set91和ImageNet。原因是：对于low-level的任务，其实set91已经包含了足够多的variability of natural images，我们底层视觉更关注图像细节纹理本身，不在乎图像具体包含了什么内容，有几个人，如何识别...但对于high-level任务，肯定是见多识广，所以larger dataset可以带来更好的结果。

针对2，可以有以下几个值得关注的点：
+ filter number：也就是network width。数量越多，performance越好，但是更加time-consuming，这是一个trade-off，但不管怎么说，比原来方法好；
+ filter size：同样也是performance和speed的trade-off
+ number of layers：在这篇文章发表的时候，还面临着网络退化的问题。所以SRCNN在加深的过程中，也遇到了很多问题。在当时是一个很open question的事情。但后来有ResNet，VDSR...等结构，加深了效果更好了。所以这里的结果参考的价值不大。

针对3，It is also worth noting that the improvement compared withthe single-channel network is not that significant (i.e., 0.07 dB). This indicates that the Cb, Cr channels barely help in improving the performance.（具体的实验很多，详见文章。这样的结论为后续使用Y channel奠定了基础）

除了以上的三点，其他是和previous SOTA模型的比较。

**总结**

总的来说，SRCNN很有开创性价值，但也有当时的局限性，仍不失为好文章。

[Table](#Table)

#### Deep Residual Learning for Image Recognition

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016 Best Paper Award.

[Paper](https://arxiv.org/abs/1512.03385)

[Kaiming He](http://kaiminghe.com/) FAIR, PhD CUHK MMLab, BEng THU.

This paper is really well-known because it provides us with a novel resolution to deal with deep networks.

Actually, ResNet is a model built for high-level CV tasks. But the concept of Residual Learning is similar to Laplacian Filtering in Digital Image Processing. We can employ this character to super-resolve images like VDSR did. 

尽管ResNet不是为了SR而生，但因为它的某些性质和SR十分契合，所以自VDSR之后，很多SR的网络里面都加入了残差学习的元素，比如identity mapping等内容...正是因为这个原因，对于学习SR的人而言，这篇文章也是必看的。

关于high-level或者这篇文章的实验部分，这里不再赘述，文章内容非常全面。这里谈谈自己对于残差学习和SR之间的见解。

![image](https://user-images.githubusercontent.com/36061421/121775649-87a51d00-cbbb-11eb-86d7-893839a07766.png)

很多人可能很疑惑，残差学习究竟在干什么？分析一下上面这张图，其实就是把主要学习对象从一个对象的全部变成了这个对象的部分（residual）。因为你会发现，这个对象会有一个identity mapping的过程，最终的output减去identity mapping的内容，就是学到的residual。也就是说，真正学习的内容是residual。

上面的过程其实和图像锐化的过程相似：

<!-- ![](http://latex.codecogs.com/svg.latex?\\g(x, y) = f(x, y) + c[\nabla^2f(x, y)]) -->

![image](https://user-images.githubusercontent.com/36061421/121776170-f1bec180-cbbd-11eb-9964-8be253714a5c.png)

上面的式子中，f(x, y)代表输入图像，g(x, y)代表锐化后的图像。显然，对于图像锐化的操作来说，重点在于带有拉普拉斯算子的那一项。这就很像残差学习。f(x, y)对应的是identity mapping的部分，g(x, y)对应着输出的部分，residual就是拉普拉斯的部分。至此，图像锐化和残差学习对应上了。

如果觉得上面抽象，那么下面是实验的结果：

假如输入是一张月亮的图像：

![moon](https://user-images.githubusercontent.com/36061421/121776402-46167100-cbbf-11eb-9510-f176340a9257.jpg)

可以明显看到，月亮的表面（坑坑洼洼的地方）的轮廓不是很清晰。这是f(x, y)。

这时候进行拉普拉斯滤波：

![moon2](https://user-images.githubusercontent.com/36061421/121776439-69412080-cbbf-11eb-81e8-3555e0f4a5f3.jpg)

可以得到一些细致的边缘。这里对应公式中的拉普拉斯项。

二者叠加，得到下面的结果：

![moon3](https://user-images.githubusercontent.com/36061421/121776451-8675ef00-cbbf-11eb-81ec-3d7dda898fbf.jpg)

看上去对比度比较差，可以自然地联想到直方图均衡化，得到最终结果：

![moon4](https://user-images.githubusercontent.com/36061421/121776466-9c83af80-cbbf-11eb-90fa-e7f499e99b72.jpg)

上面的图就已经可以满足大多数人的需求了。所以我们可以发现，在图像清晰化的过程中，拉普拉斯滤波结果的叠加很关键，这就相当于对应到残差学习中，残差的内容是很关键的。我们在超分或者清晰化图像的过程中，我们的需求就是残差的内容，这就和使用残差学习的ResNet对应上了。因此，在超分中可以很好地利用残差网络的特点。

基于这种关联性，在ResNet提出后，有了VDSR等网络。

当然了，关于residual learning带来的好处，还有很多，比如防止网络退化，防止backpropagation的时候gradient vanishing...这些内容可以详见文章；关于拉普拉斯算子的一些具体内容，以及为什么这么做可以提取图像边缘细节，还会涉及比较多的内容，详见冈萨雷斯的《数字图像处理（第三版）》，这本书非常好且详细。

[Table](#Table)

#### Accurate Image Super-Resolution Using Very Deep Convolutional Networks

Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee. Accurate Image Super-Resolution Using Very Deep Convolutional Networks. CVPR 2016.

Official Homepage: [https://cv.snu.ac.kr/research/VDSR/](https://cv.snu.ac.kr/research/VDSR/)

Notice: official code is written by **Matlab**.

However, there is still [PyTorch Version](https://github.com/twtygqyy/pytorch-vdsr).

Seoul National University (SNU)

Computer Vision Laboratory(CVLab)

Director of the Lab: [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/kmlee/) FIEEE(Class 2021), Member of KAST(the Korean Academy of Science and Technology), PhD USC

这篇文章很有意义，因为它融合了ResNet的思想，将残差学习带入到SR之中。

上面提到了，残差学习很好地解决了网络退化的问题，使得网络的层数得以进一步加深。这篇文章显然在SRCNN之后。基于SRCNN和ResNet的思想，对原有的SRCNN提出了不足和修改方法，得到了现有的VDSR。

**Contribution**

+ DNN收敛慢，但是增加learning rates可能导致梯度爆炸。本文使用了residual-learning和gradient clipping两种方法解决上面的问题；
+ VDSR可以较好地解决multi-scale的SR问题。 Cope with multiscale SR problem in a single network.

**网络结构**

![image](https://user-images.githubusercontent.com/36061421/121776959-45330e80-cbc2-11eb-859a-8aa9987e06f2.png)

Our Network Structure. We cascade a pair of layers (convolutional and nonlinear) repeatedly. An interpolated low-resolution (ILR) image goes through layers and transforms into a high-resolution (HR) image. The network predicts a residual image and the addition of ILR and the residual gives the desired output. We use 64 filters for each convolutional layer and some sample feature maps are drawn
for visualization. Most features after applying rectified linear units (ReLu) are zero.

值得注意的是：网络结构上面的灰度图是visualization的用途，正好是8 * 8 = 64 feature maps。上面的结构和前面谈到的ResNet结构，和拉普拉斯锐化的结构是相通的。

**训练**

这篇文章用的是L2范数。一种常见的思路是：假设`f(x)`是predicted image，`y`是GT，那么求解`(y - f(x))`的L2范数，作为loss function即可。但仔细想想，这里用的是residual image，我们其实没必要把整个图像拿过来求loss function，我们关注residual就好，因此：

As the input and output images are largely similar, we define a residual image `r = y - x`, where most values are likely to be zero or small. We want to predict this residual image. The loss function now becomes :

![image](https://user-images.githubusercontent.com/36061421/121777200-73fdb480-cbc3-11eb-9585-eb20945fed74.png)

where f(x) is the network prediction.

**这种只针对于residual而言的loss function，是很有意义的**。在后面的文章，比如《Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution》中，就有类似的应用。

对应到开头讲的contribution，training这块文中有提及High Learning Rates for Very Deep Networks和Adjustable Gradient Clipping，具体细节大家在自己training的时候可以细致看看，不再赘述。有意思的是一个比较：Our 20-layer network training is done within 4 hours whereas 3-layer SRCNN takes several days to train. 这凸显了VDSR的优势，回应了SRCNN中对于“the deeper, the better”的“质疑”，证明the deeper确实可以the better。

下面谈谈我很有兴趣的部分，就是contribution中的multi-scale的问题。用一种很朴素的机器学习观点来看SR问题，就是给一个LR-HR Pair，然后training。这个pair中，LR和HR应该是一个特定的scale。这样一来，train出来的网络也是specific scale。但是VDSR这里使用single network实现了multi-scale，我还是很有兴趣的。

原文的内容是：We need an economical way to store and retrieve networks. For this reason, we also train a multi-scale model. With this approach, parameters are shared across all predefined
scale factors. Training a multi-scale model is straightforward. **Training datasets for several specified scales are combined into one big dataset**.

Data preparation is similar to SRCNN with some differences. Input patch size is now equal to the size of the receptive field and images are divided into sub-images with no overlap. A mini-batch consists of 64 sub-images, **where sub-images from different scales can be in the same batch**.

上面bold的部分是数据处理的重点，也是和传统想法不同的地方。

关于这方面的实验，一开始是关于single scale。发现如果train single scale，那么没被train的scale拿去test，效果凄惨（甚至不如bicubic）。比如train 2x，拿3x去test，效果“感人”。之后作者思考，如果一个网络一口气train各种scale，那么和single scale比，会不会更有优势？于是作者根据2x，3x，4x，train了四个网络，涉及的scale分别是：{2,3,4}，{2,3}，{2,4}，{3,4}。结果发现，这个多个scale train出来网络效果真的不差（compared to single scale network）。而且，比较amazing的是：**Another pattern is that for large scales (x3; 4), our multiscale network outperforms single-scale network**。由此，作者得到结论：we observe that training multiple scales boosts the performance for large scales. 这种training strategy是有意义的。

下面是实验结果：

![image](https://user-images.githubusercontent.com/36061421/121778186-6b5bad00-cbc8-11eb-99c3-299883ead4a1.png)

具体的代码操作，还是需要看文章中开源的代码。

**训练细节**

training dataset: 291, namely BSDS 200 + T91.

test dataset: Set5, Set14, Urban 100, B100.

Training Parameters: We train all experiments over 80 epochs (9960 iterations with **batch size 64**). Learning rate was initially set to 0.1 and **then decreased by a factor of 10 every 20 epochs**. In total, the learning rate was decreased 3 times, and the learning is stopped after 80 epochs. Training takes **roughly 4 hours** on GPU Titan Z.

总结一下：本文引入residual learning和gradient clipping对于DNN训练过程中的问题解决与处理，很有意义。此外，multi-scale的training也很有意思。

[Table](#Table)

#### Deeply-Recursive Convolutional Network for Image Super-Resolution

本文使用递归神经网络来处理SR问题，这是递归神经网络第一次用于SR问题。

**Contribution**

文中提出的模型为DRCN，主要的contribution有两个：
1. recursive-supervision；这个贡献使得整个网络更好train，便于网络加深，否则会very likely面临gradient exploding和vanishing的问题；
2. skip-connection；connection的灵感源于两点：1.ResNet； 2.For SR, input and output images are **highly correlated**.

**Network Architecture**

![image](https://user-images.githubusercontent.com/36061421/121809765-b04b1680-cc90-11eb-8697-74fe687ea646.png)

网络主要分为三个部分：
+ Embedding network：这一部分之前需要将image interpolate到desired size，然后takes the input image (grayscale or RGB) and represents it as a set of feature maps. Intermediate representation used to pass information to the inference net largely depends on how the inference net internally represent its feature maps in its hidden layers. Learning this representation is done end-to-end altogether with learning other sub-networks.简单来说就是为下面的Inference network做准备。
+ Inference network：这是递归网络的核心部分。Analyzing a large image region is done by a single recursive layer. Each recursion applies the same convolution followed by a rectified linear unit. With convolution filters larger than 1 × 1, the receptive field is widened with every recursion.
+ Reconstruction net：While feature maps from the final application of the recursive layer represent the high-resolution image, transforming them (multi-channel) back into the original image space (1 or 3-channel) is necessary. reconstruction net是similar to SRCNN的最后一层的。

如果将Inference network展开，那么得到下面的结构：

![image](https://user-images.githubusercontent.com/36061421/121810024-c1e0ee00-cc91-11eb-9977-c960b8ff331c.png)

看到这里，可能有人会将Recursive Network和Recurrent Network做比较。参考[知乎](https://www.zhihu.com/question/36824148)上面的内容，我将比较好的回答放在下面。（因为目前没有深入研究这个内容，所以先使用知乎的解释）

>recurrent: 时间维度的展开，代表信息在时间维度从前往后的的传递和积累，可以类比markov假设，后面的信息的概率建立在前面信息的基础上，在神经网络结构上表现为后面的神经网络的隐藏层的输入是前面的神经网络的隐藏层的输出；recursive: 空间维度的展开，是一个树结构，比如nlp里某句话，用recurrent neural network来建模的话就是假设句子后面的词的信息和前面的词有关，而用recurxive neural network来建模的话，就是假设句子是一个树状结构，由几个部分(主语，谓语，宾语）组成，而每个部分又可以在分成几个小部分，即某一部分的信息由它的子树的信息组合而来，整句话的信息由组成这句话的几个部分组合而来。
>

通过上面的解释，可以大致了解二者的差别。如果还是很难理解，那么可以看一看下面的model结构：

![image](https://user-images.githubusercontent.com/36061421/121810200-71b65b80-cc92-11eb-8df2-386e9a1461b6.png)

这是DRCN的final version。整体来说还是像树结构的，所以称其为递归神经网络也是比较符合上面的说法。

下面讨论一下这样的网络结构下的优点和缺点以及缺点的解决。

pros: the recursive model is simple and powerful.

cons: 
+ difficult to train due to two reasons: 1.gradient vanishing; 2.gradient exploding. 至于为什么会出现gradient的问题，是因为recursive的过程中，gradient在每一层叠加的时候，都是相乘的。这样一来，层数如果很多的话，最终的gradient可能会很大也可能会很小。
+ Another known issue is that storing an exact copy of information through many recursions is not easy. In SR, output is vastly similar to input and recursive layer needs to keep the exact copy of input image for many recursions.
+ Finding the optimal number requires training many networks with different recursion depths. 这一点也很好理解。因为设置了recursive network，所以就多引入了how many layers这个超参数。实际使用的过程中，需要知道optimal是多少。

针对于第一个和第三个问题，可以通过一种手段解决，那就是：recursive supervision。

原文：**We supervise all recursions in order to alleviate the effect of vanishing/exploding gradients**. As we have assumed that the same representation can be used again and again during convolutions in the inference net, the same reconstruction net is used to predict HR images for all recursions. Our reconstruction
net now outputs D predictions and all predictions are simultaneously supervised during training (Figure 3 (a)). We use all D intermediate predictions to compute the final output. All predictions are averaged during testing. The optimal weights are automatically learned during
training. 

简单来说，就是通过每一个recursion都监督的方式，避免gradient过大或者过小的问题。具体方案见上面的final version的网络结构。相比于一开始提出的DRCN，它增加了recursive supervision的结构。

这样一来，第一个问题可以解决。那么又为什么可以解决第三个问题？

原文：**The importance of picking the optimal number of recursions is reduced as our supervision enables utilizing predictions from all intermediate layers. If recursions are too deep for the given task, we expect the weight for late predictions to be low while early predictions receive high
weights**. By looking at weights of predictions, we can figure out the marginal gain from additional recursions.

我认为这种见解是值得学习的。在每一个recursive都可以有很好的supervision之后，每一个recursion都在学习更多的内容。对于一张输入图像，它的内容终究是有限的。前几个recursion提取完image的特征之后，后面的recursion自然提取不到什么内容。因此，可以通过文中的方式解决optimal number的问题。这样一来，第三个问题也解决了。

针对于第二个问题，其实前面在讲VDSR的时候也遇到过，就是增加skip-connection就好。所以可以看到，在final version的网络结构中，有skip-connection直接从输入image到reconstruction network。

**训练**




[Table](#Table)

#### Image Super-Resolution Using Dense Skip Connections

[Paper](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)

[Imperial Vision](http://www.imperial-vision.com/)

Both of the CEO and CTO in Imperial Vision obtained PhD degree from IC.

本篇文章借鉴了DenseNet的思想：

![image](https://user-images.githubusercontent.com/36061421/121796161-a94ce580-cc49-11eb-9d6e-7ff4a5ab788d.png)

提出了SRDenseNet：

![image](https://user-images.githubusercontent.com/36061421/121796165-b23db700-cc49-11eb-93d4-c24a130b4387.png)

如上，有三种结构。对于第三种，因为增加的connection已经很多了，所以又增加了bottleneck layer。这个layer就是1 * 1的conv，目的就是reduce feature maps。

作者分析了上面的三种结构，发现3>2>1。文章中分析的是，受益于低层特征和高层特征的结合，超分辨率重建的性能得到了提升。像第三种结构把所有深度层的特征都串联起来，得到了最佳的结果，说明不同深度层的特征之间包含的信息是互补的。

[Table](#Table)

#### Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution

University of California, Merced [Vision and Learning Lab](http://vllab.ucmerced.edu/)

UCM is the newest campus of UC system. Currently, it is still under construction. 

Director of the lab : [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/) FIEEE

[LapSRN](http://vllab.ucmerced.edu/wlai24/LapSRN/); [Pytorch Version](https://github.com/twtygqyy/pytorch-LapSRN)

Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution. CVPR 2017.

在解析这篇文章之前，先来谈谈什么是拉普拉斯金字塔。

![什么是拉普拉斯金字塔和高斯金字塔](https://user-images.githubusercontent.com/36061421/121775358-ab676380-cbb9-11eb-9530-0a1d30cd3820.jpg)

上面的图来自于冈萨雷斯的《数字图像处理（第三版）》

在后面介绍这篇文章的model时，可以看到image reconstruction部分就是一个拉普拉斯金字塔。了解拉普拉斯金字塔还是很有用的，因为比较多的SR文章会涉及这个知识点。同时，你会发现这种residual的感觉就是ResNet的感觉。经过VDSR之后，大家都能更好地意识到residual learning对于SR领域的重要性。

这篇文章提出了拉普拉斯金字塔的结构，解决超分问题。文章的开头，点出了目前三个超分问题：
1. 当下的超分方法，需要将原本的image通过bicubic变成desired size，再输入到网络当中。这样increase computational complexity，此外，often results in visible reconstruction artifacts；
2. 现有方法通过`l2`范数来当做loss，会使得预测出来的高分辨率image模糊；
3. 大多数方法在复原HR图像的时候，都是通过一步实现升采样。这样做的话，增加了large scale的训练难度。In addition, existing methods cannot generate intermediate SR predictions at multiple resolutions. As a result, one needs to train a large variety of models for various applications with different desired upsampling scales and computational loads.

为了解决上面的问题，作者提出了LapSRN。这个网络可以循序渐进地进行超分，在每一个分辨率下（respective level），都会predict sub-band residuals。所谓的sub-band residuals就是respective level下，升采样得到的image和GT（ground truth）的不同。

下面是网络结构：

![image](https://user-images.githubusercontent.com/36061421/121687164-9aa0ea00-caf4-11eb-9b04-69db7b4e3ab8.png)

仔细看还是会发现，这就好像是不同level的VDSR进行cascade，下面是VDSR：

![image](https://user-images.githubusercontent.com/36061421/121687249-b73d2200-caf4-11eb-9f45-9c49b9f16f76.png)

那么这么做，有什么好处？
1. 提升了准确度：直接extract feature maps from LR，jointly optimizes the upsampling filters；
2. 速度快：LapSRN achieves real-time speed on most of the evaluated datasets；
3. Progressive reconstruction. For scenarios with limited computing resources, our 8x model can still perform 2x or 4x SR by **simply bypassing** the computation of residuals at finer levels. Existing CNN-based methods, however, do not offer such flexibility. 此外，还可以用于一些视频超分任务。这里说的simply bypassing就是把网络长度缩短，那么就可以只用于4x的SR任务，加长就可以变成8x的任务，很flexible。

下面的表格是不同方法的比较：

![image](https://user-images.githubusercontent.com/36061421/121689442-2e73b580-caf7-11eb-9017-8679fc8af001.png)

通过上面的比较，不难看出LapSRN涉及到了一开始提出的三个问题中的全部，即：computational complexity和loss function的问题。正是因为l2 loss使得overly smooth，所以使用`Charbonnier`作为loss。使用一种progressive的结构，一方面使得计算不那么difficult，另一方面可以freely truncate the network to obtain different scales on distinct purposes.

下面来简单介绍一下什么是`Charbonnier Loss`以及使用这个方法的原因（个人看法）：

![image](https://user-images.githubusercontent.com/36061421/121701771-d42d2180-cb03-11eb-9965-c08b122c9c6e.png)

这种loss很像L2范数，但仔细看看，因为epsilon取值很小，所以可以看作是L1范数（f(x) = |x|）的升级版。根据[论坛讨论的内容](https://forums.fast.ai/t/making-sense-of-charbonnier-loss/11978)，有三种看法：
1. 添加epsilon，防止数值等于0；
2. |x|(L1范数)在0不可以求微分，但是`Charbonnier Loss`可以；
3. 根据x的大小变化，和epsilon接近的时候，就是L2范数的性质，和epsilon远（比epsilon大）的时候，表现L1范数的性质。这样的设置可以利用两种范数的特性。

我个人倾向于第二种看法。对于loss function，后续还有优化（optimization）的过程，保证它不在一些特殊点不可微分，还是很重要的。至于为什么不可微分，[点击这里](https://www.shuxuele.com/calculus/differentiable.html)。

下面，我还是把`f(x) = |x|`不可微分的原因截出来：

![image](https://user-images.githubusercontent.com/36061421/121702985-ef4c6100-cb04-11eb-9b32-ee36fc47d841.png)

至于为什么`Charbonnier Loss`可以微分，这就需要自己手算一下了，很简单。

**网络结构**

从上面的网络结构图可以看出，网络是逐个level进行超分的。如果`S`代表scale factor，那么就会有![](http://latex.codecogs.com/svg.latex?\\log_{2}S)个levels。举一个例子，比如scale = 8，那么有3个sub-networks。

每一个sub-network，由两部分组成，一个是：Feature extraction，另一个是：Image reconstruction。
+ 对于feature extraction，有多个卷积层和一个转置卷积层。多个卷积层用于reconstruct a residual image at level s，转置卷积用于upsample image，以便送到下一个level。每一次upsample就是进行2x的操作，所以这一点也符合上面scale factor和S levels的关系。
+ 对于image reconstruction，The upsampled image is then combined (using element-wise summation) with the predicted residual image from the feature extraction branch to produce a high-resolution output image.

经过上面的处理，一个level就结束了，送到下一个level。下一个level继续做类似的事情，整个网络就cascade起来了。

**损失函数**

其实这个函数上面介绍过了，具体计算可以看论文。这里简单分析一下：
+ 首先为什么是这样的形式？这是因为学习的内容是**sub-band residual image**，所以loss function是这样的形式
+ 再者，我们可以看到，如果使用这样的loss function和这样的结构，**那么其实每一层都会有loss** 。假设手头有一张HR的照片，那么每一层的GT就有了，就是根据原本的GT进行downscale而来。每一层都有学习目标，每一层都是supervision。用这样的方法可以train出来LapSRN。
+ 综合上面的内容：**LapSRN就是progressive的过程，一步步进行2x操作得到目标scale** 。相比于原本直接8x的操作，LapSRN放慢了步调，一步分三步走。在放慢步调的时候，通往8x的道路上，我们可以同时得到2x和4x的结果。这就是文中说的：our 8x model can produce 2x, 4x and 8x super-resolution results **in one feed-forward pass**.

**实验部分**

这部分无非是验证拉普拉斯金字塔网络的结构有效性，loss function的有效性。详见论文。

**Limitations**

![image](https://user-images.githubusercontent.com/36061421/121763726-d7122b80-cb70-11eb-9abd-c1fa0b7fe515.png)

第一个不足是：当原本image的结构信息太少的时候，难以进行超分复原。这种情况多发生在8x降采样之后，这时图像包含的信息太少了。

第二个不足是：网络size有点大。如果要去reduce parameters，可以replace the deep convolutional layers at each level with recursive layers.

最后谈谈个人意见：如果一口气需要2x，4x，8x的结果，这个网络是方便的；如果需要large scale factor的复原，这个网络也不错。但如果只是需要2x的结果，用这个网络可能没办法得到一个非常好的结果。通篇看下来，个人觉得progressively process SR problem的想法是本文很大的卖点，如果没办法很好地利用这个性质，可能没办法发挥整个网络的最佳功效。

[Table](#Table)

#### Image Super-Resolution Using Very Deep Residual Channel Attention Networks

NEU (Boston, USA) [SmileLab](https://web.northeastern.edu/smilelab/)

Director of the lab : [Yun Fu](http://www1.ece.neu.edu/~yunfu/) FIEEE FIAPR FSPIE FOSA; PhD & MS UIUC, MEng & BS XJTU.

RCAN

Image Super-Resolution Using Very Deep Residual Channel Attention Networks. ECCV 2018.

Official Code: [https://github.com/yulunzhang/RCAN](https://github.com/yulunzhang/RCAN) (PyTorch Version)

RCAN mainly concentrates on the allocation of channel-wise attention, which should be rescaled according to different image information in a distinct image. 

Novelty: RIR(Residual in Residual) Structure, Channel Attention and RCAN(Residual Channel Attention Network) and Long Skip Connection and Short Skip Connection.

[Table](#Table)

#### Toward Real-World Single Image Super-Resolution A New Benchmark and A New Model

PolyU [Visual Computing Lab](http://www4.comp.polyu.edu.hk/~cslzhang/)

Director of the lab: [Lei Zhang](https://www4.comp.polyu.edu.hk/~cslzhang/)  DAMO Academy && PolyU FIEEE PhD & MS NWPU

Jianrui Cai, Hui Zeng, Hongwei Yong, Zisheng Cao, Lei Zhang. Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model. ICCV 2019.

Paper Inplementation:

[Pytorch](https://github.com/Alan-xw/RealSR)  Reference article : [Zhihu](https://zhuanlan.zhihu.com/p/96973488)

[Matlab](https://github.com/csjcai/RealSR) **Official**

这篇文章指出了关于现有超分方法的不足，尤其是LR-HR pairs上的问题。如果仅仅是bicubic，没有办法体现自然图像中LR和HR之间的规律，最终学到的只是如何克服bicubic这个算法而已，并没有学到真实的分辨率规律。

这篇文章前半段提出一种更好的，搜集数据的办法。作者使用两台专业相机，通过调焦的方法实现高低分辨率图像对的采集，之后alignment再training，得到better结果。

文章后半段提出一种新的网络结构。

总体来说，文章的针对性比较强。因为文中使用的提取照片的方法，和绝大多数人获取照片的方式不同。这样的话，在图片本身的性质上，就会有特别的侧重点。文章提出的关于image matching的改进和后续网络的改进，都是针对于自己制作的RealSR数据集而言。因此从更宽泛的角度而言，这篇文章的很多思想不太好延展。但是RealSR的提出确实点到了目前SR方面的痛点，所以还是非常有意义的。推荐大家看一看。

[Table](#Table)







