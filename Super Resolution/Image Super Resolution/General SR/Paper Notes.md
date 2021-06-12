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
+ [Deep Laplacian Pyramid Networks for Fast and Accurate Super-Resolution](#deep-laplacian-pyramid-networks-for-fast-and-accurate-super-resolution)
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

上面的式子中，f(x, y)代表输入图像，g(x, y)代表锐化后的图像。显然，对于图像锐化的操作来说，重点在于带有拉普拉斯算子的那一项。这就很像残差学习。f(x, y)对应的是identity mapping的部分，g(x, y)对应着输出的部分，residual就是![](http://latex.codecogs.com/svg.latex?\nabla^2f(x, y))

[Table](#Table)

#### Accurate Image Super-Resolution Using Very Deep Convolutional Networks

Jiwon Kim, Jung Kwon Lee, Kyoung Mu Lee. Accurate Image Super-Resolution Using Very Deep Convolutional Networks. CVPR 2016.

Official Homepage: [https://cv.snu.ac.kr/research/VDSR/](https://cv.snu.ac.kr/research/VDSR/)

Notice: official code is written by **Matlab**.

However, there is still [PyTorch Version](https://github.com/twtygqyy/pytorch-vdsr).

Seoul National University (SNU)

Computer Vision Laboratory(CVLab)

Director of the Lab: [Kyoung Mu Lee](https://cv.snu.ac.kr/index.php/kmlee/) FIEEE(Class 2021), Member of KAST(the Korean Academy of Science and Technology), PhD USC

这篇文章很有意义，因为它融合了ResNet的思想，将残差学习带入到SR之中。关于为什么残差学习适用于SR，之后分析。

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

[Table](#Table)







