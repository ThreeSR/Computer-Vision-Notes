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
+ [](#url)
+ [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](#image-super-resolution-using-very-deep-residual-channel-attention-networks)
+ [Toward Real-World Single Image Super-Resolution: A New Benchmark and A New Model](#toward-real-world-single-image-super-resolution-a-new-benchmark-and-a-new-model)


# Preface

The paper listed in this file will be in chronological order. I will provide you with hyperlink related to authors, their background and their labs.

Some abbreviations you may need to know before next part:

`NAE = National Academy of Engineering` `NAS = National Academy of Science` `CAS = Chinese Academy of Science` `CAE = Chinese Academy of Engineering`

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
+ 第一个是**不同channel的超分** 。以前的文章会谈论到gray-scale或者single-channel的超分，这些超分的关注点在于luminance channel。当然了，除了single-channel的内容，还有团队将RGB的各个channel进行超分，最后综合在一起。不管是single-channel还是各个channel的综合，前人并没有分析`不同channel带来的影响`和`恢复全部channel的必要性`。

SRCNN is inspired by `Image Super-Resolution via Sparse Representation`. The setting of CNN layers and 概念上面的迁移 都是和`Image Super-Resolution via Sparse Representation`对应的。

下面是SRCNN的结构图：

![image](https://user-images.githubusercontent.com/36061421/120927286-fd6c3d00-c712-11eb-8604-8863d929edbb.png)

处理的步骤：
+ 首先，将LR图像upscale到desired size，upscale的方法是`bicubic interpolation`。interpolated image用`Y`表示，这时候`Y`和ground truth（GT） image `X`是同样的size。为了表达的方便，称`Y`为LR图像。
+ 完成上面的预处理之后，网络主要完成三个步骤：
 


[Table](#Table)

#### Deep Residual Learning for Image Recognition

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. CVPR 2016 Best Paper Award.

[Paper](https://arxiv.org/abs/1512.03385)

[Kaiming He](http://kaiminghe.com/) FAIR, PhD CUHK MMLab, BEng THU.

This paper is really well-known because it provides us with a novel resolution to deal with deep networks.

Actually, ResNet is a model built for high-level CV tasks. But the concept of Residual Learning is similar to Laplacian Filtering in Digital Image Processing. We can employ this character to super-resolve images like VDSR did. 

尽管ResNet不是为了SR而生，但因为它的某些性质和SR十分契合，所以自VDSR之后，很多SR的网络里面都加入了残差学习的元素，比如identity mapping等内容...正是因为这个原因，对于学习SR的人而言，这篇文章也是必看的。

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

#### Image Super-Resolution Using Very Deep Residual Channel Attention Networks

[Table](#Table)

#### Toward Real-World Single Image Super-Resolution A New Benchmark and A New Model

[Table](#Table)







