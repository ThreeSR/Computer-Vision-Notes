# Face SR

Face Super Resolution is also named `face hallucination`（人脸幻象）. In fact, FSR is more frequently mentioned as face hallucination.

This file will contain selected papers for `face hallucination`.

**锲而舍之，朽木不折；锲而不舍，金石可镂。**

# Table

[Preface](#preface)

[Main Body](#main-body)
+ [Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis](#beyond-face-rotation-global-and-local-perception-gan-for-photorealistic-and-identity-preserving-frontal-view-synthesis)
+ [Disentangled Representation Learning GAN for Pose-Invariant Face Recognition](#disentangled-representation-learning-gan-for-pose-invariant-face-recognition)
+ [FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors](#fsrnet-end-to-end-learning-face-super-resolution-with-facial-priors)


# Preface

The paper listed in this file will be in chronological order. I will provide you with hyperlink related to authors, their background and their labs.

Some abbreviations you may need to know before next part:

`NAE = National Academy of Engineering` `NAS = National Academy of Science` `CAS = Chinese Academy of Science` `CAE = Chinese Academy of Engineering`

Notice: Chinese(Simplified) and English will be used in this file.

# Main Body

#### Beyond Face Rotation Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis

National Laboratory of Pattern Recognition(NLPR), CASIA

Member of NLPR: [Tieniu Tan](http://people.ucas.ac.cn/~tantieniu) Member of CAS, FIEEE, FIAPR, PhD & MS IC(UK), BEng XJTU

Member of NLPR: [Ran He](http://people.ucas.ac.cn/~heran) FIAPR, PhD CASIA, MS & BEng DUT

Director of NLPR:  [Tianzi Jiang](http://www.nlpr.ia.ac.cn/jiangtz/)  Chang Jiang Scholars, Member of Academia Europaea, PhD & MS ZJU, BS LZU

TPGAN

Rui Huang, Shu Zhang, Tianyu Li, Ran He. Beyond Face Rotation: Global and Local Perception GAN for Photorealistic and Identity Preserving Frontal View Synthesis. ICCV 2017.

[Pytorch Implementation](https://github.com/iwtw/pytorch-TP-GAN)

[TensorFlow Implementation(Original)](https://github.com/HRLTY/TP-GAN)

**前言**

本文提出了TPGAN模型，较好地解决了大角度侧脸对于正脸的还原，可以用于downstream的领域，比如：face recognition and attribution estimation.

尽管随着深度学习的应用，在许多benchmark datasets上，model的效果超过了人类，但是pose variations are still the **bottleneck** for many real-world application scenarios.

Existing methods that address pose variations can be divided into two categories：
+ One category tries to **adopt hand-crafted or learned pose-invariant features**
+ While the other resorts to **synthesis techniques** to recover a frontal view image from a large pose face image and then use the recovered face images for face recognition.

针对于第一种方法，可以分为traditional和deep learning：
+ traditional的做法就是often make use of robust local descriptors such as Gabor, Haar and LBP to account for local distortions and then adopt metric learning techniques to achieve pose invariance.
+ DL的方法就是often handle position variances with pooling operation and employ triplet loss or contrastive loss to ensure invariance to very large intraclass variations.

凭借着朴素的情感，不难发觉这种基于数学和描述子的方法，比较依赖于现有的信息。如果现有的信息不够sufficient，那么最终的结果不会好。简单来说，如果得到的姿态非常large，比如侧脸的转角太大，那么用这类方法恢复成正脸的难度很高，不够有效。

对于第二种方法：
+ 早期的方式是utilize 3D geometrical transformations to render a frontal view by first aligning the 2D image with either a general or an identity specific 3D model. 这样的建模虽然有一定广泛性，但缺少fine details，会导致最终结果模糊；
+ 现在有了基于数据驱动的DL方法，并且也有比较好的结果，但结果还是不够清晰，需要进一步提升。

由于人脸合成本质是ill-posed问题，所以如果能利用较多的先验知识（priors），那么可以得到更好的结果。我们可以类比一下人类是如何重建一个人的正脸的：
+ 第一步：When human try to conduct a view synthesis process, we firstly infer the global structure (or a sketch) of a frontal face based on both our prior knowledge and the observed profile. 
+ 第二步：Then our attention moves to the local areas where all facial details will be filled out.

以上是一个从大局到细节的过程。基于这样的一个过程，作者提出了TPGAN的结构：a deep architecture with two pathways (TP-GAN) for frontal view synthesis. These two pathways focus on the inference of global structure and the transformation of local texture respectively.

**Contribution**
+ We propose a human-like global and local aware GAN architecture for frontal view synthesis from a single image, which can synthesize photorealistic and identity preserving frontal view images **even under a very large pose**
+ We combine **prior knowledge** from data distribution (adversarial training) and domain knowledge of faces (symmetry and identity preserving loss) to exactly recover the lost information inherent in projecting a 3D object into a 2D image space.
+ We demonstrate the possibility of a **“recognition via generation” framework** and outperform state-of-the-art *recognition* results under a large pose. Although some deep learning methods have been proposed for face synthesis, our method is the first attempt to be effective for the recognition task with synthesized faces.

(Note: **Frontal view synthesis**, or termed as **face normalization**, is a challenging task due to its ill-posed nature.)

**网络结构**

![image](https://user-images.githubusercontent.com/36061421/122665234-a0738b00-d1d8-11eb-8fab-5f4b0bd51b79.png)

generator整体结构分为global和local texture两个pathway。

*Global*

采用encoder和decoder的结构，先对图像的high-level features进行提取，在利用decoder进行返还。

*Landmark Located Patch Network*

分为四个网路，每一个针对于脸上的一处特征。最终将这些特征的内容级联在一起。

通过上面两个pathways之后，output是合在一起的feature maps。

除了generator，肯定还有discriminator。根据Goodfellow的论文，可以得到本问题背景下的min-max problem：

![image](https://user-images.githubusercontent.com/36061421/122665455-e2510100-d1d9-11eb-82f2-cd244c53eec2.png)

**Loss function**

本文的loss由四个部分组成：

Pixel-wise Loss

![image](https://user-images.githubusercontent.com/36061421/122665504-34922200-d1da-11eb-9459-42f5f542b9a7.png)

普通的像素级L1范数，虽然可能overly smooth，但是对于加快优化和superior performance依然重要。

Symmetry Loss

Specifically, we define a symmetry loss in two spaces, i.e. **the original pixel space** and **the Laplacian image space**, which is robust to illumination changes. 这样的原因下面再说。

![image](https://user-images.githubusercontent.com/36061421/122665619-d4e84680-d1da-11eb-95f3-2e6cf907c51f.png)

这个loss利用了人脸对称的先验知识。表达式可以看出，这是对输出结果分成左右两半，对这两半的内容进行约束。这样的好处有两个方面：1.generating realistic images by encouraging a symmetrical structure. 2.accelerating the convergence of TP-GAN by providing additional back-propagation gradient to relieve self-occlusion for extreme poses.

但实际中，不太可能一张脸一定左右对称，因为：1.人脸自身问题；2.光照不同。根据作者所说：Fortunately, the pixel difference inside a local area is consistent, and the gradients of a point along all directions are largely reserved under different illuminations. Therefore, the Laplacian space is more robust to illumination changes and more indicative for face structure. 因此，上面说two spaces并且使用了拉普拉斯space。

Adversarial Loss

![image](https://user-images.githubusercontent.com/36061421/122665770-c189ab00-d1db-11eb-85b1-54b1569cb538.png)

Identity Preserving Loss

![image](https://user-images.githubusercontent.com/36061421/122665866-73c17280-d1dc-11eb-8e31-f535fad7eb30.png)

使用Light CNN，将最后两层的内容相加。light cnn用于提取high-level features，可以提取到图像本身的identity，所以可以控制这个loss的大小来控制图像的最终质量。在人脸训练或者最终形成的过程中，identity的保留也是至关重要的。

Overall Objective Function

The final synthesis loss function is a weighted sum of all the losses defined above:

![image](https://user-images.githubusercontent.com/36061421/122665932-e3cff880-d1dc-11eb-965d-da450c05b813.png)

We also impose a total variation regularization Ltv on the synthesized result to reduce spike artifacts.

注：spike artifact为图像伪影，关于什么是图像伪影，推荐两个链接：[常见的伪影](https://wangwei1237.gitbook.io/digital_video_concepts/shi-pin-zhi-liang-du-liang/4_1_0_compressionlossartifactsandvisualquality/4_1_2_commonartifacts)，[图像压缩中常见的一些压缩伪影（compresaion artifact）](https://www.jianshu.com/p/dbeec7b682f3)。后续再细说。

**实验部分**

涉及的数据集：训练集：[MultiPIE](http://www.cs.cmu.edu/afs/cs/project/PIE/MultiPie/Multi-Pie/Home.html), a large dataset with 750, 000+ images for face recognition under pose, illumination and expression changes；使用的Light CNN的训练集MS-Celeb-1M；测试集：[LFW](http://vis-www.cs.umass.edu/lfw/)。（最终的结果中，LFW可以test出fine details，但色彩基调来源于MultiPIE）

训练超参数和时间：The training of TP-GAN lasts for one day with a batch size of 10 and a learning rate of 10−4. In all our experiments, we empirically set α = 10−3, λ1 = 0.3, λ2 = 10−3, λ3 = 3 × 10−3 and λ4 = 10−4.

大多数先前的工作只能复原60°以内的pose，太大的难以复原。但通过TPGAN，可以在适当的loss设置和enough data的条件下，复原出large pose的image。值得一提的是，TPGAN的方法没有使用到3D的先验知识，是纯粹使用2D data-driven的方法。

通过上面提及的数据集，可以得到下面的结果：

![image](https://user-images.githubusercontent.com/36061421/122666640-2693cf80-d1e1-11eb-8f00-5ca4ea1b4c7d.png)

文章提出TPGAN的一个关键目的在于“recognition via generation”，在本文之前，虽有frontal view合成的方法，但是recognition基本都翻车了。文章中还根据一篇人脸幻象的study，指出：使用CNN合成的高分辨率图像，用于recognition的时候，将会导致performance降低而不是增加。为了试试能不能打破这个“魔咒”，文章进行了充分的实验。具体的实验场景设置详见论文。

文中Feature Visualization部分用于TPGAN和Light CNN在分类上面的比较，结果是TPGAN的分类效果更好：

![image](https://user-images.githubusercontent.com/36061421/122666796-0b758f80-d1e2-11eb-9b65-116158a79e89.png)

实验的最后部分是消融实验，也就是算法分析。详见文章。

小结：

本文提出的TPGAN实现了“recognition via generation”的目标，得到了好的结果，这是非常有意义的。在loss function部分，Adversarial Loss和Identity Preserving Loss对最终的输出结果起到了关键作用，是目标得以达成的重要基石。
***

#### Disentangled Representation Learning GAN for Pose-Invariant Face Recognition

[Computer Vision Lab](http://cvlab.cse.msu.edu/), MSU

[Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)   FIAPR, PhD CMU, MS ZJU, BS Beijing Information Technology Institute

Luan Tran, Xi Yin, Xiaoming Liu. Disentangled Representation Learning GAN for Pose-Invariant Face Recognition. CVPR 2017.

#### FSRNet End-to-End Learning Face Super-Resolution with Facial Priors

Nanjing University of Science and Technology [Jian Yang](http://gsmis.njust.edu.cn/open/TutorInfo.aspx?dsbh=tLbjVM9T1OzsoNduSpyHQg==&yxsh=4iVdgPyuKTE=&zydm=L-3Jh59wXco=#Wdtd)

FSRNet

Yu Chen, Ying Tai, Xiaoming Liu, Chunhua Shen, Jian Yang. FSRNet: End-to-End Learning Face Super-Resolution with Facial Priors. CVPR 2018 SPOTLIGHT Presentation.

[Official Code](https://github.com/tyshiwo/FSRNet)

Related material: [Zhihu](https://zhuanlan.zhihu.com/p/54198784)





