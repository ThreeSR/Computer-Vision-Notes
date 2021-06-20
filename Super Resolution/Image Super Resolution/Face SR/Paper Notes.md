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

整体结构分为global和local texture两个pathway。

*Global*

采用encoder和decoder的结构，先对图像的high-level features进行提取，在利用decoder进行返还。

*Landmark Located Patch Network*

分为四个网路，每一个针对于脸上的一处特征。最终将这些特征的内容级联在一起。

通过上面两个pathways之后，output是合在一起的feature maps。




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





