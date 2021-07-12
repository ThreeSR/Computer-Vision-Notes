# ML Notes for CV

This file will contain some Machine Learning(ML) knowledge for Computer Vision(CV).

***
# Table

+ [Decision Tree](#decision-tree)
+ [Support Vector Machine](#support-vector-machine)
+ [Feature Extraction and Sparse Learning](#feature-extraction-and-sparse-learning)
+ [Auto Encoder](#auto-encoder)
+ [Self-Attention](#self-attention)
+ [Transformer](#transformer)
+ [Reinforcement Learning](#reinforcement-learning)
+ [Meta Learning](#meta-learning)
+ [Life Long Learning](#life-long-learning)
+ [Network Compression](#network-compression)

***
# Preface

在阅读CV的论文时，经常会看到一些机器学习的概念。这篇笔记的目的就是记录常见的机器学习概念，便于查阅。

***
# Main Body

## Decision Tree

![image](https://user-images.githubusercontent.com/36061421/125217876-4e63e800-e2f4-11eb-97f7-311a819566cf.png)

形如上面的样式，就是决策树的基本样式。

决策树学习的基本算法：

![image](https://user-images.githubusercontent.com/36061421/125217908-5cb20400-e2f4-11eb-9a8b-4ac81dd75e91.png)

对于决策树的生成，最重要的是第8行：选择最优的划分属性。

用下式定义`信息熵`，度量样本集合的纯度。

![image](https://user-images.githubusercontent.com/36061421/125218065-a995da80-e2f4-11eb-84bc-897864b01acf.png)

数值越小，纯度越高。

用下式表达信息增益：

![image](https://user-images.githubusercontent.com/36061421/125218076-b5819c80-e2f4-11eb-83ed-20947061e211.png)

一般而言，信息增益越大，意味着使用属性a来进行划分，获得的“纯度提升”越大。

除了这种度量方法，还有别的，比如：增益率和基尼指数。详见《机器学习》（周志华）P77。

从生成之后的决策树可以看出，这棵树很容易overfitting。这是因为这棵树学习到的东西包含了较多数据集独有的特征，这些特征并不具有代表性。为了降低overfitting的影响，可以进行剪枝（pruning）操作。

剪枝分为两种：1.预剪枝；2.后剪枝。

**预剪枝**就是在决策树一步步成型的过程中，进行划分前后的精度比较。如果划分后精度可以提升，那么可以继续生成，否则令当下结点为叶子结点，结束生成。

![image](https://user-images.githubusercontent.com/36061421/125218459-a18a6a80-e2f5-11eb-84bf-2ae2214bd3aa.png)

精度的判别就是让测试集在训练集的基础上进行测试。基于训练集，有了整棵树的下一步规划，这时候拿测试集数据来比较分支前后的精度变化。

可以看出，这种做法在一定程度上可以降低过拟合的风险，此外还可以降低运算复杂度。但由于这种方法是基于贪心的本质，所以最终的结果未必是最佳的，可能有欠拟合的风险，同样导致泛化能力不佳。

**后剪枝**就是在决策树生成之后，进行剪枝。剪枝与否的标准也是在于验证集精度的变化。如果精度提升，那么可以剪枝，反之不剪。

![image](https://user-images.githubusercontent.com/36061421/125219089-cc28f300-e2f6-11eb-82a9-b58e9cab1e40.png)

后剪枝可以保留更多分支，欠拟合的风险减小。与此同时，泛化能力比较好。但最大的问题是计算开销大。

[Table](#table)
***
## Support Vector Machine
Pending...

[Table](#table)
***
## Feature Extraction and Sparse Learning

由于数据量的庞大，我们可能遇到“维数爆炸”的问题。在机器学习的过程中，并不是所有数据都在起巨大作用。在海量的数据中，我们应该挑选“有意义”的数据。因此，数据降维很重要。

数据降维一般可以采用两种手段：1.PCA等降维手段；2.特征选择。这里介绍特征选择，PCA以后介绍。

特征选择一般分为三种：
+ 1.过滤式选择：过滤式方法先对数据集进行特征选择，然后再训练学习器，特征选择过程与后续学习器无关。这相当于先用特征选择过程对初始特征进行"过滤"，再用过滤后的特征来训练模型。知名方法有Relief算法，这是针对二分类问题的；
+ 2.包裹式选择：与过滤式特征选择不考虑后续学习器不间，包裹式特征选择直接把最终将要使用的学习器的性能作为特征于集的评价准则。换言之，包裹式特征选择的目的就是为给定学习器选择最有利于其性能、"量身定做"的特征子集。LVW (Las Vegas Wrapper)是一个典型的包裹式特征选择方法。它在拉斯维加斯方法(Las Vegas method)框架下使用随机策略来进行子集搜索，并以最终分类器的误差为特征子集评价准则；
+ 3.嵌入式选择：在过滤式和包裹式特征选择方法中，特征选择过程与学习器训练过程有明显的分别;与此不同，嵌入式特征选择是将特征选择过程与学习器训练过程融为一体，两者在同一个优化过程中完成，即在学习器训练过程中自动地进行了特征选择。（在嵌入式选择的章节中，还有涉及`岭回归`等内容。）

详细内容参见《机器学习》（周志华）P249。

**稀疏表示和字典学习**

前面的“稀疏”，指的是和学习任务无关，在特征选择过程中应该被剔除掉的内容。在这里，“稀疏”的含义有些不同。指的是在矩阵中，很多元素为0。这些0元素并不是规律地成行或者成列出现。

那么，研究这种“稀疏”有什么意义？

当样本具有这样的稀疏表达形式时，对学习任务来说会有不少好处，例如：线性支持向量机之所以能在文本数据上有很好的性能，恰是由于文本数据在使用上述的字频表示后具有高度的稀疏性，使大多数问题变得线性可分。同时，稀疏样本并不会造成存储上的巨大负担，因为稀疏矩阵己有很多高效的存储方法。

值得注意的是：我们要的“稀疏”应该是适当的。举例来说，如果在从事文档分类的任务，那么《康熙字典》就是“过度稀疏”，《现代汉语常用字表》就是“恰当稀疏”。

显然，在一般的学习任务中(例如图像分类)并没有《现代汉语常用字表》可用，我们需学习出这样一个"字典"为普通稠密表达的样本找到合适的字典，将样本转化为合适的稀疏表示形式，从而使学习任务得以简化，模型复杂度得以降低，通常称为"字典学习" (dictionary learning) ，亦称"稀疏编码" (sparse coding)。

这两个称谓稍有差别，"字典学习"更侧重于学得字典的过程，而"稀疏编码"则更侧重于对样本进行稀疏表达的过程。由于两者通常是在同一个优化求解过程中完成的，因此下面我们不做进一步区分，笼统地称为字典学习。

给定数据集{Xl，X2，... ，Xm}，字典学习最简单的形式为：

![image](https://user-images.githubusercontent.com/36061421/125221392-c0d7c680-e2fa-11eb-8712-b4969e8177bf.png)

其中B∈R 为字典矩阵，k称为字典的词汇量，通常由用户指定， αi∈Rk 则是样本x∈Rd的稀疏表示。显然，上式的第一项是希望由α能很好地重构Xi，第二项则是希望αi尽量稀疏.

求解过程见P256。

**压缩感知**

现实生活中，希望通过少量信息重构全部信息。对于通信系统，如果想要恢复原始信号，需要满足奈奎斯特采样定理。那可否用一些方式，突破奈奎斯特采样定理，使用更少的内容恢复原始信号？这就是压缩感知要做的事情。

事实上，在很多应用中均可获得具有稀疏性的信号，例如图像或声音的数字信号通常在时域上不具有稀疏性，但经过傅里叶变换、余弦变换、小波变换等处理后却会转化为频域上的稀疏信号。

[Table](#table)
***

## Auto Encoder

基本结构：

![image](https://user-images.githubusercontent.com/36061421/125222843-583e1900-e2fd-11eb-9c2d-6beac503ee6b.png)

自编码器是一种self-supervised learning（自监督学习/自督导学习）的例子。对于自监督学习，就是通过不需要label的data进行学习，是无监督学习的一种。

可以发现，CycleGAN和自编码器的思想很接近。此外利用中间得到的vector，我们可以进行特征解耦（Feature Disentanglement）的操作。

这种特征解耦可以用在语音转换上：

![image](https://user-images.githubusercontent.com/36061421/125223288-1792cf80-e2fe-11eb-8501-4060d609e15f.png)

比如李宏毅老师的声音分为文字和个人腔调两部分，新垣结衣的声音也分为内容和音调两部分。两个人的语音讯息交叉，可以使得用李宏毅老师的腔调讲新垣结衣说的日文。

![image](https://user-images.githubusercontent.com/36061421/125223517-7ce6c080-e2fe-11eb-8f34-945c854ac1ba.png)

除了上面的应用，还可以用在：

**VAE**

![image](https://user-images.githubusercontent.com/36061421/125223572-9425ae00-e2fe-11eb-89d7-92086de7580e.png)

**Compression**

![image](https://user-images.githubusercontent.com/36061421/125223614-a56eba80-e2fe-11eb-99f3-362c9d17ee37.png)

下面谈谈**Representation learning**

什么是representation？

![image](https://user-images.githubusercontent.com/36061421/125223816-fa123580-e2fe-11eb-8231-f1b00f95dd77.png)

Good representations are:
1. Compact
2. Explanatory
3. Disentangled
4. Interpretable

![image](https://user-images.githubusercontent.com/36061421/125223939-2c239780-e2ff-11eb-969a-f7d71ac6fac6.png)

![image](https://user-images.githubusercontent.com/36061421/125224059-6856f800-e2ff-11eb-9375-be59d85a5505.png)

[Table](#table)
***

## Self-Attention

自注意力机制最早用于NLP领域，不同于传统的RNN（GRU,LSTM...），它可以一次性考虑非常多的语言信息，一并处理。但问题是positional encoding的方法。目前，没有最佳的positional encoding的解决方案，还在研究。

具体的self-attention相关内容，详见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)。

![image](https://user-images.githubusercontent.com/36061421/125224572-76594880-e300-11eb-8b35-4277dda6d550.png)

![image](https://user-images.githubusercontent.com/36061421/125224583-7c4f2980-e300-11eb-8b04-7f2c4592fe1d.png)

![image](https://user-images.githubusercontent.com/36061421/125224647-9557da80-e300-11eb-8b11-1d2403de040c.png)

![image](https://user-images.githubusercontent.com/36061421/125224670-9ee14280-e300-11eb-8316-35e6c65d23ad.png)

![image](https://user-images.githubusercontent.com/36061421/125224707-aa346e00-e300-11eb-8d9f-a057a653c4d5.png)

[Table](#table)

***
## Transformer

具体的Transformer相关内容，详见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)，很详细。

[Table](#table)

***
## Reinforcement Learning

类比于传统的ML步骤，可以得到RL的步骤：

![image](https://user-images.githubusercontent.com/36061421/125226348-81fa3e80-e303-11eb-89e0-db95ea0e8e36.png)

![image](https://user-images.githubusercontent.com/36061421/125226357-8888b600-e303-11eb-812e-208d0d22c231.png)

![image](https://user-images.githubusercontent.com/36061421/125226383-92121e00-e303-11eb-956e-9374f8c7c543.png)

![image](https://user-images.githubusercontent.com/36061421/125226406-9cccb300-e303-11eb-9a99-f9408f2de444.png)

强化学习就是根据一个agent跟环境（environment）进行互动，得到相应reward，从而学习的过程。这是一个不断试错的过程，最终机器可以学会什么是对的，什么应该做。所以严格来说，强化学习和前面的监督学习与非监督学习并不相同。

强化学习的难点在于如何进行optimization的过程。如上图所示，环境s会对actor进行影响，actor就是agent，reward就是agent做完action之后，根据环境得到的反馈。由于环境的不确定性，以及有时候reward不易定义，所以导致强化学习的训练是很艰难的，而且随机性很大。

在上面说到的三者中，actor是一个神经网络结构，可以拿去train。既然要train，就需要一个loss function。对于RL，loss应该来自于reward的相关定义。

下面介绍`policy gradient`：

对于reward的处理，假如当前reward只考虑一开始的reward到现在为止，没有考虑后面的，那显然是short-sighted；

如果reward是从开始到最后直接相加，也会有问题。问题在于当下的动作一般不会比最后的动作对最后的reward影响更大。所以需要设定一个衰减因子。（除了上面的操作，还需要对reward减去一个baseline）

![image](https://user-images.githubusercontent.com/36061421/125236830-36ea2680-e317-11eb-9954-e272d8e1a6ce.png)

γ就是那个衰减因子。

对reward进行设定之后，下面是policy gradient的算法步骤：

![image](https://user-images.githubusercontent.com/36061421/125236943-6e58d300-e317-11eb-8a7d-6fb78c424544.png)

可以发现：**在每一次for循环的时候，都需要得到一堆新的数据，这些数据用来当做训练数据，之后用来进行优化的步骤**。这一点不同于一般的深度学习！原因在于：强化学习需要即时地与周围环境进行互动，得到当下的数据进行后续的修正。因此必须在得到新的θ之后，即时进行交互，得到一堆新数据之后再进行下一步操作。这点很重要，也揭示了强化学习不容易训练的本质。

![image](https://user-images.githubusercontent.com/36061421/125237258-fb039100-e317-11eb-9584-316ec5ef4543.png)

**On-policy and Off-policy**

上述介绍的方法就是on-policy的方法，off-policy的不同点在于它并不是每次for循环都采集一堆数据，而是有可能用别的交互数据来train当下的模型。这样一来，可能做到不需要每次循环都采样一堆数据。

具体的off-policy方法这里不涉及，大致来说，off-policy常用`Proximal Policy Optimization (PPO)`。

**Critic**

![image](https://user-images.githubusercontent.com/36061421/125238555-f6d87300-e319-11eb-85ff-99e166cdaab8.png)

Critic的作用就是预测衰减因子的数值。如果一个trajectory可以在短时间内结束，那么reward的估算不难。但是如果一个trajectory太长，我们就不太能在短时间内得到反馈。这时候就需要critic进行短时的预测，进行“未卜先知”，预测衰减因子的数值。

那么，如何得到critic的value function呢？

一般来说，有两种方法：1.Monte Carlo (MC) based approach；2.Temporal difference (TD) approach。

pending...

[Table](#table)

***
## Meta Learning

元学习就是学习如何学习。比如在训练神经网络的时候，需要调节很多超参数，这些超参数有时候靠运气也靠经验。能否有一种方法，可以让机器自己学会怎么调参，从而解放人类呢？这就是元学习的初衷。

但现实问题是，如果需要机器自己调参，我们需要另外train一个模型，让机器学习如何调参。在train新模型的时候，我们还是需要人工调参....（该来的都会有）

那么我们这么做的目的是什么？

其实还是想要探寻一种更普适的方法，让机器学习如何调参。虽然元学习本身也要调参，但可能会训练出一个泛化能力很好的模型，实现人们一次调参，多次利用的结果。这显然也是很有意义的。

基于元学习这种思想，我们还可以“套娃”操作。比如做一个“元元学习”，在元学习的基础上进行元学习。

[Table](#table)

***
## Life Long Learning

终身学习是基于原有模型，持续地令其学习更新。（不是人文意义上的终身都在学习）

**讲到LLL，有的人会想到迁移学习。这个跟迁移学习不一样。迁移学习从A迁移到B，我们只关注fine-tuning之后，B的表现怎么样。但在LLL中，我们还很关注A的表现怎么样。**

以一种很naive的眼光看这个问题，感觉只要一直不断地输入数据，持续地增加epoch去train，就会有好结果。但其实不然。在学习新的B时，模型对A的表现会下降，这叫做`catastrophic forgetting`。那么，为什么会这样呢？

![image](https://user-images.githubusercontent.com/36061421/125227817-267d8000-e306-11eb-95a6-0e6ed9f0adcf.png)

可以这么认为，就是在train新的任务之后，数据（比如训练好的最优权重）在高维空间中发生了偏移。在新的任务的驱使下，原本的最优权重θ偏移了。导致的结果就是在新任务上表现好，在老任务上表现下降，这就是`catastrophic forgetting`（灾难性遗忘）。

解决方案之一是`Selective Synaptic Plasticity`。原理如下：

![image](https://user-images.githubusercontent.com/36061421/125228011-88d68080-e306-11eb-859a-268b8b3124f7.png)

简单来说，就是在新任务与老任务之间做一个权衡，保全二者。

具体内容见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)。

[Table](#table)

***
## Network Compression

一般来说，网络压缩有以下几种方法：

**Network Pruning**

可以对网络进行剪枝。但要注意，剪枝之后的网络也应该是可以使用矩阵运算的。因为一方面，pytorch不方便编程结构很奇怪的网络，另一方面，GPU需要矩阵进行加速，如果网路太奇怪，GPU都没办法好好加速。网络剪枝之后，需要使用原本random init的参数，不能换参数，否则train不出来。
![image](https://user-images.githubusercontent.com/36061421/125228725-e3241100-e307-11eb-958a-24e1f47543ee.png)

**Knowledge Distillation**

知识蒸馏就是让一个large教师网络（teacher）教一个small学生网络（student），然后小的学生网络学会内容的同时，网络的开销也小。这样做的样式看上去像普通的监督学习，那么这么做的意义何在？

意义在于，学生网络不光可以学到正确答案，还可以学到次正确，次次正确的内容。比如对数字1进行识别，学生网络不光可以知道正确答案是1，还可以知道有一定几率判别结果为7，相当于学生网络学到了1和7之间的相似关联。这一点是一般的监督学习没办法给予的。因为监督学习是打好了标签，没办法很准确地给出具体哪个结果的概率是多少。这就是知识蒸馏不同于平常网络的意义。

![image](https://user-images.githubusercontent.com/36061421/125229137-adcbf300-e308-11eb-8d53-1b94e0e12059.png)

**Parameter Quantization**

参数量化指的是将原本8bit的数据压缩到更小的bit，甚至是1bit。这样显然可以降低memory和computation。这种做法是利用了数据的冗余。

![image](https://user-images.githubusercontent.com/36061421/125229239-ef5c9e00-e308-11eb-8e0d-7a188cc03836.png)

**Architecture Design**

显然，在宏观的网络结构上，也可以做文章。不同于上面的pruning，这里是从layer角度入手，而不是神经元。

使用了`Depthwise Separable Convolution`。不同于一般的CNN，Each filter only considers one channel，There is no interaction between channels。

这样做的原理时`Low rank approximation`：

![image](https://user-images.githubusercontent.com/36061421/125229646-d1dc0400-e309-11eb-8768-7553fae3bc41.png)

具体内容见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)。

这种中间“bottleneck”一下的手法，和NIN还是有异曲同工之妙的。

**Dynamic Computation**

这一部分就是在不同端（比如手机端，手表端），动态地调节网络的情况。以手机端为例，当手机电量很满的时候，网络结构完整，但手机电量不够的时候，网络进行压缩，减少电量消耗。

[Table](#table)







