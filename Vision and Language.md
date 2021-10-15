# Vision & Language and Multimodal Learning

**书山有路勤为径，学海无涯苦作舟。**

本文档用于记录多模态学习。

***
# Table

+ [Term](#term)
+ [Related Research Group](#related-research-group)
+ [Self-Attention](#self-attention)
+ [Transformer](#transformer)
+ [BERT](#bert)

***

## Term

visual grounding：它需要机器在接受一张图片和一个 query（指令）之后，「指」出图片当中与这个 query 所相关的物体。也叫做referring expression comprehension

VQA = visual question answering

Image Caption是一个融合计算机视觉、自然语言处理和机器学习的综合问题，它类似于翻译一副图片为一段描述文字。 caption：n. （图片的）说明文字；（电影或电视的）字幕；（法律文件的）开端部分。

[Table](#table)

***

## Related Research Group

Microsoft Research

[MSRA](https://www.msra.cn/)

[Jifeng Dai](https://jifengdai.org/) PhD & BEng THU

Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, Jifeng Dai.VL-BERT: Pre-training of Generic Visual-Linguistic Representations. ICLR 2020.

Microsoft Dynamics 365 AI Research

THU AIR

[Jingjing Liu](https://air.tsinghua.edu.cn/EN/team-detail.html?id=66&classid=8) PhD degree in Computer Science from MIT EECS

Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, Jingjing Liu. UNITER: UNiversal Image-TExt Representation Learning. ECCV 2020.
***

Oregon State University

[Stefan Lee](http://web.engr.oregonstate.edu/~leestef/)

Jiasen Lu, Dhruv Batra, Devi Parikh, Stefan Lee. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks. arXiv preprint arXiv:1908.02265.

Supplementary Material

[Zhihu](https://zhuanlan.zhihu.com/p/101511981)

[GitHub](https://github.com/jiasenlu/vilbert_beta)
***

The University of Adelaide [V3A Lab](https://v3alab.github.io//)

[Qi Wu](http://www.qi-wu.me/) Assistant Professor, PhD in Computer Science from the University of Bath,  MS Advanced Computer System: Global Computing & Media Technology, University of Bath, BA Information and Computing Science, China Jiliang University.

Yanyuan Qiao, Chaorui Deng, Qi Wu. Referring Expression Comprehension: A Survey of Methods and Datasets. IEEE TMM.

***

UW, Seattle 

[Yejin Choi](https://homes.cs.washington.edu/~yejin/)  Ph.D. in Computer Science at Cornell University and BS in Computer Science and Engineering at Seoul National University

[Ali Farhadi](https://homes.cs.washington.edu/~ali/) 

[Allen Institute for Artificial Intelligence AI2](https://allenai.org/)

Rowan Zellers, Yonatan Bisk, Ali Farhadi, Yejin Choi. From Recognition to Cognition: Visual Commonsense Reasoning. CVPR 2019 oral.

Jae Sung Park, Chandra Bhagavatula, Roozbeh Mottaghi, Ali Farhadi, Yejin Choi. VisualCOMET: Reasoning about the Dynamic Context of a Still Image. ECCV 2020 Spotlight.

***

Cornell 

[Cornell Graphics and Vision Group](https://rgb.cs.cornell.edu/)

[Noah Snavely](http://www.cs.cornell.edu/~snavely/) Cornell Tech CSE PhD at UW, Seattle, BS at University of Arizona

Claire Yuqing Cui, Apoorv Khandelwal, Yoav Artzi, Noah Snavely, Hadar Averbuch-Elor. Who's Waldo? Linking People Across Text and Images. ICCV 2021 Oral.

[Table](#table)
***

## Self-Attention

<!-- 传统的RNN在处理文本的时候，会遇到一些问题。其中一个比较棘手的问题是：在处理机器翻译的时候，输入的文本长度并不固定。这样一来，如果我们的输入长度定得太短，那么有一些信息就没办法同时被网络考虑，导致一些上下文的关联信息遗漏；如果我们定得足够长，那对于很短的文本，会造成资源的浪费。 -->
在处理一段文本的时候，我们一般倾向于使用RNN。但RNN的问题在于，它是一个time sequence。也就是说，需要一个个时点地产生输出，接下来的这个输出再被当作下一个输入。因此，处理长文本时，没办法实现并行化（parallel）。也就是说，RNN相当于人眼从左往右扫视的过程，没办法一下子看清一整个句子。当然了，可以有双向RNN，但是双向RNN也是基于一个个time step最终得到的，而不是一下子得到。所以在parallel的方面，RNN不好用。

针对于parallel，其实可以考虑CNN。因为CNN是针对一片区域进行计算的，所以对于parallel的处理有帮助。

![image](https://user-images.githubusercontent.com/36061421/137382212-ceea9e4d-f51d-44f2-88cc-fc78d970a435.png)

如上图，如果是这样处理，随着深度的增加，后面的感受野越大，其实越能看到更大范围的句子。但这样做，深度可能会比较深，运算开销会很大。

所以综上，使用传统CNN和RNN都不太能够更好满足NLP的发展需求。

在这种时候，提出了`self-attention`的想法。

自注意力机制最早用于NLP领域，不同于传统的RNN（GRU,LSTM...），它可以一次性考虑非常多的语言信息，一并处理。但问题是positional encoding的方法。目前，没有最佳的positional encoding的解决方案，还在研究。在最早的《attention is all you need》中，作者提出的是一种`Sinusoidal`的方法。

（具体的`self-attention`相关内容，详见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)。）

![image](https://user-images.githubusercontent.com/36061421/125224572-76594880-e300-11eb-8b35-4277dda6d550.png)

![image](https://user-images.githubusercontent.com/36061421/125224583-7c4f2980-e300-11eb-8b04-7f2c4592fe1d.png)

![image](https://user-images.githubusercontent.com/36061421/125224647-9557da80-e300-11eb-8b11-1d2403de040c.png)

![image](https://user-images.githubusercontent.com/36061421/125224670-9ee14280-e300-11eb-8316-35e6c65d23ad.png)

![image](https://user-images.githubusercontent.com/36061421/125224707-aa346e00-e300-11eb-8d9f-a057a653c4d5.png)

除了上述的`self-attention`，还有`multi-head attention`：

![image](https://user-images.githubusercontent.com/36061421/137379471-ca61f5db-2971-442e-9eac-33eb85c46473.png)

这种attention，在原有的基础上，q、k、v进行了进一步的分支，乘上了各自的矩阵，所以变成了`multi-head attention`。

`Self-attention`这样的结构在`transformer`和`BERT`中均有所应用。实际上，不光是NLP领域有应用，CV领域也有。这部分内容连接了CV和NLP，是多模态学习中的重要部分。

上面展现了self-attention的运算过程，但针对其逻辑上的合理性，还没有更深入的分析。前面只是提及了为什么要有这么一种结构，这种结构究竟能为我们解决什么问题。

最后，来比较一下CNN、RNN与self-attention之间的关联。

CNN & Self-attention

如果把self-attention用在图像上面，上面的图片已经展示了这个过程。在处理过程中，一张解析度5 * 10的image，实际上是5 * 10 * 3的tensor。这时候，vector a就是一个pixel point对应的三维向量。一整个句子就是由5 * 10个pixel points组成的，每个像素点相当于一个句子中的单词。这样一来，就可以算句子中各个单词之间的关联。其实就是算像素点之间的联系，最后可以用可视化的方式展现图片信息的关联。

![image](https://user-images.githubusercontent.com/36061421/137383646-11661e40-fb68-4c6a-a8bd-8a8b931b4718.png)

上图展现了CNN与self-attention之间的关系。之前说道，attention应用在一张图上之后，可以计算得到不同pixel之间的关联。在CNN中，我们人为规定一个感受野，只允许感受野中的像素点进行卷积操作。对于self-attention，是在整张图上面进行操作，然后得到哪些点应该更相关。所以CNN是人为限定下的简化attention，attention是更复杂的CNN。

![image](https://user-images.githubusercontent.com/36061421/137384126-e6a28a36-748c-43c0-923e-1e3247ad0a93.png)

进一步而言，attention能做更多事情，而CNN是有限的。在一定条件下，self-attention可以转换为CNN。从这样的角度来看self-attention的产生，可见其并非凭空生成，而是基于现有的结构进行的延拓。

RNN & Self-attention

二者主要在两个方面不同：1.self-attention可以更好地进行parallel处理，这点在一开始就有提及。即使是双向RNN能考虑全局信息，也需要在一个个时序的过程中，存储前面的内容，不如self-attention来得直接；2.在处理句间关系的时候，self-attention聚在一起处理，但是RNN需要一个个time处理，没那么方便。

所以可以看出来，self-attention可以比RNN更加便捷。在很多时候，RNN结构已经被self-attention取代。

![image](https://user-images.githubusercontent.com/36061421/137385971-34154548-245b-45cb-a104-35f0bd15ca89.png)

如果把我们的视野放宽，其实可以发现，graph也可以使用vector set进行表示。也就是说，self-attention也可以用在graph上面。

![image](https://user-images.githubusercontent.com/36061421/137386171-81fcbd14-de4b-44f9-b82e-61c19ef10f56.png)

[Table](#table)

***

## Transformer

具体的Transformer相关内容，可以详见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)，很详细。

在处理真实问题的时候，我们经常会遇到seq2seq的问题。处理seq2seq的模型很多，今天介绍的transformer就是其中之一。

![image](https://user-images.githubusercontent.com/36061421/137391669-d261a6aa-8578-4cfe-b70d-f7c628df4cdb.png)

一般来说，seq2seq的模型都有encoder和decoder两个组成部分。最早在14年9月就已经有人提及了seq2seq model。最右侧就是transformer，可以看出来，它也是遵循encoder和decoder的基本结构。

在学完上面的self-attention之后，直观而言，应该想到transformer中的self-attention应该遵循下面的基本结构：

![image](https://user-images.githubusercontent.com/36061421/137391991-5637412b-9204-4b4b-8840-e1d67e9ba30a.png)

但其实transformer中的结构更加复杂。实际情况中，不实用原本的结构也可以。可以自行针对不同问题设定不同的结构。

![image](https://user-images.githubusercontent.com/36061421/137392184-43c3987d-e884-4b3b-bbbb-993d9d10db17.png)

上图展示了transformer中使用self-attention的过程。相较于上面naive的版本，增加了layer norm和residual的过程。layer norm不用于batch norm，它的过程就是算mean和std，最后归一化。

![image](https://user-images.githubusercontent.com/36061421/137392411-ab7ee6bd-88f1-4ecf-8e6b-222421718695.png)

宏观地来看待这个结构。我们在一开始的时候加上`Sinusoidal`的positional encoding，之后接上上述结构，可以得到编码后的结果。

如上面所说，可以不同于原本的transformer结构：

![image](https://user-images.githubusercontent.com/36061421/137392795-a0a6f5d0-b3c2-416a-ae87-ba8e4af0795d.png)

以上，就是transformer的encoder部分。

下面，来介绍一下decoder的部分：

decoder一般分为两种，一种是`Autoregressive(AT)`，另一种叫做`Not Autoregressive(NAT)`。其中，AT比较符合我们常见的序列化建模。

Autoregressive(AT)

![image](https://user-images.githubusercontent.com/36061421/137436347-63ac14ac-2a6c-4964-b33a-c8e34c72927f.png)

假如这是一个seq2seq的语音辨识系统，输入一段语音。通过encoder之后，生成四个vector。这四个vector将会送进decoder。与此同时，decoder也会有自己的输入。比如在句子开始的时候，会有BOS（begin of sentence），在句子结尾的时候，会有END。decoder在接收两方的输入之后，会生成自己的输出。这个输出经过softmax之后，会是一种概率分布。显然，最高的那一个就是本次的结果。

对于此时输出的vector，其相应的维数在不同语言中可以不同。比如中文，常见汉字就是3、4千个，那么维数控制在4000以内就够用了（独热编码）；如果是英文，可以选择词根词缀、26个字母、常用单词作为维数的考量。

AT的意思就是，后续的输入是根据前面的输出而来的。比如以下图为例：

![image](https://user-images.githubusercontent.com/36061421/137437019-ac74e7d7-01cd-48ed-a2c4-394d7d29452c.png)

前面出来的文字会参与到后面的输入之中。这时候会出来一个问题，就是如果中间某个字出了问题，后续整个序列会不会整段垮掉？这个问题后面再说，先skip。

![image](https://user-images.githubusercontent.com/36061421/137527525-fbf0491e-838b-4bcc-b768-75a35a6d6290.png)

上图就是decoder的全部构造，接下来分析一下encoder和decoder之间的差异：

![image](https://user-images.githubusercontent.com/36061421/137527732-2f4e1b03-4319-440c-9a26-cdae943b97d3.png)

如果用灰色区域遮住部分decoder，那么会发现encoder和decoder基本一样。当然了，还是有不同。比如：最后的输出是经过softmax之后的概率分布；multi-head attention是masked。softmax的存在比较好理解，接下来分析一下什么是masked multi-head attention：

这里我们稍微复习一下什么是self-attention：

![image](https://user-images.githubusercontent.com/36061421/137528168-47be8b20-4f91-4b65-868f-11250e10cba7.png)

可以看出，self-attention的输出是看过所有输入之后的产出。对于masked情况，

![image](https://user-images.githubusercontent.com/36061421/137528319-60e67894-4c8f-44b2-9811-1a7520b1e071.png)

在输出b1的时候，只能看到a1，b2只能看到a1和a2。具体而言，相较于前面的self-attention的计算，我们这样计算masked：

![image](https://user-images.githubusercontent.com/36061421/137528519-146e7eca-4284-4ff5-a216-228644d56411.png)

那么问题来了，为什么要这么做？

回想一下前面的decoder结构，我们的decoder的输入是根据前面的输出得到的。因此，对于decoder而言，它应该是一步步地得到所有input的信息，而不是一股脑地得到所有内容。这样一来，这种masked attention结构就符合我们的要求。

接下来我们会比较前面提过的NAT和现在的AT之间的差异，不过在此之前，我再补充一下关于AT的输出长度的控制。

一般来说，我们会增加一个名为`END`的token。这样一来，句子就会在某个时点停住，不会无限地输出下去。

下面，我们比较NAT与AT：

![image](https://user-images.githubusercontent.com/36061421/137529977-cb512479-eba7-43c2-942a-10016b30bd17.png)

AT不必多说，NAT相较于AT，最大的不同在于NAT可以parallel处理消息，而AT不行。AT就像是传统的RNN思维，NAT就像是self-attention的思维。给NAT一堆BOS的token，它自然会产出一整个句子，而不是像AT那样一个个字地生成。

此外，由于BOS的token是在一开始的时候就提供的，所以除了并行处理这个优点之外，NAT还有便于控制输出长度的优点。

那么我们如何知道输出的长度应该是多少呢？

有两种方法：1.训练一个分类器，用这个分类器去预测输出长度应该是多少；2.直接设定一个很大的参数，比如300。我们能预知到这组输出不可能超过300，然后进行工作。一旦看到`END` token，那么后面的内容就都忽略，保留前面的内容即可。

这部分的最后，谈谈二者的性能比较：

相较于AT，NAT的效果一般没有AT来得好。在使用了很多trick的情况下，NAT才可以达到一般的AT效果。尽管如此，NAT研究的人还是很多。（至于为什么NAT效果不太好，应该是multi-modality的问题，详见李宏毅老师关于NLP的课件）

刚刚在比较encoder与decoder的时候，我们刻意遮住了decoder与encoder的连接部分。接下来，我们要分析这个连接部分：

![image](https://user-images.githubusercontent.com/36061421/137539958-3d3180bf-3f04-4e24-aeea-170607acbb75.png)

这个部分称为`cross attention`。

![image](https://user-images.githubusercontent.com/36061421/137540337-2e95d1aa-b907-4bb1-814a-8c1fbbb63dbe.png)

具体`cross attention`的做法详见上图。decoder这边的q和encoder那边的k、v接连运算，得到结果。

关于`cross attention`的具体流程，可以进行更改。实际上，也有paper去研究如何设定`cross attention`会有更好的效果。

至此，关于transformer的结构分析可以告一段落。现在分析如何train这个模型。

前面有提及decoder的输出。在经过softmax之后，可以得到一个概率分布。之后这个分布和独热码的GT进行cross-entropy，就可以得到loss，将各个loss加起来，minimize即可。

![image](https://user-images.githubusercontent.com/36061421/137541494-5985fa03-c93a-4d0a-ba6b-dd277d9bd748.png)

可以看出，这个过程和分类问题很像。其实在挑选输出字符的时候，就是一个分类的过程。

![image](https://user-images.githubusercontent.com/36061421/137541838-0622f06a-7922-45b5-957c-7442f983ebf8.png)

整体的过程如上图所示。有几点值得注意：1.注意END的存在；2.在training过程中，decoder的输入是GT。在train的过程中，decoder已经偷看到GT的内容。但实际操作过程中，我们没办法知道完全的正确答案，decoder的下一个字符输入应该是上一个的输出。显然，这里train与test有mismatch。如何解决接下来会讲。这里使用GT作为decoder的input的做法叫做`Teacher Forcing`。

下面谈一谈使用seq2seq模型的一些tips：

1.copy mechanism

![image](https://user-images.githubusercontent.com/36061421/137543478-ec58b604-3b52-4a96-ae79-34a6d92ccfec.png)

简单来说，在聊天机器人中，机器人可以使用我们输入的词汇。他们只需要复制，没必要创造新词汇。这样一来，聊天机器人的错误率可以下降。

除了聊天机器人，这个tip的应用场景还有文本摘要。我们也同样需要在文章中复制一些词汇，然后写在摘要里面。

![image](https://user-images.githubusercontent.com/36061421/137543643-0a4da09d-1583-4387-b056-ad0a7aa8bcdb.png)

值得注意的是，可能需要百万量级的文章才能把模型train起来，不然效果不好。

2.guided attention

seq2seq模型是个黑盒子，我们不能完全知晓它在做什么。因此，在一些任务上面，这样的模型会有比较奇怪的输出。我们可以用guided attention这种强制手段去降低意外发生的可能性。也就是强制地给予该有的attention。

3.beam search

![image](https://user-images.githubusercontent.com/36061421/137544727-a2d0da51-5e49-43e8-9179-e34cbf3660f6.png)

如上，在输出的时候，我们会有很多条路径。在贪心和局部最优的问题下，我们想要尽可能地找到更好的solution。这样的技巧就是`beam search`。

![image](https://user-images.githubusercontent.com/36061421/137545178-f9ca6497-b47a-4004-8f72-807d8377880c.png)

接下来分析下一个问题。在衡量的时候，train时使用的是cross-entropy，但是在test时，使用的是BLEU score。二者其实并没有很直接的关联。train的时候如果直接用BLEU score，那么不能微分，只能用一些强化学习的手段解决（把decoder当做是agent，BLUE score当做是reward，用RL去train）。因此，train和test时会出现一些偏差。

最后，把前面挖的坑填一下。我们之前说过，train和test之间有一部分的mismatch，这个叫做exposure bias：

![image](https://user-images.githubusercontent.com/36061421/137545954-a55b8d04-9c84-4179-be06-5416ff80aa79.png)

如果train的时候都给GT，那么在test时，一旦出现错误，就会导致error propagation，一步错步步错。那要怎么解决呢？其实直接在train的时候，也加一些error即可。也就是说，没必要全部是GT进行输入，GT之中也掺杂error即可。

这种技巧称为`scheduled sampling`：

![image](https://user-images.githubusercontent.com/36061421/137546343-451ea9ce-fda0-437c-a148-e738f8aedbb0.png)

[Table](#table)

***

## BERT




