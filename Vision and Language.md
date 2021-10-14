**书山有路勤为径，学海无涯苦作舟。**

visual grounding：它需要机器在接受一张图片和一个 query（指令）之后，「指」出图片当中与这个 query 所相关的物体。也叫做referring expression comprehension

VQA = visual question answering

Image Caption是一个融合计算机视觉、自然语言处理和机器学习的综合问题，它类似于翻译一副图片为一段描述文字。 caption：n. （图片的）说明文字；（电影或电视的）字幕；（法律文件的）开端部分。


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

具体的Transformer相关内容，详见[李宏毅老师的机器学习课件](https://speech.ee.ntu.edu.tw/~hylee/ml/2021-spring.html)，很详细。



[Table](#table)



