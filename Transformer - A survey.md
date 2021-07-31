# Transformer - A survey

变形主要是 sparse attention降低复杂度

1. Fixed Patterns(FP)

    限制attention的范围 从全局变成局部 降低计算复杂度

    1. Blockwise Patterns

    把输入序列切成几个block 让attention只发生在block内 所以复杂度从$O\left(n^{2}\right)$降到$O\left(B^{2}\right)$ B是block size  但是这样切割会导致序列不连贯 attention能力受限 效果应该不太行

    1. Strided Patterns

    滑动窗口 每个token与周围相邻的几个token作attention 相邻的token范围就是window size 机制是因为自然语言在多数情况下都是局部相关 所以在一个窗口范围内作attention往往不会丢失太多信息 应该比之前的blockwise pattern好一些

    1. Compressed Patterns

    先通过卷积池化对序列进行降采样 比如用核为2 步长为2的CNN 把2个token表征成一个向量 然后再做attention 同样也能降低attention的计算复杂度  其实就是通过CNN对序列进行切分

2. Combination of Patterns

    通过结合多个访问模式来提高覆盖范围 降低了内存复杂度 multiple patterns提高了self-attention机制的覆盖率

    1. Sparse Transformer

    结合了strided和local attention： 把head的一半分给了pattern

    1. Axial Transformer

    给定一个高维张量  用一系列的self-attention分别沿着单一维度来计算

3. Learnable Patterns (LP)

    learnable patterns是对fixed patterns的扩展 fixed pattern是认为规定好一些区域 让该区域的token进行注意力计算 learnable patterns则是通过引入可学习参数 让模型自己找到划分区域

    1. reformer

    引入基于哈希的相似度度量方法将输入序列切割

    1. routing transformer

    对token向量进行k-means聚类来将整体序列分割成多个子序列

    本质上说LP与FP是一致的 都是通过将整体序列切分成子序列 attention只在子序列中进行 从而降低计算开销 LP的区域划分是通过模型学得 而FP是人为定义

4. Memory

    一般来说做multihead self-attention Q=K=V=X X为输入序列，长度为n 而在set transformer中先单独设置了m个向量（m是超参数） 然后这m个向量与X做multihead attention 得到m个临时向量（temporary memory） 然后把X与这m个临时向量再做一次multihead attention得到输出 其实就是用这m个向量将输入序列X的信息先通过attention进行压缩 再通过attention还原 抽取输入序列的特征 但是在压缩编码解码的过程中会有信息损失 所以后来的改进方法是引入全局记忆，即设置一些token 它们可以与所有的token进行注意力交互 由于这些token的数目远小于序列长度 因此也不会给计算带来负担 而且往往携带了整个输入序列的信息

5. Low rank methods

    经过softmax之后的N*N的注意力矩阵是不满秩的 所以不需要计算一个完整的注意力矩阵 因此我们可以将n*d维（n：序列长度 d：表示模型向量维度的K，V向量映射到k*d维空间 这样注意力矩阵就变成n*k维 k是超参数 所以整个的时间复杂度就降至O(n)了

6. Kernels

    以核函数变换的新形式取代原有的softmax注意力矩阵计算 将计算复杂度降至O(n)范围内

7. Recurrence

    recurrence也是fixed patterns中blockwise的一种延伸 本质上是对输入序列进行区域划分 不过它进一步的对划分后的block做了一层循环连接 通过这样的层级关系就可以把一个长序列的输入更好的表征