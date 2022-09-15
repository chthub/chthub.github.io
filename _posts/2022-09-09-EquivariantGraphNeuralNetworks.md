---
layout:     post
title:      "E(n) Equivariant Graph Neural Networks"
subtitle:   "E(n) 等变图神经网络"
date:       2022-09-09 12:00:00
author:     "Chthub"
# header-img: "imgs/image-20220909134601099.png"
catalog: true
mathjax: true
tags:
    - 等变图神经网络
---

这篇文章提出一种对平移，旋转，反射和排列等变的图神经网络 E(n)-Equivariant Graph Neural Networks

## Formulation

几何等变的图神经网络和一般的图神经网络的区别在于几何等变图神经网络在消息传递的过程中考虑到了空间坐标信息，同时要保持坐标信息对于空间变换的等变性。

<img src="https://raw.githubusercontent.com/chthub/everydaypaper/main/imgs/image-20220909134601099.png" alt="image-20220909134601099" style="zoom: 50%;" />

在这张图里面，对原图（左上角）进行旋转，节点的embedding没有发生变化，但是坐标的embedding变了。同时分别对原图和旋转之后的图消息传递之后得到的新的坐标的embedding来说，也存在一个等价的旋转变换。所以这里的等变性是一种约束，约束坐标的embedding不能随意改变，坐标的embedding要满足这种空间上的等变性。换句话说，我们希望学到的这种坐标的embedding确实是代表了空间位置信息，而不是其他的含义。此外，对于节点自身的embedding来说，它也能在更新的时候加入空间信息，使得节点的embedding更能具有空间上的唯一性，这一点在后面在自编码器的例子上的核心想法。

给定图 $\mathcal{G}=(\mathcal{V}, \mathcal{E})$，节点 $v_i \in \mathcal{V}$，边 $e_{i j} \in \mathcal{E}$，图卷积的消息传递过程如下：

$$
\begin{aligned}
\mathbf{m}_{i j} &=\phi_e\left(\mathbf{h}_i^l, \mathbf{h}_j^l, a_{i j}\right) \\
\mathbf{m}_i &=\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{i j} \\
\mathbf{h}_i^{l+1} &=\phi_h\left(\mathbf{h}_i^l, \mathbf{m}_i\right)
\end{aligned}
$$

这里 $\mathbf{h}_i^l \in \mathbb{R}^{nf}$ 是节点 $v_i$ 在 $l$ 层的embedding，$a_{i j}$ 表示边的属性，$\mathcal{N}(i)$ 是 $v_i$ 的邻居节点的集合。

本文中提出的Equivariant Graph Convolutional Layer (EGCL)的消息传递过程如下：

$$
\begin{aligned}
\mathbf{h}^{l+1}, \mathbf{x}^{l+1}&=\operatorname{EGCL}\left[\mathbf{h}^l, \mathbf{x}^l, \mathcal{E}\right]\\
\mathbf{m}_{i j} &=\phi_e\left(\mathbf{h}_i^l, \mathbf{h}_j^l,\left\|\mathbf{x}_i^l-\mathbf{x}_j^l\right\|^2, a_{i j}\right) \\
\mathbf{x}_i^{l+1} &=\mathbf{x}_i^l+C \sum_{j \neq i}\left(\mathbf{x}_i^l-\mathbf{x}_j^l\right) \phi_x\left(\mathbf{m}_{i j}\right),\; C=\frac{1}{|\mathcal{V}|-1} \\
\mathbf{m}_i &=\sum_{j \neq i} \mathbf{m}_{i j} \\
\mathbf{h}_i^{l+1} &=\phi_h\left(\mathbf{h}_i^l, \mathbf{m}_i\right)
\end{aligned}
$$

这里的$\mathbf{x}_i^l$表示节点$v_i$在$l$层的坐标的embedding。区别在于等变的GNN里面的$\mathbf{m}$引入了坐标信息，这里的坐标的embedding也是在每层都会更新。

这里有一个细节是$\mathbf{x}_i^{l+1}$的计算时考虑到了除$v_i$之外的所有其他节点，而不只是$v_i$的邻居。同时$\mathbf{m}$在聚集的时候也是在整个图上聚集而不是局限在邻居节点。

此外，在消息传递的时候还可以把节点的速度考虑进去，初始速度是$\mathbf{v}_i^{\text{init}}$：

$$
\begin{aligned}
\mathbf{h}^{l+1}, \mathbf{x}^{l+1}, \mathbf{v}^{l+1}&=\operatorname{EGCL}\left[\mathbf{h}^l, \mathbf{x}^l, \mathbf{v}^{\text {init }}, \mathcal{E}\right]\\
&\mathbf{v}_i^{l+1}=\phi_v\left(\mathbf{h}_i^l\right) \mathbf{v}_i^{\text {init }}+C \sum_{j \neq i}\left(\mathbf{x}_i^l-\mathbf{x}_j^l\right) \phi_x\left(\mathbf{m}_{i j}\right) \\
&\mathbf{x}_i^{l+1}=\mathbf{x}_i^l+\mathbf{v}_i^{l+1}
\end{aligned}
$$

注意到如果$\mathbf{v}_i^{\text{init}}=0$，那么这个式子和前面没有速度项的消息传递一样。

## 链接预测

文章中的这个模型可以用来做链接预测任务。如果一开始只提供了一个点云或者一个节点集合，此时如果以全连接的方式表示变，这种方法不能scale到节点数量很多的情况。本文提出了一种同时将消息传递和边的构建结合在一起的方法：

$$
\mathbf{m}_i=\sum_{j \in \mathcal{N}(i)} \mathbf{m}_{i j}=\sum_{j \neq i} e_{i j} \mathbf{m}_{i j}\\
e_{i j} \approx \phi_{i n f}\left(\mathbf{m}_{i j}\right),\; \phi_{i n f}: \mathbb{R}^{n f} \rightarrow[0,1]^1
$$

$\phi_{i n f}$就是一个带有sigmoid函数的线性层，对连边进行概率估计。

## N-body system

给定N个带点粒子，已知初始的电荷量，位置和速度，预测1000个时间戳后的位置。回归问题

## 图自编码器

图自编码器的解码器计算邻接矩阵出了直接内积恢复之外，还可以这样定义：

$$
\hat{A}_{i j}=g_e\left(\mathbf{z}_i, \mathbf{z}_j\right)=\frac{1}{1+\exp \left(w\left\|\mathbf{z}_i-\mathbf{z}_j\right\|^2+b\right)}
$$

直接内积的方式适合图变分自编码器，这种方式对GAE和VGAE都挺合适的。

然后是一个一般的图神经网络无法处理的一种情况：

<img src="https://raw.githubusercontent.com/chthub/everydaypaper/main/imgs/image-20220909142349775.png" alt="image-20220909142349775" style="zoom:50%;" />

对于这种4个节点连接起来的cycle graph，如果每个节点的embedding都是一样的，那么消息传递之后的节点embedding也是一样的，自编码器就无法重构节点之间的连边。

一种解决方法是在节点的embedding里面加入噪声，在这个例子里对每个节点的embedding加入了从$\mathcal{N}(\mathbf{0}, \sigma \mathbf{I})$中采样的随机噪声。这种方法的缺点是引入了额外的噪声分布，使得模型也不得不去学习这个分布。

另一种方法是在节点的embedding里面加入位置信息。(Paper: on the equivalence between positional node embeddings and structural graph representations)

然而在本文中，作者将随机采样的噪声作为位置坐标输入到模型里面，输出的节点embedding用于重构邻接矩阵。

实验结果如下，Noise-GNN就是在embedding里面加了随机噪声的GNN模型，右图是尝试过拟合训练集，pe是指训练集图的稀疏程度：

![image-20220909143644002](https://raw.githubusercontent.com/chthub/everydaypaper/main/imgs/image-20220909143644002.png)

可以看到EGNN的power是比Noise-GNN更强的。

还有一个实验是两个数据集上embedding维度的影响：

![image-20220909144035900](https://raw.githubusercontent.com/chthub/everydaypaper/main/imgs/image-20220909144035900.png)

还是维度越高，结果越好。