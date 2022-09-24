---
layout:     post
title:      "GNNExplainer: Generating Explanations for Graph Neural Networks"
subtitle:   "GNNExplainer: 生成图神经网络的解释"
date:       2022-09-24 12:00:00
author:     "Chthub"
# header-img: "imgs/image-20220909134601099.png"
catalog: true
mathjax: true
tags:
    - GNN
    - 可解释性
    - NeurIPS
---

Published on NeurIPS 2019

本文想要解释GNN的预测结果。如下图所示，对于节点 $v$， 解释$v$的预测结果分成了两部分，一个是解释$v$周围的子图，哪些节点和边对预测结果重要，哪些不重要。一部分是解释节点的特征，哪些维度的特征发挥了作用，哪些没有。

本文基于互信息最大化的框架，学习邻接矩阵和节点特征的mask，实现对预测结果的可解释性。

![image-20220924101937523](https://raw.githubusercontent.com/chthub/chthub.github.io/main/imgs/image-20220924101937523.png)

## Formulation

首先是一个关键的insight，当预测节点的标签时，GNN基于的是节点周围的邻居信息（文章中称之为computation graph）。即节点$v$的计算图$G_c(v)$告诉了GNN如何生成其embedding $\mathbf{z}$。计算图$G_c(v)$的邻接矩阵是$A_c(v) \in\lbrace 0,1\rbrace^{n \times n}$，节点的特征集合是$X_c(v)=\left\lbrace x_j \mid v_j \in G_c(v)\right\rbrace$。然后可以将GNN $\Phi$基于计算图的计算过程表示为：

$$
P_{\Phi}\left(Y \mid G_c, X_c\right)
$$

$Y\in \lbrace 1, \ldots, C\rbrace$表示节点的标签。

本文提出的GNNExplainer会对预测$\hat{y}$生成解释$(G_S,X_S^F)$。$G_S$是计算图的子图，$X_S$是对应的节点特征集合，$X_S^F$是节点的特征子集，由$F$进行mask。GNNExplainer可以用于单实例解释和多实例解释的情况。

### 单实例解释

给定节点 $v$, 我们的目标是识别对预测结果 $\hat{y}$ 重要的子图 $G_S \subseteq G_c$ 和关联的特征 $X_S=$ $\left\lbrace x_j \mid v_j \in G_S\right\rbrace$。我们使用互信息来建模这种重要性：

$$
\max _{G_S} M I\left(Y,\left(G_S, X_S\right)\right)=H(Y)-H\left(Y \mid G=G_S, X=X_S\right)
$$

当GNN训练好之后$\Phi$是固定的，所以第一项为常数。所以最大化互信息等价于最小化条件熵：

$$
H\left(Y \mid G=G_S, X=X_S\right)=-\mathbb{E}_{Y \mid G_S, X_S}\left[\log P_{\Phi}\left(Y \mid G=G_S, X=X_S\right)\right]
$$

如果 $G_S \sim \mathcal{G}$ 是随机的图变量，此时：

$$
\min _{\mathcal{G}} \mathbb{E}_{G_S \sim \mathcal{G}} H\left(Y \mid G=G_S, X=X_S\right)
$$

在凸假设下，基于Jensen不等式有如下的上界：

$$
\min _{\mathcal{G}} H\left(Y \mid G=\mathbb{E}_{\mathcal{G}}\left[G_S\right], X=X_S\right)
$$

然后这里有一些基于平均场变分近似的讨论，没看懂。

如果要回答诸如：

> why does the trained model predict a certain class label 
>
> or 
>
> how to make the trained model predict a desired class label

可以利用交叉熵修改损失函数如下：

$$
\min _M-\sum_{c=1}^C \mathbb{1}[y=c] \log P_{\Phi}\left(Y=y \mid G=A_c \odot \sigma(M), X=X_c\right)
$$

这里的$M$就是mask 矩阵，$\odot$表示逐元素的乘积。

为了学习哪些节点特征对预测 $\hat{y}$ 是重要的，GNNEXPLAINER学习一个特征选择器$F$。这里的$F$是一个mask向量。所以最终的目标函数如下：

$$
\max _{G_S, F} M I\left(Y,\left(G_S, F\right)\right)=H(Y)-H\left(Y \mid G=G_S, X=X_S^F\right)
$$

