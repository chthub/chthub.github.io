---
layout:     post
title:      "数学公式测试"
subtitle:   ""
date:       2022-09-15 12:00:00
author:     "Chthub"
# header-img: "imgs/image-20220909134601099.png"
catalog: true
mathjax: true
---

进行数学公式测试，查看渲染结果。

下面都是得要求下划线后面加空格，否则报错。

## 例一
下划线渲染不正确：

这里 $\mathbf{h}_i^l \in \mathbb{R}^{nf}$ 是节点 $v_i$ 在 $l$ 层的embedding，$a_{i j}$ 表示边的属性，$\mathcal{N}(i)$ 是 $v_i$ 的邻居节点的集合。

## 例二
大括号转义失败：

$\left\{x_i\right\}_{i=1}^n$

应该用`\lbrace`和`\rbrace`命令：

$\left\lbrace x_i\right\rbrace_{i=1}^n$

## 例三
双斜杠换行：

$$
Q\left(c_i\right)=\frac{p\left(x_i, c_i ; \theta\right)}{Const}\\
\sum_{c_i \in C}Q\left(c_i\right)=1=\frac{1}{Const}\sum_{c_i \in C}p\left(x_i, c_i ; \theta\right)\\
Const=\sum_{c_i \in C}p\left(x_i, c_i ; \theta\right)
$$

## 例四
代码块必须要空行：
$$
f(x)=\int f(x)\sum_i^nf(x)
$$

## 例五
还是下划线的问题

损失函数的第一项作者把InfoNCE加入了进来，这里$v'_i$是和$v_i$属于相同簇的增广样本，$r$是负样本的数量。总共进行$M$次聚类，每次的类簇数都不一样$K=\left\lbrace k_m\right\rbrace_{m=1}^M$。通过这种方式可以提升原型的概率估计的鲁棒性，并能编码分层的结构。

## 例六

假设现在有数据集包含 $N$ 个输入和标签对 $\mathcal{D}=\left\lbrace \mathbf{x}_n, y_n\right\rbrace_{n=1}^N$ ，标签有 $C$ 类。输入样本$\mathbf{x}_n$的表征向量由映射$\mathbf{f}(\cdot ; \boldsymbol{\Theta})$计算：
