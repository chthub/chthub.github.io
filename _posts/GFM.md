---
layout:     post
title:      "Graph foundation model"
subtitle:   ""
date:       2024-06-12 12:00:00
author:     "Chthub"
# header-img: "imgs/image-20220909134601099.png"
catalog: true
mathjax: true
tags:
    - GNN
    - LLMs
    - ICLR
---

<link rel="stylesheet" type="text/css" href="vanier.css">




{pause focus-at-unpause}
# Graph Foundation model
<p style="text-align: center;">
<br> <br> <br> <br> Hao Cheng <br> <br> Department of Biomedical Informatics  <br> The Ohio State University<br>06/12/2024
</p>

{pause #Contents up-at-unpause unfocus-at-unpause}
# Contents
- Background
- Taxonomy
  - GNN-based Models
  - LLM-based Models
  - GNN+LLM-based Models

- Summary

{pause up-at-unpause}
# Paradigms in deep learning
{#img1}
![alternative text](./img1.png)

![alternative text](./2.png)

{pause #img3 center-at-unpause up-at-unpause=img1}
<figure style="text-align: center;">
    <img src="3.png" alt="">
    <figcaption>Deep Graph Learning</figcaption>
</figure>

{pause #gfm}
**Graph foundation models (GFMs)** are pre-trained on broad graph data and can be adapted to a wide range of downstream graph tasks, expected to demonstrate **emergence** and **homogenization** capabilities.

{pause down-at-unpause}
<figure style="text-align: center;">
    <img src="4.png" alt="">
    <figcaption>Graph foundation models</figcaption>
</figure>

{pause #Emergence up-at-unpause=gfm}
**Emergence** means that the graph foundation model will exhibit some new abilities when having a large parameters or trained on more data.

{pause}
**Homogenization** means that the graph foundation model can be applied to
different formats of tasks, such as node classification, link prediction and graph classification.

{pause #advantages up-at-unpause=Emergence}
# Other advantages?
1. The reliance on unstructured text, causing the models to sometimes miss obvious logical entailments or hallucinate incorrect conclusions

{pause .block #solution1 unstatic-at-unpause}
Graphs can provide LLMs with a more explicit representation of relationships and entities, enabling them to reason more effectively and avoid hallucinations. 

{pause #img5 up-at-unpause=advantages}
<figure style="text-align: center;">
    <img src="5.png" alt="">
</figure>

{pause unstatic-at-unpause=img5}
2. LLMs are fundamentally limited by when they were trained, and it can be difficult to incorporate ‘fresh’ information about the state of the world which has changed

{pause .block #solution2 unstatic-at-unpause}
Graphs can provide LLMs with a more explicit representation of relationships and entities, enabling them to reason more effectively and avoid hallucinations. 

{pause static-at-unpause=solution1}

{pause static-at-unpause=solution2}

{pause #Taxonomy up-at-unpause}
# Taxonomy 
<figure style="text-align: center;">
    <img src="6.png" alt="">
</figure>

{pause up-at-unpause}
# GNN-based Models
## Backbone Architectures
### Message Passing-based
Each node aggregates information from its neighboring nodes, processes the information, and then sends messages to its neighbors in a series of iterative steps.
<figure style="text-align: center;">
    <img src="7.png" alt="">
</figure>

{pause}
### Graph Transformer-based
Treats the graph as if it were fully connected, meaning it considers and
measures the similarity between every pair of nodes in the graph. 

Having long-range modeling capability and strong expressive power.
<figure style="text-align: center;">
    <img src="8.png" alt="">
</figure>

{pause up-at-unpause}
## Pretraining
### Contrastive Methods
Goal: maximize mutual information between different views.

Contrastive learning in different scale:
<figure style="text-align: center;">
    <img src="9.png" alt="">
</figure>

{pause}
### Generative Methods
Goal: enable GNNs to understand the general structural and attribute semantics of graphs.

Includes: Graph Reconstruction and Property Prediction.

{pause up-at-unpause}
## Adaptation
### Fine-Tuning
Utilize a pre-training model to generate node embeddings or graph
embeddings, and subsequently fine-tune an external **task-specific layer** to generalize the pre-training model to downstream tasks.

{pause}
### Prompt-Tuning
The graph prompt not only requires the prompt “content” but also needs to know how to organize these
prompt tokens and how to insert the prompt into the original graph.

<figure style="text-align: center;">
    <img src="11.png" alt="" style="width: 80%;">
</figure>

{pause up-at-unpause}
# Current works
<figure style="text-align: center;">
    <img src="10.png" alt="" style="width: 100%;">
    <figcaption><small>From <i>Jiawei Liu et al. Towards Graph Foundation Models: A Survey and Beyond</i></small></figcaption>
</figure>

{pause up-at-unpause}
# Taxonomy 
<figure style="text-align: center;">
    <img src="6.png" alt="">
</figure>

{pause up-at-unpause}
# LLM-based models
## Backbone Architectures
{#graph2token}
### Graph to token
This approach entails the tokenization of graph information and imitates
the standardized input format of transformer-based models.
<figure style="text-align: center;">
    <img src="12.png" alt="">
</figure>

{#graph2text}
### Graph to text
{pause up-at-unpause=graph2token}

Describe graph information using natural language.
<figure style="text-align: center;">
    <img src="13.png" alt="" style="width: 90%;">
</figure>

{pause up-at-unpause=graph2text}
<figure style="text-align: center;">
    <img src="14.png" alt="" style="width: 90%;">
</figure>

{pause up-at-unpause}
## Pretraining
{#LM}
### Language Modeling (LM)
In the context of a text sequence represented as $s_{1: L}=\left[s_1, s_2, \cdots, s_L\right]$, its overall joint probability, denoted as $p\left(s_{1: L}\right)$, can be expressed as a product of conditional probabilities, as shown in equation:

$$
p\left(s_{1: L}\right)=\prod_{l=1}^L p\left(s_l \mid s_{0: l-1}\right) .
$$

Here, $s_0$ represents a distinct token signifying the commencement of the sequence. 

To model the context $s_{0: l-1}$, a neural encoder $f_{\text {nenc }}(\cdot)$ is employed, and the conditional probability is calculated as follows:

$$
p\left(s_l \mid s_{0: l-1}\right)=f_{l m}\left(f_{\text {nenc }}\left(s_{0: l-1}\right)\right) .
$$

In this equation, $f_{l m}$ represents the prediction layer. By training the network using maximum likelihood estimation (MLE) with a large corpus, we can effectively learn these probabilities.

A drawback of unidirectional language models is their encoding of contextual information for each token, which is solely based on the preceding leftward context tokens and the token itself. 

{pause up-at-unpause=LM}
### Masked Language Modeling (MLM)
In MLM, specific tokens within the input sentences are randomly masked, and the
model is then tasked with predicting these masked tokens by analyzing the contextual information in
the surrounding text. 

{pause up-at-unpause}
## Adaptation
### Manual Prompting
Manually create prefix style prompts.

### Automatic Prompting
Automatically generate prompts using LLM.

<figure style="text-align: center;">
    <img src="18.png" alt="" style="width: 70%;">
</figure>

{pause up-at-unpause}
# Related paper
<br>
<br>
<figure style="text-align: center;">
    <img src="15.png" alt="" style="width: 100%;">
</figure>

{pause up-at-unpause}
## Formulation
Let $f$ be the interface function to a generative Al model, which takes high-dimensional discrete input tokens $W$ and produces output in the same token space ( $f: W \mapsto W$ ).

Graphs $G=(V, E)$.

The goal in prompt engineering is to find the correct way to phrase a question $Q$ and graph $G$, such that the $f$ will return the corresponding answer $A, (Q \in W, A \in W)$ 

$$A=f(G, Q)$$

So the objective function is
$$
\max _{g, q} \mathbb{E}_{G, Q, S \in D} \operatorname{score}_f(g(G), q(Q), S)
$$

Here $g,q$ are encoders, $\mathrm{S}$ is a ground truth solution to $\mathrm{Q}$ in the traning dataset $D$.

<figure style="text-align: center;">
    <img src="16.png" alt="" style="width: 100%;">
</figure>

{pause up-at-unpause}
## Results of varying graph encoding function
<style>
 .container {
            display: flex;
            align-items: center; /* Vertical center alignment */
            justify-content: flex-start; /* Horizontal alignment, image on the left, text on the right */
            border: 0px solid #ccc;
            padding: 2px;
            max-width: 100%;
            margin: 50px auto;
            box-sizing: border-box;
        }

        .container img {
            margin-right: 10px; /* Space between image and text */
            max-height: 900px; /* Adjust the size as needed */
            height:10%;
        }

        .container .text {
            flex: 1; /* Allows the text to take up the remaining space */
            word-wrap: break-word;
        }
</style>

<div class="container">

<img src="17.png" alt="">

1. LLMs perform poorly on basic graph tasks

2. Simple prompts are best for simple tasks

3. Graph encoding function has significant impact on LLM reasoning
</div>

{pause up-at-unpause}
# LLMs lack a global model of a graph
<br>

They evaluate the performance of LLMs on disconnected nodes. 

In this task, they provide a graph description to the LLM, specifying the nodes and edges, and ask about the nodes that are not directly connected to a given node

The ZERO-SHOT prompting method achieved an accuracy of 0.5%, while the ZERO-COT, FEW-SHOT, COT, and COT-BAG methods achieved close to 0.0% accuracy. 

This is because the graph encoding functions primarily encode information about connected nodes, while not explicitly encoding information about nodes that are not connected.

{pause up-at-unpause}
# Current works
<figure style="text-align: center;">
    <img src="19.png" alt="" style="width: 100%;">
    <figcaption><small>From <i>Jiawei Liu et al. Towards Graph Foundation Models: A Survey and Beyond</i></small></figcaption>
</figure>

<br>
<br>

<figure style="text-align: center;">
    <img src="12.png" alt="" style="width: 100%;">
</figure>

{pause up-at-unpause}
# Taxonomy 
<figure style="text-align: center;">
    <img src="6.png" alt="">
</figure>

{pause up-at-unpause}
# GNN+LLM-based models
{#Backbone}
## Backbone Architectures
<figure style="text-align: center;">
    <img src="20.png" alt="" style="width: 100%;">
</figure>

{pause #GNN-centric up-at-unpause=Backbone}
> ### GNN-centric Methods
> Utilize LLM to extract node features from raw data and make predictions using GNN.
> 
> The textual dataset $T$ is annotated with task-specific labels Y, where $G=(V, E, T)$ and $T$ is the set of texts with each element aligned with a node in $V$. 
> 
> We can train GNN with the loss:
> $$
> \begin{aligned}
> & \operatorname{Loss}_{\mathrm{CLS}}=\mathcal{L}_\theta(\phi(\operatorname{GNN}(\operatorname{LLM}(T))), \mathrm{Y}), \\
> & \operatorname{Loss}_{\text {LINK }}=\mathcal{L}_\theta\left(\phi\left(\operatorname{GNN}\left(\operatorname{LLM}\left(T_{\text {src }}\right)\right), \operatorname{GNN}\left(\operatorname{LLM}\left(T_{\text {dst }}\right)\right), \mathrm{Y}\right)\right.
> \end{aligned}
> $$
> 
> where $\phi(\cdot)$ is the classifier for the classification task or similarity function for the link prediction task, $T_{\text {src }}$ and $T_{\mathrm{dst}}$ are the texts of the source node and the target node, respectively.
> 
> $\text{ Loss}_{\text {CLS }}$ and  $\text{Loss}_{\text {LINK }}$ are the loss of classification and link prediction task, respectively. 
> 
{pause unstatic-at-unpause=GNN-centric}
### Symmetric methods
Align the embeddings of GNN and
LLM to make better predictions or utilize the embeddings for downstream tasks.
$$
z^{l+1}=\operatorname{TRM}\left(\operatorname{CONCAT}\left(\hat{z}^l, h^l\right)\right)
$$
where TRM is the transformer, and $h^l$ is the output of $l$-th layer of transformer,$\hat{z}^l$ is the output of $l$-th layer of GNN.

{pause unstatic-at-unpause=GNN-centric}
### LLM-centric Methods
Utilize GNNs to enhance the performance of LLM

{pause up-at-unpause}
# Pretraining
## LM or MLM
...
{pause}
## Text-Text Contrastive Learning (TTCL)
$$
\operatorname{Loss}_{\mathrm{TTCL}}=\mathbf{E}_{x, y^{+}, y^{-}}\left[-\log \frac{\exp \left(k\left(x, y^{+}\right)\right)}{\exp \left(k\left(x, y^{+}\right)\right)+\exp \left(k\left(x, y^{-}\right)\right)}\right]
$$
where $\mathrm{E}$ is the expectation, $k$ is the score function, $y^{+}$is the positive sample and $y^{-}$is the negative sample. 

{pause}
## Graph-Text Contrastive Learning (GTCL)
$$
\begin{align}
\operatorname{Loss}_{\mathrm{NCE}} = -\frac{1}{N} \sum_{i=1}^N &\left[ y_i \log \left(k\left(\operatorname{LLM}\left(\mu_i\right), \operatorname{GNN}\left(\xi_i\right)\right)\right) \right. \\&+ \left. \left(1-y_i\right) \log \left(1-k\left(\operatorname{LLM}\left(\mu_i\right), \operatorname{GNN}\left(\xi_i\right)\right)\right) \right]
\end{align}
$$

where $\mu_i$ is the text representation, $\xi_i$ is the graph representation, and $k$ is a score function to predict the activity of a molecule. 

{pause up-at-unpause}
# Current works
<figure style="text-align: center;">
    <img src="21.png" alt="" style="width: 100%;">
    <figcaption><small>From <i>Jiawei Liu et al. Towards Graph Foundation Models: A Survey and Beyond</i></small></figcaption>
</figure>

<br>

<figure style="text-align: center;">
    <img src="20.png" alt="" style="width: 100%;">
</figure>

{pause up-at-unpause}
# Summary 
<figure style="text-align: center;">
    <img src="6.png" alt="">
</figure>

## Next step
1. Propose model architectures that go beyond the Transformer

2. Other methods for traning and adaptation, such as knowledge distillation, reinforcement learning from human feedback (RLHF) and model editing

3. Lack of killer Applications

4. Safety