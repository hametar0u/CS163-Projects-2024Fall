---
layout: post
comments: true
title: Team 42 - ASL Fingerspelling
author: Jeffrey Kwan, Selina Song, Ishita Ghosh, Jason Cheng
date: 2024-12-10
---


> This block is a brief introduction of your project. You can put your abstract here or any headers you want the readers to know.


<!--more-->
{: class="table-of-content"}
* TOC
{:toc}

## Introduction
250,000 to 500,000 people use American Sign Language (ASL) in the US, making it the third most commonly used language after English and Spanish. ASL interpretation is important because it removes barriers to communication for deaf or hearing-impaired people, allowing more interactions between them and hearing people who do not know ASL. It also expedites communication during emergency situations by removing the need for an interpreter, and provides alternative accessibility options for smart home devices and mobile applications.

There are 5 components to a sign language: hand shape, location, movement, orientation, and non-manual. Non-manual components include facial expressions and slight nuances not directly measured through hand position that add meaning to the phrase, and often is the distinction between different words that have the same sign.

The scope of our project will be to extract words and phrases from fingerspelling for ASL which uses hand shape, orientation, and movement, especially letters that include moving parts. We will take as input short video sequences and classify them live, returning as output the prediction of the word, letter-by-letter. We will use the dataset called Chicago Fingerspelling in the Wild(+) which includes 55,232 fingerspelling sequences signed by 260 signers.
	
We evaluate and compare 3 different implementations of CSLR, Continuous Sign Language Recognition: MiCT-RaNet, a spatio-temporal approach, C^2ST, which takes advantage of textual contextual information, and Open Pose landmarks. We implement the MiCT-RaNet approach and modify it based on [INSERT IMPLEMENTATION OVERVIEW].

## Deep Learning Implementations
### 1. MiCT-RANet
#### Part 1: MiCT
In 3D convolutions, kernels slide along spatial and temporal dimensions of the input to make 3D spatio-temporal feature maps. The tensor size: `T x H x W x C` which stands for temporal duration, height, width, number of channels respectively. The output is `T' x H' x W' x N` where N=number of filters. The main problem with 3D convolutions are that they can be difficult to optimize with high memory usage and cost. The solution in this approach by Zhou et al. is a Mixed Convolution with 2D and 3D CNNs. MiCT is mixed in 2 ways, first by concatenating connections and adding cross-domain residual connections. The input is passed through a 2D convolution to extract static features then put through a 3D convolution and a cross-domain residual so the 3D convolution only needs to learn residual information along the temporal dimension. This reduces the total number of 3D convolutions needed, reducing model size, while maintaining high performance. Compared to other models like C3D which uses 8 3D convolutions, MiCT uses 4 3D convolutions with better performance and lower memory usage.

#### Part 2: RANet
RaNet is short for "Recurrent Attention Network". 

#### Part 3: Connectionist Temporal Classification
Not sure if this notation will be consistent but fuck it

At inference time, the model generates a frame-by-frame letter sequence $$Z$$ and collapses it to prediction $$\hat{Y}$$ via a label collapsing function $$\mathbb{B}$$. This typically involves some combination of removing blanks and deduplication.

At train time, MiCT-RANet optimizes the following loss function:

$$\mathcal{L}_{CTC} = -\log(\sum_Z P(Z \mid G))$$ (equation ??)

where $$P(Z \mid G) = \prod_sP(z_s \mid G)$$. Intuitively, the loss is the negative log likelihood of summing all the possible alignments (aka frame-by-frame letter sequences) $$Z$$.

### 2. C2ST
Zhang et. al (2023) argue that conventional CSLR methods don’t take advantage of contextual dependencies between words. This is because the CTC loss makes an independence assumption between gloss tokens [^1] - looking at equation ?? we see that each token $$z_s$$ is generated independent of each other.

To this end, they propose a framework that extracts and fuses contextual information from the text sequence with the spatio-temporal features, achieving SOTA performance of 17.7 WER on the Phoenix-2014 dataset, among others. Here is a general workflow (also see figure ?):

Let $$X = \{x_i\}_{i = 1}^N$$ be the input video with $$N$$ frames, which is divided into $$S$$ chunks, and $$Z = \{z_i\}_{i=1}^S$$ be the predicted alignment.


1. Pass $$X$$ into the spatio-temporal module to obtain visual features $$L = \{l_s\}_{s = 1}^S$$.
2. Let $$z_0 = \oslash$$. For time step $$s$$ in range $$\{1 .. S\}$$:
   1. Feed gloss sequence generated so far $$Y_{<s} = \{y_i\}_{i = 0}^{s-1}$$ into the language module to encode sequence feature $$b_{<s}$$. 
   2. Sum visual and sequence features $$l_s$$ and $$b_{<s}$$ to get fused feature $$j_s = l_s + b_{<s}$$.
   3. Generate $$z_s$$ by feeding $$j_s$$ into the classifier and using a decoding algorithm (such as beam search).
3. The final step of training and inference proceeds the same way as in CTC. More details on the loss function in a later section.

[^1]: In the context of fingerspelling, a gloss token is a single character, but in general, this could be a word or a part of a word. We will be using letter sequence and gloss sequence interchangeably in this report.
Spatio-Temporal Module Details
The paper uses the Swin-Transformer as the vision backbone for the spatial module, a 1D-TCN for the local temporal module, and a Bi-LSTM for extracting global temporal features.
Language Module Details
The paper uses a pre-trained BERT-base as the language model, fine-tuned on the target gloss sequences in the CSLR train-set. An adaptor layer is used to account for the variance difference in gloss sequences and human grammar.
Loss function details
The loss function used in the paper is a three-part loss function:
$$\mathcal{L} = \mathcal{L}_{CST}^c + \mathcal{L}_{CST}^v + \mathcal{L}_{KL}$$

$$\mathcal{L}_{CST}$$ is the negative log likelihood of the target sequence normalized by STC term:

$$\mathcal{L}_{CST} = -\log(\sum_Z \frac{P(Z \mid J)}{STC(Z, Y)})$$

Where $$p(Z \mid J) = \prod_s P(z_s \mid z_{<s}, J)$$. Here, $$P(Z \mid J)$$ is the probability of an “alignment” $$Z$$ as estimated by the decoder, and summing up all possible alignment paths $$Z$$ gives us the probability of the target sequence. The paper does this for the chunk and video levels (pass in local visual features $L$ for chunk level and global features $$G$$ for video level), hence $$\mathcal{L}_{CST}^c, \mathcal{L}_{CST}^v$$. The STC function is unspecified, but could be the edit distance between the ground-truth sequence $$Y$$ and predicted sequence $$Z$$ after removing the blanks. This helps pay more attention to potential sequences predicted by the model that have a smaller sequence-level error. Note that $$P(z_s \mid z_{<s}, J)$$ breaks the independence assumption by conditioning the decoder model on the previous tokens $$z_{<s}$$ as well. 

$$\mathcal{L}_{KL}$$ is a KL-divergence term to ensure consistency between chunk-level features $$J^c$$ and video-level features $$J^v$$.


### 3. OpenPose
Brock et al. utilized OpenPose to recognize non-manual content in a continuous Japanese sign language with two main components. First, they used supervised learning to do automatic temporal segmentation with a binary random forest classifier, classifying 0 for transitions and 1 for signs. After frame wise label prediction and segment proposal, they used a segment wise word classification with a CNN, then translating those predicted segment labels to sentence translation.

## MiCT-RANet Implementation Details
Training code and fine-tuned model weights were not given, so we had to train from scratch. Due to compute constraints, we were only able to train on 1000 sequences and validate on 981. Originally following the paper’s hyperparameters, we found that SGD with `lr=0.01` and `momentum=0.9` caused exploding gradients, so we switched to Adam with `lr=0.001`. However, the model would now consistently predict empty sequences, and could not rectify that.











## Basic Syntax
### Image
Please create a folder with the name of your team id under /assets/images/, put all your images into the folder and reference the images in your main content.

You can add an image to your survey like this:
![YOLO]({{ '/assets/images/team42/MCN_loss2.png' | relative_url }})
{: style="width: 400px; max-width: 100%;"}
*Fig 1. YOLO: An object detection method in computer vision* [1].

Please cite the image if it is taken from other people's work.


### Table
Here is an example for creating tables, including alignment syntax.

|             | column 1    |  column 2     |
| :---        |    :----:   |          ---: |
| row1        | Text        | Text          |
| row2        | Text        | Text          |



### Code Block
```
# This is a sample code block
import torch
print (torch.__version__)
```


### Formula
Please use latex to generate formulas, such as:

$$
\tilde{\mathbf{z}}^{(t)}_i = \frac{\alpha \tilde{\mathbf{z}}^{(t-1)}_i + (1-\alpha) \mathbf{z}_i}{1-\alpha^t}
$$

or you can write in-text formula $$y = wx + b$$.

### More Markdown Syntax
You can find more Markdown syntax at [this page](https://www.markdownguide.org/basic-syntax/).

## Reference
Please make sure to cite properly in your work, for example:

[1] Redmon, Joseph, et al. "You only look once: Unified, real-time object detection." *Proceedings of the IEEE conference on computer vision and pattern recognition*. 2016.

MODIFY BIB TO FIT FORMAT OF [1]

[2] Zhou et. al, MiCT: Mixed 3D/2D Convolutional Tube for Human Action Recognition (2018)
Prikhodko, A., Grif, M., & Bakaev, M. (2020). Sign language recognition based on notations and neural networks.

[3] Brock, H., Farag, I., & Nakadai, K. (2020). Recognition of non-manual content in continuous Japanese sign language. Sensors (Switzerland), 20(19), 1–21. 

[4] Koller, O., et al. "Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers." *Computer Vision and Image Understanding*. 2015.

[5] Zhang, H, et al. "C2ST: Cross-modal Contextualized Sequence Transduction for Continuous Sign Language Recognition," *2023 IEEE/CVF International Conference on Computer Vision (ICCV)*. 2023.

---
