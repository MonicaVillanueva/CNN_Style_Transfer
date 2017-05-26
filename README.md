CNN_Style_Transfer
===================


Authors
-------
This repository is being developed as part of the course [Deep Learning in Data Science (DD2424)](https://www.kth.se/social/course/DD2424/) at [KTH Royal Institute of Technology](http://kth.se), in the Spring 17 P4 round.

| Author               | GitHub                                            |
|:---------------------|:--------------------------------------------------|
| Sergio López | [Serloapl](https://github.com/Serloapl) |
| Mónica Villanueva | [MonicaVillanueva](https://github.com/MonicaVillanueva)     |
| Diego Yus | [DYusL](https://github.com/DYusL)       |

Content
-------

 - [style_transfer](https://github.com/MonicaVillanueva/CNN_Style_Transfer/tree/master/style_transfer): Code to run in local. It includes the hierarchy of folders to run the program sucessfully (mind the weights in aditional resources). For more information take a look at the readme.md.
 - [TegnerScripts](https://github.com/MonicaVillanueva/CNN_Style_Transfer/tree/master/TegnerScripts): version of the previous code used to run the program on [PDC](https://www.pdc.kth.se/).
 - [Reference papers](https://github.com/MonicaVillanueva/CNN_Style_Transfer/tree/master/Reference%20papers): Study of the literature related with the topic. The papers are highlighted and commented.
 - [Report](https://github.com/MonicaVillanueva/CNN_Style_Transfer/tree/master/Report): Includes the reading version of the overleaf project used and a PDF version of the final report.

----------


Description
-------------
The idea of the project is to extract the style of a painting and transfer it to another to a photography without losing its content, so that it generates images that combine the content and style from two different sources. We will try to reproduce the original concept, developed in the paper [“A Neural Algorithm of Artistic Style”](https://arxiv.org/pdf/1508.06576.pdf), understand the underlying theory and explore the quality of the output, as well as its limitations.

As suggested by the paper, we will use a **Convolutional Neural Network (VGG)** and gradient descent on a loss function defined as the squared-error loss between the feature representation of the painting and the photography. With this, we can trade between importance of the content and the style.


Dataset
-------
In general, and contrary to most usual machine learning projects, this project does not require the use of an extensive dataset. Instead, we will only make use of “original photographies”, that we desire to transform, and “painting images”, that will provide the style the characteristics of which we want to extract.
Therefore, and in order to facilitate the final qualitative evaluation, we will make use of the same or similar photographies and paintings used in the paper mentioned above.

Libraries
-------
The code will be developed employing **Tensor Flow**, an open library for machine learning originally created by Google.

Experiments
-------
 - Replicate the result of the papers using the same painting and same photograph used by them.
 - Replicate the different reconstructions of content and style in different layeres
 - Replicate the combinations of content-style (alpha/beta ratio) influence exhibited in the paper.
 - Test the transfer on video.
 
Evaluation
-------
There will not be an exhaustive quantitative evaluation. A qualitative approach will be carried out, comparing the obtained results with their counterparts on the original paper, in order to evaluate the success of the replication process.

----------

Aditional resources
-------------------

 - The weights for the VGG16 network in tensorflow format have been downloaded from [Davi Frossard's post on VGG](https://www.cs.toronto.edu/~frossard/post/vgg16/). It has not been included in this repo due to GitHub limitations.

> **Note:**

> - This already the **final version** of the repository.


