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


> **Note:**

> - This repo in currently under development.
> - The **final version** of it will be reached at the begining of **June 2017**.


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
 - Replicate the different combinations of content-style influence exhibited in the paper.
 - Apply the style of different paintings to one photograph and check if we can get similar results to the paper.
 - Test if different elements in a photograph (objects, background) are affected in the same way by the application of the painting style.
 
Evaluation
-------
There will not be an exhaustive quantitative evaluation. A qualitative approach will be carried out, comparing the obtained results with their counterparts on the original paper, in order to evaluate the success of the replication process.

