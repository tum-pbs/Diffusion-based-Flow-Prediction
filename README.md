<h1 align="center">
    Diffusion-based-Flow-Prediction
</h1>
<h6 align="center">Official implementation of AIAA Journal paper</h6>
<h3 align="center">"Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models"</h3>

<p align="center">
  [<a href="https://arc.aiaa.org/doi/10.2514/1.J063440">AIAA Journal</a>]•[<a href="https://arxiv.org/abs/2312.05320">Arxiv</a>]•[<a href=https://github.com/tum-pbs/Diffusion-based-Flow-Prediction/blob/main/assets/ICML_Poster.pdf>ICML Workshop Poster</a>]•[Blog (coming soon)]
</p>
<img src="./assets/main.svg" style="zoom: 50%;" />

This repository contains a framework for **uncertainty prediction** of Reynolds-averaged Navier-Stokes flows around airfoils using **Denoising Diffusion Probabilistic Models (DDPM)**. It features code for generating a dataset evaluating the simulation uncertainty induced by simulation parameters, and training code for diffusion models, in addition to baselines using **Bayesian neural networks (BNNs)** and **heteroscedastic uncertainty models**.




## Paper Info

<h4 align="center">Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models</h4>
<h6 align="center"><a href="mailto:qiang7.liu@tum.de">Qiang Liu</a> and <a href="mailto:nils.thuerey@tum.de">Nils Thuerey</a></h6>

<h6 align="center">
    <img src="assets/TUM.svg" width="32"> Technical University of Munich
</h6>
***Abstract***: Leveraging neural networks as surrogate models for turbulence simulation is a topic of growing interest. At the same time, embodying the inherent uncertainty of simulations in the predictions of surrogate models remains very challenging. The present study makes a first attempt to use denoising diffusion probabilistic models (DDPMs) to train an uncertainty-aware surrogate model for turbulence simulations. Due to its prevalence, the simulation of flows around airfoils with various shapes, Reynolds numbers, and angles of attack is chosen as the learning objective. Our results show that DDPMs can successfully capture the whole distribution of solutions and, as a consequence, accurately estimate the uncertainty of the simulations. The performance of DDPMs is also compared with varying baselines in the form of Bayesian neura networks and heteroscedastic models. Experiments demonstrate that DDPMs outperformthe other methods regarding a variety of accuracy metrics. Besides, it offers the advantageof providing access to the complete distributions of uncertainties rather than providing a set of parameters. As such, it can yield realistic and detailed samples from the distribution of solutions. 

***Read from:*** [<a href="https://arc.aiaa.org/doi/10.2514/1.J063440">AIAA Journal</a>]•[<a href="https://arxiv.org/abs/2312.05320">Arxiv</a>]

***Cite as:*** 

```latex
@article{Liu2024DDPM,
author = {Liu, Qiang and Thuerey, Nils},
title = {Uncertainty-Aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models},
journal = {AIAA Journal},
doi = {10.2514/1.J063440},
}
# volume and number information is currently not available
```



## Code Guide

### Notebooks

We provide several notebooks to show how to use our code:

* [generate_dataset.ipynb](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction/blob/main/generate_dataset.ipynb): How to use OpenFOAM to generate dataset in parallel.

* [process_dataset.ipynb](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction/blob/main/process_dataset.ipynb): How to post-process the dataset for training and analysis.

* [train_networks.ipynb](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction/blob/main/train_networks.ipynb): How to train networks for diffusion models, BNNs and heteroscedastic uncertainty models.

* [sample.ipynb](https://github.com/tum-pbs/Diffusion-based-Flow-Prediction/blob/main/sample.ipynb): How to sample from the solutions using diffusion models, BNNs and heteroscedastic uncertainty models.

By using these note books, you will be able to reproduce the results in our manuscript step by step. You can also run any of the notebooks directly, and we have prepared the corresponding preamble file.

### Datasets and Pre-trained Models

The full training data set is available for download at [here](https://mediatum.ub.tum.de/1731896). In this repository, we also provide a reduced version of our dataset corresponding to the single-parameter test in our manuscript. For more details, please see `\datasets\1_parameter` and `process_dataset.ipynb`. You can also use `generate_dataset.ipynb` to generate this dataset.

The pre-trained network weights can be found in `pre_trained` folder, where you can find the weights trained with different random seeds and the corresponding network configuration. You can refer to `sample.ipynb` to see the details on how to use them.

Please let us know if you find anything doesn't work in the repository.



## Additional information

Our work focuses on the probabilistic prediction of airfoil flows to evaluate the inherent uncertainty of flow simulation. For more research on deterministic prediction, please check out our previous work:

* [Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction): Airfoil flow predictions with UNet.

* [Coord-Trans-Encoding](https://github.com/tum-pbs/coord-trans-encoding): High-accuracy predictions of airfoil flows with adaptive meshes.

The application of diffusion models in the field of fluid dynamics is a very ascendant direction. If you are interested in this, you can refer to our other work below:

* [SMDP](https://github.com/tum-pbs/SMDP): Solving Inverse Physics Problems with Score Matching.

* [autoreg-pde-diffusion](https://github.com/tum-pbs/autoreg-pde-diffusion): Prediction of PDE Simulations using Autoregressive Conditional Diffusion Models (ACDMs).

Other physics-based deep learning works of our group can be found at https://ge.in.tum.de/publications/.