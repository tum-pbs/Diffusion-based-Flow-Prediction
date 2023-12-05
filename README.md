# Diffusion-based-Flow-Prediction

This repository contains a framework for uncertainty prediction of Reynolds-averaged Navier-Stokes flows around airfoils using Denoising Diffusion Probabilistic Models (DDPM). It features code for generating a dataset evaluating the simulation uncertainty induced by simulation parameters, and training code for diffusion models, in addition to baselines using bayesian neural networks (BNNs) and heteroscedastic uncertainty models.



<img src="./pic.svg" style="zoom: 50%;" />



Full details can be found in our arxiv paper "Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models" : https://arxiv.org/XXXXX

Authors: Qiang Liu and Nils Thuerey

_Full abstract:_ Leveraging neural networks as surrogate models for turbulence simulation is a topic of growing interest. At the same time, embodying the inherent uncertainty of simulations in the predictions of surrogate models remains very challenging. The present study makes a first attempt to use denoising diffusion probabilistic models (DDPMs) to train an uncertainty-aware surrogate model for turbulence simulations. Due to its prevalence, the simulation of flows around airfoils with various shapes, Reynolds numbers, and angles of attack is chosen as the learning objective. Our results show that DDPMs can successfully capture the whole  distribution of solutions and, as a consequence, accurately estimate the uncertainty of the simulations. The performance of DDPMs is also compared with varying baselines in the form of Bayesian neural networks and heteroscedastic models. Experiments demonstrate that DDPMs outperform the other methods regarding a variety of accuracy metrics. Besides, it offers the advantage of providing access to the complete distributions of uncertainties rather than providing a set of parameters. As such, it can yield realistic and detailed samples from the distribution of solutions. All source codes and datasets utilized in this study are publicly available.  



# Additional information

Our work focuses on the probabilistic prediction of airfoil flows to evaluate the inherent uncertainty of flow simulation. For more research on deterministic prediction, please check out our previous work:

* [Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction): Airfoil flow predictions with UNet.

* [Coord-Trans-Encoding](https://github.com/tum-pbs/coord-trans-encoding): High-accuracy predictions of airfoil flow with C-shaped meshes.

Other physics-based deep learning works of our group can be found at https://ge.in.tum.de/publications/.



If you find this repository useful, please cite our paper via:

```latex
@misc{dbfp2023,
      title={Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models}, 
      author={Qiang Liu, Nils Thuerey},
      year={2023},
      eprint={XXXXXX},
      archivePrefix={arXiv},
}
```

# A quick how-to

We provide several notebooks to show how to use our code:

* generate_dataset.ipynb: How to use OpenFOAM to generate dataset in parallel.

* process_dataset.ipynb (in progress): How to split the dataset according to the number of samples and simulation parameters.

* train_network.ipynb (in progress): How to train networks for diffusion models, BNNs and heteroscedastic uncertainty models.

* sample.ipynb (in progress): How to sample from the solutions using diffusion models, BNNs and heteroscedastic uncertainty models.

# Datasets and Pre-trained Models

The full training data set will be availalbe for download here shortly.

The pre-trained network weights can be found in `pre_trained` folder, where you can find the weights trained with different random seeds and the corresponding network configuration. You can refer to `sample.ipynb` to see the details on how to use them.

------

Please let us know if you find anything doesn't work in the repository.
