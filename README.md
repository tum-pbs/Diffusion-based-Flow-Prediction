# Diffusion-based-Flow-Prediction

*Diffusion-based flow prediction (DBFP)* is a framework for uncertainty prediction of Reynolds-averaged Navier-Stokes flows around airfoils using Denoising Diffusion Probabilistic Models(DDPM). It contains generation code for a dataset evaluating the simulation uncertainty induced by simulation parameters, and network training code for diffusion models, bayesian neural networks(BNNs) and heteroscedastic uncertainty models.



<img src="./pic.svg" style="zoom: 50%;" />



Full details can be found in our arxiv paper "Uncertainty-aware Surrogate Models for Airfoil Flow Simulations with Denoising Diffusion Probabilistic Models" : https://arxiv.org/XXXXX



Author: Qiang Liu and Nils Thuerey



Our current work focuses on the probabilistic prediction of airfoil flows to evaluate the inherent uncertainty of flow simulation. For more research on deterministic prediction, please check out our previous work:

* [Deep-Flow-Prediction](https://github.com/thunil/Deep-Flow-Prediction): Airfoil flow predictions with UNet.

* [Coord-Trans-Encoding](https://github.com/tum-pbs/coord-trans-encoding): High-accuracy predictions of airfoil flow with C-shaped meshes.

Other physics-based deep learning works of our group can be found at https://ge.in.tum.de/publications/.



Our paper is currently being under review. If you find this repository useful, please cite our paper via:

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

Besides generation code, we also provide links to download the training dataset in the accompanying paper:

The pre-trained network weights can be found in `pre_trained` folder, where you can find the weights trained with different random seeds and the corresponding network configuration. You can refer to `sample.ipynb` to see the details on how to use them.

------

Please let us know if you find anything doesn't work in the repository.