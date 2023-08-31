# Automatic Parameterization for Aerodynamic Shape Optimization via Deep Geometric Learning
Codes for the published paper on AIAA Aviation Forum 2023

[[paper]](https://arc.aiaa.org/doi/10.2514/6.2023-3471) 
[[preprint]](https://infoscience.epfl.ch/record/302199?ln=en) 
[[video]](https://doi.org/10.2514/6.2023-3471.vid)

## Dependencies
Please refer to [requirements.txt]() for the packages needed for this project. Please note that 
the codes use NVIDIA GPU and Pytorch for acceleration by default.

## Get Started
We provide commands so that you can try out DMM and LSM models in a few minutes.

```
# to obtain a DMM model that parameterizes NACA-3414
mkdir exp_dmm_2d
python DMM_airfoil_2d.py -workspaceDir exp_dmm_2d -type naca -profile 3414 -reconIter 600 2>&1 | tee exp_dmm_2d/log.log
# or to obtain a parameterization of CLARK-Y
python DMM_airfoil_2d.py -workspaceDir exp_dmm_2d -type uiuc -profile clarky-il -reconIter 600 2>&1 | tee exp_dmm_2d/log.log

# to train and obtain a LSM model
mkdir exp_lsm_2d
python LSM_airfoil_2d_train.py -workspaceDir exp_lsm_2d -trainingEpoch 21 -visualize 2>&1 | tee exp_lsm_2d/log.log
```

## Citation
If you find this project is useful, please cite:
```
@inbook{doi:10.2514/6.2023-3471,
  author = {Zhen Wei and Pascal Fua and Michaël Bauerheim},
  title = {Automatic Parameterization for Aerodynamic Shape Optimization via Deep Geometric Learning},
  booktitle = {AIAA AVIATION 2023 Forum},
  publisher = {AIAA},
  doi = {10.2514/6.2023-3471},
  URL = {https://arc.aiaa.org/doi/abs/10.2514/6.2023-3471}
}
```

Additionally, if you find the Latent Space Model (LSM) is useful, please cite at the same time:
```
@article{doi:10.2514/1.J062533,
  author = {Wei, Zhen and Guillard, Benoit and Fua, Pascal and Chapin, Vincent and Bauerheim, Michaël},
  title = {Latent Representation of Computational Fluid Dynamics Meshes and Application to Airfoil Aerodynamics},
  journal = {AIAA Journal},
  year = {2023},
  doi = {10.2514/1.J062533},
  URL = { https://doi.org/10.2514/1.J062533}
}
```

## Licence
This project has a BSD-style licence, as found in the [LICENCE](https://github.com/kfxw/CFD_Mesh_Parameterization/blob/main/LICENSE) file.
Redistributions and use of this project is permittet for academic purposes only. No commercial use is allowed.
