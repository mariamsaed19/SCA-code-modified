© 2024. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration.


<h1> Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures</h1>


This repo contains experiments on **CelebA**, **Medical MNIST**, **MNIST**, **Fashion MNIST**, and **CIFAR10** datasets using our proposed **Sparse Coding Architecture (SCA) and many other state-of-the-art ML Privacy Defense baselines**. In particular, it contains files to ***replicate all experiments*** using our **SCA defense** as well as **many other SOTA defenses** **on Plug-&-Play Attack (updated with the latest StyleGAN3), End-to-End Networks, and Split Networks**. 

*Note that the goal is for defenses to achieve* ***poor*** *reconstruction metrics (i.e., lower PSNR, lower SSIM, and higher FID), indicating that the model inversion attack failed to accurately reconstruct training data examples.*

- **To replicate Plug-&-Play attack experiments** for all **10 defense baselines** (including **our SCA defense)** and **5 datasets**, The `Pnp_stylegan3` directory contains all code files  on the **Plug and Play attack setup**. 

- **To replicate End-to-end and split network attack experiments**,`CelebA`, `MedMNIST`, `MNIST`, `FMNIST`, and `CIFAR10` directories contain code files for the **End-to-end and Split attack setups for the following eight benchmarks** on the CelebA, Medical MNIST, MNIST, Fashion MNIST, and CIFAR10 datasets, respectively. The defenses in these directories are SCA, No-Defense, Laplacian Noise (Titcombe et al 2021), Gaussian-Noise, GAN, Gong et al. 2023 (with and without continual learning), and Sparse-Standard (Teti et al. 2022). **The code files for the remaining three benchmarks (Wang et al., Peng et al., and Hayes et al.'s DP-based defense) for both End to end and Split attack setups reside** in the three directories (Mi-Reg, Bido_Def, and Opacus-Dp-SGD), respectively.


**Each code file is named as follows:** `Dataset-Name_Attack-Setup-Name_Defense-Benchmark-Abbreviation.`


**A few notations used in many code file names (see paper for details) are:**

- `lca2` > **Our sparse coding architecture (SCA)**
- `nod` > No Defense; 
- `gan` > Gan Defense;
- `gn` > Gaussian Noise; 
- `bgan` > Gong et al. GAN-based defense with continual learning; 
- `wogan` > Gong et al. without continual learning; 
- `ln` > Titcombe et al. laplacian noise; 
- `dp` > Hayes et al. defense; 
- `bido` > Peng et al. defense;
- `mi-reg` > Wang et al. defense; 

**3 Attack Setting abbreviations in code filenames:**

- `pnp` > Plug-&-Play attack updated to use the latest StyleGAN3
- `etn` > End to end attack setting, 
- `split` > Split network attack

**For example, code files named with** `cifar10_etn_gn` denote code files for the Gaussian noise-based defense on the CIFAR10 dataset in end-to-end attack setup. In addition to the 5 dataset names (CIFAR10, MNIST, MedMNIST, FMNIST, and CelebA) and 10 baselines, we have:



**To start with the Plug and Play attack in the `Pnp_stylegan3,' directory, the following commands can help to get the required libraries and datasets:**


```
####git clone https://github.com/NVlabs/stylegan3.git
####pip install ninja
####pip install torchmetrics
####unzip 'YOUR_PATH/DATASETxyz.zip'
```

***Before running code, UPDATE the code line that loads the dataset and also the line that saves the results so that they match the directory where you downloaded the dataset and your desired output directory.***

**To replicate UMAP representations of linear, convolution and sparse coding layers**, we provide all codes in the `other` directory.

***For the lambda value experiment, vary the lambda value in the corresponding code file.

To run a code file, one has to install the conda environment, PyTorch, and other required packages.
Once all installation is complete, one can run the following commands to activate the conda env and finally **run the shell script (i.e., test.sh) provided to execute a Python code file**.

```
####module load miniconda3
####source activate your_env
####sbatch test.sh
```

# License
SCA is provided under a BSD license with a "modifications must be indicated" clause. See the LICENSE file for the full text. SCA is covered under the LCA-PyTorch package, known internally as LA-CC-23-064.
