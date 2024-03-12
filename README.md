© 2024. Triad National Security, LLC. All rights reserved.

This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S.
Department of Energy/National Nuclear Security Administration.


<h1> Improving Robustness to Model Inversion Attacks via Sparse Coding Architectures</h1>


This repo contains experiments on CelebA, Medical MNIST, MNIST, Fashion MNIST, and CIFAR10 datasets using our proposed Sparse Coding Architecture (SCA) and other standard baselines. In particular, it contains files to implement and test different SOTA defenses as well as our proposed SCA on Plug and Play Attack, End-to-end Networks, and Split Networks. Note that the goal is for defenses to achieve *poor* reconstruction metrics (i.e., lower PSNR, lower SSIM, and higher FID), indicating that the model inversion attack failed to accurately reconstruct training data examples. 

To summarize, our `Pnp_stylegan3' directory contains all code files on five datasets for all SOTA benchmarks, as well as our SCA on Plug and Play attack setup. In addition, CelebA, MedMNIST, MNIST, FMNIST, and CIFAR10 directories contain code files for the End-to-end and Split attack setups for the following eight benchmarks on the  CelebA, Medical MNIST, MNIST, Fashion MNIST, and CIFAR10 datasets, respectively. These benchmarks are no defense, laplacian noise, gaussian noise, Gan, Gan++ with and without continual learning, sparse standard, and our SCA. The code files for the remaining three benchmarks (Wang, Peng, and Hayes) for both End to end and Split attack setups reside in the three directories (Mi-Reg, Bido_Def, and Opacus-Dp-SGD), respectively.

A few notations used in many code file names are: nod> No Def; gan > Gan Def; bgan> Gong et al. GAN with continual learning; wogan> Gong et al. without continual learning; gn> Gaussian Noise; ln> Titcombe et al. laplacian noise; dp> Hayes et al. defense; bido> Peng et al. defense; mi-reg> Wang et al. defense; lca1> Sparse Standard baseline ; lca2> Our proposed multiplayer sparse coding architecture (SCA)


Each code file is named as follows: ``Dataset Name_Ataack Setup Name_Defense Benchmark Name." For example, code files with `cifar10_etn_gn' denote code files for the Gaussian noise-based defense on the CIFAR10 dataset in end-to-end attack setup.

To start with the Plug and Play attack in the `Pnp_stylegan3,' the following commands can help to get the required libraries and datasets.


```
####git clone https://github.com/NVlabs/stylegan3.git
####pip install ninja
####pip install torchmetrics
####unzip 'YOUR_PATH/DATASETxyz.zip'
```

"other" directory contains code to plot the Umap representation of linear, convolution and sparse coding layers.

***For lambda value experiment, vary the lambda value in the corresponding code file.

To run a code file first one has to install conda environment, pytorch, and other required packages.
Once all installation is complete, one can run the following commands to activate the conda env and finally run the shell script (i.e., test.sh) provided to execute a Python code file.

```
####module load miniconda3
####source activate your_env
####sbatch test.sh
```

# License
SCA is provided under a BSD license with a "modifications must be indicated" clause. See the LICENSE file for the full text. SCA is covered under the LCA-PyTorch package, known internally as LA-CC-23-064.
