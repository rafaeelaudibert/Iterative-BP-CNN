# Code Specifications

> This was adapted from the PDF available [here](https://github.com/rafaeelaudibert/Iterative-BP-CNN/blob/master/ReadMe.pdf)

The code contains three modules to carry out simulations of the iterative BP-CNN architecture. The three modules are generating data, training the network and testing the BP-CNN performance. The entries of these modules are all in `main.py`. To complete one round of simulation, the three modules need to be executed in order. To run these modules, a file containing the matrix `$sigma^{1/2}$` is also needed. This file can be generated using the file [`noise/generate_cov_1_2.m`](./noise/generate_cov_1_2.m). The detailed description of these modules are as follows:

## Generating data

This module is executed by setting `top_config.function` to `GenData`. The outputs of this module are two files containing the training dataset and the validation dataset. The validation dataset is to test the network performance during training.

## Training the Network

This module is executed by setting `top_config.function` to `Train`. This module will train the network and save the trained parameters in files.

## Testing the BP-CNN performance

This module is executed by setting `top_config.function` to `Simulation`. This module tests the BP-CNN performance by simulating channel encoding, transmission and channel decoding through the BP-CNN architecture. The bit error rates under different channel SNRs are calculated and saved in a file.

> If you have other questions about the codes, please feel free to contact the paper's author. Their email address is [lfbeyond@mail.ustc.edu.cn](mailto:lfbeyond@mail.ustc.edu.cn)
