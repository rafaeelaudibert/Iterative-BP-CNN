#  #################################################################
#  Python code to reproduce works from iterative BP-CNN.
#
#  Codes have been tested successfully on Python 3.8.10 with TensorFlow 2.
#
#  References:
#   Fei Liang, Cong Shen and Feng Wu, "An Iterative BP-CNN Architecture for Channel Decoding under Correlated Noise", IEEE JSTSP
#
#  Written by Fei Liang (lfbeyond@mail.ustc.edu.cn, liang.fei@outlook.com)
#  Adapted to TF2 and Python3.8 by Rafael Baldasso Audibert (baudibert@rhrk.uni-kl.de, rbaudibert@inf.ufrgs.br, me@rafaaudibert.dev)
#  #################################################################


import numpy as np
import click


@click.command()
@click.option(
    "--function",
    "-f",
    help="Which function will be used on this program",
    type=click.Choice(["GenData", "Train", "Simulation"]),
)
@click.option("--n-code", "-n", "N", help="N variable for the code", default=576, show_default=True)
@click.option("--k-code", "-k", "K", help="K variable for the code", default=432, show_default=True)
@click.option(
    "--epoch-num", "-e", help="How many iterations would be run to train the network", default=200000, show_default=True
)
@click.option("--correlation", "-c", help="Correlation parameter for the colored noise", default=0.5, show_default=True)
@click.option(
    "--denoisers",
    "-d",
    "cnn_net_number",
    help="The number of CNN denoisers in the final simulation",
    default=1,
    show_default=True,
)
@click.option("--layers", "-l", "layer_num", help="Number of layers in the CNN", default=4, show_default=True)
@click.option(
    "--filter-sizes",
    help="The convolutional filter sizes. The length of this list should be equal to `layer_num`. It should be a comma-separated list",
    default="9,3,3,15",
)
@click.option(
    "--feature-maps",
    "feature_map_nums",
    help="The feature maps sizes. The length of this list should be equal to `layer_num`. The last value should be 1. It should be a comma-separated list",
    default="64,32,16,1",
)
@click.option("--restore-from-file", "-r", help="Should restore the network from a file", default=False)
@click.option("--normality-test", "-t", help="Should enable the normality test", default=True)
@click.option("--normality-lambda", "-a", help="The lambda value used on the normality", default=1.0)
@click.option(
    "--SNR-set",
    help="The set of SNRs used in the network. Should be a comma separated list",
    default="0,0.5,1,1.5,2,2.5,3",
    show_default=True,
)
@click.option("--update-llr-with-epdf", default=False, show_default=True)
@click.option(
    "--analyze-res-noise",
    help="Whether to update the initial LLR of the next BP decoding with the empirical distribution. Otherwise, the LLR is updated by viewing the residual noise follows a Gaussian distritbution",
    default=True,
    show_default=True,
)
@click.option(
    "--same-model-all-nets",
    help="denote whether the same model parameters for all denoising networks. If true and cnn_net_number > 1, we are testing the performance of iteration between a BP and a denoising network.",
    default=False,
    show_default=True,
)
def main(
    function,
    N,
    K,
    epoch_num,
    correlation,
    layer_num,
    cnn_net_number,
    filter_sizes,
    feature_map_nums,
    restore_from_file,
    normality_test,
    normality_lambda,
    SNR_set,
    update_llr_with_epdf,
    analyze_res_noise,
    same_model_all_nets,
):
    # We are importing only here to make `--help` faster
    import LinearBlkCodes as lbc
    import Iterative_BP_CNN as ibd
    import ConvNet
    import DataIO

    file_G = f"./LDPC_matrix/LDPC_gen_mat_{N}_{K}.txt"
    file_H = f"./LDPC_matrix/LDPC_chk_mat_{N}_{K}.txt"
    blk_len = N

    correlation_file = f"./noise/cov_1_2_corr_para{correlation:.2f}.dat"
    correlation_file_simu = correlation_file

    # BP decoding
    BP_iter_nums_gen_data = np.array([5])  # Number of iterations
    BP_iter_nums_simu = np.array([5, 5])

    # the convolutional filter size. The length of this list should be equal to the layer number
    filter_sizes = list(map(int, filter_sizes.split(",")))
    assert len(filter_sizes) == layer_num, "Filter sizes length must be equal to the number of layers"
    filter_sizes = np.array(filter_sizes)

    feature_map_nums = list(map(int, feature_map_nums.split(",")))
    assert len(feature_map_nums) == layer_num, "Feature maps nums length must be equal to the number of layers"
    assert feature_map_nums[-1] == 1, "Last feature map num should be 1"
    feature_map_nums = np.array(feature_map_nums)

    # cnn config
    currently_trained_net_id = 0  # denote the cnn denoiser which is in training currently

    # differentiate models trained with the same configurations. Its length should be equal to cnn_net_number. model_id[i] denotes the index of
    # the ith network in the BP-CNN-BP-CNN-... structure.
    model_id = np.array([0])

    # Used both for training and validation
    SNR_set = np.array(list(map(float, SNR_set.split(","))), dtype=np.float32)

    network_configuration = {
        "restore_layers": layer_num if restore_from_file else 0,
        "save_layers": layer_num,
        "total_layers": layer_num,  # The input layer is not included but the output layer is included
        "feature_length": blk_len,
        "label_length": blk_len,
        "filter_sizes": filter_sizes,
        "feature_map_nums": feature_map_nums,
        "layer_num": layer_num,
        "model_folder": "./model",
        "residual_noise_property_folder": "./model",
    }

    training_configuration = {
        "corr_para": correlation,
        "currently_trained_net_id": currently_trained_net_id,
        "training_sample_num": 1999200,  # It should be a multiple of training_minibatch_size
        "epoch_num": epoch_num,
        "training_minibatch_size": 1400,
        "SNR_set_gen_data": SNR_set,
        "training_feature_file": f"./data/training/EstNoise_before_cnn{currently_trained_net_id}.dat",
        "training_label_file": "./data/training/RealNoise.dat",
        "test_sample_num": 105000,  # Should be a multiple of test_minibatch_size
        "test_minibatch_size": 3500,
        "test_feature_file": f"./data/test/EstNoise_before_cnn{currently_trained_net_id}.dat",
        "test_label_file": "./data/test/RealNoise.dat",
        "normality_test_enabled": normality_test,
        "normality_lambda": normality_lambda,
    }

    assert (
        training_configuration["test_sample_num"] % training_configuration["test_minibatch_size"] == 0
    ), "Total_test_samples must be a multiple of test_minibatch_size!"
    assert (
        training_configuration["training_sample_num"] % training_configuration["training_minibatch_size"] == 0
    ), "Total_training_samples must be a multiple of training_minibatch_size!"
    assert (
        training_configuration["training_minibatch_size"] % np.size(training_configuration["SNR_set_gen_data"]) == 0
        and training_configuration["test_minibatch_size"] % np.size(training_configuration["SNR_set_gen_data"]) == 0
    ), "A batch of training or test data should contains equal amount of data under different CSNRs!"

    code = lbc.LDPC(N, K, file_G, file_H)

    if function == "GenData":
        noise_io = DataIO.NoiseIO(N, False, None, correlation_file)

        # Generate training and testing data
        ibd.generate_noise_samples(
            "Training",
            code,
            network_configuration,
            training_configuration,
            BP_iter_nums_gen_data,
            currently_trained_net_id,
            noise_io,
            model_id,
            update_llr_with_epdf,
        )

        ibd.generate_noise_samples(
            "Test",
            code,
            network_configuration,
            training_configuration,
            BP_iter_nums_gen_data,
            currently_trained_net_id,
            noise_io,
            model_id,
            update_llr_with_epdf,
        )
    elif function == "Train":
        net_id = currently_trained_net_id
        conv_net = ConvNet.ConvNet(network_configuration, training_configuration, net_id)
        conv_net.train_network(model_id)
    elif function == "Simulation":
        batch_size = 5000
        if analyze_res_noise:
            simutimes_for_anal_res_power = int(np.ceil(5e6 / float(K * batch_size)) * batch_size)
            ibd.analyze_residual_noise(
                code,
                N,
                network_configuration,
                simutimes_for_anal_res_power,
                batch_size,
                update_llr_with_epdf,
                currently_trained_net_id,
                model_id,
                BP_iter_nums_gen_data,
                correlation_file,
                SNR_set,
            )

        simutimes_range = np.array(
            [
                np.ceil(1e7 / float(K * batch_size)) * batch_size,
                np.ceil(1e8 / float(K * batch_size)) * batch_size,
            ],
            np.int32,
        )

        ibd.simulation_colored_noise(
            code,
            N,
            network_configuration,
            simutimes_range,
            1000,
            batch_size,
            SNR_set,
            BP_iter_nums_simu,
            correlation_file,
            cnn_net_number,
            model_id,
            same_model_all_nets,
            update_llr_with_epdf,
        )


if __name__ == "__main__":
    main()
