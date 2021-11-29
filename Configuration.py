import numpy as np


class TopConfig:
    """
    TopConfig defines some top configurations. Other configurations are set based on TopConfig.
    """

    def __init__(self):
        # select functions to be executed, including generating data(GenData), training(Train), and simulation(Simulation)
        self.function = "Train"

        # code
        self.N_code = 576
        self.K_code = 432
        self.file_G = f"./LDPC_matrix/LDPC_gen_mat_{self.N_code}_{self.K_code}.txt"
        self.file_H = f"./LDPC_matrix/LDPC_chk_mat_{self.N_code}_{self.K_code}.txt"

        # noise information
        self.blk_len = self.N_code
        self.corr_para = 0.5  # correlation parameters of the colored noise

        # correlation parameters for simulation. this should be equal to corr_para. If not, it is used to test the model robustness.
        self.corr_para_simu = self.corr_para
        self.cov_1_2_file = f"./noise/cov_1_2_corr_para{self.corr_para:.2f}.dat"
        self.cov_1_2_file_simu = self.cov_1_2_file

        # BP decoding
        self.BP_iter_nums_gen_data = np.array([5])  # Number of iterations
        self.BP_iter_nums_simu = np.array([5, 5])

        # cnn config
        # denote the cnn denoiser which is in training currently
        self.currently_trained_net_id = 0
        self.cnn_net_number = 1  # the number of cnn denoisers in final simulation
        self.layer_num = 4  # the number of cnn layers

        # the convolutional filter size. The length of this list should be equal to the layer number
        self.filter_sizes = np.array([9, 3, 3, 15])
        self.feature_map_nums = np.array([64, 32, 16, 1])  # The last element must be 1
        # whether to restore previous saved network for training
        self.restore_network_from_file = False
        # differentiate models trained with the same configurations. Its length should be equal to cnn_net_number. model_id[i] denotes the index of
        #  the ith network in the BP-CNN-BP-CNN-... structure.
        self.model_id = np.array([0])

        # Training
        self.normality_test_enabled = True
        self.normality_lambda = 1
        self.SNR_set_gen_data = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], dtype=np.float32)

        # Simulation
        self.eval_SNRs = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3], np.float32)
        # denote whether the same model parameters for all denoising networks. If true and cnn_net_number > 1, we are testing the performance
        self.same_model_all_nets = False
        #  of iteration between a BP and a denoising network.
        self.analyze_res_noise = True
        # whether to update the initial LLR of the next BP decoding with the empirical distribution. Otherwise, the LLR is updated by
        self.update_llr_with_epdf = False
        # viewing the residual noise follows a Gaussian distritbution

    def parse_cmd_line(self, argv):
        if len(argv) == 1:
            return

        id = 1
        while id < len(argv):
            if argv[id] == "-Func":
                self.function = argv[id + 1]
                print(f"Function is set to {self.function}")
            # noise information
            elif argv[id] == "-CorrPara":
                self.corr_para = float(argv[id + 1])
                self.cov_1_2_file = f"./noise/cov_1_2_corr_para{self.corr_para:.2f}.dat"
                print(f"Corr para is set to {self.corr_para:.2f}")
            # Simulation
            elif argv[id] == "-UpdateLLR_Epdf":
                self.update_llr_with_epdf = argv[id + 1] == "True"
            elif argv[id] == "-EvalSNR":
                self.eval_SNRs = np.fromstring(argv[id + 1], np.float32, sep=" ")
                print(f"eval_SNRs is set to {np.array2string(self.eval_SNRs)}")
            elif argv[id] == "-AnalResNoise":
                self.analyze_res_noise = argv[id + 1] == "True"
                print("analyze_res_noise is set to {self.analyze_res_noise}")
            elif argv[id] == "-SimuCorrPara":
                self.corr_para_simu = float(argv[id + 1])
                self.cov_1_2_file_simu = f"./noise/cov_1_2_corr_para{self.corr_para_simu:.2f}.dat"
                print("Corr para for simulation is set to {self.corr_para_simu:.2f}")
            elif argv[id] == "-SameModelAllNets":
                self.same_model_all_nets = argv[id + 1] == "True"
                print(f"same_model_all_nets is set to {self.same_model_all_nets}")
            # BP decoding
            elif argv[id] == "-BP_IterForGenData":
                self.BP_iter_nums_gen_data = np.fromstring(argv[id + 1], np.int32, sep=" ")
                print(f"BP iter for gen data is set to: {np.array2string(self.BP_iter_nums_gen_data)}")
            elif argv[id] == "-BP_IterForSimu":
                self.BP_iter_nums_simu = np.fromstring(argv[id + 1], np.int32, sep=" ")
                print(f"BP iter for simulation is set to: { np.array2string(self.BP_iter_nums_simu)}")
            # CNN config
            elif argv[id] == "-NetNumber":
                self.cnn_net_number = np.int32(argv[id + 1])
            elif argv[id] == "-CNN_Layer":
                self.layer_num = np.int32(argv[id + 1])
                print(f"CNN layer number is set to {self.layer_num}")
            elif argv[id] == "-FilterSize":
                self.filter_sizes = np.fromstring(argv[id + 1], np.int32, sep=" ")
                print(f"Filter sizes are set to {np.array2string(self.filter_sizes)}")
            elif argv[id] == "-FeatureMap":
                self.feature_map_nums = np.fromstring(argv[id + 1], np.int32, sep=" ")
                print(f"Feature map numbers are set to {np.array2string(self.feature_map_nums)}")
            # training
            elif argv[id] == "-ModelId":
                self.model_id = np.fromstring(argv[id + 1], np.int32, sep=" ")
                print(f"Model id is set to {np.array2string(self.model_id)}")
            elif argv[id] == "-NormTest":
                self.normality_test_enabled = argv[id + 1] == "True"
                print(f"Normality test: {self.normality_test_enabled}")
            elif argv[id] == "-NormLambda":
                self.normality_lambda = np.float32(argv[id + 1])
                print(f"Normality lambda is set to {self.normality_lambda}")
            elif argv[id] == "-SNR_GenData":
                self.SNR_set_gen_data = np.fromstring(argv[id + 1], dtype=np.float32, sep=" ")
                print(f"SNR set for generating data is set to {np.array2string(self.SNR_set_gen_data)}.")
            else:
                print(f"Invalid parameter {argv[id]}%s!")
                exit(0)
            id += 2


# class for network configurations
class NetConfig:
    def __init__(self, top_config):
        # network parameters
        self.restore_layers = top_config.layer_num if top_config.restore_network_from_file else 0
        self.save_layers = top_config.layer_num

        # the input layer is not included but the output layer is included
        self.total_layers = top_config.layer_num
        self.feature_length = top_config.blk_len
        self.label_length = top_config.blk_len

        # conv net parameters
        self.filter_sizes = top_config.filter_sizes
        self.feature_map_nums = top_config.feature_map_nums
        self.layer_num = top_config.layer_num

        self.model_folder = "./model/"
        self.residual_noise_property_folder = self.model_folder


class TrainingConfig:
    def __init__(self, top_config):

        # cov^(1/2) file
        self.corr_para = top_config.corr_para

        self.currently_trained_net_id = top_config.currently_trained_net_id

        # training data information
        # the number of training samples. It should be a multiple of training_minibatch_size
        self.training_sample_num = 1999200
        # training parameters
        self.epoch_num = 200000  # the number of training iterations.
        # one mini-batch contains equal amount of data generated under different CSNR.
        self.training_minibatch_size = 1400
        self.SNR_set_gen_data = top_config.SNR_set_gen_data
        # the data in the feature file is the network input.
        # the data in the label file is the ground truth.
        self.training_feature_file = f"./data/training/EstNoise_before_cnn{self.currently_trained_net_id}.dat"
        self.training_label_file = "./data/training/RealNoise.dat"

        # test data information
        self.test_sample_num = 105000  # it should be a multiple of test_minibatch_size
        self.test_minibatch_size = 3500
        self.test_feature_file = f"./data/test/EstNoise_before_cnn{self.currently_trained_net_id}.dat"
        self.test_label_file = "./data/test/RealNoise.dat"

        # normality test
        self.normality_test_enabled = top_config.normality_test_enabled
        self.normality_lambda = top_config.normality_lambda

        # parameter check
        if self.test_sample_num % self.test_minibatch_size != 0:
            print("Total_test_samples must be a multiple of test_minibatch_size!")
            exit(0)
        if self.training_sample_num % self.training_minibatch_size != 0:
            print("Total_training_samples must be a multiple of training_minibatch_size!")
            exit(0)
        if (
            self.training_minibatch_size % np.size(self.SNR_set_gen_data) != 0
            or self.test_minibatch_size % np.size(self.SNR_set_gen_data) != 0
        ):
            print("A batch of training or test data should contains equal amount of data under different CSNRs!")
            exit(0)
