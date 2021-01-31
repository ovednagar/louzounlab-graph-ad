import os
import pickle
from anomaly_picker import SimpleAnomalyPicker
from beta_calculator import LinearContext, LinearMeanContext
from features_meta_ad import ANOMALY_DETECTION_FEATURES, MOTIF_FEATURES
from features_picker import PearsonFeaturePicker
from graph_score import KnnScore, GmmScore, LocalOutlierFactorScore
from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from parameters import AdParams
from temporal_graph import TemporalGraph
import numpy as np
# np.seterr(all='raise')


class AnomalyDetectionOperationResearch:
    def __init__(self, params: AdParams, name):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0], "..")
        self._data_path = os.path.join(self._base_dir, "INPUT_DATA", params.database.DATABASE_FILE)
        self._params = params
        self._data_name = params.database.DATABASE_NAME
        self._logger = PrintLogger("Anomaly logger")
        self._temporal_graph = self._build_temporal_graph()
        self._ground_truth = self._load_ground_truth(self._params.database.GROUND_TRUTH)
        self._idx_to_name = list(self._temporal_graph.graph_names())
        self._name_to_idx = {name: idx for idx, name in enumerate(self._idx_to_name)}
        self._out = open(os.path.join("..", name), "wt")
        self._out.write(",".join(["FN", "TN", "TP", "FP", "recall", "precision", "specificity", "F1",
                                  self._params.attr_string()]) + "\n")
        self._build()

    def _load_ground_truth(self, gd):
        if type(gd) is list:
            return {self._temporal_graph.name_to_index(g_id): 1 for g_id in gd}
        elif type(gd) is dict:
            return {self._temporal_graph.name_to_index(g_id): float(val) for g_id, val in gd.items()}
        return None

    def _build_temporal_graph(self):
        database_name = self._params.database.DATABASE_NAME + "_" + str(self._params.max_connected)\
                        + "_" + str(self._params.directed)
        vec_pkl_path = os.path.join(self._base_dir, "pkl", "temporal_graphs", database_name + "_tg.pkl")
        if os.path.exists(vec_pkl_path):
            self._logger.info("loading pkl file - temoral_graphs")
            tg = pickle.load(open(vec_pkl_path, "rb"))
        else:
            tg = TemporalGraph(database_name, self._data_path, self._params.database.DATE_FORMAT,
                               self._params.database.TIME_COL, self._params.database.SRC_COL,
                               self._params.database.DST_COL, weight_col=self._params.database.WEIGHT_COL,
                               weeks=self._params.database.WEEK_SPLIT, days=self._params.database.DAY_SPLIT,
                               hours=self._params.database.HOUR_SPLIT, minutes=self._params.database.MIN_SPLIT,
                               seconds=self._params.database.SEC_SPLIT, directed=self._params.directed,
                               logger=self._logger).to_multi_graph()
            tg.suspend_logger()
            pickle.dump(tg, open(vec_pkl_path, "wb"))
        tg.wake_logger()
        return tg

    def _calc_matrix(self):
        database_name = self._params.database.DATABASE_NAME + "_" + str(self._params.max_connected) + "_" + str(
            self._params.directed)
        mat_pkl_path = os.path.join(self._base_dir, "pkl", "vectors", database_name + "_matrix_log" +
                                    str(self._params.log) + ".pkl")
        if os.path.exists(mat_pkl_path):
            self._logger.info("loading pkl file - graph_matrix")
            return pickle.load(open(mat_pkl_path, "rb"))

        gnx_to_vec = {}
        # create dir for database
        pkl_dir = os.path.join(self._base_dir, "pkl", "features")
        database_pkl_dir = os.path.join(pkl_dir, database_name)
        if database_name not in os.listdir(pkl_dir):
            os.mkdir(database_pkl_dir)

        for gnx_name, gnx in zip(self._temporal_graph.graph_names(), self._temporal_graph.graphs()):
            # create dir for specific graph features
            gnx_name_path = gnx_name.replace(':', '_')
            gnx_name_path = gnx_name_path.replace('/', '_')
            gnx_path = os.path.join(database_pkl_dir, gnx_name_path)
            if gnx_name_path not in os.listdir(database_pkl_dir):
                os.mkdir(gnx_path)

            gnx_ftr = GraphFeatures(gnx, self._params.features, dir_path=gnx_path, logger=self._logger,
                                    is_max_connected=self._params.max_connected)
            gnx_ftr.build(should_dump=True, force_build=self._params.FORCE_REBUILD_FEATURES)  # build features
            # calc motif ratio vector
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).as_matrix(norm_func=log_norm)

        pickle.dump(gnx_to_vec, open(mat_pkl_path, "wb"))
        return gnx_to_vec

    def _calc_vec(self):
        database_name = self._params.database.DATABASE_NAME + "_" + \
                        str(self._params.max_connected) + "_" + str(self._params.directed)
        vec_pkl_path = os.path.join(self._base_dir, "pkl", "vectors", database_name + "_vectors_log_" +
                                    str(self._params.log) + ".pkl")
        if os.path.exists(vec_pkl_path):
            self._logger.info("loading pkl file - graph_vectors")
            return pickle.load(open(vec_pkl_path, "rb"))

        # create dir for database
        pkl_dir = os.path.join(self._base_dir, "pkl", "features")
        database_pkl_dir = os.path.join(pkl_dir, database_name)
        if database_name not in os.listdir(pkl_dir):
            os.mkdir(database_pkl_dir)

        gnx_to_vec = {}
        for gnx_name, gnx in zip(self._temporal_graph.graph_names(), self._temporal_graph.graphs()):
            # create dir for specific graph features
            gnx_path = os.path.join(database_pkl_dir, gnx_name)
            if gnx_name not in os.listdir(database_pkl_dir):
                os.mkdir(gnx_path)

            gnx_ftr = GraphFeatures(gnx, self._params.features, dir_path=gnx_path, logger=self._logger,
                                    is_max_connected=self._params.max_connected)
            gnx_ftr.build(should_dump=True, force_build=self._params.FORCE_REBUILD_FEATURES)  # build features
            # calc motif ratio vector
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).activate_motif_ratio_vec(norm_func=log_norm)

        pickle.dump(gnx_to_vec, open(vec_pkl_path, "wb"))
        return gnx_to_vec

    def _build(self):
        for lg in [True, False]:
            self._params.log = lg
            for vec_type in ["mean_regression", "regression"]:  # motif_ratio
                self._params.vec_type = vec_type
                self.features = ANOMALY_DETECTION_FEATURES if self._params.vec_type == "regression" else MOTIF_FEATURES,
                if self._params.vec_type == "regression" or self._params.vec_type == "mean_regression":
                    mx_dict = self._calc_matrix()
                    concat_mx = np.vstack([mx for name, mx in mx_dict.items()])
                    for ftr_pairs in [3, 4, 5]:  # [1, 2, 3, 4, 5, 10] [5, 10, 15, 20, 25, 30, 40, 45, 50]: # [1, 2, 3, 4, 5, 10, 15, 20, 25, 50, 70, 90, 110, 130, 150, 170, 200]: #  [5, 10, 15, 20, 25, 30, 40, 45, 50]
                        self._params.ftr_pairs = ftr_pairs
                        for identical in [0.99]:  #  [0.7, 0.8, 0.9, 0.95, 0.99] [0.7, 0.8, 0.9, 0.95, 0.99]
                            self._params.identical_bar = identical

                            pearson_picker = PearsonFeaturePicker(concat_mx, size=self._params.ftr_pairs,
                                                                  logger=self._logger, identical_bar=self._params.identical_bar)
                            for win in list(range(25, min(100, self._temporal_graph.number_of_graphs()), 25)):
                                self._params.window_correlation = win
                                best_pairs = pearson_picker.best_pairs()
                                if best_pairs is None:
                                    continue
                                if self._params.vec_type == "regression":
                                    beta = LinearContext(self._temporal_graph, mx_dict, best_pairs, window_size=self._params.window_correlation)
                                else:
                                    beta = LinearMeanContext(self._temporal_graph, mx_dict, best_pairs, window_size=self._params.window_correlation)
                                beta_matrix = beta.beta_matrix()
                                self._pick_anomalies(beta_matrix)

                elif self._params.vec_type == "motif_ratio":
                    self._graph_to_vec = self._calc_vec()
                    beta_matrix = np.vstack([self._graph_to_vec[name] for name in self._temporal_graph.graph_names()])
                    self._pick_anomalies(beta_matrix)

    def _pick_anomalies(self, beta_matrix):
            for score_type in ["knn", "gmm", "local_outlier"]:
                self._params.score_type = score_type
                if self._params.score_type == "knn":
                    for win in list(range(25, min(100, self._temporal_graph.number_of_graphs()), 25)):
                        self._params.window_score = win
                        for k in list(range(5, min(win, 50) - 1, 5)):
                            self._params.KNN_k = k
                            score = KnnScore(beta_matrix, self._params.KNN_k, self._data_name,
                                             window_size=self._params.window_score)
                            anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                                                                 num_anomalies=self._params.n_outliers)
                            truth = [self._temporal_graph.name_to_index(g_id) for g_id in
                                     self._params.database.GROUND_TRUTH] if self._params.database.GROUND_TRUTH else None
                            FN, TN, TP, FP, recall, precision, specificity, F1 = anomaly_picker.build(truth=truth)
                            self._out.write(",".join([str(FN), str(TN), str(TP), str(FP), str(recall), str(precision),
                                                      str(specificity), str(F1), self._params.attr_val_string()]) + "\n")

                elif self._params.score_type == "gmm":
                    for win in list(range(25, min(100, self._temporal_graph.number_of_graphs()), 25)):
                        self._params.window_score = win
                        for comp in [1, 2, 3, 4, 5]:
                            self._params.n_components = comp
                            score = GmmScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                                             n_components=self._params.n_components)
                            anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                                                                 num_anomalies=self._params.n_outliers)
                            truth = [self._temporal_graph.name_to_index(g_id) for g_id in
                                     self._params.database.GROUND_TRUTH] if self._params.database.GROUND_TRUTH else None
                            FN, TN, TP, FP, recall, precision, specificity, F1 = anomaly_picker.build(truth=truth)
                        self._out.write(",".join([str(FN), str(TN), str(TP), str(FP), str(recall), str(precision),
                                                  str(specificity), str(F1), self._params.attr_val_string()]) + "\n")

                elif self._params.score_type == "local_outlier":
                    for win in list(range(25, min(100, self._temporal_graph.number_of_graphs()), 25)):
                        self._params.window_score = win
                        for neighbors in list(range(5, min(win, 50), 5)):
                            self._params.n_neighbors = neighbors
                            score = LocalOutlierFactorScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                                                            n_neighbors=self._params.n_neighbors)
                            anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                                                                 num_anomalies=self._params.n_outliers)
                            truth = [self._temporal_graph.name_to_index(g_id) for g_id in
                                     self._params.database.GROUND_TRUTH] if self._params.database.GROUND_TRUTH else None
                            FN, TN, TP, FP, recall, precision, specificity, F1 = anomaly_picker.build(truth=truth)
                            self._out.write(",".join([str(FN), str(TN), str(TP), str(FP), str(recall), str(precision),
                                                      str(specificity), str(F1), self._params.attr_val_string()]) + "\n")

                # elif self._params.score_type == "gmm":
                #     for win in list(range(25, min(100, self._temporal_graph.number_of_graphs()), 25)):
                #         self._params.window_score = win
                #         for comp in [1, 2, 3, 4, 5]:
                #             recalls, precisions = [], []
                #             measures = {}
                #             for n_outlier in n_outliers:
                #                 self._params.n_components = comp
                #                 score = GmmScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                #                                  n_components=self._params.n_components)
                #                 anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                #                                                      num_anomalies=self._params.n_outliers)
                #                 truth = [self._temporal_graph.name_to_index(g_id) for g_id in
                #                          self._params.database.GROUND_TRUTH] if self._params.database.GROUND_TRUTH else None
                #                 FN, TN, TP, FP, recall, precision, specificity, F1 = anomaly_picker.build(truth=truth)
                #                 recalls.append(recall), precisions.append(precision)
                #                 if n_outlier == p:
                #                     measures['FN'], measures['TN'], measures['TP'], measures['FP'], measures['recall'],\
                #                     measures['precision'], measures['F1'] = FN, TN, TP, FP, recall, precision, F1
                #             # self._out.write(",".join([str(FN), str(TN), str(TP), str(FP), str(recall), str(precision),
                #             #                           str(specificity), str(F1), self._params.attr_val_string()]) + "\n")
                #             self._out.write(",".join([str(measures['FN']), str(measures['TN']), str(measures['FP']),
                #                                       str(measures['TP']), str(measures['FP']), str(measures['recall']),
                #                                       str(measures['precision']), str(measures['F1']),
                #                                       self._params.attr_val_string()]) + "\n")
if __name__ == "__main__":
    from parameters import DEFAULT_PARAMS, EnronParams, OldEnronParams, TwitterSecurityParams, SecRepoParams, \
        RealityMiningParams
    import os

    enron_old = {'KNN_k': 20, 'directed': True, 'ftr_pairs': 30, 'identical_bar': 0.95, 'log': True,
                 'max_connected': False,
                 'n_components': 4, 'n_neighbors': 45, 'score_type': 'gmm', 'vec_type': 'mean_regression',
                 'window_correlation': 50, 'window_score': 50}
    enron = {'KNN_k': 45, 'directed': True, 'ftr_pairs': 2, 'identical_bar': 0.9, 'log': True, 'max_connected': False,
             'n_components': 5, 'n_neighbors': 25, 'score_type': 'knn', 'vec_type': 'mean_regression',
             'window_correlation': 25, 'window_score': 75}
    Reality_Mining = {'KNN_k': 45, 'directed': True, 'ftr_pairs': 2, 'identical_bar': 0.99, 'log': True,
                      'max_connected': False,
                      'n_components': 5, 'n_neighbors': 45, 'score_type': 'knn', 'vec_type': 'mean_regression',
                      'window_correlation': 25, 'window_score': 75}
    TwitterSecurity = {'KNN_k': 45, 'directed': True, 'ftr_pairs': 10, 'identical_bar': 0.95, 'log': True,
                      'max_connected': False,
                      'n_components': 5, 'n_neighbors': 45, 'score_type': 'knn', 'vec_type': 'mean_regression',
                      'window_correlation': 25, 'window_score': 75}
    SecRepo = {'KNN_k': 45, 'directed': True, 'ftr_pairs': 10, 'identical_bar': 0.99, 'log': True,
                      'max_connected': False,
                      'n_components': 5, 'n_neighbors': 45, 'score_type': 'knn', 'vec_type': 'mean_regression',
                      'window_correlation': 25, 'window_score': 75}
    Darpa = {'KNN_k': 45, 'directed': True, 'ftr_pairs': 10, 'identical_bar': 0.99, 'log': True,
                      'max_connected': False,
                      'n_components': 5, 'n_neighbors': 45, 'score_type': 'knn', 'vec_type': 'mean_regression',
                      'window_correlation': 25, 'window_score': 75}
    dataset_params = {'Darpa': Darpa, 'SecRepo': SecRepo, 'enron_old': enron_old, 'enron': enron, 'Reality_Mining': Reality_Mining, 'TwitterSecurity': TwitterSecurity}
    if not os.path.exists("operation_research_results"):
        os.mkdir("operation_research_results")
    # for param in [AdParams(SecRepoParams(), dataset_params, 'SecRepo'), AdParams(RealityMiningParams(), dataset_params, 'Reality_Mining'),  # ,AdParams(SecRepoParams())]: #
    #                AdParams(TwitterSecurityParams(), dataset_params, 'TwitterSecurity'), AdParams(OldEnronParams(), dataset_params, 'enron_old')]:
    #     out_file_path = os.path.join("operation_research_results",
    #                                  param.database.DATABASE_NAME + "_operation_research_9_12_20.csv")
    #     AnomalyDetectionOperationResearch(param, out_file_path)
    from parameters import DarpaParams
    for param in [AdParams(DarpaParams(),  dataset_params, 'Darpa')]:
        out_file_path = os.path.join("operation_research_results",
                                     param.database.DATABASE_NAME + "_operation_research_27_01_21_stam.csv")
        AnomalyDetectionOperationResearch(param, out_file_path)
    # A = AnomalyDetection()
    # A.build_manipulations()

