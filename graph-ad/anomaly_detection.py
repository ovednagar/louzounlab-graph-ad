import os
import pickle
from anomaly_picker import SimpleAnomalyPicker
from beta_calculator import LinearContext, LinearMeanContext
from features_picker import PearsonFeaturePicker
from graph_score import KnnScore, GmmScore, LocalOutlierFactorScore
from features_processor import FeaturesProcessor, log_norm
from graph_features import GraphFeatures
from loggers import PrintLogger
from parameters import AdParams
from temporal_graph import TemporalGraph
import numpy as np
# np.seterr(all='raise')


class AnomalyDetection:
    def __init__(self, params: AdParams):
        self._base_dir = __file__.replace("/", os.sep)
        self._base_dir = os.path.join(self._base_dir.rsplit(os.sep, 1)[0])
        self._data_path = os.path.join(self._base_dir, "INPUT_DATA", params.database.DATABASE_FILE)
        self._params = params
        self._data_name = params.database.DATABASE_NAME
        self._logger = PrintLogger("Anomaly logger")
        self._temporal_graph = self._build_temporal_graph()
        self._ground_truth = self._load_ground_truth(self._params.database.GROUND_TRUTH)
        # self._temporal_graph.filter(
        #         lambda x: False if self._temporal_graph.node_count(x) < 20 else True,
        #         func_input="graph_name")
        self._idx_to_name = list(self._temporal_graph.graph_names())
        self._name_to_idx = {name: idx for idx, name in enumerate(self._idx_to_name)}

        if self._params.vec_type == "motif_ratio":
            self._build_second_method()
        elif self._params.vec_type == "regression":
            self._build_first_method()
        elif self._params.vec_type == "mean_regression":
            self._build_third_method()

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
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).as_matrix(norm_func=log_norm) if self._params.log else \
                FeaturesProcessor(gnx_ftr).as_matrix()

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
            gnx_to_vec[gnx_name] = FeaturesProcessor(gnx_ftr).activate_motif_ratio_vec(norm_func=log_norm)\
                if self._params.log else FeaturesProcessor(gnx_ftr).activate_motif_ratio_vec()

        pickle.dump(gnx_to_vec, open(vec_pkl_path, "wb"))
        return gnx_to_vec

    def _build_first_method(self):
        mx_dict = self._calc_matrix()
        concat_mx = np.vstack([mx for name, mx in mx_dict.items()])
        pearson_picker = PearsonFeaturePicker(concat_mx, size=self._params.ftr_pairs,
                                              logger=self._logger, identical_bar=self._params.identical_bar)
        best_pairs = pearson_picker.best_pairs()
        beta = LinearContext(self._temporal_graph, mx_dict, best_pairs, window_size=self._params.window_correlation)
        beta_matrix = beta.beta_matrix()
        if self._params.score_type == "knn":
            score = KnnScore(beta_matrix, self._params.KNN_k, self._data_name,
                             window_size=self._params.window_score)
        elif self._params.score_type == "gmm":
            score = GmmScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                             n_components=self._params.n_components)
        else:   # self._params["score_type"] == "local_outlier":
            score = LocalOutlierFactorScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                                            n_neighbors=self._params.n_neighbors)
        anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                                             num_anomalies=self._params.n_outliers)
        anomaly_picker.build()
        anomaly_picker.plot_anomalies_bokeh(self._params.anomalies_file_name, truth=self._ground_truth,
                                            info_text=self._params.tostring())

    def _build_second_method(self):
        self._graph_to_vec = self._calc_vec()
        self._graph_matrix = np.vstack([self._graph_to_vec[name] for name in self._temporal_graph.graph_names()])
        if self._params.log:
            self._graph_matrix = log_norm(self._graph_matrix)

        if self._params.score_type == "knn":
            score = KnnScore(self._graph_matrix, self._params.KNN_k, self._data_name,
                             window_size=self._params.window_score)
        elif self._params.score_type == "gmm":
            score = GmmScore(self._graph_matrix, self._data_name, window_size=self._params.window_score,
                             n_components=self._params.n_components)
        else:   # self._params["score_type"] == "local_outlier":
            score = LocalOutlierFactorScore(self._graph_matrix, self._data_name,
                                            window_size=self._params.window_score,
                                            n_neighbors=self._params.n_neighbors)

        anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                                             num_anomalies=self._params.n_outliers)
        anomaly_picker.build()
        anomaly_picker.plot_anomalies_bokeh(self._params.anomalies_file_name, truth=self._ground_truth,
                                            info_text=self._params.tostring())

    def _build_third_method(self):
        mx_dict = self._calc_matrix()
        concat_mx = np.vstack([mx for name, mx in mx_dict.items()])
        pearson_picker = PearsonFeaturePicker(concat_mx, size=self._params.ftr_pairs,
                                              logger=self._logger, identical_bar=self._params.identical_bar)
        best_pairs = pearson_picker.best_pairs()
        beta = LinearMeanContext(self._temporal_graph, mx_dict, best_pairs, window_size=self._params.window_correlation)
        beta_matrix = beta.beta_matrix()
        if self._params.score_type == "knn":
            score = KnnScore(beta_matrix, self._params.KNN_k, self._data_name,
                             window_size=self._params.window_score)
        elif self._params.score_type == "gmm":
            score = GmmScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                             n_components=self._params.n_components)
        else:   # self._params["score_type"] == "local_outlier":
            score = LocalOutlierFactorScore(beta_matrix, self._data_name, window_size=self._params.window_score,
                                            n_neighbors=self._params.n_neighbors)
        anomaly_picker = SimpleAnomalyPicker(self._temporal_graph, score.score_list(), self._data_name,
                                             num_anomalies=self._params.n_outliers)
        anomaly_picker.build()
        anomaly_picker.plot_anomalies_bokeh(self._params.anomalies_file_name, truth=self._ground_truth,
                                            info_text=self._params.tostring())


if __name__ == "__main__":
    from parameters import DEFAULT_PARAMS
    AnomalyDetection(DEFAULT_PARAMS)
    # A = AnomalyDetection()
    # A.build_manipulations()

