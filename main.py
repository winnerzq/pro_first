# -*- coding: utf-8 -*-
import os

os.environ['OMP_NUM_THREADS'] = '1'

from data_loader import load_data
from graph_builder import build_spatial_graph
from hmm_feature_extractor import extract_hmm_features
from dataset_builder import build_dataset
from model import EnhancedGCN
from trainer import train_and_evaluate
from utils import create_train_test_split, create_pyg_data, plot_results
from config import LOCATIONS, WINDOW_SIZE, HMM_STATES, SAVE_PATH

if __name__ == "__main__":
    # 数据加载
    data_dict, site_names = load_data('chaohu-processed-data.xlsx')

    # 构建空间图
    edge_index, edge_weight = build_spatial_graph(LOCATIONS, site_names)

    # 特征工程
    hmm_feat_array, _ = extract_hmm_features(data_dict, n_states=HMM_STATES, window_size=WINDOW_SIZE)
    X, y = build_dataset(data_dict, hmm_feat_array, window_size=WINDOW_SIZE)

    # 数据集划分
    train_mask = create_train_test_split(site_names)

    # 构建PyG Data
    data = create_pyg_data(X, y, edge_index, edge_weight, train_mask)

    # 初始化模型
    model = EnhancedGCN(in_dim=data.x.shape[1])

    # 训练与评估
    trained_model, losses = train_and_evaluate(model, data, site_names, save_path=SAVE_PATH)

    # 可视化结果
    trained_model.eval()
    with torch.no_grad():
        pred = trained_model(data).cpu().numpy()
        y_true = data.y.cpu().numpy()

    plot_results(y_true, pred, data.train_mask, losses)