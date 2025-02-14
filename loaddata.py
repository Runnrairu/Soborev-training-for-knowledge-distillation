# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, load_diabetes

def load_data(test_size=0.2, random_state=42, dataset_type='regression'):
    """
    データを読み込み、訓練データとテストデータに分割し、TensorFlowのテンソルに変換します。
    
    引数:
    test_size (float): テストデータの割合（デフォルトは0.2）
    random_state (int): データ分割時の乱数シード（デフォルトは42）
    dataset_type (str): 使用するデータセットの種類 ('regression' または 'diabetes')
    
    戻り値:
    X_train, X_test, y_train, y_test: 訓練データとテストデータ (TensorFlowテンソル)
    """
    
    if dataset_type == 'regression':
        # 回帰問題用のサンプルデータを生成（make_regressionを使う）
        X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=random_state)
    elif dataset_type == 'diabetes':
        # 糖尿病のデータセットを読み込む（scikit-learn の糖尿病データセット）
        diabetes = load_diabetes()
        X, y = diabetes.data, diabetes.target
    else:
        raise ValueError("Invalid dataset_type. Choose 'regression' or 'diabetes'.")
    
    # データの分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # NumPy 配列から TensorFlow テンソルに変換
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    
    return X_train, X_test, y_train, y_test