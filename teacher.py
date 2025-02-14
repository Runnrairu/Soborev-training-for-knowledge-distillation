# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from loaddata import load_data  # loaddata.pyからデータをインポート
import matplotlib.pyplot as plt  # グラフ描画のためにmatplotlibをインポート

# MLPモデルの定義
class MLP(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(hidden_dim, activation='relu', input_shape=(hidden_dim,))
        self.dense3 = tf.keras.layers.Dense(output_dim)
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

# モデルを保存する関数
def save_model(model, path):
    model.save(path, save_format='tf')

# モデルをロードする関数
def load_model(path):
    return tf.keras.models.load_model(path)

# 学習用関数
def train_model(X_train, y_train, model, criterion, optimizer, batch_size=32, epochs=200):
    # 損失を保存するリスト
    loss_list = []

    # 学習ループ
    for epoch in range(epochs):
        # バッチごとの処理
        epoch_loss = 0.0
        for batch_idx in range(0, len(X_train), batch_size):
            batch_inputs = X_train[batch_idx:batch_idx + batch_size]
            batch_targets = y_train[batch_idx:batch_idx + batch_size]
            
            with tf.GradientTape() as tape:
                outputs = model(batch_inputs)
                loss = criterion(batch_targets, outputs)  # 回帰なのでそのまま
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / (len(X_train) // batch_size)
        loss_list.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

    return loss_list

# モデルの出力の勾配を計算する関数
def calculate_model_output_gradient(model, X_train):
    with tf.GradientTape() as tape:
        tape.watch(X_train)  # X_trainも勾配計算対象としてウォッチ
        outputs = model(X_train)  # モデルの出力計算
    gradients = tape.gradient(outputs, X_train)  # モデルの出力 f(X) に対する X_train の勾配を計算
    return gradients

# 損失のグラフを描画する関数
def plot_loss(loss_list):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(loss_list) + 1), loss_list, label="Training Loss")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss During Training')
    plt.legend()
    plt.grid(True)
    plt.show()

# メイン関数
def gen_teacher():
    # データのロード
    X_train, X_test, y_train, y_test = load_data(dataset_type='regression')

    # モデルの設定
    input_dim = X_train.shape[1]  # 特徴量の数
    hidden_dim = 512  # 隠れ層の次元
    output_dim = 1  # 回帰なので出力は1次元

    # モデルのインスタンス化
    model_f = MLP(input_dim, hidden_dim, output_dim)

    # 損失関数とオプティマイザ
    criterion = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # 学習を実行
    loss_list = train_model(X_train, y_train, model_f, criterion, optimizer, batch_size=32, epochs=200)

    # 損失のグラフを表示
    plot_loss(loss_list)

    # モデルの保存（SavedModel形式で保存）
    save_model(model_f, 'mlp_model')
    print("モデルを保存しました：mlp_model")

    # 学習後、モデルの出力に対する入力の勾配を計算
    gradients = calculate_model_output_gradient(model_f, X_train)
    print("勾配計算結果（X_trainに対する勾配）:")
    print(gradients.numpy())

    # モデルの出力Y_fも計算
    Y_f = model_f(X_train)
    print("モデルの出力Y_f:")
    print(Y_f.numpy())

    # 勾配Y_dot_trainも計算
    Y_dot_train = gradients.numpy()
    print("モデルの出力Y_dot_train（勾配）:")
    print(Y_dot_train)

    return X_train, X_test, y_train, y_test, model_f, Y_f, Y_dot_train

# 実行
if __name__ == "__main__":
    gen_teacher()