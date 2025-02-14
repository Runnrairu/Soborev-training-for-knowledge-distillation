# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from teacher import gen_teacher  # teacherモジュールのインポート

# MLPモデルの定義（g用）
class MLP_G(tf.keras.Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP_G, self).__init__()
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='swish', input_shape=(input_dim,))
        self.dense2 = tf.keras.layers.Dense(output_dim)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        return x

# gの損失関数（fとの比較）
def sobolev_loss(f_outputs, g_outputs, f_gradients, g_gradients):
    # 出力の差を計算
    output_difference = tf.reduce_mean(tf.square(f_outputs - g_outputs))
    print("Mean output_difference:", tf.reduce_mean(tf.abs(output_difference)).numpy())
    # 勾配の差を計算
    gradient_difference =5* tf.reduce_mean(tf.square(f_gradients - g_gradients))
    print("Mean gradient_difference:", tf.reduce_mean(tf.abs(gradient_difference)).numpy())

    
    # 総合的な損失（出力差と勾配差の和）
    loss = output_difference# + gradient_difference
    
    return loss

# gの学習用関数
def train_g(X_train, y_train, model_f, model_g, Y_f, Y_dot_train, criterion, optimizer, batch_size=32, epochs=1000):
    # 学習ループ
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx in range(0, len(X_train), batch_size):
            batch_inputs = X_train[batch_idx:batch_idx + batch_size]
            batch_targets = y_train[batch_idx:batch_idx + batch_size]
            
            # f(X)とgrad f(X)を用意
            f_outputs = Y_f[batch_idx:batch_idx + batch_size]
            f_gradients = Y_dot_train[batch_idx:batch_idx + batch_size]

            with tf.GradientTape() as tape:
                g_outputs = model_g(batch_inputs)  # g(X)の出力

                # g(X)の勾配を計算
                with tf.GradientTape() as tape_grad:
                    tape_grad.watch(batch_inputs)
                    g_outputs = model_g(batch_inputs)
                g_gradients = tape_grad.gradient(g_outputs, batch_inputs)

                # Sobolev損失を計算
                loss = sobolev_loss(f_outputs, g_outputs, f_gradients, g_gradients)

            gradients = tape.gradient(loss, model_g.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model_g.trainable_variables))
            
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / (len(X_train) // batch_size)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss}")

    return model_g

# モデルの評価（MSE計算）
def evaluate_model(model_f, model_g, X_test, y_test, Y_f, Y_dot_train):
    f_outputs = model_f(X_test)
    g_outputs = model_g(X_test)
    f_gradients = Y_dot_train

    # f_outputsとy_testをnumpy配列に変換してMSEを計算
    f_outputs_numpy = f_outputs.numpy()
    y_test_numpy = y_test.numpy()

    # MSE（f(X)とg(X)の比較）
    mse_f = mean_squared_error(y_test_numpy, f_outputs_numpy)
    mse_g = mean_squared_error(y_test_numpy, g_outputs.numpy())  # g_outputsもnumpyに変換

    return mse_f, mse_g
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
def soborev_train(X_train, X_test, y_train, y_test, model_f, Y_f, Y_dot_train):
    input_dim = X_train.shape[1]  # 特徴量の数
    hidden_dim = 64 # 隠れ層の次元
    output_dim = 1  # 回帰なので出力は1次元

    # gモデルのインスタンス化
    model_g = MLP_G(input_dim, hidden_dim, output_dim)

    # 損失関数とオプティマイザ
    criterion = sobolev_loss
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # gを学習
    model_g = train_g(X_train, y_train, model_f, model_g, Y_f, Y_dot_train, criterion, optimizer, epochs=1000)

    # gとfの評価
    mse_f, mse_g = evaluate_model(model_f, model_g, X_test, y_test, Y_f, Y_dot_train)

    print(f"fモデルのMSE: {mse_f}")
    print(f"gモデルのMSE: {mse_g}")

    return model_g, mse_f, mse_g

# 実行
if __name__ == "__main__":
    # 必要なデータを受け取る部分
    X_train, X_test, y_train, y_test, model_f, Y_f, Y_dot_train = gen_teacher()  # gen_teacher()でデータをロード

    # Sobolev訓練
    soborev_train(X_train, X_test, y_train, y_test, model_f, Y_f, Y_dot_train)