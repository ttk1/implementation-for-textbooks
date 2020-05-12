import argparse
import os
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.eager import context


# seedの固定
np.random.seed(555)


class FRNN(tf.keras.Model):
    """エンコーダー
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # v2.0 alpha版ではtf.keras.layersとの完全な互換性がない
        # 一部v1.xのクラスを使用している
        self.lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_dim,
                                                      state_is_tuple=True)

    def call(self, input_seq):
        batch_size = input_seq.shape[1]
        # LSTMの状態初期化
        state = self.lstm.zero_state(batch_size, tf.float32)

        X = tf.transpose(input_seq, [1, 0, 2])
        _, state = tf.compat.v1.nn.dynamic_rnn(self.lstm,
                                               X,
                                               initial_state=state)
        return state


class BRNN(tf.keras.Model):
    """デコーダ―
    """

    def __init__(self, hidden_dim, n_dim_obs=1, training=True):
        super().__init__()
        # デコーダ―側のLSTM
        self.lstm = \
            tf.compat.v1.nn.rnn_cell.LSTMCell(num_units=hidden_dim,
                                              state_is_tuple=True)
        # 隠れ層の状態をもとに出力値を計算するための全結合層
        self.out_linear = tf.keras.layers.Dense(
            n_dim_obs,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))
        # 教師データを入力しLSTMの隠れ層の次元に合わせる全結合層
        self.shape_linear = tf.keras.layers.Dense(
            hidden_dim,
            activation=tf.tanh,
            kernel_initializer=tf.random_uniform_initializer(-0.1, 0.1))
        self.n_dim_obs = n_dim_obs
        self.training = training

    def predict(self, state, seq_size):
        """推論用のメソッド
        """

        outputs = []
        # t時点の出力
        # stateは[hidden_state, cell_state]を保持している
        output = self.out_linear(state[0])
        outputs.insert(0, output)
        for _ in range(1, seq_size):
            # 次元をLSTMの隠れ状態に合わせる
            inp = self.shape_linear(output)
            _, state = self.lstm(inp, state)
            # 隠れ状態から出力値を算出する
            output = self.out_linear(state[0])
            outputs.insert(0, output)
        return outputs

    def train(self, input_seq, state, seq_size):
        """訓練用のメソッド
        predictとほぼ同じ処理だが、観測データを使ってデコードする
        """

        outputs = []
        output = self.out_linear(state[0])
        outputs.insert(0, output)
        for t in reversed(range(1, seq_size)):
            inp = self.shape_linear(input_seq[t])
            out, state = self.lstm(inp, state)
            output = self.out_linear(state[0])
            outputs.insert(0, output)
        return outputs

    def call(self, f_lstm_latest_state, seq_size, input_seq=None):
        batch_size = f_lstm_latest_state[0].shape[0]
        # LSTMの状態の初期化
        self.lstm.zero_state(batch_size, tf.float32)

        # エンコーダーの最後の状態をデコーダ―の初期状態とする
        state = f_lstm_latest_state[0]

        # 訓練
        if self.training:
            outputs = self.train(input_seq, f_lstm_latest_state, seq_size)
        # 推論
        else:
            outputs = self.predict(f_lstm_latest_state, seq_size)
        return tf.stack(outputs, axis=0)


class EncDecAD(tf.keras.Model):
    """EncDec-ADの本体
    """

    def __init__(self,
                 hidden_dim,
                 n_dim_obs=1,
                 training=True):
        super().__init__()

        # 観測値の次元
        self.n_dim_obs = n_dim_obs
        # エンコーダー
        self.f_lstm = FRNN(hidden_dim)
        # デコーダー
        self.b_lstm = BRNN(hidden_dim, n_dim_obs, training)
        self.trainig = training

    def reset(self, training):
        self.b_lstm.training = training

    def call(self, input_seq, training=True):
        # エンコーダー/デコーダ―のtrainingオプションの初期化
        self.reset(training)

        batch_size = input_seq.shape[1]
        seq_size = input_seq.shape[0]

        # エンコーダー側の処理の実行
        h = self.f_lstm(input_seq)
        # デコーダー側の処理の実行
        outputs = self.b_lstm(h, seq_size, input_seq=input_seq)
        return outputs


def loss_fn(model, inputs, targets, training):
    """損失計算メソッド
    """
    seq_len, b_size, n_dim_obs = inputs.shape
    labels = tf.reshape(targets, [-1])
    outputs = model(inputs, training)
    # データ長に影響されるためmean_squared_errorは使えない
    # 愚直に二乗平方和を計算する
    individual_losses = tf.math.reduce_sum(
        tf.math.squared_difference(outputs, targets), axis=1)
    loss = tf.math.reduce_sum(individual_losses)
    return loss, outputs


def anomaly_score(outputs, targets, normal_data):
    """異常スコアの計算メソッド
    """
    seq_length, batch_size, n_dim_obs = targets.shape

    eval_residual = np.abs(outputs - targets)
    normal_residual = np.abs(outputs - normal_data)

    res_mu = normal_residual.mean(axis=(0,1))
    res_sig = normal_residual.std(axis=(0,1))
    res_sig_inv = np.linalg.pinv(res_sig) if res_sig.shape[0]>1 else 1/res_sig

    diff_mu = (eval_residual - res_mu).transpose(1,0,2).reshape(-1, n_dim_obs)
    if len(res_sig_inv)==1:
        scores = diff_mu / res_sig_inv * diff_mu
    else:
        scores = diff_mu.dot(res_sig_inv) * diff_mu
    return scores.sum(axis=1)


def _divide_into_batches(data, batch_size):
    """系列データをEncDec-ADへの入力用に成形する
    """
    n_time, n_dim_obs = data.shape
    nbatch = n_time // batch_size
    data = data[:nbatch * batch_size]
    data = data.reshape(batch_size, -1, n_dim_obs).transpose((1,0,2))
    return data


def _get_batch(data, i, seq_len):
    """バッチ毎にデータを取得する
    """
    slen = min(seq_len, data.shape[0] - i)
    inputs = data[i:i + slen]
    target = inputs.copy()
    # tensorflowのopに変換
    # dataは変数ではないのでconstantとする
    return tf.constant(inputs), tf.constant(target)


def evaluate(args, model, eval_data, train_data, training=False):
    """エポック毎の評価
    損失と異常スコアを計算
    """
    total_loss = 0.0
    total_batches = 0
    start = time.time()
    l_scores = []
    for _, i in enumerate(range(0, eval_data.shape[0], args.seq_len)):
        # バッチ毎にデータを取得
        inp, target = _get_batch(eval_data, i, args.seq_len)
        # 損失計算
        loss, outputs = loss_fn(model, inp, target, training=training)
        total_loss += loss.numpy()
        total_batches += 1

        _, batch_size, _= inp.shape
        # 異常スコアの計算
        scores = anomaly_score(outputs, target, train_data[:, :batch_size])
        l_scores.append(scores)

    time_in_ms = (time.time() - start) * 1000
    sys.stderr.write("eval loss %.2f (eval took %d ms)\n" %
                   (total_loss / total_batches, time_in_ms))
    return total_loss, l_scores, outputs


def train(model, optimizer, train_data, sequence_length, clip_ratio,
          training=True):
    """1エポック分の学習
    """

    def model_loss(inputs, targets):
        return loss_fn(model, inputs, targets, training=training)[0]

    total_time = 0
    batch_start_idx_range = range(0, train_data.shape[0]-1, sequence_length)
    for batch, i in enumerate(batch_start_idx_range):
        # バッチ毎にデータを取得
        train_seq, train_target = _get_batch(train_data, i, sequence_length)
        start = time.time()
        with tf.GradientTape() as tape:
            loss, _ = loss_fn(model, train_seq, train_target, training)
        # 勾配計算
        grads = tape.gradient(loss, model.trainable_variables)

        # パラメタ更新
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_time += (time.time() - start)
        if batch % 10 == 0:
            time_in_ms = (total_time * 1000) / (batch + 1)
            sys.stderr.write(
                "batch %d: training loss %.2f, avg step time %d ms\n" %
                    (batch, model_loss(train_seq, train_target).numpy(),
                     time_in_ms))


class Datasets(object):
    """ダミーデータの生成
    """

    def __init__(self):
        # 訓練データ
        t = np.linspace(0, 5*np.pi, 500)
        self.train = 10 * np.sin(t).reshape(-1,1)
        self.train = np.tile(np.abs(self.train), (32, 1)).astype('f')

        # テストデータ
        t = np.linspace(0, 4*np.pi, 400)
        self.valid = 10 * np.sin(t).reshape(-1,1)
        self.valid = np.concatenate(
            (np.random.randn(100).reshape(100,1), self.valid),
            axis=0)
        self.valid = np.tile(np.abs(self.valid), (4, 1)).astype('f')


def main(args):
    if not args.data_path:
        raise ValueError("Must specify --data-path")
    # データセットの読み出し
    data = Datasets()
    # EncDec-ADへの入力用に成形した訓練データの作成
    train_data = _divide_into_batches(data.train, args.batch_size)
    # EncDec-ADへの入力用に成形したテストデータの作成
    eval_data = _divide_into_batches(data.valid, args.eval_batch_size)

    # GPUの有無の確認
    have_gpu = context.num_gpus() > 0

    # デバイスの割り当て（GPUデバイスが検出されない場合は使わない）
    with tf.device("/device:GPU:0" if have_gpu else None):
        # 学習係数
        # 学習係数は変化するのでVariableで定義
        learning_rate = tf.Variable(args.learning_rate, name="learning_rate")
        sys.stderr.write("learning_rate=%f\n" % learning_rate.numpy())
        # EncDecADクラスのインスタンス作成
        model = EncDecAD(args.hidden_dim, args.training)
        # オプティマイザーオブジェクトの作成
        optimizer = tf.keras.optimizers.Adam(learning_rate)

        best_loss = None
        cnt = 0
        # エポック毎のループ
        for _ in range(args.epoch):
            # 訓練
            train(model, optimizer, train_data, args.seq_len, args.clip)
            # 評価
            eval_loss, l_scores, outputs = evaluate(args,
                                                    model,
                                                    eval_data,
                                                    train_data)
            # テストデータを使った評価での損失が下がった場合
            if not best_loss or eval_loss < best_loss:
                best_loss = eval_loss
                cnt = 0
            # テストデータを使った評価での損失が下がらなかった場合
            else:
                cnt += 1
                # 6回連続で下がらなかった場合
                if cnt>5:
                    # 学習係数を1/1.2倍する
                    learning_rate.assign(max(learning_rate/1.2, .002))
                    sys.stderr.write(
                        "eval_loss did not reduce in this epoch, "
                        "changing learning rate to %f for the next epoch\n" %
                            learning_rate.numpy())
                    cnt = 0

        scores = l_scores[0]
        for score in l_scores[1:]:
            scores = np.concatenate((scores, score))
        # 結果の表示
        plt.plot(data.train[:len(data.valid)],
                 color='g', alpha=0.5, label="normal")
        plt.plot(data.valid,
                 ':', color='b', alpha=0.5, label="anomalous")
        plt.plot(outputs.numpy().transpose(1,0,2).flatten(),
                 '--', color='y', label="predict")
        plt.plot(scores, '+-',  color='r', alpha=0.3, label="score")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        default="./src/chapter04/simple-examples/data",
        help="Data directory path")
    parser.add_argument(
        "--learning-rate", type=float, default=.05, help="Learning rate.")
    parser.add_argument(
        "--epoch", type=int, default=200, help="Number of epoches.")
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--eval-batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument(
        "--seq-len", type=int, default=500, help="Sequence length.")
    parser.add_argument(
        "--hidden-dim", type=int, default=10, help="Hidden layer dimension.")
    parser.add_argument(
        "--clip", type=float, default=0.3, help="Gradient clipping ratio.")
    parser.add_argument(
        "--training", type=bool, default=True, help="Training or not.")

    args, unparsed = parser.parse_known_args()
    main(args)
