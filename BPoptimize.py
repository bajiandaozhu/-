import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut
from multiprocessing import Pool
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')
# ================= 数据准备 =================
data = {
    'temperature': [365, 340, 315, 290, 265, 240],
    'lifespan': [12.0114, 12.7956, 13.1671, 13.8114, 14.4386, 15.1809]
}
df = pd.DataFrame(data)


# ================= 超参数优化设置 =================
class PSOHyperTuner:
    def __init__(self, data, n_particles=10, max_iter=30, cv=LeaveOneOut()):
        self.data = data
        self.cv = cv
        self.n_particles = n_particles
        self.max_iter = max_iter

        # 超参数搜索空间
        self.param_ranges = {
            'hidden': (4, 16),  # 隐藏层神经元数量范围（整数）
            'lr': (0.001, 0.1),  # 学习率范围（对数尺度）
            'epochs': (1000, 20000)  # 训练周期范围（整数）
        }

        # PSO参数
        self.w = 0.7  # 惯性权重
        self.c1 = 1.4  # 个体学习因子
        self.c2 = 1.5  # 社会学习因子

    def _init_particles(self):
        particles = []
        for _ in range(self.n_particles):
            particle = {
                'hidden': np.random.randint(*self.param_ranges['hidden']),
                'lr': 10 ** np.random.uniform(np.log10(self.param_ranges['lr'][0]),
                                              np.log10(self.param_ranges['lr'][1])),
                'epochs': np.random.randint(*self.param_ranges['epochs']),
                'velocity': [0, 0, 0],
                'best_loss': float('inf'),
                'best_params': None
            }
            particles.append(particle)
        return particles

    def _evaluate(self, params):
        """执行交叉验证评估超参数"""
        loo = LeaveOneOut()
        losses = []

        for train_idx, val_idx in loo.split(self.data):
            # 数据分割
            train_data = self.data.iloc[train_idx]
            val_data = self.data.iloc[val_idx]

            # 归一化处理
            x_scaler = MinMaxScaler(feature_range=(-1, 1))
            y_scaler = MinMaxScaler(feature_range=(-1, 1))

            x_train = x_scaler.fit_transform(train_data[['temperature']].values)
            y_train = y_scaler.fit_transform(train_data[['lifespan']].values)
            x_val = x_scaler.transform(val_data[['temperature']].values)
            y_val = y_scaler.transform(val_data[['lifespan']].values)

            # 训练模型
            model = BPModel(
                hidden_units=params['hidden'],
                learning_rate=params['lr'],
                max_epochs=params['epochs']
            )
            model.fit(x_train.T, y_train.T)

            # 验证预测
            pred = model.predict(x_val.T)
            loss = np.mean((pred - y_val.T) ** 2)
            losses.append(loss)

        return np.mean(losses)

    def optimize(self):
        particles = self._init_particles()
        global_best = {'loss': float('inf'), 'params': None}
        history = []

        for iter in range(self.max_iter):
            print(f"Iteration {iter + 1}/{self.max_iter}")

            # 并行评估粒子
            with Pool() as p:
                losses = p.map(self._evaluate, particles)

            # 更新个体最优和全局最优
            for i, loss in enumerate(losses):
                if loss < particles[i]['best_loss']:
                    particles[i]['best_loss'] = loss
                    particles[i]['best_params'] = particles[i].copy()

                if loss < global_best['loss']:
                    global_best['loss'] = loss
                    global_best['params'] = particles[i].copy()

            # 更新粒子速度和位置
            for p in particles:
                # 速度更新
                r1, r2 = np.random.rand(2)
                v_old = p['velocity']

                # 参数索引顺序：[hidden, lr, epochs]
                new_v = [
                    self.w * v_old[0] + self.c1 * r1 * (p['best_params']['hidden'] - p['hidden']) +
                    self.c2 * r2 * (global_best['params']['hidden'] - p['hidden']),

                    self.w * v_old[1] + self.c1 * r1 * (np.log10(p['best_params']['lr']) - np.log10(p['lr'])) +
                    self.c2 * r2 * (np.log10(global_best['params']['lr']) - np.log10(p['lr'])),

                    self.w * v_old[2] + self.c1 * r1 * (p['best_params']['epochs'] - p['epochs']) +
                    self.c2 * r2 * (global_best['params']['epochs'] - p['epochs'])
                ]

                # 位置更新
                p['hidden'] = int(np.clip(p['hidden'] + new_v[0], *self.param_ranges['hidden']))
                p['lr'] = 10 ** (np.log10(p['lr']) + new_v[1]).clip(*np.log10(self.param_ranges['lr']))
                p['epochs'] = int(np.clip(p['epochs'] + new_v[2], *self.param_ranges['epochs']))
                p['velocity'] = new_v

            history.append(global_best['loss'])
            print(f"Best Loss: {global_best['loss']:.6f}")

        return global_best['params'], history


# ================= BP神经网络模型 =================
class BPModel:
    def __init__(self, hidden_units=8, learning_rate=0.01, max_epochs=10000):
        self.hidden_units = hidden_units
        self.lr = learning_rate
        self.max_epochs = max_epochs

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X, Y):
        # 初始化权重
        np.random.seed(42)
        self.W1 = np.random.randn(self.hidden_units, X.shape[0]) * 0.1
        self.b1 = np.zeros((self.hidden_units, 1))
        self.W2 = np.random.randn(Y.shape[0], self.hidden_units) * 0.1
        self.b2 = np.zeros((Y.shape[0], 1))

        for epoch in range(self.max_epochs):
            # 前向传播
            self.hidden = self._sigmoid(self.W1 @ X + self.b1)
            output = self.W2 @ self.hidden + self.b2

            # 计算损失
            loss = np.mean((output - Y) ** 2)

            # 反向传播
            d_output = (output - Y) / X.shape[1]
            dW2 = d_output @ self.hidden.T
            db2 = np.sum(d_output, axis=1, keepdims=True)

            d_hidden = self.W2.T @ d_output * self._sigmoid_derivative(self.hidden)
            dW1 = d_hidden @ X.T
            db1 = np.sum(d_hidden, axis=1, keepdims=True)

            # 更新参数
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1

            if loss < 1e-5:
                break

    def predict(self, X):
        hidden = self._sigmoid(self.W1 @ X + self.b1)
        return self.W2 @ hidden + self.b2


# ================= 主程序 =================
if __name__ == "__main__":
    # 超参数优化
    tuner = PSOHyperTuner(df, n_particles=8, max_iter=20)
    best_params, loss_history = tuner.optimize()

    print("\n=== 最优超参数 ===")
    print(f"隐藏层神经元数: {best_params['hidden']}")
    print(f"学习率: {best_params['lr']:.6f}")
    print(f"训练周期: {best_params['epochs']}")

    # 使用最优参数训练最终模型
    x_scaler = MinMaxScaler(feature_range=(-1, 1))
    y_scaler = MinMaxScaler(feature_range=(-1, 1))
    x = x_scaler.fit_transform(df[['temperature']].values)
    y = y_scaler.fit_transform(df[['lifespan']].values)

    final_model = BPModel(
        hidden_units=best_params['hidden'],
        learning_rate=best_params['lr'],
        max_epochs=best_params['epochs']
    )
    final_model.fit(x.T, y.T)

    # 可视化预测结果
    plt.figure(figsize=(12, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, 'b-o')
    plt.title('PSO Optimization Process')
    plt.xlabel('Iteration')
    plt.ylabel('Validation Loss (MSE)')

    # 预测曲线
    plt.subplot(1, 2, 2)
    test_temps = np.linspace(200, 400, 50).reshape(-1, 1)
    test_scaled = x_scaler.transform(test_temps)
    pred_scaled = final_model.predict(test_scaled.T)
    pred = y_scaler.inverse_transform(pred_scaled.T)

    plt.plot(df['temperature'], df['lifespan'], 'ro', label='真实数据')
    plt.plot(test_temps, pred, 'b-', label='预测曲线')
    plt.title('温度-寿命预测曲线')
    plt.xlabel('温度 (℃)')
    plt.ylabel('寿命 (小时)')
    plt.legend()
    plt.tight_layout()
    plt.show()


    # 215℃预测
    def predict(temp):
        scaled_temp = x_scaler.transform([[temp]])
        pred_scaled = final_model.predict(scaled_temp.T)
        return y_scaler.inverse_transform(pred_scaled.T)[0][0]


    print("\n=== 特别预测 ===")
    print(f"温度 215℃ 的预测寿命: {predict(215):.2f} 小时")









