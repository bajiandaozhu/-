import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from pyswarm import pso
import matplotlib
matplotlib.rc("font", family='Microsoft YaHei')

# ================= 数据准备 =================
data = {
    #'temperature': [365, 340, 315, 290, 265, 240],
    'stresses': [365, 340, 315, 290, 265],
    #'lifespan': [12.01, 12.79, 13.17, 13.81, 14.44, 15.18]
    'lifespan': [12.01, 12.79, 13.17, 13.81, 14.44]
}
df = pd.DataFrame(data)

# ================= 数据预处理 =================
X = df[['stresses']]
y = df['lifespan']

# 标准化处理
x_scaler = StandardScaler()
y_scaler = StandardScaler()
X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y.values.reshape(-1, 1))

# 划分训练测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# ================= PSO优化 =================
mse_history = []  # 记录优化过程

def svr_pso(params):
    global mse_history
    C, epsilon, kernel_index = params
    kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
    kernel = kernel_list[min(int(kernel_index), 3)]  # 确保索引不越界

    model = SVR(C=C, epsilon=epsilon, kernel=kernel)
    scores = cross_val_score(model, X_train, y_train.ravel(),
                             cv=2, scoring='neg_mean_squared_error')
    mse = -scores.mean()
    mse_history.append(mse)
    return mse

# 执行优化
lb = [1, 0.025, 0]
ub = [100, 0.1, 3]
best_params, _ = pso(svr_pso, lb, ub, swarmsize=10, maxiter=10)

# ================= 模型训练 =================
C_best, epsilon_best, kernel_index = best_params
kernel_best = ['linear', 'rbf', 'poly', 'sigmoid'][int(kernel_index)]
best_svr = SVR(C=C_best, epsilon=epsilon_best, kernel=kernel_best).fit(X_train, y_train.ravel())

# ================= 结果可视化 =================
plt.figure(figsize=(12, 5))

# 优化过程可视化
plt.subplot(1, 2, 1)
plt.plot(mse_history, 'b-o', markersize=4)
plt.title('Parameter search process')  #超参数搜索过程
plt.xlabel('Parameter combination number')  #参数组合序号
plt.ylabel('verification loss (MSE)')  #验证损失 (MSE)
plt.grid(True)

# 预测曲线可视化
plt.subplot(1, 2, 2)
test_temps = np.linspace(200, 400, 50).reshape(-1, 1)
# 转换为DataFrame保持特征名称
test_df = pd.DataFrame(test_temps, columns=['stresses'])  # 新增关键修改
test_scaled = x_scaler.transform(test_df)  # 使用DataFrame转换
pred_scaled = best_svr.predict(test_scaled)
pred = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))

plt.plot(df['stresses'], df['lifespan'], 'ro', label='real data')
plt.plot(test_temps, pred, 'b-', label='forecast curve')
plt.title('stresses-life relationship prediction')
plt.xlabel('stresses (MPa)')
plt.ylabel('life (lnN)')
plt.legend()
plt.tight_layout()
plt.savefig('C:/Users/huyingqiang/Desktop/可靠性/SCI论文/1.png')
plt.show()

# ================= 结果输出 =================
print("\n=== 最优参数组合 ===")
print(f"正则化参数 C: {best_svr.C:.4f}")
print(f"核函数类型: {best_svr.kernel}")
print(f"不敏感带 epsilon: {best_svr.epsilon:.4f}")

# 处理gamma参数类型问题
gamma_value = best_svr.gamma
if isinstance(gamma_value, str):
    print(f"核参数 gamma: {gamma_value}")
else:
    print(f"核参数 gamma: {gamma_value:.4f}")

# 预测函数
def svr_predict(temp):
    # 转换为带特征名称的DataFrame
    input_df = pd.DataFrame([[temp]], columns=['stresses'])  # 新增关键修改
    scaled_temp = x_scaler.transform(input_df)
    pred_scaled = best_svr.predict(scaled_temp)
    return y_scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]

# 测试结果对比
y_pred = y_scaler.inverse_transform(best_svr.predict(X_test).reshape(-1, 1))
print("\n=== 测试集表现 ===")
print(pd.DataFrame({
    '实际值': y_scaler.inverse_transform(y_test).flatten(),
    '预测值': y_pred.flatten(),
    '误差(%)': np.abs(y_scaler.inverse_transform(y_test).flatten() - y_pred.flatten()) /
             y_scaler.inverse_transform(y_test).flatten() * 100
}))

print("\n=== 特别预测 ===")
print(f"压力 215MPa 的预测寿命: {svr_predict(240):.4f} lnN")
