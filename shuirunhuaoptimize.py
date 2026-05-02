import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.layers import Conv1D, Dense, Flatten, Dropout, BatchNormalization, MaxPooling1D, \
    GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import LeaveOneOut
import matplotlib

matplotlib.rc("font", family='Microsoft YaHei')

# ================== 数据准备 ==================
data = {
    '摩擦系数': [0.08, 0.09, 0.09, 0.1, 0.21, 0.33, 0.4, 0.38, 0.54, 0.18, 0.25, 0.45],
    '温度': [21, 20, 21, 23, 26, 27, 29, 29, 32, 23, 26, 34],
    '噪声': [42, 40, 43, 42, 48, 52, 58, 62, 60, 45, 50, 58],
    '振动': [0.1, 0.2, 0.1, 0.2, 0.5, 0.8, 1.2, 4, 2.8, 1.5, 2.8, 3.4],
    '标签': [
        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0],
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]]
}
df = pd.DataFrame(data)

status_mapping = {(0, 0, 0, 0): 0, (1, 0, 0, 0): 1, (0, 1, 0, 0): 2,
                  (0, 0, 1, 0): 3, (0, 0, 0, 1): 4}
status_map = {
    0: "正常轴承",
    1: "橡胶内衬裂纹",
    2: "艇轴弯曲",
    3: "橡胶内衬疲劳断裂",
    4: "橡胶内衬磨损"
}
y = np.array([status_mapping[tuple(label)] for label in df['标签']])
X_original = df[['摩擦系数', '温度', '噪声', '振动']].values

# ================== 数据预处理 ==================


def preprocess_data(X, y):
    if np.any(np.isnan(X)):
        print("警告：输入特征包含缺失值，正在进行填充。")
        X = np.nan_to_num(X)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


X_scaled, y, scaler = preprocess_data(X_original, y)


# ================== 增强模型架构 ==================
class EnhancedHybridModel(tf.keras.Model):
    def __init__(self, input_shape):
        super().__init__()
        self.base = tf.keras.Sequential([
            Conv1D(32, 3, activation='relu', padding='same',
                   kernel_regularizer=l2(0.001), input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(2),
            Conv1D(64, 3, activation='relu', padding='same',
                   kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            GlobalAveragePooling1D(),
            Dense(256, kernel_regularizer=l2(0.001)),
            Dropout(0.5)
        ])
        self.domain_adapter = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dropout(0.3)
        ])
        self.classifier = tf.keras.Sequential([
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(5, activation='softmax')
        ])
        self.build((None,) + input_shape)

    def call(self, inputs):
        x = self.base(inputs)
        x = self.domain_adapter(x)
        return self.classifier(x)


# ================== 改进训练流程 ==================
class EnhancedHybridTrainer:
    def __init__(self, input_shape):
        self.model = EnhancedHybridModel(input_shape)
        self.adv_generator = tf.keras.Sequential([
            Conv1D(16, 2, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            Conv1D(64, 2, activation='relu', padding='same'),
            Conv1D(input_shape[-1], 2, activation='tanh', padding='same')
        ])

        # 动态学习率
        class_lr = ExponentialDecay(0.0001, 1000, 0.9)
        gen_lr = ExponentialDecay(0.0002, 1000, 0.9)
        self.class_optimizer = Adam(class_lr)
        self.gen_optimizer = Adam(gen_lr)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

    def compute_loss(self, y_true, y_pred):
        main_loss = self.loss_fn(y_true, y_pred)
        reg_loss = tf.add_n(self.model.losses)
        return main_loss + 0.1 * reg_loss  # 调整正则化权重

    def train_step(self, X_batch, y_batch):
        with tf.GradientTape(persistent=True) as tape:
            # 数据增强：添加随机噪声
            X_aug = X_batch + tf.random.normal(tf.shape(X_batch), stddev=0.01)

            # 生成对抗扰动
            perturbation = self.adv_generator(X_aug) * 0.1  # 限制扰动幅度
            X_adv = X_aug + perturbation

            # 混合数据
            X_mixed = tf.concat([X_aug, X_adv], axis=0)
            y_mixed = tf.concat([y_batch, y_batch], axis=0)

            # 前向传播
            y_pred = self.model(X_mixed)
            total_loss = self.compute_loss(y_mixed, y_pred)

        # 更新分类器
        class_grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.class_optimizer.apply_gradients(zip(class_grads, self.model.trainable_variables))

        # 更新生成器（梯度反转）
        gen_grads = tape.gradient(total_loss, self.adv_generator.trainable_variables)
        gen_grads = [-g for g in gen_grads]  # 反转梯度方向
        self.gen_optimizer.apply_gradients(zip(gen_grads, self.adv_generator.trainable_variables))

        del tape
        return total_loss.numpy()


# ================== 增强训练流程 ==================
def enhanced_training(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    input_shape = (X.shape[1], 1)
    X_3d = X_scaled.reshape(-1, *input_shape).astype(np.float32)

    trainer = EnhancedHybridTrainer(input_shape)
    loo = LeaveOneOut()
    accuracies = []
    all_y_true = []
    all_y_pred = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(loo.split(X)):
        X_train, X_val = X_3d[train_idx], X_3d[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # 创建数据管道
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.shuffle(1024).batch(32).prefetch(2)

        best_val_loss = np.inf
        wait = 0
        patience = 200
        train_loss = []
        val_loss_history = []

        for epoch in range(4000):
            # 训练阶段
            epoch_loss = []
            for X_batch, y_batch in train_dataset:
                loss = trainer.train_step(X_batch, y_batch)
                epoch_loss.append(loss)

            # 验证阶段
            val_pred = trainer.model(X_val)
            val_loss = trainer.loss_fn(y_val, val_pred).numpy()
            val_loss_history.append(val_loss)

            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

            train_loss.append(np.mean(epoch_loss))
            if (epoch + 1) % 100 == 0:
                print(
                    f"Fold {fold + 1} Epoch {epoch + 1} | Train Loss: {np.mean(epoch_loss):.4f} | Val Loss: {val_loss:.4f}")

        # 最终验证评估
        y_pred = np.argmax(trainer.model(X_val), axis=1)
        acc = accuracy_score(y_val, y_pred)

        fold_histories.append({'train_loss': train_loss, 'val_loss': val_loss_history})
        all_y_true.append(y_val[0])
        all_y_pred.append(y_pred[0])
        accuracies.append(acc)
        print(f"Fold {fold + 1} | Acc: {acc:.2%}")

    # ========== 可视化增强 ==========
    plt.figure(figsize=(14, 6))

    # 损失曲线
    plt.subplot(1, 2, 1)
    for i, hist in enumerate(fold_histories):
        plt.plot(hist['train_loss'], color='purple', alpha=0.3, label='Train Loss' if i == 0 else "")
        plt.plot(hist['val_loss'], color='orange', alpha=0.3, label='Val Loss' if i == 0 else "")
    plt.title('Training/Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(alpha=0.3)

    # 混淆矩阵
    plt.subplot(1, 2, 2)
    cm = confusion_matrix(all_y_true, all_y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(status_map.values()))
    disp.plot(cmap='Oranges', ax=plt.gca(), values_format='d')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"\n平均准确率: {np.mean(accuracies):.2%}")
    return trainer.model, scaler


# ================== 执行训练 ==================
final_model, final_scaler = enhanced_training(X_original, y)


# ================== 诊断接口 ==================
def diagnose(features, case_num):
    scaled = final_scaler.transform([features])
    input_3d = scaled.reshape(1, scaled.shape[1], 1)
    proba = final_model.predict(input_3d, verbose=0)

    fig = plt.figure(figsize=(10, 5), num=f"Diagnosis {case_num}")
    colors = ['#4C72B0'] * 5
    max_idx = np.argmax(proba)
    colors[max_idx] = '#DD8452'

    bars = plt.bar(status_map.values(), proba[0], color=colors)
    plt.ylim(0, 1)
    plt.title(f'案例 {case_num} 故障概率分布', pad=20)
    plt.ylabel('Probability')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 f'{height:.1%}', ha='center', va='bottom')

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return {
        'pred_dist': np.round(proba[0], 3),
        'diagnosis': status_map[max_idx],
        'confidence': f"{proba[0][max_idx] * 100:.1f}%"
    }


# ================== 测试案例 ==================
test_cases = [
    [0.09, 20, 40, 0.2],  # 正常
    [0.09, 21, 43, 0.1],  # 正常
    [0.4, 29, 58, 1.2],  # 裂纹
    [0.25, 26, 50, 2.8],  # 磨损
    [0.33, 27, 52, 0.8],  # 裂纹
    [0.18, 23, 45, 1.5],  # 磨损
    [0.38, 29, 62, 4],  # 弯曲
    [0.54, 32, 60, 2.8]  # 断裂
]

print("\n诊断测试结果:")
for i, case in enumerate(test_cases, 1):
    print(f"\n案例 {i}: {case}")
    result = diagnose(case, i)
    print(f"结论: {result['diagnosis']} ({result['confidence']})")
    plt.close('all')