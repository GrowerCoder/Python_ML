# ml_linear_regression.py
"""
Linear Regression Implementation (Univariate & Multivariate)
Includes both Gradient Descent and Normal Equation solutions
Datasets: ex1data1.txt (univariate) and ex1data2.txt (multivariate)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
    """通用线性回归模型"""
    def __init__(self, method='gradient_descent', alpha=0.01, iterations=1000):
        """
        初始化线性回归模型
        Parameters:
            method - 优化方法: 'gradient_descent' 或 'normal_equation'
            alpha - 学习率（仅梯度下降需要）
            iterations - 迭代次数（仅梯度下降需要）
        """
        self.method = method
        self.alpha = alpha
        self.iterations = iterations
        self.theta = None
        self.cost_history = []
        self.feature_stats = {}  # 存储特征标准化参数
        
    @staticmethod
    def add_ones_column(X: np.ndarray) -> np.ndarray:
        """添加全1列作为x0特征"""
        return np.insert(X, 0, 1, axis=1)
    
    def feature_normalize(self, X: np.ndarray) -> np.ndarray:
        """特征标准化（Z-score标准化）"""
        if not self.feature_stats:
            # 添加微小值防止除零
            self.feature_stats = {
                'mean': X.mean(axis=0),
                'std': X.std(axis=0) + 1e-8
            }
        return (X - self.feature_stats['mean']) / self.feature_stats['std']
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算当前参数的损失值"""
        m = len(X)
        if self.theta is None or np.isnan(self.theta).any():
            return np.inf
        predictions = X @ self.theta.T
        return np.sum((predictions - y.reshape(-1, 1))**2) / (2 * m)
    
    def gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """执行批量梯度下降"""
        m = len(X)
        y = y.reshape(-1, 1)
        
        for i in range(self.iterations):
            predictions = X @ self.theta.T
            error = predictions - y
            gradients = (X.T @ error) / m
            self.theta -= self.alpha * gradients.T
            
            # 计算并存储损失值
            cost = self.compute_cost(X, y)
            if np.isnan(cost):
                print(f"警告: 第{i}次迭代出现NaN值，提前终止")
                break
            self.cost_history.append(cost)
            
            # 动态调整学习率
            if len(self.cost_history) > 1 and self.cost_history[-1] > self.cost_history[-2]:
                self.alpha *= 0.5  # 损失增加时减小学习率
    
    def normal_equation(self, X: np.ndarray, y: np.ndarray):
        """正规方程解法"""
        # 使用伪逆提高数值稳定性
        self.theta = np.linalg.pinv(X.T @ X) @ X.T @ y.reshape(-1, 1)
    
    def fit(self, X: np.ndarray, y: np.ndarray, normalize=False):
        """
        训练模型
        Parameters:
            X - 特征矩阵
            y - 目标向量
            normalize - 是否进行特征标准化
        """
        # 数据预处理
        X = X.copy()
        if normalize:
            X = self.feature_normalize(X)
        X = self.add_ones_column(X)
        
        # 初始化参数
        self.theta = np.zeros((1, X.shape[1]))
        
        # 选择优化方法
        if self.method == 'gradient_descent':
            self.gradient_descent(X, y)
        elif self.method == 'normal_equation':
            self.normal_equation(X, y)
        else:
            raise ValueError("不支持的优化方法")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """使用训练好的参数进行预测"""
        if self.feature_stats:
            X = (X - self.feature_stats['mean']) / self.feature_stats['std']
        X = self.add_ones_column(X)
        return X @ self.theta.T
    
    def plot_training(self, X: np.ndarray, y: np.ndarray):
        """可视化训练结果（仅适用于单变量）"""
        if X.shape[1] != 1:  # 原始特征维度检查
            raise ValueError("可视化仅支持单变量")
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 回归线可视化
        X_with_bias = self.add_ones_column(X)
        x_vals = np.array([X.min(), X.max()])
        y_pred = self.predict(x_vals)
        ax1.scatter(X, y, label='训练数据')
        ax1.plot(x_vals, y_pred, 'r', label='回归线')
        ax1.set_title('特征 vs 目标值')
        ax1.legend()
        
        # 损失函数曲线
        if self.method == 'gradient_descent':
            ax2.plot(range(len(self.cost_history)), self.cost_history, 'b')
            ax2.set_title('损失函数变化曲线')
            ax2.set_xlabel('迭代次数')
            ax2.set_ylabel('损失值')
        
        plt.tight_layout()
        plt.show()

# ----------------- Main Program Execution -----------------
if __name__ == "__main__":
    # ========== 单变量线性回归 ==========
    # 加载数据
    data1 = pd.read_csv('ml_linear_regression/ex1data1.txt', names=['Population', 'Profit'])
    X1 = data1['Population'].values.reshape(-1, 1)
    y1 = data1['Profit'].values
    
    # 训练模型（添加标准化）
    model_univariate = LinearRegression(method='gradient_descent', alpha=0.01, iterations=1000)
    model_univariate.fit(X1, y1, normalize=True)  # 添加标准化
    print(f"单变量回归参数:\n{model_univariate.theta}")
    
    # 结果可视化
    model_univariate.plot_training(X1, y1)

    # ========== 多变量线性回归 ==========
    # 加载数据
    data2 = pd.read_csv('ml_linear_regression/ex1data2.txt', names=['Size', 'Bedrooms', 'Price'])
    X2 = data2[['Size', 'Bedrooms']].values
    y2 = data2['Price'].values
    
    # 使用两种方法比较
    model_gd = LinearRegression(method='gradient_descent', alpha=0.01, iterations=400)
    model_eqn = LinearRegression(method='normal_equation')
    
    model_gd.fit(X2, y2, normalize=True)
    model_eqn.fit(X2, y2, normalize=True)
    
    print("\n多变量回归结果比较:")
    print(f"梯度下降参数:\n{model_gd.theta}")
    print(f"正规方程参数:\n{model_eqn.theta}")