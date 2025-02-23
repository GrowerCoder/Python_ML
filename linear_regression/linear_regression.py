import numpy as np
import pandas as pd

class LinearRegression:
    def __init__(self, file_path, learning_rate=0.01, iterations=1000):
        # 保持原有数据加载和预处理逻辑
        self.data = pd.read_csv(file_path, header=None, dtype=float)
        if not all(dtype.kind == 'f' for dtype in self.data.dtypes):
            raise ValueError("Data contains non-float values. All elements must be floating-point numbers.")
            
        # 保持原有标准化逻辑
        self.data = (self.data - self.data.mean()) / self.data.std()
        self.data.insert(0, 'Ones', 1)
        
        # 保持原有矩阵生成逻辑
        cols = self.data.shape[1]
        self.X = np.matrix(self.data.iloc[:, 0:cols-1].values)
        self.y = np.matrix(self.data.iloc[:, cols-1:cols].values)
        self.theta = np.matrix(np.zeros(cols-1))
        
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def compute_cost(self):
        error = self.X * self.theta.T - self.y
        inner_term = np.power(error, 2)
        return np.sum(inner_term) / (2 * len(self.X))
    
    def gradient_descent(self):
        """calculating using gradient descent"""
        cost_history = []
        for _ in range(self.iterations):
            gradient = (self.X.T * (self.X * self.theta.T - self.y)) / len(self.X)
            self.theta = self.theta - (self.learning_rate * gradient).T
            cost_history.append(self.compute_cost())
        return self.theta, cost_history[-1]
    
    def normal_equation(self):
        '''calaculating using normal equation'''
        return np.linalg.inv(self.X.T * self.X) * self.X.T * self.y

if __name__ == "__main__":
    file_path = '/Users/jinchengguo/Python_ML/linear_regression/data.csv'  # 修正文件路径为实际数据文件
    learning_rate = 0.01
    iterations = 1000
    model = LinearRegression(file_path, learning_rate, iterations)
    theta, cost = model.gradient_descent()
    print(f"Final Parameters: θ0 = {theta[0,0]:.4f}, θ1 = {theta[0,1]:.4f}")
    print(f"Minimum Cost: {cost:.4f}")
    