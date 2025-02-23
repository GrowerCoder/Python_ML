Linear Regression

# supervised and unsupervised learning
**supervised learning**
    a set of training data is given, and the model is trained to predict the output for new data.
**unsupervised learning**
    a set of training data is given, and the model is trained to find the pattern of the data.

# linear regression
linear regression is a supervised learning algorithm.
the model is a linear function of the input.
the model is trained to predict the output for new data.
the model is trained to find the best fit line for the data.
##
In the example [data](/Users/jinchengguo/Python_ML/linear_regression/data.csv) , theres a set of training data, which contains three columns: the first two are factors, and the third is the output. We need to find the best fit line for the data. 
## Cost function
the cost function is a function of the predicted output and the actual output. Suppose given a set of training data, the cost function is defined as:
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$$
 where $h_{\theta}(x)$ is the predicted output, $y$ is the actual output, $m$ is the number of training data.
- where $h_{\theta}(x)$ is the predicted output, which is defined as $$h_{\theta}(x) = \theta_0 + \theta_1x$$ for single factor, or $$h_{\theta}(x) = \theta_0 + \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n$$ for multiple factors.
- $\theta_0$ is the intercept of the line, and $\theta_1$ is the slope of the line.
- $x^{(i)}$ is the $i$-th training data, which contains the factor and the output.
- **we aim to minimize the cost function.**

## Gradient Descent
Gradient Descent is an optimization algorithm. It is used to find the minimum of a function. (In this case, the cost function)

- first calculate the gradient of the cost function, which is $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})x_j^{(i)}$$
- or in a vector form : $$\nabla J(\theta) = \frac{1}{m} X^T (h_{\theta}(X) - y)$$
- then update the parameters using the gradient, which is $$\theta = \theta - \alpha \nabla J(\theta)$$
- repeat the above steps until the cost function is minimized.
- $\alpha$ is the learning rate







