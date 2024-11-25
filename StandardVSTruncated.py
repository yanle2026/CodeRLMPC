import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm, norm

# 参数定义
mu, sigma = 0, 1
lower, upper = -1, 1

# 普通正态分布
x = np.linspace(-3, 3, 500)
normal_pdf = norm.pdf(x, loc=mu, scale=sigma)

# 截断正态分布
a, b = (lower - mu) / sigma, (upper - mu) / sigma
trunc_pdf = truncnorm.pdf(x, a, b, loc=mu, scale=sigma)

# 绘制对比图
plt.plot(x, normal_pdf, label="Normal Distribution", linestyle="--", color="blue")
plt.plot(x, trunc_pdf, label="Truncated Normal Distribution", color="red")
plt.axvline(lower, color="green", linestyle=":")
plt.axvline(upper, color="green", linestyle=":")
plt.title("Normal vs Truncated Normal Distribution")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.show()
