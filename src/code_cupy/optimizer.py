import cupy as cp


class Optim:
    def __init__(self, params, lr: float = 0.01):
        # 标准化参数结构为 List[List[Dict]]
        self.params = [p if isinstance(p, list) else [p] for p in params]
        self.lr = lr

    def zero_grad(self):
        for param_group in self.params:
            for param in param_group:
                if "grad" in param and param["grad"] is not None:
                    param["grad"][...] = 0.0


class SGD(Optim):
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.9):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = []

        # 构建速度缓冲区
        for param_group in self.params:
            group_vel = []
            for param in param_group:
                group_vel.append(cp.zeros_like(param["val"]))
            self.velocities.append(group_vel)

    def step(self):
        for param_group, vel_group in zip(self.params, self.velocities):
            for param, vel in zip(param_group, vel_group):
                grad = param.get("grad", param.get("grads"))  # 兼容不同命名
                if grad is None:
                    continue

                # 动量更新
                vel[...] = self.momentum * vel + grad
                param["val"][...] -= self.lr * vel


class Adam(Optim):
    def __init__(self, params, lr=0.001):
        super().__init__(params, lr)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

        # 初始化动量参数
        self.m = []
        self.v = []
        for param_group in self.params:
            group_m = []
            group_v = []
            for param in param_group:
                group_m.append(cp.zeros_like(param["val"]))
                group_v.append(cp.zeros_like(param["val"]))
            self.m.append(group_m)
            self.v.append(group_v)

    def step(self):
        self.t += 1
        for param_group, m_group, v_group in zip(self.params, self.m, self.v):
            for param, m, v in zip(param_group, m_group, v_group):
                grad = param.get("grad", param.get("grads"))  # 兼容不同命名
                if grad is None:
                    continue

                # 更新一阶矩估计
                m[...] = self.beta1 * m + (1 - self.beta1) * grad
                # 更新二阶矩估计
                v[...] = self.beta2 * v + (1 - self.beta2) * cp.square(grad)

                # 偏差修正
                m_hat = m / cp.power(1 - self.beta1, self.t)
                v_hat = v / cp.power(1 - self.beta2, self.t)

                # 参数更新
                param["val"][...] -= self.lr * m_hat / (cp.sqrt(v_hat) + self.eps)
