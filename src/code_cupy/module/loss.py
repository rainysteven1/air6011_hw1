from src.code_cupy.module.base import Module
import cupy as cp


def onehot(x, num_classes=10):
    y = cp.zeros([len(x), num_classes])
    y[cp.arange(len(x)), x] = 1
    return y


class CrossEntropyLoss(Module):
    def forward(self, x_logit, x_target):
        self.x_logit = x_logit
        self.x_target = x_target

        # softmax
        x_logit_sub = cp.exp(x_logit - cp.max(x_logit, axis=1, keepdims=True))
        x_softmax = x_logit_sub / cp.sum(x_logit_sub, axis=1, keepdims=True)
        x_softmax = cp.clip(x_softmax, a_min=1e-15, a_max=None)  # to avoid zero values
        self.x_softmax = x_softmax  # save for backward

        # loss of each item
        loss_x = -cp.log(x_softmax)[cp.arange(len(x_target)), x_target]

        # average
        return loss_x.mean()

    def backward(self, dy):
        return dy * (self.x_softmax - onehot(self.x_target)) / len(self.x_logit)


class FocalLoss(Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = cp.asarray(alpha) if not isinstance(alpha, cp.ndarray) else alpha
        self.gamma = gamma
        self.eps = 1e-8
        self.probs = None
        self.targets = None

    def forward(self, logits, targets):
        """Icput Shape: logits (N,C), targets (N,)"""
        self.targets = targets

        probs = self.softmax(logits, axis=1)
        probs = cp.clip(probs, self.eps, 1.0 - self.eps)  # 数值稳定
        self.probs = probs

        batch_size = logits.shape[0]
        probs_true = probs[cp.arange(batch_size), targets]

        modulating_factor = cp.power(1 - probs_true, self.gamma)
        alpha_t = self.alpha[targets] if self.alpha.ndim > 0 else self.alpha

        loss_per_sample = -alpha_t * modulating_factor * cp.log(probs_true)

        return loss_per_sample.mean()

    def backward(self, d_output):
        probs = self.probs
        targets = self.targets
        batch_size, _ = probs.shape

        one_hot = cp.zeros_like(probs)
        one_hot[cp.arange(batch_size), targets] = 1.0

        p_t = probs[cp.arange(batch_size), targets]
        alpha_t = self.alpha[targets] if self.alpha.ndim > 0 else self.alpha

        grad_factor = alpha_t * (1 - p_t) ** self.gamma
        log_p_t = cp.log(p_t)
        d_ce = probs - one_hot

        modulating = self.gamma * p_t * log_p_t + (1 - p_t)
        grad = grad_factor.reshape(-1, 1) * modulating.reshape(-1, 1) * d_ce

        return grad * d_output

    @staticmethod
    def softmax(x, axis):
        x_exp = cp.exp(x - x.max(axis=axis, keepdims=True))
        return x_exp / x_exp.sum(axis=axis, keepdims=True)
