from typing import Tuple, Union
from src.code_cupy.module.base import Module
import cupy as cp


class _WeightBiasLayer(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.in_channels = in_features
        self.out_channels = out_features

        self.weight = None
        self.bias = cp.zeros(self.out_channels) if bias else None

        self.d_weight = None
        self.d_bias = None

    @property
    def params(self):
        result = [dict(val=self.weight, grad=self.d_weight)]
        if self.bias is not None:
            result.append([dict(val=self.bias, grad=self.d_bias)])
        return result


class _BatchNorm(Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.num_features = num_features
        self.eps = 1e-5
        self.momentum = 0.1

        self.gamma = cp.ones(num_features)
        self.beta = cp.zeros(num_features)

        self.d_gamma = cp.ones(num_features)
        self.d_beta = cp.zeros(num_features)

        self.running_mean = cp.zeros(num_features)
        self.running_var = cp.ones(num_features)

        self.x_centered = None
        self.x_hat = None
        self.inv_std = None
        self.training = True
        self.input_shape = None

    @property
    def params(self):
        return [
            dict(val=self.gamma, grads=self.d_gamma),
            dict(val=self.beta, grads=self.d_beta),
        ]


class AdaptiveAvgPool2d(Module):
    def __init__(self, output_size: Union[int, Tuple[int]]):
        """
        Args:
            output_size: 目标输出尺寸 (int或tuple)
        """
        super().__init__()
        self.output_size = (
            (output_size, output_size) if isinstance(output_size, int) else output_size
        )
        self.input_shape = None
        self.slice_params = None

    def forward(self, x):
        self.input_shape = x.shape
        N, C, H_in, W_in = x.shape
        H_out, W_out = self.output_size

        output = cp.zeros((N, C, H_out, W_out), dtype=x.dtype)
        self.slice_params = cp.zeros((H_out, W_out, 4), dtype=int)

        for i in range(H_out):
            for j in range(W_out):
                start_h = i * H_in // H_out
                end_h = (i + 1) * H_in // H_out if i < H_out - 1 else H_in
                start_w = j * W_in // W_out
                end_w = (j + 1) * W_in // W_out if j < W_out - 1 else W_in

                output[:, :, i, j] = x[:, :, start_h:end_h, start_w:end_w].mean(
                    axis=(2, 3)
                )

                self.slice_params[i, j, 0] = start_h
                self.slice_params[i, j, 1] = end_h
                self.slice_params[i, j, 2] = start_w
                self.slice_params[i, j, 3] = end_w

        return output

    def backward(self, d_output):
        _, _, H_out, W_out = d_output.shape
        d_input = cp.zeros(self.input_shape, dtype=d_output.dtype)

        for i in range(H_out):
            for j in range(W_out):
                # 按索引提取坐标
                start_h = self.slice_params[i, j, 0]
                end_h = self.slice_params[i, j, 1]
                start_w = self.slice_params[i, j, 2]
                end_w = self.slice_params[i, j, 3]

                area_size = (end_h - start_h) * (end_w - start_w)
                d_input[:, :, start_h:end_h, start_w:end_w] += (
                    d_output[:, :, i : i + 1, j : j + 1] / area_size
                )

        return d_input


class BatchNorm1d(_BatchNorm):
    def forward(self, x):
        self.input_shape = x.shape

        x_reshaped = x.reshape(-1, self.num_features) if x.ndim == 3 else x

        if self.training:
            mean = x_reshaped.mean(axis=0)  # (C,)
            var = x_reshaped.var(axis=0)  # (C,)

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.inv_std = 1.0 / cp.sqrt(var + self.eps)
        self.x_centered = x_reshaped - mean
        self.x_hat = self.x_centered * self.inv_std

        out = self.gamma * self.x_hat + self.beta
        return out.reshape(self.input_shape)

    def backward(self, d_output):
        d_output_reshaped = (
            d_output.reshape(-1, self.num_features) if d_output.ndim == 3 else d_output
        )
        N = d_output_reshaped.shape[0]

        d_gamma = (d_output_reshaped * self.x_hat).sum(axis=0)
        d_beta = d_output_reshaped.sum(axis=0)

        dx_hat = d_output_reshaped * self.gamma
        d_var = (
            -0.5 * (dx_hat * self.x_centered).sum(axis=0) * cp.power(self.inv_std, 3)
        )
        d_mean = -(dx_hat * self.inv_std).sum(
            axis=0
        ) - 2 * d_var * self.x_centered.mean(axis=0)

        dx = dx_hat * self.inv_std + d_var * 2 * self.x_centered / N + d_mean / N

        # 保存梯度
        self.d_gamma = d_gamma
        self.d_beta = d_beta

        return dx.reshape(self.input_shape)


class BatchNorm2d(_BatchNorm):
    def forward(self, x):
        """Input shape: (N, C, H, W)"""
        if self._training:
            mean = x.mean(axis=(0, 2, 3))
            var = x.var(axis=(0, 2, 3))

            self.running_mean = (
                1 - self.momentum
            ) * self.running_mean + self.momentum * mean
            self.running_var = (
                1 - self.momentum
            ) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var

        self.inv_std = 1.0 / cp.sqrt(var + self.eps)
        self.x_centered = x - mean.reshape(1, -1, 1, 1)
        self.x_hat = self.x_centered * self.inv_std.reshape(1, -1, 1, 1)

        return self.gamma.reshape(1, -1, 1, 1) * self.x_hat + self.beta.reshape(
            1, -1, 1, 1
        )

    def backward(self, d_output):
        batch, _, out_h, out_w = d_output.shape

        d_gamma = (d_output * self.x_hat).sum(axis=(0, 2, 3))
        d_beta = d_output.sum(axis=(0, 2, 3))

        self.d_gamma = d_gamma
        self.d_beta = d_beta

        dx_hat = d_output * self.gamma.reshape(1, -1, 1, 1)
        d_var = (
            -0.5
            * (dx_hat * self.x_centered).sum(axis=(0, 2, 3))
            * (cp.power(self.inv_std, 3))
        )
        d_mean = -(dx_hat * self.inv_std.reshape(1, -1, 1, 1)).sum(
            axis=(0, 2, 3)
        ) - 2 * d_var * self.x_centered.mean(axis=(0, 2, 3))

        return (
            dx_hat * self.inv_std.reshape(1, -1, 1, 1)
            + d_var.reshape(1, -1, 1, 1) * 2 * self.x_centered / (batch * out_h * out_w)
            + d_mean.reshape(1, -1, 1, 1) / (batch * out_h * out_w)
        )


class Conv2d(_WeightBiasLayer):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__(in_channels, out_channels, bias)
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.x_padded_shape = None
        self.__init_kaiming()

    def __init_kaiming(self):
        fan_in = self.in_channels * self.kernel_size[0] * self.kernel_size[1]
        self.weight = cp.random.randn(
            self.out_channels, self.in_channels, *self.kernel_size
        ) * cp.sqrt(2.0 / fan_in)
        self.d_weight = cp.zeros_like(self.weight)

        if self.bias is not None:
            self.bias = cp.zeros(self.out_channels)
            self.d_bias = cp.zeros_like(self.bias)

    def forward(self, x):
        batch, _, in_h, in_w = x.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        # 计算输出尺寸
        out_h = (in_h + 2 * pad_h - k_h) // stride_h + 1
        out_w = (in_w + 2 * pad_w - k_w) // stride_w + 1

        # 填充输入
        x_padded = cp.pad(
            x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )
        self.x_padded_shape = x_padded.shape  # 保存填充后的形状

        # im2col展开
        self.input_col = self.__im2col(x_padded, k_h, k_w, stride_h, stride_w)

        # 矩阵乘法计算输出
        weight_col = self.weight.reshape(self.out_channels, -1)
        out = weight_col @ self.input_col.T
        out = out.reshape(batch, self.out_channels, out_h, out_w)

        return out + self.bias.reshape(1, -1, 1, 1) if self.bias is not None else out

    def backward(self, d_output):
        batch, out_c, out_h, out_w = d_output.shape
        k_h, k_w = self.kernel_size

        # 计算权重梯度（保持不变）
        d_output_flat = d_output.reshape(batch, out_c, -1)  # (batch, out_c, L)
        L = d_output_flat.shape[2]
        input_col_3d = self.input_col.reshape(batch, L, -1)  # (batch, L, in_c*k_h*k_w)
        self.d_weight = (d_output_flat @ input_col_3d).sum(axis=0)
        self.d_weight = self.d_weight.reshape(
            self.out_channels, self.in_channels, k_h, k_w
        )

        # 计算偏置梯度（保持不变）
        if self.bias is not None:
            self.d_bias = d_output.sum(axis=(0, 2, 3))

        # 关键修复：调整输入梯度计算逻辑
        weight_col = self.weight.reshape(self.out_channels, -1)  # (out_c, in_c*k_h*k_w)

        # 调整矩阵乘法顺序和维度重塑
        d_output_reshaped = d_output_flat.transpose(0, 2, 1)  # (batch, L, out_c)
        d_input_col = (
            d_output_reshaped.reshape(-1, out_c) @ weight_col
        )  # (batch*L, in_c*k_h*k_w)
        d_input_col = d_input_col.T  # (in_c*k_h*k_w, batch*L)

        # 按照im2col的展开顺序重塑
        d_input_col = d_input_col.reshape(
            self.in_channels * k_h * k_w, batch, out_h, out_w
        ).transpose(1, 0, 2, 3)  # (batch, in_c*k_h*k_w, out_h, out_w)

        return self.__col2im(
            d_input_col,
            self.x_padded_shape,  # 输入填充后的形状 (batch, in_c, padded_h, padded_w)
            k_h,
            k_w,
            self.stride[0],
            self.padding[0],
        )

    @staticmethod
    def __im2col(x, k_h, k_w, stride_h, stride_w):
        batch, in_c, in_h, in_w = x.shape
        out_h = (in_h - k_h) // stride_h + 1
        out_w = (in_w - k_w) // stride_w + 1

        # 使用跨距技巧高效展开
        cols = cp.lib.stride_tricks.as_strided(
            x,
            shape=(batch, in_c, out_h, out_w, k_h, k_w),
            strides=(
                x.strides[0],
                x.strides[1],
                stride_h * x.strides[2],
                stride_w * x.strides[3],
                x.strides[2],
                x.strides[3],
            ),
        )
        return cols.transpose(0, 2, 3, 1, 4, 5).reshape(batch * out_h * out_w, -1)

    @staticmethod
    def __col2im(grad_col, input_shape, k_h, k_w, stride, pad):
        """将梯度从列展开形式还原到输入特征图形状"""
        batch, in_c, in_h_padded, in_w_padded = input_shape
        grad_images = cp.zeros((batch, in_c, in_h_padded, in_w_padded))

        # 计算有效输出尺寸（与im2col一致）
        out_h = (in_h_padded - k_h) // stride + 1
        out_w = (in_w_padded - k_w) // stride + 1

        # 将梯度重塑为 (batch, in_c, k_h, k_w, out_h, out_w)
        grad_reshaped = grad_col.reshape(batch, in_c, k_h, k_w, out_h, out_w)

        # 累加梯度到对应位置
        for y in range(k_h):
            y_start = y * stride
            y_end = y_start + out_h * stride
            for x in range(k_w):
                x_start = x * stride
                x_end = x_start + out_w * stride
                grad_images[:, :, y_start:y_end:stride, x_start:x_end:stride] += (
                    grad_reshaped[:, :, y, x, :, :]
                )

        # 裁剪填充部分
        return grad_images[:, :, pad : in_h_padded - pad, pad : in_w_padded - pad]


class Dropout(Module):
    def __init__(self, p: float):
        super().__init__()
        assert 0 <= p < 1
        self.p = p
        self.mask = None

    def forward(self, x):
        if self._training and self.p > 0:
            self.mask = cp.random.rand(*x.shape) >= self.p
            return x * self.mask * 1.0 / (1.0 - self.p)
        else:
            return x

    def backward(self, d_output):
        if self._training and self.p > 0:
            return d_output * self.mask * 1.0 / (1.0 - self.p)
        else:
            return d_output


class Flatten(Module):
    def __init__(self, start_dim: int = 1):
        super().__init__()
        self.start_dim = start_dim
        self.input_shape = None

    def forward(self, x):
        """
        Input shape: (N, C, H, W)

        Output shape: (N, C*H*W)
        """
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_output):
        return d_output.reshape(self.input_shape)


class Linear(_WeightBiasLayer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__(in_features, out_features, bias)

        self.__init_kaiming()

    def __init_kaiming(self):
        scale = cp.sqrt(2.0 / self.in_channels)
        self.weight = cp.random.randn(self.out_channels, self.in_channels) * scale

    def forward(self, x):
        super().forward(x)
        y = cp.einsum("bi,oi->bo", x, self.weight)
        return y + self.bias if self.bias is not None else y

    def backward(self, d_output):
        self.d_weight = cp.einsum("bo,bi->oi", d_output, self.input)
        if self.bias is not None:
            self.d_bias = d_output.sum(axis=0)
        return cp.einsum("bo,oi->bi", d_output, self.weight)


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        )
        self.stride = stride if stride else self.kernel_size
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.input_shape = None
        self.max_indices = None

    def forward(self, x):
        """Input shape: (N, C, H, W)"""
        self.input_shape = x.shape
        N, C, H, W = x.shape
        k_h, k_w = self.kernel_size
        stride_h, stride_w = self.stride
        pad_h, pad_w = self.padding

        H_out = (H + 2 * pad_h - k_h) // stride_h + 1
        W_out = (W + 2 * pad_w - k_w) // stride_w + 1

        x_padded = cp.pad(
            x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant"
        )

        x_strided = cp.lib.stride_tricks.as_strided(
            x_padded,
            shape=(N, C, H_out, W_out, k_h, k_w),
            strides=(
                x_padded.strides[0],
                x_padded.strides[1],
                stride_h * x_padded.strides[2],
                stride_w * x_padded.strides[3],
                x_padded.strides[2],
                x_padded.strides[3],
            ),
        )

        window_flat = x_strided.reshape(N, C, H_out, W_out, -1)
        max_indices_flat = window_flat.argmax(axis=4)
        kh_idx = max_indices_flat // k_w
        kw_idx = max_indices_flat % k_w

        n_idx, c_idx, i_idx, j_idx = cp.indices((N, C, H_out, W_out))
        h_padded = i_idx * stride_h + kh_idx
        w_padded = j_idx * stride_w + kw_idx

        self.max_indices = (
            n_idx.ravel(),
            c_idx.ravel(),
            h_padded.ravel(),
            w_padded.ravel(),
        )

        return window_flat.max(axis=4).reshape(N, C, H_out, W_out)

    def backward(self, d_output):
        N, C, H, W = self.input_shape
        pad_h, pad_w = self.padding

        d_input = cp.zeros((N, C, H + 2 * pad_h, W + 2 * pad_w), dtype=d_output.dtype)
        cp.add.at(d_input, self.max_indices, d_output.ravel())

        if pad_h > 0 or pad_w > 0:
            d_input = d_input[:, :, pad_h:-pad_h, pad_w:-pad_w]

        return d_input
