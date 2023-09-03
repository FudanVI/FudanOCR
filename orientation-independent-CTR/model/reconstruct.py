import torch
import torch.nn as nn

def easy_deconv(in_dims, out_dims, kernel, stride=1, groups=1, bias=True, dilation=1):
    if isinstance(kernel, int):
        kernel = (kernel, kernel)
    if isinstance(stride, int):
        stride = (stride, stride)

    c_in, h_in, w_in = in_dims
    c_out, h_out, w_out = out_dims

    padding = [0, 0]
    output_padding = [0, 0]

    lhs_0 = -h_out + (h_in - 1) * stride[0] + kernel[0]  # = 2p[0] - o[0]
    if lhs_0 % 2 == 0:
        padding[0] = lhs_0 // 2
    else:
        padding[0] = lhs_0 // 2 + 1
        output_padding[0] = 1

    lhs_1 = -w_out + (w_in - 1) * stride[1] + kernel[1]  # = 2p[1] - o[1]
    if lhs_1 % 2 == 0:
        padding[1] = lhs_1 // 2
    else:
        padding[1] = lhs_1 // 2 + 1
        output_padding[1] = 1

    return torch.nn.ConvTranspose2d(
        in_channels=c_in,
        out_channels=c_out,
        kernel_size=kernel,
        stride=stride,
        padding=tuple(padding),
        output_padding=tuple(output_padding),
        groups=groups,
        bias=bias,
        dilation=dilation
    )


class GeneratorIMG_V(nn.Module):
    def __init__(self):
        super(GeneratorIMG_V, self).__init__()

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 16, 2), out_dims=(512, 32, 4), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 32, 4), out_dims=(256, 64, 8), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 64, 8), out_dims=(128, 128, 16), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 128, 16), out_dims=(64, 256, 32), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_5', in_dims=(64, 256, 32), out_dims=(3, 256, 32), kernel=5, stride=2, actf=nn.ReLU())

    def forward(self, x):
        x = x.permute(0, 1, 3, 2).flip(dims=[3])
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            x = deconv_layer(x)
        return x

class GeneratorIMG_H(nn.Module):
    def __init__(self):
        super(GeneratorIMG_H, self).__init__()
        self.deconv_layers = nn.ModuleList()
        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 2, 16), out_dims=(512, 4, 32), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 4, 32), out_dims=(256, 8, 64), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 8, 64), out_dims=(128, 16, 128), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 16, 128), out_dims=(64, 32, 256), kernel=5, stride=2, actf=nn.Tanh())
        add_deconv('g_deconv_5', in_dims=(64, 32, 256), out_dims=(3, 32, 256), kernel=5, stride=2, actf=nn.Tanh())

    def forward(self, x):
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            x = deconv_layer(x)
        return x

class GeneratorIMG_H_1(nn.Module):
    def __init__(self):
        super(GeneratorIMG_H_1, self).__init__()

        self.deconv_layers = nn.ModuleList()

        def add_deconv(name, in_dims, out_dims, kernel, stride, actf):
            self.deconv_layers.add_module(
                name,
                nn.Sequential(
                    easy_deconv(
                        in_dims=in_dims,
                        out_dims=out_dims,
                        kernel=kernel,
                        stride=stride,
                    ),
                    actf
                )
            )

        add_deconv('g_deconv_1', in_dims=(1024, 2, 2), out_dims=(512, 4, 4), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_2', in_dims=(512, 4, 4), out_dims=(256, 8, 8), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_3', in_dims=(256, 8, 8), out_dims=(128, 16, 16), kernel=5, stride=2, actf=nn.ReLU())
        add_deconv('g_deconv_4', in_dims=(128, 16, 16), out_dims=(64, 32, 32), kernel=5, stride=2, actf=nn.Tanh())
        add_deconv('g_deconv_5', in_dims=(64, 32, 32), out_dims=(3, 32, 32), kernel=5, stride=2, actf=nn.Tanh())

    def forward(self, x):
        for i, deconv_layer in enumerate(self.deconv_layers, 1):
            x = deconv_layer(x)
        return x

if __name__ == '__main__':
    g = GeneratorIMG_V()
    img = torch.randn((32, 1024, 2, 16))
    res = g(img)
    print(res.size())
