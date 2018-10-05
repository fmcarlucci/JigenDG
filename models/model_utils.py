from torch.autograd import Function


class GradientKillerLayer(Function):
    @staticmethod
    def forward(ctx, x, **kwargs):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return None, None


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val

        return output, None