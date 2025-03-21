from models.fc_model import FullyConnectedClassifier
from models.cnn_model import ConvolutionalClassifier

def get_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))

    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0

    try:
        for p in parameters:
            param_norm = p.data.norm(norm_type)
            total_norm += param_norm ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

def get_model(config):
    if config.model == "fc":
        model = FullyConnectedClassifier(28 ** 2, 10)
    elif config.model == "cnn":
        model = ConvolutionalClassifier(10)
    else:
        raise NotImplementedError("You need to specify model name.")

    return model
