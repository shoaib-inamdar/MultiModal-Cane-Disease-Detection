import torch


def compute_flops(model, input_shapes):
    from torchinfo import summary

    stats = summary(
        model,
        input_data=[torch.rand(*input_shapes[0]), torch.rand(*input_shapes[1])],
        verbose=0,
    )
    return {
        "total_params": stats.total_params,
        "trainable_params": stats.trainable_params,
        "total_mult_adds": stats.total_mult_adds,
    }
