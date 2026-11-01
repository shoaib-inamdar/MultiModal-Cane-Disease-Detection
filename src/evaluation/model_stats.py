import torch


def compute_flops(
    model: torch.nn.Module,
    input_shapes: list[tuple],
    device: str = "cpu",
) -> dict:
    """
    Estimate FLOPs (MACs) and parameter counts for a model.

    Args:
        model        : nn.Module to profile
        input_shapes : list of tuples, one per model input
                       e.g. [(1, 3, 224, 224), (1, 4)]
        device       : "cuda" or "cpu" -- must match the model device

    Returns dict:
        total_params     : int
        trainable_params : int
        total_mult_adds  : int  (MACs -- multiply-accumulate ops)
    """
    from torchinfo import summary

    dummy_inputs = [torch.rand(*shape).to(device) for shape in input_shapes]

    stats = summary(
        model,
        input_data=dummy_inputs,
        verbose=0,
    )
    return {
        "total_params": stats.total_params,
        "trainable_params": stats.trainable_params,
        "total_mult_adds": stats.total_mult_adds,
    }
