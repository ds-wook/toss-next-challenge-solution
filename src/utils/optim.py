from torch import nn


def create_parameter_groups(
    model: nn.Module,
    base_lr: float,
    fm_lr_multiplier: float = 1.0,
    deep_lr_multiplier: float = 1.0,
    cin_lr_multiplier: float = 1.0,
    encoder_lr_multiplier: float = 1.0,
):
    """
    Create parameter groups with different learning rates for different model components.

    Args:
        model: The model instance
        base_lr: Base learning rate
        fm_lr_multiplier: Learning rate multiplier for FM components (default: 1.0)
        deep_lr_multiplier: Learning rate multiplier for Deep/MLP components (default: 1.0)
        cin_lr_multiplier: Learning rate multiplier for CIN components (default: 1.0)
        encoder_lr_multiplier: Learning rate multiplier for Encoder components (default: 1.0)
    Returns:
        List of parameter groups for optimizer
    """
    # Define parameter name patterns for each component
    fm_param_patterns = [
        "categorical_linear",
        "numerical_linear",
        "categorical_embeddings",
        "numerical_embeddings",
        "bias",
    ]
    deep_param_patterns = ["mlp"]
    cin_param_patterns = ["cin.conv_layers", "cin_output"]
    encoder_param_patterns = ["encoder"]

    # Collect parameters for each component
    fm_params = []
    deep_params = []
    cin_params = []
    encoder_params = []
    other_params = []

    for name, param in model.named_parameters():
        if any(pattern in name for pattern in fm_param_patterns):
            fm_params.append(param)
        elif any(pattern in name for pattern in deep_param_patterns):
            deep_params.append(param)
        elif any(pattern in name for pattern in cin_param_patterns):
            cin_params.append(param)
        elif any(pattern in name for pattern in encoder_param_patterns):
            encoder_params.append(param)
        else:
            other_params.append(param)

    # Create parameter groups
    param_groups = []

    if fm_params:
        param_groups.append(
            {
                "params": fm_params,
                "lr": base_lr * fm_lr_multiplier,
                "name": "FM_components",
            }
        )

    if deep_params:
        param_groups.append(
            {
                "params": deep_params,
                "lr": base_lr * deep_lr_multiplier,
                "name": "Deep_components",
            }
        )

    if cin_params:
        param_groups.append(
            {
                "params": cin_params,
                "lr": base_lr * cin_lr_multiplier,
                "name": "CIN_components",
            }
        )

    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": base_lr * encoder_lr_multiplier,
                "name": "Encoder_components",
            }
        )

    if other_params:
        param_groups.append(
            {"params": other_params, "lr": base_lr, "name": "Other_components"}
        )

    return param_groups
