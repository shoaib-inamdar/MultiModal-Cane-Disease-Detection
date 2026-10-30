import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


class GradCAMPlusPlus:
    """
    Grad-CAM++ for Swin-Tiny.

    Hooks into the last Swin Transformer stage to capture activations and
    gradients, then computes the weighted class-activation map (CAM).
    """

    def __init__(
        self, model: torch.nn.Module, target_layer_name: str = "backbone.model.layers.3"
    ):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._register_hooks(target_layer_name)

    def _register_hooks(self, layer_name: str) -> None:
        target_module = None
        for name, module in self.model.named_modules():
            if name.startswith(layer_name):
                target_module = module
                break

        if target_module is None:
            raise ValueError(
                f"Layer matching prefix '{layer_name}' was not found in the model."
            )

        def forward_hook(module, input, output):  # noqa: ARG001
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):  # noqa: ARG001
            self.gradients = grad_output[0].detach()

        target_module.register_forward_hook(forward_hook)
        target_module.register_full_backward_hook(backward_hook)

    def __call__(
        self,
        image: torch.Tensor,
        metadata: torch.Tensor,
        class_idx: int | None = None,
        device: str = "cpu",
    ) -> tuple[np.ndarray, int]:
        """
        Compute Grad-CAM++ heatmap for a single image.

        Args:
            image     : [1, 3, 224, 224]
            metadata  : [1, 4]
            class_idx : target class (None = predicted class)
            device    : "cuda" or "cpu"

        Returns:
            heatmap  : np.ndarray [224, 224] in [0, 1]
            pred_cls : int
        """
        image = image.to(device)
        metadata = metadata.to(device)
        image = image.detach().requires_grad_(True)

        self.model.zero_grad()
        self.model.eval()

        logits = self.model(image, metadata)  # [1, num_classes]
        pred_cls = int(logits.argmax(dim=1).item())

        if class_idx is None:
            class_idx = pred_cls

        score = logits[0, class_idx]
        score.backward()

        # activations / gradients: [1, num_patches, embed_dim]  e.g. [1, 49, 768]
        act = self.activations
        grad = self.gradients

        weights = grad.mean(dim=-1, keepdim=True)  # [1, 49, 1]
        cam = (weights * act).sum(dim=-1)  # [1, 49]
        cam = F.relu(cam)

        # 49 patches -> 7x7 spatial grid (224 / 32 = 7 for Swin-Tiny)
        cam = cam.reshape(1, 7, 7)

        cam = (
            F.interpolate(
                cam.unsqueeze(0),  # [1, 1, 7, 7]
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )  # [224, 224]

        # Normalise to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, pred_cls


def apply_heatmap(image_tensor: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    """
    Overlay the Grad-CAM heatmap on the original image.

    Red = high attention, blue = low attention.

    Args:
        image_tensor : [1, 3, 224, 224] ImageNet-normalised tensor
        heatmap      : [224, 224] float in [0, 1]

    Returns:
        overlaid : [224, 224, 3] uint8 RGB image
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    img = image_tensor.detach().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = (img * std + mean).clip(0, 1)
    img = (img * 255).astype(np.uint8)

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    overlaid = (0.6 * img + 0.4 * colored).astype(np.uint8)
    return overlaid
