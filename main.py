import torch
import torchvision
import torchvision.models as models


def main():
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # I'm using ResNet18 as a pretrained model.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = model.to(device)
    model.eval()

    # Running a dummy forward pass on the GPU/CPU to confirm everything works
    x = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        y = model(x)

    print("ResNet18 forward pass OK.")
    print("Output shape:", tuple(y.shape))


if __name__ == "__main__":
    main()
