import torch
from models import CustomLightCNN, CustomDeepCNN, CustomGAPCNN, CustomMultiScaleCNN, CustomResidualCNN, CustomDepthwiseSepCNN

def get_params_and_channels(model, name):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"=== {name} ===")
    print(f"Total Trainable Parameters (Weights/Biases): {trainable_params:,}")
    
    # Analyze the architecture layers to list out "neurons" (channel features/hidden units)
    # Let's inspect the linear (FC) layers specifically since they represent traditional "neurons".
    fc_layers = []
    for layer_name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            fc_layers.append(f"{layer_name} (Input features: {module.in_features}, Output classes/neurons: {module.out_features})")
    
    print("Fully Connected Neurons:")
    for fc in fc_layers:
        print(f"  - {fc}")
    print()

if __name__ == "__main__":
    get_params_and_channels(CustomLightCNN(), "CustomLightCNN (Model 1)")
    get_params_and_channels(CustomDeepCNN(), "CustomDeepCNN (Model 2)")
    get_params_and_channels(CustomGAPCNN(), "CustomGAPCNN (Model 3)")
    get_params_and_channels(CustomMultiScaleCNN(), "CustomMultiScaleCNN (Model 4)")
    get_params_and_channels(CustomResidualCNN(), "CustomResidualCNN (Model 5)")
    get_params_and_channels(CustomDepthwiseSepCNN(), "CustomDepthwiseSepCNN (Model 6)")
