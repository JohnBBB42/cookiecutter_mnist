import torch
import torchvision

model = torchvision.models.resnet18(weights=None)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
onnx_model = torch.onnx.dynamo_export(
    model,
    dummy_input,
    export_options=torch.onnx.ExportOptions(dynamic_shapes=True),
)
onnx_model.save("resnet18.onnx")
