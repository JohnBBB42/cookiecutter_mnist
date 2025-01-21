import torch
import torchvision
import pytorch_lightning as pl
import onnx
import onnxruntime

class LitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.eval()

    def forward(self, x):
        return self.model(x)

model = LitModel()
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
model.to_onnx(
    file_path="resnet18.onnx",
    input_sample=dummy_input,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
