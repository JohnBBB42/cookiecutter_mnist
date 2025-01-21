import sys
import time
from statistics import mean, stdev

import onnxruntime as ort
import torch
import torchvision


def timing_decorator(func, function_repeat: int = 10, timing_repeat: int = 5):
    """Decorator that times the execution of a function."""

    def wrapper(*args, **kwargs):
        timing_results = []
        for _ in range(timing_repeat):
            start_time = time.time()
            for _ in range(function_repeat):
                result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            timing_results.append(elapsed_time)
        print(f"Avg +- Stddev: {mean(timing_results):0.3f} +- {stdev(timing_results):0.3f} seconds")
        return result

    return wrapper


# Define and prepare the model
model = torchvision.models.resnet18(weights=None)  # No pre-trained weights

def initialize_weights(module):
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.xavier_uniform_(module.weight)  # Initialize weights
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)  # Initialize biases to zero

model.apply(initialize_weights)  # Apply weight initialization
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with dynamic axes
if sys.platform == "win32":
    torch.onnx.export(
        model,
        dummy_input,
        "resnet18.onnx",
        input_names=["l_x_"],
        output_names=["output"],
        dynamic_axes={"l_x_": {0: "batch_size", 2: "height", 3: "width"}},  # Enable dynamic height and width
        opset_version=18,
    )
else:
    torch.onnx.dynamo_export(
        model,
        dummy_input,
        export_options=torch.onnx.ExportOptions(dynamic_shapes=True),  # Enables dynamic shapes
    ).save("resnet18.onnx")

ort_session = ort.InferenceSession("resnet18.onnx")
for input in ort_session.get_inputs():
    print(f"Input name: {input.name}, shape: {input.shape}, type: {input.type}")

@timing_decorator
def torch_predict(image) -> None:
    """Predict using PyTorch model."""
    model(image)


@timing_decorator
def onnx_predict(image) -> None:
    """Predict using ONNX model."""
    ort_session.run(None, {"l_x_": image.numpy()})


if __name__ == "__main__":
    for size in [224, 448, 896]:
        dummy_input = torch.randn(1, 3, size, size)
        print(f"Image size: {size}")
        torch_predict(dummy_input)
        onnx_predict(dummy_input)
