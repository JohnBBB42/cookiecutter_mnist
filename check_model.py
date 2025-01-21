import onnx
model = onnx.load("resnet18.onnx")
onnx.checker.check_model(model)
print(onnx.helper.printable_graph(model.graph))
