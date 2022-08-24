import torch
import torchvision.models as models
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# model = models.inception_v3(pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
# torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
# model = torch.load("/home2/zhangya9/onnx_models/efficientnet-b0-pytorch/efficientnet-b0.pth")
# model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_resnet50', pretrained=True)
model.eval()
# from efficientnet_pytorch import EfficientNet
# model = EfficientNet.from_pretrained('efficientnet-b0')

print(model)

# Let's create a dummy input tensor  
dummy_input = torch.randn(1,3,224,224, requires_grad=True)  

# Export the model   
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "/home2/zhangya9/onnx_models/ResNet/v1.5/resnet50-pytorch.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=7,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input'],   # the model's input names 
        output_names = ['output'], # the model's output names 
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                            'output' : {0 : 'batch_size'}}) 
print(" ") 
print('Model has been converted to ONNX') 
