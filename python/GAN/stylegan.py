from PIL import Image
import matplotlib.pylab as plt
path2content= "./mainimg.jpg"
path2style= "./styleimg.jpg"
content_img = Image.open(path2content)
style_img = Image.open(path2style)

import torchvision.transforms as transforms

h, w = 256, 384 
mean_rgb = (0.485, 0.456, 0.406)
std_rgb = (0.229, 0.224, 0.225)

transformer = transforms.Compose([
                    transforms.Resize((h,w)),  
                    transforms.ToTensor(),
                    transforms.Normalize(mean_rgb, std_rgb)]) 

content_tensor = transformer(content_img)
print(content_tensor.shape, content_tensor.requires_grad)

style_tensor = transformer(style_img)
print(style_tensor.shape, style_tensor.requires_grad)

input_tensor = content_tensor.clone().requires_grad_(True)
print(input_tensor.shape, input_tensor.requires_grad)

import torch

def imgtensor2pil(img_tensor):
    img_tensor_c = img_tensor.clone().detach()
    img_tensor_c*=torch.tensor(std_rgb).view(3, 1,1)
    img_tensor_c+=torch.tensor(mean_rgb).view(3, 1,1)
    img_tensor_c = img_tensor_c.clamp(0,1)
    img_pil=to_pil_image(img_tensor_c)
    return img_pil


from torchvision.transforms.functional import to_pil_image
import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_vgg = models.vgg19(pretrained=True).features.to(device).eval()
for param in model_vgg.parameters():
    param.requires_grad_(False)   
print(model_vgg)

def get_features(x, model, layers):
    features = {}
    for name, layer in enumerate(model.children()):
        x = layer(x)
        if str(name) in layers:
            features[layers[str(name)]] = x
    return features

def gram_matrix(x):
    n, c, h, w = x.size()
    x = x.view(n*c, h * w)
    gram = torch.mm(x, x.t())
    return gram

import torch.nn.functional as F

def get_content_loss(pred_features, target_features, layer):
    target= target_features[layer]
    pred = pred_features [layer]
    loss = F.mse_loss(pred, target)
    return loss

def get_style_loss(pred_features, target_features, style_layers_dict):  
    loss = 0
    for layer in style_layers_dict:
        pred_fea = pred_features[layer]
        pred_gram = gram_matrix(pred_fea)
        n, c, h, w = pred_fea.shape
        target_gram = gram_matrix (target_features[layer])
        layer_loss = style_layers_dict[layer] *  F.mse_loss(pred_gram, target_gram)
        loss += layer_loss/ (n* c * h * w)
    return loss

feature_layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '21': 'conv4_2',  
                  '28': 'conv5_1'}

con_tensor = content_tensor.unsqueeze(0).to(device)
sty_tensor = style_tensor.unsqueeze(0).to(device)

content_features = get_features(con_tensor, model_vgg, feature_layers)
style_features = get_features(sty_tensor, model_vgg, feature_layers)

for key in content_features.keys():
    print(content_features[key].shape)

from torch import optim

input_tensor = con_tensor.clone().requires_grad_(True)
optimizer = optim.Adam([input_tensor], lr=0.01)

num_epochs = 500
content_weight = 1e1
style_weight = 1e4
content_layer = "conv5_1"
style_layers_dict = { 'conv1_1': 0.75,
                      'conv2_1': 0.5,
                      'conv3_1': 0.25,
                      'conv4_1': 0.25,
                      'conv5_1': 0.25}

for epoch in range(num_epochs+1):
    optimizer.zero_grad()
    input_features = get_features(input_tensor, model_vgg, feature_layers)
    content_loss = get_content_loss (input_features, content_features, content_layer)
    style_loss = get_style_loss(input_features, style_features, style_layers_dict)
    neural_loss = content_weight * content_loss + style_weight * style_loss
    neural_loss.backward(retain_graph=True)
    optimizer.step()
    
    if epoch % 100 == 0:
        print('epoch {}, content loss: {:.2}, style loss {:.2}'.format(
          epoch,content_loss, style_loss))

print(imgtensor2pil(input_tensor[0].cpu()))
plt.imshow(imgtensor2pil(input_tensor[0].cpu()))
imgtensor2pil(input_tensor[0].cpu()).save("./result/result.jpg")














