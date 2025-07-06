
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import torchvision.models as models

class_names=['Front_Breakage', 'Front_Crushed', 'Front_Normal', 'Rear_Breakage', 'Rear_Crushed', 'Rear_Normal']
trained_model=None

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')
        # Freeze all layers
        for param in self.model.parameters():
            param.requires_grad = False
        # Unfreezing 4th layer
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        in_features = self.model.fc.in_features

        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x

def predict(image_path):
    image= Image.open(image_path).convert('RGB')
    transform=transforms.Compose([
        transforms.Resize((224,224)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
      ])
    image_tensor=transform(image).unsqueeze(0) #tensor size will be(3,224,224), but our model is trained on batch size
                     #our model tensor size is (x,3,224,224), so we have to unsqueeze

    global trained_model
    if trained_model is None:
        trained_model = CarClassifierResNet()
        trained_model.load_state_dict(torch.load('model\saved_model.pth',map_location='cpu'))

        trained_model.eval()
    with torch.no_grad():
        output = trained_model(image_tensor)

        _,predicted_class=torch.max(output,1)
        return class_names[predicted_class.item()]




