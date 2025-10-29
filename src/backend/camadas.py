import torch

state_dict = torch.load("finalv1.pth", map_location="cpu")

# Ver todas as chaves
print(state_dict.keys())

# Ver o tamanho da última camada fc2
print(state_dict['fc2.weight'].shape)  # shape = [num_classes, 512]
print(state_dict['fc2.bias'].shape)    # shape = [num_classes]
