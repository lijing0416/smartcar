import torch
state_dict = torch.load("./checkpoint/actor_actor_1024_2114.pth", map_location="cpu")
for k,v in state_dict.items():
    print(k, v.shape)
    break
