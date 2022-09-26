import torch
import torch.nn as nn
import torch.nn.functional as F


# define the actor network
class Actor(nn.Module):
    def __init__(self, args, agent_id):
        super(Actor, self).__init__()
        self.max_action = args.high_action
        self.conv1 = nn.Sequential( # input shape(3,84,84)
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2), # output shape(16,84,84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape(16,42,42)
        )
        self.conv2 = nn.Sequential( # input shape(16,42,42)
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2), # output shape(32,42,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape(32,21,21)
        )
        
        self.conv3 = nn.Sequential( # input shape(32,21,21)
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2), # output shape(64,21,21)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3) # output shape(64,7,7)
        )

        self.out = nn.Linear(64*7*7 ,args.action_shape[agent_id])
        # self.fc1 = nn.Linear(args.obs_shape[agent_id], 64)
        # self.fc2 = nn.Linear(64, 64)
        # self.fc3 = nn.Linear(64, 64)
        # self.action_out = nn.Linear(64, args.action_shape[agent_id])

    def forward(self, x: torch.Tensor):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # actions = self.max_action * torch.tanh(self.action_out(x))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        actions = self.max_action * torch.tanh(self.out(x))

        return actions


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.max_action = args.high_action

        self.conv1 = nn.Sequential( # input shape(3*20=60,84,84)
            nn.Conv2d(in_channels=60,out_channels=64,kernel_size=5,stride=1,padding=2), # output shape(64,84,84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape(16,42,42)
        )
        self.conv2 = nn.Sequential( # input shape(16,42,42)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2), # output shape(32,42,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape(32,21,21)
        )
        
        self.conv3 = nn.Sequential( # input shape(32,21,21)
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=5,stride=1,padding=2), # output shape(64,21,21)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3) # output shape(64,7,7)
        )

        self.out = nn.Linear(64*7*7+20 ,1)

    def forward(self, state, action):
        cated_state = torch.cat(state, dim=1)
        x = F.relu(self.conv1(cated_state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        action = torch.cat(action, dim=1)
        x = torch.cat([x, action], dim=-1)
        q_value = self.out(x)
        return q_value

class SActor(nn.Module):
    def __init__(self, args, agent_id):
        super(SActor, self).__init__()
        self.max_action = args.high_action
        self.conv1 = nn.Sequential( # input shape(3*20=60,84,84)
            nn.Conv2d(in_channels=60,out_channels=128,kernel_size=5,stride=1,padding=2), # output shape(16,84,84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(128,42,42)
        )
        self.conv2 = nn.Sequential( # input shape(128,42,42)
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=5,stride=1,padding=2), # output shape(32,42,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2) # output shape(256,21,21)
        )
        
        self.conv3 = nn.Sequential(  # input shape(256,21,21)
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=5,stride=1,padding=2), # output shape(64,21,21)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3) # output shape(512,7,7)
        )

        self.out = nn.Linear(512*7*7 ,20)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        actions = self.max_action * torch.tanh(self.out(x))

        return actions


class SCritic(nn.Module):
    def __init__(self, args):
        super(SCritic, self).__init__()
        self.max_action = args.high_action

        self.conv1 = nn.Sequential(  # input shape(3*20=60,84,84)
            nn.Conv2d(in_channels=60, out_channels=128, kernel_size=5,
                      stride=1, padding=2),  # output shape(64,84,84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(128,42,42)
        )
        self.conv2 = nn.Sequential(  # input shape(128,42,42)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5,
                      stride=1, padding=2),  # output shape(32,42,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(256,21,21)
        )

        self.conv3 = nn.Sequential(  # input shape(256,21,21)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5,
                      stride=1, padding=2),  # output shape(512,21,21)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)  # output shape(512,7,7)
        )

        self.out = nn.Linear(512*7*7+20, 1)

    def forward(self, state, action):
        # cated_state = torch.cat(state, dim=1)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # action = torch.cat(action, dim=1)
        x = torch.cat([x, action], dim=-1)
        q_value = self.out(x)
        return q_value


class TCritic(nn.Module):
    def __init__(self):
        super(TCritic, self).__init__()
        
        self.conv1 = nn.Sequential(  # input shape(3*20=60,84,84)
            nn.Conv2d(in_channels=60, out_channels=128, kernel_size=5,
                      stride=1, padding=2),  # output shape(64,84,84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(128,42,42)
        )
        self.conv2 = nn.Sequential(  # input shape(128,42,42)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5,
                      stride=1, padding=2),  # output shape(32,42,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(256,21,21)
        )
        self.conv3 = nn.Sequential(  # input shape(256,21,21)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5,
                      stride=1, padding=2),  # output shape(512,21,21)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)  # output shape(512,7,7)
        )
        self.out = nn.Linear(512*7*7+20, 1)


        self.conv1_c = nn.Sequential(  # input shape(3*20=60,84,84)
            nn.Conv2d(in_channels=60, out_channels=128, kernel_size=5,
                      stride=1, padding=2),  # output shape(64,84,84)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(128,42,42)
        )
        self.conv2_c = nn.Sequential(  # input shape(128,42,42)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5,
                      stride=1, padding=2),  # output shape(32,42,42)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)  # output shape(256,21,21)
        )

        self.conv3_c = nn.Sequential(  # input shape(256,21,21)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5,
                      stride=1, padding=2),  # output shape(512,21,21)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3)  # output shape(512,7,7)
        )
        self.out_c = nn.Linear(512*7*7+20, 1)

    def forward(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.cat([x, action], dim=-1)
        q1_value = self.out(x)

        y= F.relu(self.conv1(state))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = y.view(y.size(0), -1)
        y = torch.cat([y, action], dim=-1)
        q2_value = self.out(y)
        return q1_value, q2_value
    
    def Q1(self, state, action):
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # action = torch.cat(action, dim=1)
        x = torch.cat([x, action], dim=-1)
        q_value = self.out(x)
        return q_value
