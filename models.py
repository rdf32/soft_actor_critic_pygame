import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        
    def forward(self, state):
        x = nn.functional.relu(self.conv1(state))
        x = nn.functional.relu(self.conv2(x))
        return x
    
class DActor(nn.Module):
    def __init__(self, action_bounds, params):
        super(DActor, self).__init__()
        self.entropy_lr = params['entropy_lr']
        self.log_std_min = params['log_std_min']
        self.log_std_max = params['log_std_max']
        self.env_min, self.env_max = action_bounds
        self.fc1 = nn.Linear(in_features=32*32, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.output_layer_mean = nn.Linear(256, len(self.env_max))
        self.output_layer_log_std = nn.Linear(256, len(self.env_max))
        
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(self.env_min,
                                    device=self.device, 
                                    dtype=torch.float32)

        self.env_max = torch.tensor(self.env_max,
                                    device=self.device, 
                                    dtype=torch.float32)
        
        self.nn_min = torch.tanh(torch.Tensor([float('-inf')])).to(self.device)
        self.nn_max = torch.tanh(torch.Tensor([float('inf')])).to(self.device)
        self.rescale_fn = lambda x: (x - self.nn_min) * (self.env_max - self.env_min) / (self.nn_max - self.nn_min) + self.env_min
        self.target_entropy = -np.prod(self.env_max.shape)
        self.logalpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam([self.logalpha], lr=self.entropy_lr)

    def forward(self, state):
        x = nn.functional.relu(self.fc1(state))
        x = nn.functional.relu(self.fc2(x))
        xmean = self.output_layer_mean(x)
        xlog_std = torch.clamp(self.output_layer_log_std(x), self.log_std_min, self.log_std_max)
        return xmean, xlog_std
    
    def full_pass(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        pi_s = Normal(mean, log_std.exp())
        pre_tanh_action = pi_s.rsample()
        tanh_action = torch.tanh(pre_tanh_action)
        action = self.rescale_fn(tanh_action)
        log_prob = pi_s.log_prob(pre_tanh_action) - torch.log((1 - tanh_action.pow(2)).clamp(0, 1) + epsilon)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, self.rescale_fn(torch.tanh(mean))
    
    def _update_exploration_ratio(self, greedy_action, action_taken):
        env_min, env_max = self.env_min.cpu().numpy(), self.env_max.cpu().numpy()
        self.exploration_ratio = np.mean(abs((greedy_action - action_taken)/(env_max - env_min)))

    def _get_actions(self, state):
        mean, log_std = self.forward(state)

        action = self.rescale_fn(torch.tanh(Normal(mean, log_std.exp()).sample()))
        greedy_action = self.rescale_fn(torch.tanh(mean))
        random_action = np.random.uniform(low=self.env_min.cpu().numpy(),
                                          high=self.env_max.cpu().numpy())

        action_shape = self.env_max.cpu().numpy().shape
        action = action.detach().cpu().numpy().reshape(action_shape)
        greedy_action = greedy_action.detach().cpu().numpy().reshape(action_shape)
        random_action = random_action.reshape(action_shape)

        return action, greedy_action, random_action

    def select_random_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, random_action)
        return random_action

    def select_greedy_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, greedy_action)
        return greedy_action

    def select_action(self, state):
        action, greedy_action, random_action = self._get_actions(state)
        self._update_exploration_ratio(greedy_action, action)
        return action

class DQNetwork(nn.Module):
    def __init__(self, action_dim, action_bounds):
        super(DQNetwork, self).__init__()
        self.env_min, self.env_max = action_bounds
        self.fc1 = nn.Linear(in_features=32*32+action_dim, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=1)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)
        
    def forward(self, state, action):
        x = nn.functional.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = nn.functional.relu(self.fc2(x))
        x = self.out(x)
        return x
    
class CActor(DActor):
    def __init__(self, action_bounds, params):
        super().__init__(action_bounds, params)
        self.fc1 = nn.Linear(in_features=32*14*14, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.cnn = CNN()

    def forward(self, state):
        x = self.cnn(state)
        x = x.view(-1, 32*14*14)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        xmean = self.output_layer_mean(x)
        xlog_std = torch.clamp(self.output_layer_log_std(x), self.log_std_min, self.log_std_max)
        return xmean, xlog_std
    
class CQNetwork(DQNetwork):
    def __init__(self, action_dim, action_bounds):
        super().__init__(action_dim, action_bounds)
        self.cnn = CNN()
    
    def forward(self, state, action):
        x = self.cnn(state)
        x = x.view(-1, 32*14*14)
        x = nn.functional.relu(self.fc1(torch.cat((state, action), dim=1)))
        x = nn.functional.relu(self.fc2(x))
        x = self.out(x)
        return x
    
class SAC():
    def __init__(self, pmodel_func, poptimizer_func, vmodel_func, voptimizer_func,
                  init_func, process_func, iteration_func, params):
        
        self.pmodel_func = pmodel_func
        self.poptimizer_func = poptimizer_func
        self.vmodel_func = vmodel_func
        self.voptimizer_func = voptimizer_func
        self.init_func = init_func
        self.process_func = process_func
        self.iteration_func = iteration_func
        self.params = params

    
