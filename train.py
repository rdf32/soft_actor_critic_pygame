import os
import sys
import time
import copy
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
from itertools import count
from collections import deque
from dataclasses import dataclass
from functools import partial
from models import SAC, DActor, CActor, DQNetwork, CQNetwork 
from simulation import *

LEAVE_PRINT_EVERY_N_SECS = 300
ERASE_LINE = '\x1b[2K'
EPS = 1e-6
BEEP = lambda: os.system("printf '\a'")
RESULTS_DIR = os.path.join('..', 'results')
SEEDS = (12, 34, 56, 78, 90)

buffer = 5
screen_width, screen_height = 256, 256
dimensions = Screen(screen_height, screen_width)
background = cv2.resize(cv2.cvtColor(cv2.imread('images/background.jpg'), cv2.COLOR_BGR2RGB), (screen_width, screen_height))
crosshair_image = cv2.resize(cv2.cvtColor(cv2.imread('images/crosshair.png'), cv2.COLOR_BGR2RGB), (int(.3*screen_width), int(.3*screen_width)))
crosshair = np.ma.array(crosshair_image, mask=crosshair_image == 0)

config = {

    "large": {
        "xs": [116, 158, 7, 210, 59],
        "ys": int(.15*screen_height),
        "number": 5,
        "scale": int(.15*screen_width),
        "points": 10
        },
    "medium": {
        "xs": [19, 142, 82, 195, 225],
        "ys": int(.25*screen_height),
        "number": 5,
        "scale": int(.07*screen_width),
        "points": 25
        },
    "small": {
        "xs": [106, 54, 193, 19, 174],
        "ys": int(.35*screen_height),
        "number": 5,
        "scale": int(.05*screen_width),
        "points": 30
        }
}

@dataclass
class Screen:
    height: int
    width: int

@dataclass
class ReplayBuffer:
    buffer: deque
    batch_size: int

@dataclass
class Experience:
    state: np.ndarray
    action: np.ndarray
    reward: int
    next_state: np.ndarray
    done: bool

def dnn_init(targets):
    return draw_targets(background, targets)[1].astype(np.float32), False

def dnn_iterate(action, targets):
    reward, done = get_reward(targets, scale(action, dimensions))
    _, next_statem = draw_targets(background, targets)
    return np.array([reward]), next_statem, np.array([done])

def dnn_process(state):
    return (cv2.resize(state, (128, 128)).reshape(32, 32, 4, 4).sum(axis=(2,3)).ravel() / 16.) + np.random.normal(0, .1, 32*32)

def cnn_init(targets):
    return draw_targets(background, targets)[0], False

def cnn_iterate(action, targets):
    reward, done = get_reward(targets, scale(action, dimensions))
    next_state, _ = draw_targets(background, targets)
    draw_crosshair(next_state, crosshair, scale(action, dimensions))
    return np.array([reward]), next_state, np.array([done])

def cnn_process(state):
    return cv2.resize(state, (128, 128)).T / 255.

def unpack(batch, attribute):
    return torch.cat([torch.from_numpy(getattr(sample, attribute).astype(np.float32)).unsqueeze(0) for sample in batch])

def load(minibatch):
    return (unpack(minibatch, 'state'), unpack(minibatch, 'action'),
             unpack(minibatch, 'reward'), unpack(minibatch, 'next_state'), unpack(minibatch, 'done'))

def soft_update(target, online, tau):
    for target_param, param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

def policy_target(ovmodela, ovmodelb, states, actions):
    return torch.min(ovmodela(states, actions), 
                     ovmodelb(states, actions))

def policy_loss(q_sa, logpi_s, alpha):
    return (alpha * logpi_s - q_sa).mean()

def policy_update(pmodel, poptimizer, ploss, maxgrad):
    poptimizer.zero_grad()
    ploss.backward()
    torch.nn.utils.clip_grad_norm_(pmodel.parameters(), maxgrad)        
    poptimizer.step()

def alpha_target(pmodel, logpi):
    return (logpi + pmodel.target_entropy).detach()

def alpha_loss(pmodel, alphat):
    return -(pmodel.logalpha * alphat).mean()

def alpha_update(pmodel, alphal):
    pmodel.alpha_optimizer.zero_grad()
    alphal.backward()
    pmodel.alpha_optimizer.step()
    return pmodel.logalpha.exp()

def Q_target(pmodel, tvmodela, tvmodelb, rewards, nstates, dones, alpha, gamma):
    ap, logpi_sp, _ = pmodel.full_pass(nstates)
    q_spap_a = tvmodela(nstates, ap)
    q_spap_b = tvmodelb(nstates, ap)
    q_spap = torch.min(q_spap_a, q_spap_b) - alpha * logpi_sp
    return (rewards + gamma * q_spap * dones).detach()

def Q_loss(omodel, states, actions, targetq):
    return (omodel(states, actions) - targetq).pow(2).mul(0.5).mean()

def Q_update(q_loss, omodel, optimizer, maxgrad):
    optimizer.zero_grad()
    q_loss.backward()
    torch.nn.utils.clip_grad_norm_(omodel.parameters(), maxgrad)
    optimizer.step()

class SACT():
    def __init__(self, sac, eparams, seed):
        self.sac = sac
        self.tau = sac.params['tau']
        self.rbuffer = ReplayBuffer(deque(maxlen=sac.params['buffer_size']), sac.params['batch_size'])
        self.nwarmup = eparams['nwarmup']
        self.tupdate = eparams['tupdate']
        self.gamma = eparams['gamma']
        self.seed = seed
        self.params = eparams

    def optimize_model(self, state_batch, action_batch, reward_batch, next_state_batch, done_batch):

        # policy loss
        current_actions, logpi_s, _ = self.policy_model.full_pass(state_batch)
        alpha = alpha_update(self.policy_model, alpha_loss(self.policy_model, alpha_target(self.policy_model, logpi_s)))
        current_q_sa = policy_target(self.online_value_model_a, self.online_value_model_b, state_batch, current_actions)
        policy_update(self.policy_model, self.policy_optimizer, policy_loss(current_q_sa, logpi_s, alpha), self.sac.policy_maxgrad)
        
        # Q loss
        targetq = Q_target(self.policy_model, self.target_value_model_a, self.target_value_model_b,
                          reward_batch, next_state_batch, done_batch, alpha, self.gamma)
        Q_update(Q_loss(self.online_value_model_a, next_state_batch, action_batch, targetq),
                  self.online_value_model_a, self.value_optimizer_a, self.value_maxgrad)
        Q_update(Q_loss(self.online_value_model_b, next_state_batch, action_batch, targetq),
                  self.online_value_model_b, self.value_optimizer_b, self.value_maxgrad)

    def update_vnetworks(self, tau=None):
        tau = self.tau if tau is None else tau
        soft_update(self.target_value_model_a, self.online_value_model_a, tau)
        soft_update(self.target_value_model_b, self.online_value_model_b, tau)

    def episode_init(self, evaluation=False):
        if evaluation:
            return (0, copy.deepcopy(eval_targets))
        self.episode_reward.append(0.0)
        self.episode_timestep.append(0.0)
        self.episode_exploration.append(0.0)
        return (0, time.time(), validate_targets(background, generate_groups(dimensions, config), dimensions, config))
        
    def interaction_step(self, state, targets, train=True):
        if train:
            if len(self.rbuffer.buffer) < (self.rbuffer.batch_size * self.nwarmup):
                action = self.policy_model.select_random_action(torch.FloatTensor(self.sac.process_func(state)))
            else:
                action = self.policy_model.select_action(torch.FloatTensor(self.sac.process_func(state)))
        else:
            action = self.policy_model.select_greedy_action(torch.FloatTensor(self.sac.process_func(state)))
        return Experience(state, action, *self.sac.iteration_func(action, targets))
    
    def update_results(self, experience, train=True):
        if train:
            self.rbuffer.buffer.append(experience)
            self.episode_reward[-1] += experience.reward
            self.episode_timestep[-1] += 1
            self.episode_exploration[-1] += self.policy_model.exploration_ratio
        return experience.next_state, experience.done

    def train(self, actor=None, avalue=None, bvalue=None):
        training_start, last_debug_time = time.time(), float('-inf')
        torch.manual_seed(self.seed) ; np.random.seed(self.seed) ; random.seed(self.seed)

        self.episode_timestep = []
        self.episode_reward = []
        self.episode_seconds = []
        self.evaluation_scores = []        
        self.episode_exploration = []

        self.policy_model = self.sac.pmodel_func() if actor is None else actor
        self.online_value_model_a = self.sac.vmodel_func() if avalue is None else avalue
        self.online_value_model_b = self.sac.vmodel_func() if bvalue is None else bvalue

        self.target_value_model_a = self.sac.vmodel_func()
        self.target_value_model_b = self.sac.vmodel_func()
        self.update_vnetworks(tau=1.0)

        self.policy_optimizer = self.sac.poptimizer_func(self.policy_model, self.sac.params['policy_lr'])
        self.value_optimizer_a = self.sac.voptimizer_func(self.online_value_model_a, self.sac.params['value_lr'])
        self.value_optimizer_b = self.sac.voptimizer_func(self.online_value_model_b, self.sac.params['value_lr'])

        training_time = 0
        episode_results = []
        result = np.full((self.params['max_episodes'], 5), np.nan)
        best_score = -np.inf

        for episode in range(1, self.params['max_episodes'] + 1):
            step, episode_start, targets = self.episode_init()
            state, done = self.sac.init_func(targets)
            for _ in count():
                state, done = self.update_results(self.interaction_step(state, targets))
                if len(self.rbuffer.buffer) > (self.rbuffer.batch_size * self.nwarmup):
                    self.optimize_model(*load(random.sample(self.rbuffer.buffer, self.rbuffer.batch_size)))
                if np.sum(self.episode_timestep) % self.params['tupdate'] == 0:
                    self.update_vnetworks()
                if done or (step > self.params['max_steps']): break
                step += 1

            episode_elapsed = time.time() - episode_start
            self.episode_seconds.append(episode_elapsed)
            training_time += episode_elapsed

            evaluation_score, _ = self.evaluate(max_steps=self.params['max_steps'])
            if evaluation_score > best_score:
                print(f'saving model.. Score: {evaluation_score}, {episode}')
                torch.save(self.policy_model.state_dict(), f"{self.params['out_dir']}/actor_{experiment}.pt")
                torch.save(self.online_value_model_a.state_dict(), f"{self.params['out_dir']}/critica_{experiment}.pt")
                torch.save(self.online_value_model_b.state_dict(), f"{self.params['out_dir']}/criticb_{experiment}.pt")
                best_score = evaluation_score
            if episode % self.params['tupdate'] == 0:
                print(f'saving model epoch.. Score: {evaluation_score}, {episode}')
                torch.save(self.policy_model.state_dict(), f"{self.params['out_dir']}/actor_{experiment}_{episode}.pt")
                torch.save(self.online_value_model_a.state_dict(), f"{self.params['out_dir']}/critica_{experiment}_{episode}.pt")
                torch.save(self.online_value_model_b.state_dict(), f"{self.params['out_dir']}/criticb_{experiment}_{episode}.pt")
            total_step = int(np.sum(self.episode_timestep))
            self.evaluation_scores.append(evaluation_score)


            mean_10_reward = np.mean(self.episode_reward[-10:])
            std_10_reward = np.std(self.episode_reward[-10:])
            mean_100_reward = np.mean(self.episode_reward[-100:])
            std_100_reward = np.std(self.episode_reward[-100:])
            mean_100_eval_score = np.mean(np.ma.masked_invalid(self.evaluation_scores[-100:]))
            std_100_eval_score = np.std(np.ma.masked_invalid(self.evaluation_scores[-100:]))
            lst_100_exp_rat = np.array(self.episode_exploration[-100:])/np.array(self.episode_timestep[-100:])
            mean_100_exp_rat = np.mean(lst_100_exp_rat)
            std_100_exp_rat = np.std(lst_100_exp_rat)
            episode_results.append({"mean_10_reward": mean_10_reward, "std_10_reward": std_10_reward, "mean_100_reward": mean_100_reward,
                              "std_100_reward": std_100_reward, "mean_100_eval_score": mean_100_eval_score, "std_100_eval_score": std_100_eval_score,
                              "mean_100_exp_rat": mean_100_exp_rat, "std_100_exp_rat": std_100_exp_rat, "mean_10_eval": np.mean(self.evaluation_scores[-10:])})
            df = pd.DataFrame(episode_results)
            df.to_csv(f"{self.params['out_dir']}/results_{experiment}.csv", index=False)
            
            wallclock_elapsed = time.time() - training_start
            result[episode-1] = total_step, mean_100_reward, mean_100_eval_score, training_time, wallclock_elapsed
            reached_debug_time = time.time() - last_debug_time >= LEAVE_PRINT_EVERY_N_SECS

            elapsed_str = time.strftime("%H:%M:%S", time.gmtime(time.time() - training_start))
            debug_message = 'el {}, ep {:04}, ts {:07}, '
            debug_message += 'ar 10 {:05.1f}\u00B1{:05.1f}, '
            debug_message += '100 {:05.1f}\u00B1{:05.1f}, '
            debug_message += 'ex 100 {:02.1f}\u00B1{:02.1f}, '
            debug_message += 'ev {:05}\u00B1{:05}'
            debug_message = debug_message.format(
                elapsed_str, episode-1, total_step, mean_10_reward, std_10_reward, 
                mean_100_reward, std_100_reward, mean_100_exp_rat, std_100_exp_rat,
                mean_100_eval_score, std_100_eval_score)
            print(debug_message, end='\r', flush=True)

            reached_max_episodes = episode >= self.params['max_episodes']
            reached_goal_mean_reward = mean_100_eval_score >= self.params['goal_reward']
            training_is_over = reached_max_episodes or reached_goal_mean_reward
            if reached_debug_time or training_is_over:
                print(ERASE_LINE + debug_message, flush=True)
                last_debug_time = time.time()
            if training_is_over:
                if reached_max_episodes: print(u'--> reached_max_episodes \u2715')
                if reached_goal_mean_reward: print(u'--> reached_goal_mean_reward \u2713')
                print(f'saving model final model')
                torch.save(self.policy_model.state_dict(), f"{self.params['out_dir']}/actor_{experiment}_end.pt")
                torch.save(self.online_value_model_a.state_dict(), f"{self.params['out_dir']}/critica_{experiment}_end.pt")
                torch.save(self.online_value_model_b.state_dict(), f"{self.params['out_dir']}/criticb_{experiment}_end.pt")
                break
                
        final_eval_score, score_std = self.evaluate(self.policy_model, n_episodes=100)
        wallclock_time = time.time() - training_start
        print('Training complete.')
        print('Final evaluation score {:.2f}\u00B1{:.2f} in {:.2f}s training time,'
              ' {:.2f}s wall-clock time.\n'.format(
                  final_eval_score, score_std, training_time, wallclock_time))
  
        return result, final_eval_score, training_time, wallclock_time

    def evaluate(self, n_episodes=1, max_steps=200):
        step = 0
        rewards = []
        for _ in range(n_episodes):
            step, targets = self.episode_init(evaluation=True)
            state, done = self.sac.init_func(targets)
            rewards.append(0)
            for _ in count():
                experience = self.interaction_step(state, targets, train=False)
                state, done = self.update_results(experience, train=False)
                rewards[-1] += experience.reward
                step += 1
                if done or (step > max_steps): break
        return np.mean(rewards), np.std(rewards)

class Environment():
    def __init__(self, state_init, process_func, iteration_func):
        self.state_init = state_init
        self.process_func = process_func
        self.iteration_func = iteration_func
        
class Evaluator():
    def __init__(self, path, env):
        self.path = path
        self.env = env
        
    def interaction_step(self, state, targets):
        print(np.unique(self.env.process_func(state)))
        action = self.agent.select_greedy_action(torch.FloatTensor(self.env.process_func(state)))
        reward, next_state, done = self.env.iteration_func(action, targets)
        return Experience(state.copy(), action.copy(), reward, next_state.copy(), done)
    
    def update_results(self, experience):
        return experience.next_state.copy(), experience.done
    
    def evaluate(self, n_episodes, max_steps):
        total_rewards = []
        for _ in tqdm(range(n_episodes)):
            self.agent = DActor(action_bounds, mparams)
            self.agent.load_state_dict(torch.load(self.path))
            rewards = []
            targets = copy.deepcopy(eval_targets)
            state, done = self.env.state_init(targets)
            rewards.append(0)
            for step in range(max_steps):
                experience = self.interaction_step(state, targets)
                state, done = self.update_results(experience)
                print(experience.reward)
                rewards[-1] += experience.reward
                if done or (step == (max_steps - 1)):
                    total_rewards.append(np.sum(np.array(rewards)))
                    break
        return total_rewards
    
# devaluation = Evaluator('models/actor_000008_520.pt', Environment(dnn_init, dnn_process, dnn_iterate))
# devaluation.evaluate(1, 50)

eparams = {
    'nwarmup': 10,
    'tupdate': 40,
    'gamma': 0.99,
    'max_steps': 200,
    'max_episodes': 2000,
    'goal_reward': 450,
    'out_dir': os.getcwd()
}
mparams = {
    "log_std_min": -20, 
    "log_std_max": 2,
    "buffer_size": 16384,
    "batch_size": 512,
    "tau": 0.001,
    "entropy_lr": 0.0001,
    "policy_lr": 0.0004,
    "value_lr": 0.001,
    "policy_maxgrad": float('inf'),
    "value_maxgrad": float('inf')
}

experiment = "000007"
action_dim = 2
action_bounds = (np.array([0., 0.], dtype=np.float32), np.array([1., 1.], dtype=np.float32))
eval_targets = validate_targets(background, generate_groups(dimensions, config), dimensions, config)

    
if __name__ == '__main__':
    mmodels = str(sys.argv[1])
    if mmodels == "dnn":
        dpmodel_func = partial(lambda action_bounds, params: DActor(action_bounds, params),
                                action_bounds=action_bounds, params=mparams)
        dpoptimizer_func = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        dvmodel_func = partial(lambda action_dim, action_bounds: DQNetwork(action_dim, action_bounds),
                                action_dim=action_dim, action_bounds=action_bounds)
        dvoptimizer_func = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        ASAC = SAC(dpmodel_func, dpoptimizer_func, dvmodel_func, dvoptimizer_func,
                dnn_init, dnn_process, dnn_iterate, mparams)
        
    elif mmodels == "cnn":
        cvmodel_func = partial(lambda action_dim, action_bounds: CQNetwork(action_dim, action_bounds),
                                action_dim=action_dim, action_bounds=action_bounds)
        cvoptimizer_func = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        cpoptimizer_func = lambda net, lr: optim.Adam(net.parameters(), lr=lr)
        cpmodel_func = partial(lambda action_bounds, params: CActor(action_bounds, params), 
                            action_bounds=action_bounds, params=mparams)
        ASAC = SAC(cpmodel_func, cpoptimizer_func, cvmodel_func, cvoptimizer_func,
                    cnn_init, cnn_process, cnn_iterate, mparams)
    
    sac_results = []
    best_agent, best_eval_score = None, float('-inf')
    for seed in SEEDS:
        agent = SACT(ASAC, eparams, seed)
        result, final_eval_score, training_time, wallclock_time = agent.train()
        sac_results.append(result)
        if final_eval_score > best_eval_score:
            best_eval_score = final_eval_score
            best_agent = agent
    sac_results = np.array(sac_results)
    _ = BEEP()

