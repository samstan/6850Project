import numpy as np
from tqdm import tqdm

def truncate(state):
    new_state = state
    if state<0:
        new_state = 0
    elif state>9:
        new_state = 9
    return new_state

def randomize(action, state, neighbor_states):
    '''neighbor-dependent transition'''
    return int(np.random.normal(state+action, np.sum(neighbor_states)))

class Env():
    def __init__(self, num_agents = 4):

        self.state_space = [np.arange(10) for i in range(num_agents)]

        self.action_space = [np.array([-1,0,1]) for i in range(num_agents)]

        self.num_agents = num_agents

        self.states = [None for i in range(num_agents)]

        self.neighbor_dict = {0:[1], 1:[0,2], 2:[1,3], 3:[2]}

    def reset(self):
        for i in range(self.num_agents):
            self.states[i] = 2*i + 1

    def get_neighbor_states(self, agent):
        neighbors = self.neighbor_dict[agent]
        return [self.states[n] for n in neighbors]

    
    def intersection(self):
        res = 0
        for i in range(self.num_agents):
            for j in range(i+1,self.num_agents):
                if self.states[i] == self.states[j]:
                    res+=1
        return res


    def step(self,actions):
        action_list = [-1,0,1]
        for i in range(self.num_agents):
            self.states[i] = truncate(randomize(action_list[actions[i]], self.states[i], self.get_neighbor_states(i)))
        return self.intersection() * 2

    def reward(self, states, actions):
        new_states = states
        action_list = [-1,0,1]
        for i in range(len(states)):
            new_states[i] = truncate(randomize(action_list[actions[i]], states[i], self.get_neighbor_states(i)))
        rews = [0 for i in range(self.num_agents)]
        for i in range(self.num_agents):
            for j in range(self.num_agents):
                if i!=j and new_states[i] == new_states[j]:
                        rews[i]+=1
        return new_states, rews

        
class ValueIter():

    def __init__(self, env, T = 1):
        self.env = env
        self.Q = np.zeros((10, 10, 10, 10, 3 , 3, 3, 3))
        self.V = np.zeros((10,10,10,10))
        self.T = T
        self.samples = {}
        self.rewards = {}
        self.policy = {}

    def update_V(self):
        for s0 in range(10):
            for s1 in range(10):
                for s2 in range(10):
                    for s3 in range(10):
                        self.V[s0,s1,s2,s3] = np.min(self.Q[s0,s1,s2,s3,:,:,:,:])

    def sample(self):
        for s0 in range(10):
            for s1 in range(10):
                for s2 in range(10):
                    for s3 in range(10):
                        for a0 in range(3):
                            for a1 in range(3):
                                for a2 in range(3):
                                    for a3 in range(3):
                                        self.samples[(s0,s1,s2,s3,a0,a1,a2,a3)] = []
                                        self.rewards[(s0,s1,s2,s3,a0,a1,a2,a3)] = []
                                        if self.T>=1:
                                            for i in range(self.T):
                                                samp,rew = self.env.reward([s0, s1, s2, s3], [a0, a1, a2, a3])
                                                self.samples[(s0,s1,s2,s3,a0,a1,a2,a3)].append(samp)
                                                self.rewards[(s0,s1,s2,s3,a0,a1,a2,a3)].append(rew)
                                        else:
                                            if np.random.random()<self.T:
                                                samp,rew = self.env.reward([s0, s1, s2, s3], [a0, a1, a2, a3])
                                                self.samples[(s0,s1,s2,s3,a0,a1,a2,a3)].append(samp)
                                                self.rewards[(s0,s1,s2,s3,a0,a1,a2,a3)].append(rew) 

    def iterate(self, max_iter = 10):
        for i in tqdm(range(max_iter)):
            for s0 in range(10):
                for s1 in range(10):
                    for s2 in range(10):
                        for s3 in range(10):
                            for a0 in range(3):
                                for a1 in range(3):
                                    for a2 in range(3):
                                        for a3 in range(3):
                                            newQ = 0
                                            T_ = len(self.rewards[(s0,s1,s2,s3,a0,a1,a2,a3)])
                                            for t in range(T_):
                                                immediate = np.sum(self.rewards[(s0,s1,s2,s3,a0,a1,a2,a3)][t])
                                                s0_,s1_,s2_,s3_ = self.samples[(s0,s1,s2,s3,a0,a1,a2,a3)][t]
                                                maximizer = self.V[s0_,s1_,s2_,s3_]
                                                newQ += (immediate+maximizer)/T_
                                            if newQ>0:
                                                self.Q[s0, s1, s2, s3, a0, a1, a2, a3] = newQ
            self.update_V()

    def get_subsamples(self):
        res0 = {}
        rew0 = {}
        res1 = {}
        rew1 = {}
        res2 = {}
        rew2 = {}
        res3 = {}
        rew3 = {}
        for s0 in range(10):
            for s1 in range(10):
                for a0 in range(3):
                    res0[(s0,s1,a0)] = []
                    rew0[(s0,s1,a0)] = []
        for s0 in range(10):
            for s1 in range(10):
                for s2 in range(10):
                    for a1 in range(3):
                        res1[(s0,s1,s2,a1)] = []
                        rew1[(s0,s1,s2,a1)] = []
        for s1 in range(10):
            for s2 in range(10):
                for s3 in range(10):
                    for a2 in range(3):
                        res2[(s1,s2,s3,a2)] = []
                        rew2[(s1,s2,s3,a2)] = []
        for s2 in range(10):
            for s3 in range(10):
                for a3 in range(3):
                    res3[(s2,s3,a3)] = []   
                    rew3[(s2,s3,a3)] = []  

        for s0 in range(10):
            for s1 in range(10):
                for s2 in range(10):
                    for s3 in range(10):
                        for a0 in range(3):
                            for a1 in range(3):
                                for a2 in range(3):
                                    for a3 in range(3):
                                        T_ = len(self.samples[(s0,s1,s2,s3,a0,a1,a2,a3)])
                                        for t in range(T_):
                                            temp = self.samples[(s0,s1,s2,s3,a0,a1,a2,a3)][t]
                                            res0[(s0,s1,a0)].append(temp[0])
                                            res1[(s0,s1,s2,a1)].append(temp[1])
                                            res2[(s1,s2,s3,a2)].append(temp[2])
                                            res3[(s2,s3,a3)].append(temp[3])
                                            tmp2 = self.rewards[(s0,s1,s2,s3,a0,a1,a2,a3)][t]
                                            rew0[(s0,s1,a0)].append(tmp2[0])
                                            rew1[(s0,s1,s2,a1)].append(tmp2[1])
                                            rew2[(s1,s2,s3,a2)].append(tmp2[2])
                                            rew3[(s2,s3,a3)].append(tmp2[3])

        return res0, rew0, res1, rew1, res2, rew2, res3, rew3

    def iterate_network(self, max_iter = 10):
        res0, rew0, res1, rew1, res2, rew2, res3, rew3 = self.get_subsamples()
        for i in tqdm(range(max_iter)):
            for s0 in range(10):
                for s1 in range(10):
                    for s2 in range(10):
                        for s3 in range(10):
                            for a0 in range(3):
                                for a1 in range(3):
                                    for a2 in range(3):
                                        for a3 in range(3):
                                            states0  = res0[(s0,s1,a0)]
                                            rewards0 = rew0[(s0,s1,a0)]
                                            states1 = res1[(s0,s1,s2,a1)]
                                            rewards1 = rew1[(s0,s1,s2,a1)]
                                            states2 = res2[(s1,s2,s3,a2)]
                                            rewards2 = rew2[(s1,s2,s3,a2)]
                                            states3 = res3[(s2,s3,a3)]
                                            rewards3 = rew3[(s2,s3,a3)]
                                            T_ = min(len(states0),len(states1),len(states2),len(states3))
                                            # for t in range(T_):
                                            #     s0_, s1_, s2_, s3_ = states0[t], states1[t], states2[t],states3[t]
                                            #     immediate = np.sum([rewards0[t],rewards1[t], rewards2[t], rewards3[t]])
                                            #     maximizer = int(np.max(self.Q[s0_,s1_,s2_,s3_,:,:,:,:]))
                                            #     newQ += (immediate+maximizer)/T_
                                            immediate = (np.sum(rewards0[:T_]) + np.sum(rewards1[:T_]) + np.sum(rewards2[:T_]) + np.sum(rewards3[:T_]))/T_
                                            # maximizers = [self.V[states0[t],states1[t], states2[t],states3[t]] for t in range(T_)]
                                            maximizers = self.V[states0[:T_],states1[:T_], states2[:T_],states3[:T_]]
                                            maximizer = np.sum(maximizers)/T_
                                            if immediate+maximizer>0:
                                                self.Q[s0, s1, s2, s3, a0, a1, a2, a3] = immediate+maximizer
            self.update_V()
    def get_policy(self):
        for s0 in range(10):
            for s1 in range(10):
                for s2 in range(10):
                    for s3 in range(10):
                        self.policy[(s0,s1,s2,s3)] = np.unravel_index(np.argmin(self.Q[s0,s1,s2,s3,:,:,:,:]), self.Q[s0,s1,s2,s3,:,:,:,:].shape)

if __name__ == "__main__":
    np.random.seed(6850)
    e = Env()
    e.reset()
    v = ValueIter(e, 0.01)
    v.sample()
    v.iterate(max_iter = 10)
    v.get_policy()

    H = 100
    e.reset()
    reward = 0
    for h in range(H):
        reward += e.step(v.policy[tuple(e.states)])
    
    print(reward)


    