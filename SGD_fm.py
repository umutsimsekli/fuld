import math
import torch
from torch.optim.optimizer import Optimizer
import scipy.io as sio
import numpy as np

# Adapted from AdamW implementation in https://github.com/egg-west/AdamW-pytorch/blob/master/adamW.py

# In order to use the optimizer with 1<alpha<2, the Matlab/Octave script precompute_grad.m needs to be run. 

class SGD_fm(Optimizer):

    def __init__(self, params, alp=1.0, lr=1e-3, gam=0.9, eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= gam:
            raise ValueError("Invalid gamma value: {}".format(gam))
        defaults = dict(alp=alp, lr=lr, gam=gam, eps=eps)

        self.v_range = []
        self.grad_precomp = []

        super(SGD_fm, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_fm, self).__setstate__(state)

    def precompute_grad(self,v):
        inds = np.digitize(v.numpy(), self.v_range)
        grad = torch.from_numpy(self.grad_precomp[inds])
        grad = grad / 2
        return grad.float()

    def step(self):
        """Performs a single optimization step.
        """
        loss = None

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('SGD_fm does not support sparse gradients')

                state = self.state[p]
                gam = group['gam']
                alp = group['alp']

                # State initialization
                if len(state) == 0:
                    #print('initializing opt')
                    state['step'] = 0
                    # V(t) in the paper
                    state['momentum'] = 0.001 * torch.randn_like(p.data)
                    if( alp>1.0 and alp<2.0):
                        mat_contents = sio.loadmat("./precomputed_grads/"+f"{alp:2.4f}"+".mat")
                        self.v_range      = np.squeeze(mat_contents['v_range'])
                        self.grad_precomp = np.squeeze(mat_contents['grad'])
                    

                momentum = state['momentum']
                state['step'] += 1

                eta = group['lr'] 

                momentum.mul_(1- eta*gam).add_(-eta, grad)
                state['momentum'] = momentum

                if(alp == 1.0):
                    p.data.add_(eta, momentum/(momentum**2+1))
                elif(alp == 2.0):
                    p.data.add_(eta, momentum)
                else:
                    p.data.add_(eta, self.precompute_grad(momentum))

        return loss

