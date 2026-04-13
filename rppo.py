import os
import numpy as np
import torch
import torch.nn as nn

ACTIONS=("L45","L22","FW","R22","R45")
STATE_DIM=18
N_ACTIONS=5

_GRU_HIDDEN=128
_FC_EMBED=64
_MODEL=None
_H=None
_STEP=0
_MAX_EP_STEPS=1000

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Sequential(
            nn.Linear(STATE_DIM,_FC_EMBED),nn.Tanh(),
            nn.Linear(_FC_EMBED,_FC_EMBED),nn.Tanh()
        )
        self.gru=nn.GRUCell(_FC_EMBED,_GRU_HIDDEN)
        self.actor=nn.Sequential(
            nn.Linear(_GRU_HIDDEN,_FC_EMBED),nn.Tanh(),
            nn.Linear(_FC_EMBED,N_ACTIONS)
        )

    def forward(self,obs,h):
        e=self.embed(obs)
        h=self.gru(e,h)
        logits=self.actor(h)
        return logits,h

def _load():
    global _MODEL,_H
    if _MODEL is not None:
        return
    path=os.path.join(os.path.dirname(__file__),"rppo_weights.pth")
    ckpt=torch.load(path,map_location="cpu")
    net=Net()
    net.load_state_dict(ckpt["net"], strict=False)
    net.eval()
    _MODEL=net
    _H=torch.zeros(1,_GRU_HIDDEN)

def _reset_hidden():
    global _H
    _H=torch.zeros(1,_GRU_HIDDEN)

def sensor_policy(obs):
    near=obs[0:8]
    far=obs[8:16]
    ir=obs[16]
    stuck=obs[17]

    if ir:
        return "FW"
    if near[3] or near[4]:
        return "FW"
    if far[3] or far[4]:
        return "FW"
    if near[0] or near[1] or far[0]:
        return "L22"
    if near[6] or near[7] or far[7]:
        return "R22"
    if stuck:
        return "L45"
    return "L45"

def policy(obs,rng):
    global _STEP,_H
    _load()

    # reset at episode start
    if _STEP==0:
        _reset_hidden()

    # if no sensors → search
    if np.sum(obs[:16])==0:
        action="L45"
    else:
        with torch.no_grad():
            obs_t=torch.FloatTensor(obs).unsqueeze(0)
            logits,_H=_MODEL(obs_t,_H)
            a=int(logits.argmax(1).item())
            action=ACTIONS[a]

    _STEP+=1
    if _STEP>=_MAX_EP_STEPS:
        _STEP=0

    return action