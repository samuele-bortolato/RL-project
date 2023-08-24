import torch
import matplotlib.pyplot as plt

def simlog():
    pass

def simexp():
    pass

def two_hot_encode(expecter_target_value, smr, sr, shr, smoothing=1e-2, device='cuda'):
    # create targets for critic
    simlog = (expecter_target_value.abs()+1).log()*expecter_target_value.sign()
    y = torch.zeros(len(simlog), sr, device=device)
    y[torch.arange(len(simlog), dtype=torch.long),(simlog*shr/smr+shr).floor().long().clip(0,sr-1)] = 1-(simlog.clip(-smr,smr)*shr/smr+shr).frac()
    y[torch.arange(len(simlog), dtype=torch.long),((simlog.clip(-smr,smr)*shr/smr+shr).floor().long()+1).clip(0,sr-1)] = (simlog.clip(-smr,smr)*shr/smr+shr).frac()

    # soft targets
    y = y*(1-smoothing) + torch.ones_like(y)/sr*smoothing 
    return y

def update_target_model(model, target_model, decay=1e-2):
    model_dict = model.state_dict()
    target_model_dict = target_model.state_dict()
    for weight_key, target_weight_key in zip(model_dict.keys(),target_model_dict.keys()):
        target_model_dict[target_weight_key] = (1-decay)*target_model_dict[target_weight_key] + decay*model_dict[weight_key]
    target_model.load_state_dict(target_model_dict)