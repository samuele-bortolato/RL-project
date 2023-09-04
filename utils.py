import torch
import matplotlib.pyplot as plt
import numpy as np


def compute_dist_discrete(actions):
    action_probs = actions.softmax(-1)
    dist = torch.distributions.Categorical(action_probs)
    return dist

def compute_dist_continuous(actions):
    
    means = actions[:,:2]
    correlations = actions[:,2:-2].tanh()*(1-1e-5)
    variances = actions[:,-2:].exp()

    b,n = means.shape
    corr_matrix = torch.zeros(b, n, n, device=means.device)
    indices = torch.tril_indices(n, n, offset=-1)
    corr_matrix[..., indices[0], indices[1]] = correlations
    corr_matrix[..., indices[1], indices[0]] = correlations
    corr_matrix[..., torch.arange(n), torch.arange(n)] = 1
    cov_matrix = corr_matrix * variances[...,None] * variances[:,None]
    
    dist = torch.distributions.multivariate_normal.MultivariateNormal(means, cov_matrix)
    
    return dist


def decode_action_discrete(dist, sampled_action, dec_x, dec_y):
    if len(sampled_action.shape)==2:
        sampled_action = sampled_action.squeeze(-1)
    log_prob = dist.log_prob(sampled_action)
    sampled_action_decoded = torch.stack([dec_x[sampled_action.long()],dec_y[sampled_action.long()]],1)
    return sampled_action_decoded, log_prob

def decode_action_continuous(dist, sampled_action):
    log_prob_sample = dist.log_prob(sampled_action)
    sampled_action_decoded = sampled_action.tanh()
    log_prob = log_prob_sample - torch.log( 1- sampled_action_decoded.square() + 1e-8 ).sum(-1) # correct accounting for the tanh transform
    return sampled_action_decoded, log_prob



def two_hot_encode(expecter_target_value, smr, sr, shr, smoothing=1e-2, device='cuda'):
    # create targets for critic
    simlog = (expecter_target_value.abs()+1).log()*expecter_target_value.sign()
    y = torch.zeros(len(simlog), sr, device=device)
    y[torch.arange(len(simlog), dtype=torch.long),(simlog*shr/smr+shr).floor().long().clip(0,sr-1)] = 1-(simlog.clip(-smr,smr)*shr/smr+shr).frac()
    y[torch.arange(len(simlog), dtype=torch.long),((simlog.clip(-smr,smr)*shr/smr+shr).floor().long()+1).clip(0,sr-1)] = (simlog.clip(-smr,smr)*shr/smr+shr).frac()

    # soft targets (to increase stability)
    y = y*(1-smoothing) + torch.ones_like(y)/sr*smoothing 
    return y

def update_target_model(model, target_model, decay=1e-2):
    model_dict = model.state_dict()
    target_model_dict = target_model.state_dict()
    for weight_key, target_weight_key in zip(model_dict.keys(),target_model_dict.keys()):
        target_model_dict[target_weight_key] = (1-decay)*target_model_dict[target_weight_key] + decay*model_dict[weight_key]
    target_model.load_state_dict(target_model_dict)

def decode_value_two_hot(bin_values, Vs,):
    Values = (torch.softmax(Vs,1)@bin_values[:,None])[:,0]
    return Values

def critic_error_func_two_hot( simlog_max_range, simlog_res, simlog_half_res, smoothing, X,Y,):
    y = two_hot_encode(Y, simlog_max_range, simlog_res, simlog_half_res, smoothing=smoothing)
    return torch.nn.functional.cross_entropy(X, y, reduction='none')

def critic_error_func_normal(X,Y):
    if len(X.shape)==2:
        X=X.squeeze(1)
    if len(Y.shape)==2:
        Y=Y.squeeze(1)
    return torch.nn.functional.mse_loss(X, Y, reduction='none')


def update_target_models(actor, V, Q, target_actor, target_V, target_Q, decay=1e-3):
    update_target_model(model=actor, target_model=target_actor,decay=decay)
    update_target_model(model=V, target_model=target_V, decay=decay)
    update_target_model(model=Q, target_model=target_Q, decay=decay)

def flatten_sequences(X, removelast=False):
    for i in range(len(X)):
        if removelast:
            X[i] = X[i][:,:-1]
        s = X[i].shape
        if len(s)==2:
            X[i] = X[i].reshape(s[0]*s[1])
        else:
            X[i] = X[i].reshape(s[0]*s[1],-1)
    return X

def reshape_sequences(X, shape):
    for i in range(len(X)):
        if X[i].shape.numel()==np.prod(shape):
            X[i] = X[i].reshape(shape)
        else:
            X[i] = X[i].reshape(shape+(-1,))
    return X

def get_batch(X, batch_size, b_idx):
    start = b_idx*batch_size
    end = (b_idx+1)*batch_size
    for i in range(len(X)):
        X[i] = X[i][start:end]
    return X

def initialize_zeros(shape, n, device):
    X=[]
    for _ in range(n):
        X.append(torch.zeros(shape, device=device))
    return X

def update_batched(X, U, batch_size, b_idx):
    start = b_idx*batch_size
    end = (b_idx+1)*batch_size
    for i in range(len(X)):
        X[i][start:end] = U[i]
    return X

def validation_plots(tb_writer, st, batch_size, V, target_V, actor, decode_values, discrete_actions, dec_x, dec_y, fig, ax, i):

    # compute the value function and the action probability in each point of the validation plane
    Values = []
    V_t = []
    A = []
    for b in range((len(st)+batch_size-1)//batch_size):
        stb = st[b*batch_size:(b+1)*batch_size]
        _, v, _ = V(stb.reshape(stb.shape[0],-1).cuda())
        _, vt, _ = target_V(stb.reshape(stb.shape[0],-1).cuda())
        a, _, _ = actor(stb.reshape(stb.shape[0],-1).cuda())
        Values.append(v)
        V_t.append(vt)
        A.append(a)
    Values = torch.concat(Values,0)
    A = torch.concat(A,0)
    V_t = torch.concat(V_t,0)

    # decode values
    Values = decode_values(Values).cpu()
    V_t = decode_values(V_t).cpu()

    tb_writer.add_image('V', (Values.reshape(1,100,100)/2+0.5).clip(0,1)*(1-1e-5), i)
    tb_writer.add_image('V_t', (V_t.reshape(1,100,100)/2+0.5).clip(0,1)*(1-1e-5), i)

    # plot the mean.tanh() (not the real mean of the distribution of the tanh)
    pos = st[:,0].reshape(100,100,-1)[::2,::2]
    A = A.reshape(100,100,-1)[::2,::2]

    if discrete_actions:
        action_probs = A.softmax(-1)
        u = action_probs@dec_x[:,None]
        v = action_probs@dec_y[:,None]
        A = torch.concat([u,v],-1)

    plt.quiver(pos[...,1].flatten(), -pos[...,0].flatten(), A[...,1].tanh().detach().cpu().flatten(), -A[...,0].tanh().detach().cpu().flatten(), color='g',scale=50, headwidth=2)
    ax.axis('off')
    plt.gca().set_aspect('equal')
    plt.subplots_adjust(0,0,1,1,0,0)
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.clf()

    tb_writer.add_image('Policy visualization', np.transpose(data,(2,0,1)) , i)