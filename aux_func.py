import torch
import torch.nn.functional as F

'''
The negative log-likelihood loss is computed by taking the element-wise logarithm of Q and multiplying it by the weight tensor, 
and then applying the F.nll_loss function from PyTorch to the resulting tensor and the labels tensor, 
which are the true labels for the given inputs.


'''
def NL_loss(f, labels):
    Q_1 = 1 - F.softmax(f, dim=1) # softmax of f
    Q = F.softmax(Q_1, dim=1)
    weight = 1 - Q
    out = weight * torch.log(Q)
    return F.nll_loss(out, labels)

# The entropy_loss function measures the entropy of the predicted probability distribution. 
# It aims to maximize the entropy of the output probabilities, which can encourage the network to output less confident predictions. 
# The entropy loss can be used as a regularization technique to prevent overfitting.
def entropy_loss(p):
    p = F.softmax(p, dim=1)
    epsilon = 1e-5
    return (-1 * torch.sum(p * torch.log(p + epsilon))) / p.shape[0]


