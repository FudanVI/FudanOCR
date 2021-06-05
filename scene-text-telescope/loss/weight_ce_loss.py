import torch
import numpy as np
import pickle as pkl

standard_alphebet = '-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
standard_dict = {}
for index in range(len(standard_alphebet)):
    standard_dict[standard_alphebet[index]] = index

def load_confuse_matrix():
    f = open('./dataset/mydata/confuse.pkl', 'rb')
    data = pkl.load(f)
    f.close()
    number = data[:10]
    upper = data[10:36]
    lower = data[36:]
    end = np.ones((1,62))
    pad = np.ones((63,1))
    rearrange_data = np.concatenate((end, number, lower, upper), axis=0)
    rearrange_data = np.concatenate((pad, rearrange_data), axis=1)
    rearrange_data = 1 / rearrange_data
    rearrange_data[rearrange_data==np.inf] = 1
    rearrange_data = torch.Tensor(rearrange_data).cuda()

    lower_alpha = 'abcdefghijklmnopqrstuvwxyz'
    # upper_alpha = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i in range(63):
        for j in range(63):
            if i != j and standard_alphebet[j] in lower_alpha:
                rearrange_data[i][j] = max(rearrange_data[i][j], rearrange_data[i][j+26])
    rearrange_data = rearrange_data[:37,:37]

    return rearrange_data

weight_table = load_confuse_matrix()
def weight_cross_entropy(pred, gt):
    global weight_table
    batch = gt.shape[0]
    weight = weight_table[gt]
    pred_exp = torch.exp(pred)
    pred_exp_weight = weight * pred_exp
    loss = 0
    for i in range(len(gt)):
        loss -= torch.log(pred_exp_weight[i][gt[i]] / torch.sum(pred_exp_weight, 1)[i])
    return loss / batch


if __name__ == '__main__':
    print(load_confuse_matrix().shape)
