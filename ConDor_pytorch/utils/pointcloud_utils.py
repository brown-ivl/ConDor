import torch
from utils.group_points import gather_idx
import numpy as np
import h5py

def diameter(x, axis=-2, keepdims=True):
    return torch.max(x, dim=axis, keepdims=keepdims).values - torch.min(x, dim=axis, keepdims=keepdims).values

def kdtree_indexing(x, depth=None, return_idx = False):
    num_points = x.shape[1]
    #assert isPowerOfTwo(num_points)
    if depth is None:
        depth = int(np.log(num_points) / np.log(2.) + 0.1)
    y = x
    batch_idx = torch.arange(x.shape[0]).to(torch.int32).to(x.device)
    batch_idx = torch.reshape(batch_idx, (-1, 1))
    batch_idx = batch_idx.repeat(1, x.shape[1])

    points_idx = torch.arange(num_points).type_as(x).to(torch.int64)
    points_idx = torch.reshape(points_idx, (1, -1, 1))
    points_idx = points_idx.repeat(x.shape[0], 1, 1)



    for i in range(depth):
        y_shape = list(y.shape)
        diam = diameter(y)
        split_idx = torch.argmax(diam, dim=-1).to(torch.long).to(x.device)
        split_idx = split_idx.repeat(1, y.shape[1])
        idx = torch.arange(y.shape[0]).to(torch.long).to(x.device)
        idx = idx.unsqueeze(-1)
        idx = idx.repeat(1, y.shape[1])
        branch_idx = torch.arange(y.shape[1]).to(torch.long).to(x.device)
        branch_idx = branch_idx.unsqueeze(0)
        branch_idx = branch_idx.repeat(y.shape[0], 1)
        split_idx = torch.stack([idx, branch_idx, split_idx], dim=-1)
        # print(split_idx, split_idx.shape)
        # Gather implementation required
        # m = tf.gather_nd(y, split_idx)
        # print(y.shape)
        m = gather_idx(y, split_idx)
        # print(m.shape)
        sort_idx = torch.argsort(m, dim=-1)
        sort_idx = torch.stack([idx, sort_idx], dim=-1)
        # Gather required
        points_idx = gather_idx(points_idx, sort_idx)
        points_idx = torch.reshape(points_idx, (-1, int(y.shape[1] // 2), 1))
        # Gather
        y = gather_idx(y, sort_idx)
        y = torch.reshape(y, (-1, int(y.shape[1] // 2), 3))


    
    y = torch.reshape(y, x.shape)
    if not return_idx:
        return y

    points_idx = torch.reshape(points_idx, (x.shape[0], x.shape[1]))
    points_idx_inv = torch.argsort(points_idx, dim=-1)
    points_idx = torch.stack([batch_idx, points_idx], dim=-1)
    points_idx_inv = torch.stack([batch_idx, points_idx_inv], dim=-1)

    return y, points_idx, points_idx_inv


if __name__=="__main__":

    # x = (torch.ones((2, 1024, 3)) * torch.reshape(torch.arange(1024), (1, -1, 1))).cuda()
    # x = torch.randn((2, 1024, 3)).cuda()
    filename = "/home/rahul/research/data/sapien_processed/train_refrigerator.h5"
    f = h5py.File(filename, "r")
    x = torch.from_numpy(f["data"][:2]).cuda()
    # x2 = torch.from_numpy(f["data"][2:4]).cuda()
    y, kd, kd_2 = kdtree_indexing(x, return_idx = True)
    print(x, x.shape, y, y.shape)

    print(kd, kd.shape)
    print(kd_2, kd_2.shape)
