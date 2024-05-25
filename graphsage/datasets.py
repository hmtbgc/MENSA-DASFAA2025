import scipy
import os
import numpy as np
import json
import torch
import dgl
from sklearn.preprocessing import StandardScaler

class CustomDataset(object):
    def __init__(self, raw_dir, name):
        super(CustomDataset, self).__init__()
        dataset_path = os.path.join(raw_dir, name)
        adj_full = scipy.sparse.load_npz(os.path.join(dataset_path, 'adj_full.npz')).astype(np.bool_)
        role = json.load(open(os.path.join(dataset_path, 'role.json')))
        feats = np.load(os.path.join(dataset_path, 'feats.npy'))
        class_map = json.load(open(os.path.join(dataset_path, 'class_map.json')))
        class_map = {int(k): v for k, v in class_map.items()}
        scaler = StandardScaler()
        scaler.fit(feats)
        feats = scaler.transform(feats)
        feats = feats.astype(np.float32)
        
        train_idx = torch.tensor(role['tr'])
        val_idx = torch.tensor(role['va'])
        test_idx = torch.tensor(role['te'])
        feats = torch.tensor(feats)
        g = dgl.from_scipy(adj_full)
        
        g.ndata['feat'] = feats
        train_mask = torch.zeros(g.num_nodes()).bool()
        val_mask = torch.zeros(g.num_nodes()).bool()
        test_mask = torch.zeros(g.num_nodes()).bool()
        train_mask[train_idx] = 1
        val_mask[val_idx] = 1
        test_mask[test_idx] = 1
        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask
        
        label = []
        for i in range(g.num_nodes()):
            label.append(class_map[i])
        label = torch.tensor(label)
        
        g.ndata['label'] = label
        
        self.g = g
        if (len(label.shape) > 1):
            self.num_classes = label.shape[1]
        else:
            self.num_classes = torch.unique(label).shape[0]
        
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        assert index == 0
        return self.g
    
    
class AmazonDataset(CustomDataset):
    def __init__(self, raw_dir):
        super(AmazonDataset, self).__init__(raw_dir, 'amazon')
        
class YelpDataset(CustomDataset):
    def __init__(self, raw_dir):
        super(YelpDataset, self).__init__(raw_dir, 'yelp')
        
class RedditDataset(CustomDataset):
    def __init__(self, raw_dir):
        super(RedditDataset, self).__init__(raw_dir, 'reddit')
        
class FlickrDataset(CustomDataset):
    def __init__(self, raw_dir):
        super(FlickrDataset, self).__init__(raw_dir, 'flickr')