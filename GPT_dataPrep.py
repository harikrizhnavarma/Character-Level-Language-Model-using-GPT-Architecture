import torch
from hyper_parameters import TRAIN_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.model_selection import train_test_split

class DataPrep:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

        with open(self.dataset_name, 'r', encoding = 'utf-8') as f:
            self.text = f.read()
        
        self.char = sorted(list(set(self.text)))
        self.vocab_size = len(self.char)

        # create mapping by the characters
        str_to_int = {ch:index for index, ch in enumerate(self.char)} 
        int_to_str = {index:ch for index, ch in enumerate(self.char)}

        self.encode = lambda z: [str_to_int[c] for c in z]
        self.decode = lambda z: ''.join([int_to_str[c] for c in z])

        data = torch.tensor(self.encode(self.text), dtype = torch.long)

        data_len = len(data)
        train_split = int(data_len * TRAIN_SIZE)
        self.train_data = data[:train_split]
        self.val_data = data[train_split:]

    def get_batch(self, split, block_size, batch_size):

        input_data = (self.train_data if split == 'train' else self.val_data if split == 'validate' else None) 
        ix = torch.randint(len(input_data) - block_size, (batch_size, ))
                                                                        
        inputs = (torch.stack([input_data[i:i+block_size] for i in ix])).to(device)
        targets = (torch.stack([input_data[i+1:i+block_size + 1] for i in ix])).to(device)

        return inputs, targets