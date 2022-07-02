#!/usr/bin/env python
# coding: utf-8

# In[13]:


import torch
import torch.nn as nn

class CLFModel(nn.Module):
    """remove output size variable?  dont think its needed
    instead of all these inputs, see what can be replaced by just taking dimensions of an input 
    vector
    """
    def __init__(self, input_size, hidden_size, batch_size, n_classes):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.lstm1 = nn.LSTMCell(1, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.n_classes)
        
    def forward(self, x):
        """x and y are taken in the method"""
        h_t = torch.zeros(x.size(0), self.hidden_size)
        c_t = torch.zeros(x.size(0), self.hidden_size)
        h_t2 = torch.zeros(x.size(0), self.hidden_size)
        c_t2 = torch.zeros(x.size(0), self.hidden_size)
        
        # outputs = torch.empty(1, 1000, 10)
        # outputs = []
        for x_t in x.split(1, dim=1):
            h_t, c_t = self.lstm1(x_t, (h_t, c_t)) # updates states in lstm layer 1
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2)) # updates states in lstm layer 2

            # output = self.linear(h_t2)
            # outputs += output
            # outputs = torch.stack(outputs, dim=1)
        outputs = self.linear(h_t2)
        return outputs


# In[ ]:




