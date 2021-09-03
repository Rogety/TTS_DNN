
import math 
import os
import torch as t
import torch.nn as nn 
import torch.nn.functional as f
import torch.utils.data as data
import numpy as np

class Gaussian(nn.Module):
    def __init__(self, input_size, output_size):
        super(Gaussian, self).__init__()
        self.mean = nn.Linear(input_size, output_size)
        self.var = nn.Linear(input_size, output_size)

    def forward(self, x):
        mean = self.mean(x)
        var = t.exp( self.var(x) ) + 0.01
        return [mean, var]

    def log_density(self, x, ans):
        mean, var = self(x)
        a = -0.5 * t.sum( t.log( 2* math.pi * var ), 1) 
        b = -0.5 * t.sum( (ans-mean)**2 / var , 1)
        return a + b
    
    def density(self, x, ans):
        mean, var = self(x)
        a = t.prod( 2 * math.pi * var, 1)
        b = -0.5 * t.sum( (ans-mean)**2 / var, 1)
        return (a**-0.5) * t.exp( b )

class GaussianMixtures(nn.Module):
    def __init__(self, input_size, n_categories, output_size, output_vfloor=None ):
        super(GaussianMixtures, self).__init__()
        self.output_size = output_size
        self.n_categories = n_categories
        self.means = nn.Linear(input_size, n_categories * output_size)
        self.variances = nn.Linear(input_size, n_categories * output_size)
        self.weights = nn.Linear(input_size, n_categories)
        if output_vfloor is None:
            self.register_buffer("vfloor", t.full( (output_size,), 0.01 ) )
        else:
            #print(output_vfloor)
            self.register_buffer("vfloor", output_vfloor*0.01)
            
        
    def forward(self, x):
        weights = f.log_softmax( self.weights(x), dim=1 ) #(batch, n_categories)
        means = self.means(x).view(-1, self.n_categories, self.output_size )
        variances = (t.exp( self.variances(x) )).view(-1, self.n_categories, self.output_size ) + self.vfloor
        return (weights, means, variances)
        
    def forward_maximum_mixture(self, x):
        weights, means, variances = self(x)
        max_indices_2d = weights.argmax(dim=1, keepdim=True) #(batch, 1)
        max_indices_3d = max_indices_2d.view(-1,1,1).expand(-1,-1, self.output_size)
        w = t.gather(weights, 1, max_indices_2d).squeeze(dim=1) #(batch)
        m = t.gather(means, 1, max_indices_3d).squeeze(dim=1) #(batch, output_size)
        v = t.gather(variances, 1, max_indices_3d).squeeze(dim=1) #(batch, output_size)
        return (w, m, v, max_indices_2d.squeeze(1))

    def forward_selected_mixture(self, x, indices):
        weights, means, variances = self(x)
        indices_2d = indices.view(-1,1) # (batch, 1)
        indices_3d = indices.view(-1,1,1).expand(-1, 1,self.output_size) #(batch, 1, output_size)
        w = t.gather( weights, 1, indices_2d ).squeeze(1) #(batch)
        m = t.gather( means, 1, indices_3d ).squeeze(1) #(batch, output)
        v = t.gather( variances, 1, indices_3d ).squeeze(1) #(batch, output)
        return (w, m, v)

    def forward_argmax(self, x):
        weights = f.log_softmax( self.weights(x), dim=1 ) #(batch, n_categories)
        max_indices = weights.argmax(dim=1, keepdim=False) #(batch)
        return max_indices

class LogGaussianDensity(nn.Module):
    def __init__(self, weight, mean, var):
        super(LogGaussianDensity, self).__init__()
        self.register_buffer("weight", weight)
        self.register_buffer("mean", mean)
        self.register_buffer("var", var)
    
    def forward( self, ans ):
        m = self.mean
        v = self.var
        w = self.weight
        a = -0.5 * t.sum( t.log( 2* math.pi * v ), 1)  #(batch)
        b = -0.5 * t.sum( (ans-m)**2 / v, 1) #(batch)
        return a + b + w
    
    def get_means( self ):
        return self.mean
    
class AcousticModel(nn.Module):
    def __init__(self, input_size, hidden_size, n_categories, output_size, vfloor=None):
        super(AcousticModel, self).__init__()
        self.hidden_0 = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.density = GaussianMixtures( hidden_size, n_categories, output_size, vfloor)
        
    def forward(self, x):
        h0 = f.tanh( self.hidden_0(x) )
        h1 = f.tanh( self.hidden_1(h0) )
        h2 = f.tanh( self.hidden_2(h1) )
        return self.density(h2)
    
    def forward_maximum_mixture(self, x):
        h0 = f.tanh( self.hidden_0(x) )
        h1 = f.tanh( self.hidden_1(h0) )
        h2 = f.tanh( self.hidden_2(h1) )
        return self.density.forward_maximum_mixture(h2)   
    
    def forward_selected_mixture(self, x, indices):
        h0 = f.tanh( self.hidden_0(x) )
        h1 = f.tanh( self.hidden_1(h0) )
        h2 = f.tanh( self.hidden_2(h1) )
        return self.density.forward_selected_mixture(h2, indices)
    
    def to_log_density(self, x):
        h0 = f.tanh( self.hidden_0(x) )
        h1 = f.tanh( self.hidden_1(h0) )
        h2 = f.tanh( self.hidden_2(h1) )
        w, m, v, idx = self.density.forward_maximum_mixture(h2)    
        return (LogGaussianDensity(w.detach() ,m.detach() ,v.detach()), idx)


class AcousticModelV3(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, variance=None):
        super(AcousticModelV3, self).__init__()
        self.hidden_0 = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        
        if variance is None:
            variance = t.ones(output_size)

        self.register_buffer( "variance", variance )

    def forward( self, x ):
        h0 = t.sigmoid( self.hidden_0(x) )
        h1 = t.sigmoid( self.hidden_1(h0) )
        h2 = t.sigmoid( self.hidden_2(h1) )
        mean = self.mean(h2) 
        return mean
           
class AcousticModelAdapt(nn.Module):
    def __init__(self, source_model, variance=None):
        super(AcousticModelAdapt, self).__init__()
        n_output = source_model.mean.out_features
        for param in source_model.parameters():
            param.requires_grad = False
        self.source_model= source_model
        self.trans0 = nn.Linear(n_output, n_output)
        if variance is None:
            variance = source_model.variance
        self.register_buffer( "variance", variance )
    
    def forward(self, x):
        out = self.source_model(x)
        out = self.trans0(out)
        return out
    
    def to_acoustic_model_v3(self):
        model = self.source_model
        bias = (self.trans0.weight.mv(self.source_model.mean.bias) + self.trans0.bias).detach()
        weight = self.trans0.weight.mm(self.source_model.mean.weight).detach()
        
        model.mean.weight = nn.Parameter(weight)
        model.mean.bias = nn.Parameter(bias)
        model.register_buffer( "variance", self.variance )
        
        for param in model.parameters():
            param.requires_grad = False
        
        for param in model.mean.parameters():
            param.requires_grad = True

        return model

class DurationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DurationModel, self).__init__()
        self.hidden_0 = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)     
        self.hidden_2 = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        h0 = t.tanh( self.hidden_0(input) )
        h1 = t.tanh( self.hidden_1(h0) )
        h2 = f.log_softmax( self.hidden_2(h1), dim=1)
        return h2

class DurationModelV2(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DurationModelV2, self).__init__()
        self.hidden_0 = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)     
        self.hidden_2 = nn.Linear(hidden_size, 1)

    def forward(self, input):
        h0 = t.tanh( self.hidden_0(input) )
        h1 = t.tanh( self.hidden_1(h0) )
        h2 = self.hidden_2(h1)
        return t.exp(h2)


def log_gaussian_density(w, m, v, x):
    a = t.log(2*math.pi* v)
    #print(a.size())
    b = -0.5 * a.sum() #(1) # t.sum( a, 1)  #(batch)
    c = -0.5 * t.sum( (x-m)**2 / v, 1) #(batch)
    return b + c + w

# input: (time, size)
# outpt: (time, size)
def delta(x):
    h = x.t()
    h = t.unsqueeze( h, 1 )
    h = f.pad( h , (1,1) )
    w = t.Tensor([[[-0.5,0,0.5]]])
    h = f.conv1d( h, w )
    h = t.squeeze(h,1)
    h = h.t()
    return h

def delta2(x):
    h = x.t()
    h = t.unsqueeze( h, 1 )
    h = f.pad( h , (1,1) )
    w = t.Tensor([[[1.0,-2.0,1.0]]])
    h = f.conv1d( h, w )
    h = t.squeeze(h,1)
    h = h.t()
    return h

# input: (time)
# output: (time)
def smooth(x):
    h = t.unsqueeze( x, 0 ) #(1, time)
    h = t.unsqueeze( h, 1 ) #(1, 1, time)
    h = f.pad( h , (5,5) )
    w = t.Tensor([[[1,1,1,1,1,1,1,1,1,1,1]]])
    h = f.conv1d( h, w )
    h = t.squeeze(h,1)
    h = t.squeeze(h,0)
    return h/11



class Slice(data.Dataset):
    def __init__(self, dataset, offset, length):
        super(Slice, self).__init__()
        self.dataset = dataset
        self.offset = offset
        self.length = length
    
    def __getitem__(self, idx):
        if type(idx) is slice:
            s = slice( idx.start + self.offset, idx.stop + self.offset)
            return self.dataset[ s ]
        else:
            return self.dataset[ idx + self.offset ]

    def __len__(self):
        return self.length

def split_dataset( dataset, ratio=0.7):
    length = len(dataset)
    train_len = int(length*ratio)
    valid_len = length - train_len
    train_set = Slice(dataset, 0, train_len)
    valid_set = Slice(dataset, train_len, valid_len)
    return train_set, valid_set


def savebin( tensor, fid ):
    t = tensor.contiguous().numpy()
    t.tofile(fid)

# c_hat: (length, mgc_order), float
# c:(length, mgc_order), float
# uv:(length), byte

def make_dataset_old( in_fn, in_order, out_fn, out_order):
    in_length = int( os.stat( in_fn ).st_size / in_order / 4 )
    out_length = int( os.stat( out_fn ).st_size / out_order / 4 )
    
    if in_length != out_length:
        raise "inconsistent sizes"

    input = t.from_numpy( np.memmap(in_fn, dtype='float32', mode='r', shape=(in_length,in_order ) ) )
    output = t.from_numpy( np.memmap(out_fn, dtype='float32', mode='r', shape=(out_length,out_order ) ) )
    return data.TensorDataset( input, output )


def make_dataset( in_fn, in_order, in_dtype, out_fn, out_order, out_dtype):
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)

    in_length = int( os.stat( in_fn ).st_size / in_order / in_dtype.itemsize )
    out_length = int( os.stat( out_fn ).st_size / out_order / out_dtype.itemsize )
    
    if in_length != out_length:
        raise "inconsistent sizes"

    input = t.from_numpy( np.memmap(in_fn, dtype=in_dtype, mode='r', shape=(in_length,in_order ) ) )
    output = t.from_numpy( np.memmap(out_fn, dtype=out_dtype, mode='r', shape=(out_length,out_order ) ) )

    return data.TensorDataset( input, output )


def load_bin( filename, dtype):
    dtype = np.dtype(dtype)
    length = int( os.stat(filename).st_size/dtype.itemsize )
    return t.from_numpy( np.memmap(filename, dtype=dtype, mode='r', shape=(length,)))