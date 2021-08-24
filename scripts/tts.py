import math
import os
import torch as t
import torch.nn as nn
import torch.nn.functional as f
import torch.utils.data as data
import numpy as np


def log_gaussian_density(w, m, v, x):
    a = t.log(2*math.pi* v)
    #print(a.size())
    b = -0.5 * a.sum() #(1) # t.sum( a, 1)  #(batch)
    c = -0.5 * t.sum( (x-m)**2 / v, 1) #(batch)
    return b + c + w

def padding(input , max_length):

    re_length = max_length - input.shape[0]
    m = t.zeros( (re_length , input.shape[1]) )
    re_input = t.cat([input, m], axis=0)

    return re_input

def make_dataset( in_fn, in_order, in_dtype, out_fn, out_order, out_dtype):
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)

    in_length = int( os.stat( in_fn ).st_size / in_order / in_dtype.itemsize )
    out_length = int( os.stat( out_fn ).st_size / out_order / out_dtype.itemsize )

    if in_length != out_length:
        print("in_length:{},out_length:{}".format(in_length , out_length))
        raise "inconsistent sizes"

    print("input:",in_fn)
    print("output:",out_fn)
    input = t.from_numpy( np.memmap(in_fn, dtype=in_dtype, mode='r', shape=(in_length,in_order ) ) )
    output = t.from_numpy( np.memmap(out_fn, dtype=out_dtype, mode='r', shape=(out_length,out_order ) ) )
    #print("input:",input)
    #print("output:",output)
    print("input.shape : " , input.shape) ## [538630 , 1386]
    print("output.shape : " , output.shape) ## [538630 ,  79]

    return data.TensorDataset( input, output )

def make_dataset_lstm( in_fn, in_order, in_dtype, out_fn, out_order, out_dtype, train_model):
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)


    ## (1123 , 1070 , 1386)
    if train_model == 0 :
        padded_input = t.zeros( (1,1070,1386) )
        max_length = 1070 ##dummy
    else :
        padded_input = t.zeros( (1,360,1382) )
        max_length = 360
    for dirPath, dirNames, fileNames in os.walk(in_fn):
        for item in fileNames :
            print(item)
            in_filepath = os.path.join(dirPath,item)
            in_length = int( os.stat(in_filepath).st_size / in_order / in_dtype.itemsize )
            input = t.from_numpy( np.memmap(in_filepath, dtype=in_dtype, mode='r', shape=(in_length,in_order ) ) )
            z = padding(input,max_length).unsqueeze(0)
            padded_input = t.cat( (padded_input,z) , dim=0 )


    ## (1123 , 1070 ,79 )
    if train_model == 0 :
        padded_output = t.zeros( (1,1070,79) )
        max_length = 1070
    else :
        padded_output = t.zeros( (1,360,1) )
        max_length = 360
    for dirPath, dirNames, fileNames in os.walk(out_fn):
        for item in fileNames :
            print(item)
            out_filepath = os.path.join(dirPath,item)
            out_length = int( os.stat(out_filepath).st_size / out_order / out_dtype.itemsize )
            output = t.from_numpy( np.memmap(out_filepath, dtype=out_dtype, mode='r', shape=(out_length,out_order ) ) )
            z = padding(output,max_length).unsqueeze(0)
            padded_output = t.cat( (padded_output,z) , dim=0 )

    print(padded_input.shape)
    print(padded_output.shape)

    '''
    fake_input = t.rand(10,1070,1386)
    fake_output = t.rand(10,1070,79)
    #fake_input = t.rand(10,1070,1382)
    #fake_output = t.rand(10,1070,1)
    print(fake_input.shape)
    print(fake_output.shape)

    return data.TensorDataset( fake_input, fake_output )
    '''
    return data.TensorDataset( padded_input, padded_output )

def make_dataset_lstm_dur( in_fn, in_order, in_dtype, out_fn, out_order, out_dtype):
    in_dtype = np.dtype(in_dtype)
    out_dtype = np.dtype(out_dtype)


    ## (1123 , 1070 , 1382)
    padded_input = t.zeros( (1,1070,1382) ) ##dummy
    for dirPath, dirNames, fileNames in os.walk(in_fn):
        max_length = 1070
        for item in fileNames :
            in_filepath = os.path.join(dirPath,item)
            in_length = int( os.stat(in_filepath).st_size / in_order / in_dtype.itemsize )
            input = t.from_numpy( np.memmap(in_filepath, dtype=in_dtype, mode='r', shape=(in_length,in_order ) ) )
            #print(input.dtype)
            z = padding(input,max_length).unsqueeze(0)
            padded_input = t.cat( (padded_input,z) , dim=0 )
            print(item)

    ## (1123 , 1070 ,1 )
    padded_output = t.zeros( (1,1070,1) )
    for dirPath, dirNames, fileNames in os.walk(out_fn):
        max_length = 1070
        for item in fileNames :
            out_filepath = os.path.join(dirPath,item)
            out_length = int( os.stat(out_filepath).st_size / in_order / out_dtype.itemsize )
            output = t.from_numpy( np.memmap(out_filepath, dtype=out_dtype, mode='r', shape=(out_length,out_order ) ) )
            #print(input.dtype)

            z = padding(output,max_length).unsqueeze(0)
            padded_output = t.cat( (padded_output,z) , dim=0 )
            print(item)

    print(padded_input.shape)
    print(padded_output.shape)
    '''
    #fake_input = t.rand(10,1070,1386)
    #fake_output = t.rand(10,1070,79)
    fake_input = t.rand(10,1070,1382)
    fake_output = t.rand(10,1070,1)
    #print(fake_input.shape)
    #print(fake_output.shape)

    return data.TensorDataset( fake_input, fake_output )
    '''
    return data.TensorDataset( padded_input, padded_output )

class AcousticModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, variance=None):
        super(AcousticModel, self).__init__()
        self.hidden_0 = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        #self.hidden_3 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)
        ## 4 layer and sigmoid best
        if variance is None:
            variance = t.ones(output_size)

        self.register_buffer( "variance", variance )

    def forward( self, x ):
        h0 = t.sigmoid( self.hidden_0(x) )
        h0_d = self.dropout(h0)
        h1 = t.sigmoid( self.hidden_1(h0_d) )
        h1_d = self.dropout(h1)
        h2 = t.sigmoid( self.hidden_2(h1_d) )
        #h3 = t.sigmoid( self.hidden_3(h2) )
        mean = self.mean(h2)
        return mean

class DurationModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DurationModel, self).__init__()
        self.hidden_0 = nn.Linear(input_size, hidden_size)
        self.hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size, hidden_size)
        self.hidden_3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.5)
        # 3 layer nad relu best
    def forward(self, input):
        h0 = t.sigmoid( self.hidden_0(input) )
        h0_d = self.dropout(h0)
        h1 = t.sigmoid( self.hidden_1(h0_d) )
        h1_d = self.dropout(h1)
        h2 = t.sigmoid( self.hidden_1(h1_d) )
        h3 = self.hidden_3(h2)
        return t.exp(h3)

class AcousticModel_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, variance=None):
        super(AcousticModel_LSTM,self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Linear(hidden_size,79)
        if variance is None:
            variance = t.ones(output_size)
        self.register_buffer( "variance", variance )
    def forward(self,x):
        r_out , (h_n,h_c)=self.rnn(x,None)
        #print("r_out :",r_out.shape)
        out=self.out(r_out[:,:,:])
        #print("out :",out.shape)
        return out

class DurationModel_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DurationModel_LSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.out=nn.Linear(hidden_size,1)
        # 3 layer nad relu best
    def forward(self,x):
        r_out , (h_n,h_c)=self.rnn(x,None)
        #print("r_out :",r_out.shape)
        out=self.out(r_out[:,:,:])
        #print("out :",out.shape)
        return out
