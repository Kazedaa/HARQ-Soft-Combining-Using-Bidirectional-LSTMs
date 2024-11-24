import torch
import komm

def ber(pred,target):
    size = torch.numel(target)
    return torch.sum(pred!=target).item()/size

def train_loop(batch, criterion, optimizer, model):
    
    y,x1,x2 = torch.unbind(batch,dim=1) 
    x = torch.concat([x1,x2], axis =1)
    optimizer.zero_grad()        
    y_pred = model(x)       
    loss = criterion(y_pred, y)
    ber_value = ber(torch.round(y_pred.detach()),y)
    loss.backward()  
    optimizer.step()
    
    return loss.item(),ber_value

def test_loop(batch, criterion, model):
    
    with torch.no_grad():
        model.eval()
        y,x1,x2 = torch.unbind(batch, dim = 1)
        x = torch.concat([x1,x2], axis =1)
        y_pred = model(x)
        ber_value = ber(torch.round(y_pred.detach()),y)
        loss = criterion(y_pred, y)
        
        return loss.item(),ber_value
    
def data_gen(snr,data_bits,num_messages,mod):
    awgn = komm.AWGNChannel(snr=10**(snr / 10),signal_power=5.0)
    data_signal = [mod.modulate(data_bits[i]) for i in range(num_messages)]
    data_noisy_signal1 = [awgn(data_signal[i]) for i in range(num_messages)]
    data_noisy_signal2 = [awgn(data_signal[i]) for i in range(num_messages)]
    data_received_bits1 = [mod.demodulate(data_noisy_signal1[i]) for i in range(num_messages)]
    data_received_bits2 = [mod.demodulate(data_noisy_signal2[i]) for i in range(num_messages)]

    data_bits = torch.tensor(data_bits,dtype=torch.float32)
    data_received_bits1 = torch.tensor(data_received_bits1,dtype=torch.float32)
    data_received_bits2 = torch.tensor(data_received_bits2,dtype=torch.float32)
    return torch.stack((data_bits,data_received_bits1,data_received_bits2),axis=1)