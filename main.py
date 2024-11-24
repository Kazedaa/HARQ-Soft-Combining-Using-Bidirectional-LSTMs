import torch
import komm

from DataGenerator import Data
from Autoencoder import AutoencoderModel
from utils import train_loop, test_loop
from plots import Plots

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--modulation', '-m', type = str, dest = "mod", help='Choose Modulation Technique["qam", "ask", "psk", "pam"]')
args = parser.parse_args()
if args.mod == "qam":
    mod = komm.QAModulation(16)
elif args.mod == "ask":
    mod = komm.ASKModulation(16)
elif args.mod == "psk":
    mod = komm.PSKModulation(16)
elif args.mod == "pam":
    mod = komm.PAModulation(16)
else:
    print('Choose Modulation Technique["qam", "ask", "psk", "pam"]')
    exit()

#Constants 
SNR = 2
NUM_SYMBOLS = 4
INPUT_SIZE = 2*NUM_SYMBOLS*8
HIDDEN_SIZE = 256
NUM_LAYERS = 1
NUM_EPOCHS = 20
BATCH_SIZE_TRAIN = 64
BATCH_SIZE_TEST = 8
NUM_MESSAGES_TRAIN = 2048
NUM_MESSAGES_TEST = 256

awgn = komm.AWGNChannel(snr=10**(SNR / 10),signal_power=5.0)

data = Data(modulator = mod, 
            noise_ch = awgn, 
            num_symbols=NUM_SYMBOLS, 
            num_messages_train=NUM_MESSAGES_TRAIN, 
            num_messages_test=NUM_MESSAGES_TEST,
            batch_size_train=BATCH_SIZE_TRAIN,
            batch_size_test=BATCH_SIZE_TEST
            )

train_dataloader = data.train_dataloader
test_dataloader = data.test_dataloader
train_bits = data.train_bits
test_bits = data.test_bits

model = AutoencoderModel(input_size = INPUT_SIZE, hidden_size= HIDDEN_SIZE, num_layers = NUM_LAYERS)

criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


train_losses = []
test_losses = []
train_bers = []
test_bers = []
for epoch in range(NUM_EPOCHS):
    train_l = 0.0
    test_l = 0.0
    train_b = 0.0
    test_b = 0.0
    count  = 0
    for train_batch,test_batch in iter(zip(train_dataloader,test_dataloader)):

        train_loss,train_ber = train_loop(batch = train_batch,criterion = criterion,optimizer = optimizer,model = model)
        test_loss,test_ber = test_loop(batch = test_batch,criterion = criterion,model = model)
        
        train_l += train_loss
        test_l += test_loss
        train_b += train_ber
        test_b += test_ber
        count += 1
        
    train_losses.append(train_l / count)
    test_losses.append(test_l / count)
    train_bers.append(train_b / count)
    test_bers.append(test_b / count)
        
    print(f"Epoch {epoch + 1} : Training Loss : {train_l / count} Training BER : {train_b / count}  Validation Loss : {test_l / count} Validation BER : {test_b / count}")


Plots(
    args.mod,
    model,
    NUM_MESSAGES_TRAIN,
    NUM_MESSAGES_TEST,
    train_dataloader,
    train_losses,
    test_losses,
    train_bers,
    test_bers,
    train_bits,
    test_bits
).plot()