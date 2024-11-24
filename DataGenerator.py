import random
import torch
from torch.utils.data import DataLoader
import komm

import warnings
warnings.filterwarnings("ignore")

class Data():
    def __init__(self, modulator, noise_ch, num_symbols, num_messages_train, num_messages_test, batch_size_train, batch_size_test):
        self.mod = modulator
        self.noise_ch = noise_ch
        self.num_symbols = num_symbols
        self.num_messages_train = num_messages_train
        self.num_messages_test = num_messages_test
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.train_dataloader = None
        self.test_dataloader = None
        self.train_bits = None
        self.test_bits = None
        self.data()

    def data(self):
        train_bytes = [[random.randint(0,255) for i in range(self.num_symbols)] for j in range(self.num_messages_train)]
        train_bits = [list(map(int,list(''.join(format(byte,'08b') for byte in train_bytes[j]))))for j in range(self.num_messages_train)]
        train_signal = [self.mod.modulate(train_bits[i]) for i in range(self.num_messages_train)]
        train_noisy_signal1 = [self.noise_ch(train_signal[i]) for i in range(self.num_messages_train)]
        train_noisy_signal2 = [self.noise_ch(train_signal[i]) for i in range(self.num_messages_train)]
        train_received_bits1 = [self.mod.demodulate(train_noisy_signal1[i]) for i in range(self.num_messages_train)]
        train_received_bits2 = [self.mod.demodulate(train_noisy_signal2[i]) for i in range(self.num_messages_train)]

        train_bits = torch.tensor(train_bits,dtype=torch.float32)
        train_received_bits1 = torch.tensor(train_received_bits1,dtype=torch.float32)
        train_received_bits2 = torch.tensor(train_received_bits2,dtype=torch.float32)
        train_data = torch.stack((train_bits,train_received_bits1,train_received_bits2),axis=1)

        test_bytes = [[random.randint(0,255) for i in range(self.num_symbols)] for j in range(self.num_messages_test)]
        test_bits = [list(map(int,list(''.join(format(byte,'08b') for byte in test_bytes[j]))))for j in range(self.num_messages_test)]
        test_signal = [self.mod.modulate(test_bits[i]) for i in range(self.num_messages_test)]
        test_noisy_signal1 = [self.noise_ch(test_signal[i]) for i in range(self.num_messages_test)]
        test_noisy_signal2 = [self.noise_ch(test_signal[i]) for i in range(self.num_messages_test)]
        test_received_bits1 = [self.mod.demodulate(test_noisy_signal1[i]) for i in range(self.num_messages_test)]
        test_received_bits2 = [self.mod.demodulate(test_noisy_signal2[i]) for i in range(self.num_messages_test)]

        test_bits = torch.tensor(test_bits,dtype=torch.float32)
        test_received_bits1 = torch.tensor(test_received_bits1,dtype=torch.float32)
        test_received_bits2 = torch.tensor(test_received_bits2,dtype=torch.float32)
        test_data = torch.stack((test_bits,test_received_bits1,test_received_bits2),axis=1)

        self.train_dataloader = DataLoader(train_data, batch_size=self.batch_size_train, shuffle=False, num_workers=4)
        self.test_dataloader = DataLoader(test_data, batch_size=self.batch_size_test, shuffle=False, num_workers=4)
        self.train_bits = train_bits
        self.test_bits = test_bits