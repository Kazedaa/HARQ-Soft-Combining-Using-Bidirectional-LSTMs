import os
import torch
import komm
import matplotlib.pyplot as plt
import numpy as np
from utils import ber, data_gen

class Plots():
    def __init__(self,
                  mod, 
                  model, 
                  num_messages_train, 
                  num_messages_test, 
                  train_dataloader, 
                  train_losses, 
                  test_losses, 
                  train_bers, 
                  test_bers, 
                  train_bits, 
                  test_bits,
                  ):
        self.mod = mod.upper()
        self.model = model
        self.NUM_MESSAGES_TRAIN = num_messages_train
        self.NUM_MESSAGES_TEST = num_messages_test
        self.train_dataloader = train_dataloader
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.train_bers = train_bers
        self.test_bers = test_bers
        self.train_bits = train_bits
        self.test_bits = test_bits


    def plot(self):
        os.makedirs(f"{self.mod}",exist_ok=True)
        self.loss_fig()
        self.ber_fig()
        self.noise_vs_frame_fig()
        self.snr_vs_ber(self.train_bits, self.NUM_MESSAGES_TRAIN, mode = "Train")
        self.snr_vs_ber(self.test_bits, self.NUM_MESSAGES_TEST, mode = "Test")

    def loss_fig(self):
        plt.figure()
        plt.plot([i for i in range(1,len(self.train_losses)+1)], self.train_losses, label = "Training Loss")
        plt.plot([i for i in range(1,len(self.test_losses)+1)] , self.test_losses, label = "Validation Loss", color = 'red',marker = 'o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self.mod}/loss")

    def ber_fig(self):
        plt.figure()
        plt.plot([i for i in range(1,len(self.train_bers)+1)], self.train_bers, label = "Training Loss")
        plt.plot([i for i in range(1,len(self.test_bers)+1)] , self.test_bers, label = "Validation Loss", color = 'red',marker = 'o')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"{self.mod}/ber")

    def noise1_vs_frame1(self,mod, noise_pred_train, noise_x1_train):
        plt.figure()
        plt.plot([i for i in range(1,len(noise_pred_train)+1)], noise_pred_train, label = "Predicted Noise", color = 'red', marker = 'o')
        plt.plot([i for i in range(1,len(noise_x1_train)+1)] , noise_x1_train, label = "First Part Noise")
        plt.xlabel("Frame")
        plt.ylabel("Noise1")
        plt.legend()
        plt.savefig(f"{mod}/noise1")

    def noise2_vs_frame2(self, mod, noise_pred_train, noise_x2_train):
        plt.figure()
        plt.plot([i for i in range(1,len(noise_pred_train)+1)], noise_pred_train, label = "Predicted Noise", color = 'red', marker = 'o')
        plt.plot([i for i in range(1,len(noise_x2_train)+1)] , noise_x2_train, label = "Second Part Noise")
        plt.xlabel("Frame")
        plt.ylabel("Noise2")
        plt.legend()
        plt.savefig(f"{mod}/noise2")

    def noise_vs_frame_fig(self):
        noise_pred_train = []
        noise_x1_train = []
        noise_x2_train = []
        with torch.no_grad():
            self.model.eval()
            for x in self.train_dataloader:
                y,x1,x2 = torch.unbind(x, dim = 1)
                x = torch.concat([x1,x2], axis =1)
                y_pred = self.model(x).round()
                noise_pred_train.append(ber(y_pred,y))
                noise_x1_train.append(ber(x1,y))
                noise_x2_train.append(ber(x2,y))
        self.noise1_vs_frame1(self.mod, noise_pred_train, noise_x1_train)
        self.noise2_vs_frame2(self.mod, noise_pred_train, noise_x2_train)

    def snr_vs_ber(self, bits, num_messages, mode ):
        qam = komm.QAModulation(16)
        ask = komm.ASKModulation(16)
        psk = komm.PSKModulation(16)
        pam = komm.PAModulation(16)
        with torch.no_grad():
            self.model.eval()
            ber_QAM=[]
            ber_PAM=[]
            ber_ASK=[]
            ber_PSK=[]
            for snr in np.arange(-4,30,1):
                data_QAM=data_gen(snr,bits.detach().numpy(),num_messages,qam)
                data_QAM = torch.unbind(data_QAM,axis = 1)
                data_PAM=data_gen(snr,bits.detach().numpy(),num_messages,pam)
                data_PAM = torch.unbind(data_PAM,axis = 1)
                data_PSK=data_gen(snr,bits.detach().numpy(),num_messages,psk)
                data_PSK = torch.unbind(data_PSK,axis = 1)
                data_ASK=data_gen(snr,bits.detach().numpy(),num_messages,ask)
                data_ASK = torch.unbind(data_ASK,axis = 1)
                y_pred_QAM=self.model(torch.concat([data_QAM[1],data_QAM[2]],axis=1))
                y_pred_PAM=self.model(torch.concat([data_PAM[1],data_PAM[2]],axis=1))
                y_pred_PSK=self.model(torch.concat([data_PSK[1],data_PSK[2]],axis=1))
                y_pred_ASK=self.model(torch.concat([data_ASK[1],data_ASK[2]],axis=1))
                ber_PSK.append(ber(torch.round(y_pred_PSK.detach()),data_PSK[0]))
                ber_PAM.append(ber(torch.round(y_pred_PAM.detach()),data_PAM[0]))
                ber_ASK.append(ber(torch.round(y_pred_ASK.detach()),data_ASK[0]))
                ber_QAM.append(ber(torch.round(y_pred_QAM.detach()),data_QAM[0]))

        plt.figure()
        plt.plot(np.arange(-4,30,1),ber_QAM,label='QAM',color='red',marker='o')
        plt.plot(np.arange(-4,30,1),ber_PAM,label="PAM",color='blue',marker='s')
        plt.plot(np.arange(-4,30,1),ber_ASK,label="ASK",color='green',marker='^')
        plt.plot(np.arange(-4,30,1),ber_PSK,label="PSK",color='orange',marker='*')
        plt.yscale('logit')
        plt.xlabel("SNR (dB)")
        plt.ylabel(f"{mode}BER")
        plt.legend()
        plt.savefig(f"{self.mod}/{self.mod}_{mode}")
