from __future__ import division
import numpy as np
import scipy.interpolate 
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import math
import os

carrier = 64
num_symbol = 50
CP = carrier//4
Pilot = 8
bitNum = 2
SNRdb = 20

allCarriers = np.arange(carrier)  # indices of all subcarriers ([0, 1, ... K-1])
print(allCarriers)
pilotCarriers = allCarriers[::carrier//Pilot] # Pilots is every (K/P)th carrier.
print(pilotCarriers)
dataCarriers = np.delete(allCarriers, pilotCarriers)
print(dataCarriers)
payloadBits_per_OFDM = len(dataCarriers) * bitNum
print(payloadBits_per_OFDM)

mapping_table = {
    (0,0) : -1-1j,
    (0,1) : -1+1j,
    (1,0) : 1-1j,
    (1,1) : 1+1j,
}

demapping_table = {v : k for k, v in mapping_table.items()}

def Modulation(bits):
    bit_r = bits.reshape((int(len(bits)/bitNum), bitNum))
    return (2*bit_r[:,0]-1)+1j*(2*bit_r[:,1]-1)                                    # This is just for QAM modulation

def IDFT(OFDM_data):
    return np.fft.ifft(OFDM_data)

def addCP(OFDM_time):
    cp = OFDM_time[-CP:]               # take the last CP samples ...
    return np.hstack([cp, OFDM_time])  # ... and add them to the beginning


def channel(signal, channelResponse, SNRdb):
    convolved = np.convolve(signal, channelResponse)
    signal_power = np.mean(abs(convolved ** 2))
    sigma2 = signal_power * 10 ** (-SNRdb / 10)  # calculate noise power based on signal power and SNR

    # Generate complex noise with given variance
    noise = np.sqrt(sigma2 / 2) * (np.random.randn(*convolved.shape) + 1j * np.random.randn(*convolved.shape))
    return convolved + noise

def removeCP(signal):
    return signal[CP:(CP+carrier)]
def ofdm_simulate(codeword, channelResponse,SNRdb):
    bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))

    OFDM_data = np.zeros(carrier, dtype=complex)
    print(OFDM_data)
    OFDM_data[pilotCarriers] = pilotValue
    print(OFDM_data)

    print(bits)

    QAMbits = Modulation(bits)

    print(QAMbits)
    OFDM_data[dataCarriers] = QAMbits

    OFDM_time = IDFT(OFDM_data)

    print(OFDM_time)

    OFDM_withCP = addCP(OFDM_time)

    print(OFDM_withCP)

    OFDM_TX = OFDM_withCP

    OFDM_RX = channel(OFDM_TX, channelResponse, SNRdb)

    print(OFDM_RX)

    OFDM_RX_noCP = removeCP(OFDM_RX)

    print("------------------------------------------------------------------------")
    print(OFDM_RX_noCP)

    # ----- target inputs ---
    symbol = np.zeros(carrier, dtype=complex)
    codeword_qam = Modulation(codeword)

    symbol = codeword_qam

    OFDM_data_codeword = symbol
    OFDM_time_codeword = IDFT(OFDM_data_codeword)
    OFDM_withCP_cordword = addCP(OFDM_time_codeword)

    OFDM_RX_codeword = channel(OFDM_withCP_cordword, channelResponse, SNRdb)
    OFDM_RX_noCP_codeword = removeCP(OFDM_RX_codeword)

    print("------------------------------------------------------------------------")
    print(OFDM_RX_noCP_codeword)

    return np.concatenate((np.concatenate((np.real(OFDM_RX_noCP), np.imag(OFDM_RX_noCP))),
                           np.concatenate((np.real(OFDM_RX_noCP_codeword), np.imag(OFDM_RX_noCP_codeword))))), abs(
        channelResponse)  # sparse_mask

Pilot_file_name = 'Pilot_'+str(Pilot)
if os.path.isfile(Pilot_file_name):
    print ('Load Training Pilots txt')
    # load file
    bits = np.loadtxt(Pilot_file_name, delimiter=',')
else:
    # write file
    bits = np.random.binomial(n=1, p=0.5, size=(K*mu, ))
    np.savetxt(Pilot_file_name, bits, delimiter=',')


pilotValue = Modulation(bits)


H_folder_train = 'H_dataset_fewer/'
channel_response_set_train = []
train_idx_low = 1
train_idx_high = 21
for train_idx in range(train_idx_low, train_idx_high):
    print("Processing the ", train_idx, "th document")
    H_file = H_folder_train + str(train_idx) + '.txt'
    with open(H_file) as f:
        for line in f:
            numbers_str = line.split()
            numbers_float = [float(x) for x in numbers_str]
            h_response = np.asarray(numbers_float[0:int(len(numbers_float) / 2)]) + 1j * np.asarray(
                numbers_float[int(len(numbers_float) / 2):len(numbers_float)])
            channel_response_set_train.append(h_response)


input_samples = []
input_labels = []
bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM, ))
channel_response = channel_response_set_train[np.random.randint(0,len(channel_response_set_train))]
signal_output, para = ofdm_simulate(bits,channel_response,SNRdb)
print("signal_output is: " + str(signal_output))
print("para is: " + str(para))
input_labels.append(bits[16:32])
input_samples.append(signal_output)
print(input_labels)
print(input_samples)