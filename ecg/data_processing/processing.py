from tqdm import tqdm
import os
import torch
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import resample
import xmltodict as xtd
from argparse import ArgumentParser

class Normalisation(object):
    """
    Time series normalisation.
    """
    def __init__(self, mode="group_wise", groups=[3, 6, 12]) -> None:
        self.mode = mode
        self.groups = groups

    def __call__(self, sample) -> np.array:
        sample_dtype = sample.dtype

        if self.mode == "sample_wise":
            mean = np.mean(sample)
            var = np.var(sample)
        
        elif self.mode == "channel_wise":
            mean = np.mean(sample, axis=-1, keepdims=True)
            var = np.var(sample, axis=-1, keepdims=True)
        
        elif self.mode == "group_wise":
            mean = []
            var = []

            lower_bound = 0
            for idx in self.groups:
                mean_group = np.mean(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                mean_group = np.repeat(mean_group, repeats=int(idx-lower_bound), axis=0)
                var_group = np.var(sample[lower_bound:idx], axis=(0, 1), keepdims=True)
                var_group = np.repeat(var_group, repeats=int(idx-lower_bound), axis=0)
                lower_bound = idx

                mean.extend(mean_group)
                var.extend(var_group)

            mean = np.array(mean, dtype=sample_dtype)
            var = np.array(var, dtype=sample_dtype)

        normalised_sample = (sample - mean) / (var + 1.e-12)**.5

        return normalised_sample
    

def baseline_als(y, lam=1e8, p=1e-2, niter=10):
    """
    Asymmetric Least Squares Smoothing, i.e. asymmetric weighting of deviations to correct a baseline 
    while retaining the signal peak information.
    Refernce: Paul H. C. Eilers and Hans F.M. Boelens, Baseline Correction with Asymmetric Least Squares Smoothing (2005).
    """
    L = len(y)
    D = sparse.diags([1,-2,1], [0,-1,-2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

def resample_signal(signal: torch.Tensor, current_freq: float = 500, target_freq: float = 400):
    """
    Resample a tensor of shape (C, L) to the target frequency.

    Args:
        signal (torch.Tensor): A tensor of shape (C, L),
        where C is the number of channels and L is the length of the signal.
        current_freq (float): The current frequency of the signal in Hz.
        target_freq (float): The desired frequency in Hz.

    Returns:
        torch.Tensor: A tensor of shape (C, new_L) resampled to the target frequency.
    """
    num_channels, signal_length = signal.shape
    target_length = int(signal_length * target_freq / current_freq)
    resampled = np.array([resample(channel, target_length) for channel in signal.numpy()])
    signal = torch.tensor(resampled, dtype=signal.dtype)
    return signal


def process_ecg(sample): # expected input shape: (num_channels, sig_len)
    # remove nan
    sample = np.nan_to_num(sample)
    
    # clamp
    sample_std = sample.std()
    sample = np.clip(sample, a_min=-4*sample_std, a_max=4*sample_std)

    # remove baseline wander
    baselines = np.zeros_like(sample)
    for lead in range(sample.shape[0]):
        baselines[lead] = baseline_als(sample[lead], lam=1e7, p=0.3, niter=5)
    sample = sample - baselines

    # normalise 
    transform = Normalisation(mode="group_wise", groups=[3, 6, 12])
    sample = transform(sample)

    return sample

# ---- for reading xml files ----
def read_ecg(fname):
    lead_order = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # run example
    ecg, median_ecg, md = import_ecg(fname, lead_order)
    
    return ecg, median_ecg, md

def import_ecg(fname, lead_order):
    f = open(fname, "rt")
    raw_input = f.read()
    f.close()

    # extract ECG and metadata
    ecg, median_ecg, md = parse_xml(raw_input, lead_order)
    md['filename'] = fname

    return ecg.float().numpy(), median_ecg.float().numpy(), md

def parse_xml(input_data, lead_order):
    """ Takes input as raw xml data read from file (Cardiosoft specification), returns list of 12 lead waveforms
    and metadata. """
    md = {}  # metadata dictionary
    data = xtd.parse(input_data)['CardiologyXML']

    full_lead_nodes = [['StripData', 'WaveformData'], ['Strip', 'StripData', 'WaveformData']]
    median_lead_nodes = [['RestingECGMeasurements', 'MedianSamples', 'WaveformData'], []]
    full_leads = get_lead_data(data, full_lead_nodes, lead_order)
    median_leads = get_lead_data(data, median_lead_nodes, lead_order)

    try:
        md = get_metadata(data, md)
    except Exception:
        md = {}
        
    md['lead order'] = lead_order

    return torch.tensor(full_leads, dtype=torch.float32), torch.tensor(median_leads, dtype=torch.float32), md

def get_lead_data(data, nodes, lead_order):
    leads = [[] for i in range(12)]
    raw_lead_data = get_xml_node(data, nodes)

    # check number of leads
    if len(raw_lead_data) != 12:
        print('Warning: only {} leads found'.format(len(raw_lead_data)))

    # split data into 12 leads and convert string into individual values
    else:
        for i in range(len(raw_lead_data)):
            lead_n = lead_order.index(raw_lead_data[i]['@lead'].upper())
            lead_data_string = raw_lead_data[i]['#text']
            lead_vals = [int(x) for x in lead_data_string.split(",")]
            leads[lead_n] = lead_vals

    return leads

def get_xml_node(data, node_list):
    output = data
    try:
        for x in node_list[0]:
            output = output[x]
    except KeyError:
        output = data[:]
        try:
            for x in node_list[1]:
                output = output[x]
        except:
            raise ValueError('No lead data found!')
    return output

def get_metadata(data, md):
    md['sample rate'] = float(data['StripData']['SampleRate']['#text'])
    md['t scale'] = 1. / md['sample rate']
    md['v scale'] = float(data['StripData']['Resolution']['#text'])
    md['filter 50Hz'] = data['FilterSetting']['Filter50Hz']
    md['filter 60Hz'] = data['FilterSetting']['Filter60Hz']
    md['low pass'] = float(data['FilterSetting']['LowPass']['#text'])
    md['high pass'] = float(data['FilterSetting']['HighPass']['#text'])

    meas = data['RestingECGMeasurements']
    md['Heart rate'] = meas['VentricularRate']['#text'] + ' ' + meas['VentricularRate']['@units']
    md['P duration'] = meas['PDuration']['#text'] + ' ' + meas['PDuration']['@units']
    md['PR interval'] = meas['PQInterval']['#text'] + ' ' + meas['PQInterval']['@units']
    md['QRS duration'] = meas['QRSDuration']['#text'] + ' ' + meas['QRSDuration']['@units']
    md['QT interval'] = meas['QTInterval']['#text'] + ' ' + meas['QTInterval']['@units']
    md['QTc interval'] = meas['QTCInterval']['#text'] + ' ' + meas['QTCInterval']['@units']
    md['P axis'] = meas['PAxis']['#text'] + ' ' + meas['PAxis']['@units']
    md['R axis'] = meas['RAxis']['#text'] + ' ' + meas['RAxis']['@units']
    md['T axis'] = meas['TAxis']['#text'] + ' ' + meas['TAxis']['@units']
    return md

if __name__ == '__main__':
    argparse = ArgumentParser()
    argparse.add_argument('--input_dir', type=str)
    argparse.add_argument('--output_dir', type=str)
    args = argparse.parse_args()

    INPUT_DIR = args.input_dir
    OUTPUT_DIR = args.output_dir
    
    failed = []
    for f in tqdm(os.listdir(INPUT_DIR)):
        try:
            ecg, _, metadata = read_ecg(os.path.join(INPUT_DIR, f)) # ecg: (V, T)
            ecg = torch.from_numpy(ecg)
            proc_ecg = process_ecg(ecg)

            torch.save(torch.from_numpy(proc_ecg), os.path.join(OUTPUT_DIR, f'{f}.pt'))
        
        except:
            print(f'Failed to process {f}')
            failed.append(f)
