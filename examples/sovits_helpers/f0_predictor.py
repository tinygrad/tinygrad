import numpy as np
import parselmouth

# from https://github.com/svc-develop-team/so-vits-svc/
class PMF0Predictor:
    def __init__(self,hop_length=512,f0_min=50,f0_max=1100,sampling_rate=44100):
        self.hop_length = hop_length
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.sampling_rate = sampling_rate
        self.name = "pm"
    
    def interpolate_f0(self,f0):
        vuv_vector = np.zeros_like(f0, dtype=np.float32)
        vuv_vector[f0 > 0.0] = 1.0
        vuv_vector[f0 <= 0.0] = 0.0
    
        nzindex = np.nonzero(f0)[0]
        data = f0[nzindex]
        nzindex = nzindex.astype(np.float32)
        time_org = self.hop_length / self.sampling_rate * nzindex
        time_frame = np.arange(f0.shape[0]) * self.hop_length / self.sampling_rate

        if data.shape[0] <= 0:
            return np.zeros(f0.shape[0], dtype=np.float32),vuv_vector

        if data.shape[0] == 1:
            return np.ones(f0.shape[0], dtype=np.float32) * f0[0],vuv_vector

        f0 = np.interp(time_frame, time_org, data, left=data[0], right=data[-1])
        
        return f0,vuv_vector

    def compute_f0(self,wav,p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0,uv = self.interpolate_f0(f0)
        return f0

    def compute_f0_uv(self,wav,p_len=None):
        x = wav
        if p_len is None:
            p_len = x.shape[0]//self.hop_length
        else:
            assert abs(p_len-x.shape[0]//self.hop_length) < 4, "pad length error"
        time_step = self.hop_length / self.sampling_rate * 1000
        f0 = parselmouth.Sound(x, self.sampling_rate).to_pitch_ac(
            time_step=time_step / 1000, voicing_threshold=0.6,
            pitch_floor=self.f0_min, pitch_ceiling=self.f0_max).selected_array['frequency']

        pad_size=(p_len - len(f0) + 1) // 2
        if(pad_size>0 or p_len - len(f0) - pad_size>0):
            f0 = np.pad(f0,[[pad_size,p_len - len(f0) - pad_size]], mode='constant')
        f0,uv = self.interpolate_f0(f0)
        return f0,uv
