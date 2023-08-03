import librosa
from tinygrad.tensor import Tensor
import torchaudio

# from https://github.com/svc-develop-team/so-vits-svc/
class Slicer:
    def __init__(self,
                 sr: int,
                 threshold: float = -40.,
                 min_length: int = 5000,
                 min_interval: int = 300,
                 hop_size: int = 20,
                 max_sil_kept: int = 5000):
        if not min_length >= min_interval >= hop_size:
            raise ValueError('The following condition must be satisfied: min_length >= min_interval >= hop_size')
        if not max_sil_kept >= hop_size:
            raise ValueError('The following condition must be satisfied: max_sil_kept >= hop_size')
        min_interval = sr * min_interval / 1000
        self.threshold = 10 ** (threshold / 20.)
        self.hop_size = round(sr * hop_size / 1000)
        self.win_size = min(round(min_interval), 4 * self.hop_size)
        self.min_length = round(sr * min_length / 1000 / self.hop_size)
        self.min_interval = round(min_interval / self.hop_size)
        self.max_sil_kept = round(sr * max_sil_kept / 1000 / self.hop_size)

    def _apply_slice(self, waveform, begin, end):
        if len(waveform.shape) > 1:
            return waveform[:, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)]
        else:
            return waveform[begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)]

    def slice(self, waveform):
        if len(waveform.shape) > 1:
            samples = librosa.to_mono(waveform)
        else:
            samples = waveform
        if samples.shape[0] <= self.min_length:
            return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
        rms_list = librosa.feature.rms(y=samples, frame_length=self.win_size, hop_length=self.hop_size).squeeze(0)
        sil_tags = []
        silence_start = None
        clip_start = 0
        for i, rms in enumerate(rms_list):
            # Keep looping while frame is silent.
            if rms < self.threshold:
                # Record start of silent frames.
                if silence_start is None:
                    silence_start = i
                continue
            # Keep looping while frame is not silent and silence start has not been recorded.
            if silence_start is None:
                continue
            # Clear recorded silence start if interval is not enough or clip is too short
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = i - silence_start >= self.min_interval and i - clip_start >= self.min_length
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue
            # Need slicing. Record the range of silent frames to be removed.
            if i - silence_start <= self.max_sil_kept:
                pos = rms_list[silence_start: i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                pos = rms_list[i - self.max_sil_kept: silence_start + self.max_sil_kept + 1].argmin()
                pos += i - self.max_sil_kept
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                pos_l = rms_list[silence_start: silence_start + self.max_sil_kept + 1].argmin() + silence_start
                pos_r = rms_list[i - self.max_sil_kept: i + 1].argmin() + i - self.max_sil_kept
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None
        # Deal with trailing silence.
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start: silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))
        # Apply and return slices.
        if len(sil_tags) == 0:
            return {"0": {"slice": False, "split_time": f"0,{len(waveform)}"}}
        else:
            chunks = []
            if sil_tags[0][0]:
                chunks.append(
                    {"slice": False, "split_time": f"0,{min(waveform.shape[0], sil_tags[0][0] * self.hop_size)}"})
            for i in range(0, len(sil_tags)):
                if i:
                    chunks.append({"slice": False,
                                   "split_time": f"{sil_tags[i - 1][1] * self.hop_size},{min(waveform.shape[0], sil_tags[i][0] * self.hop_size)}"})
                chunks.append({"slice": True,
                               "split_time": f"{sil_tags[i][0] * self.hop_size},{min(waveform.shape[0], sil_tags[i][1] * self.hop_size)}"})
            if sil_tags[-1][1] * self.hop_size < len(waveform):
                chunks.append({"slice": False, "split_time": f"{sil_tags[-1][1] * self.hop_size},{len(waveform)}"})
            chunk_dict = {}
            for i in range(len(chunks)):
                chunk_dict[str(i)] = chunks[i]
            return chunk_dict

def cut(audio_path, db_thresh=-30, min_len=5000):
    audio, sr = librosa.load(audio_path, sr=None)
    slicer = Slicer(
        sr=sr,
        threshold=db_thresh,
        min_length=min_len
    )
    chunks = slicer.slice(audio)
    return chunks

def chunks2audio(audio_path, chunks):
    chunks = dict(chunks)
    audio, sr = torchaudio.load(audio_path)
    audio = Tensor(audio.numpy())
    if len(audio.shape) == 2 and audio.shape[1] >= 2:
        audio = audio.mean(0).unsqueeze(0)
    audio = audio.numpy()[0]
    result = []
    for k, v in chunks.items():
        tag = v["split_time"].split(",")
        if tag[0] != tag[1]:
            result.append((v["slice"], audio[int(tag[0]):int(tag[1])]))
    return result, sr
