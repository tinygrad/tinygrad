#%%
from model import RNNT
from data import load_data, iterate
from tinygrad import Tensor

ci,maxX,maxY = load_data()

X,Y = next(iterate(ci))

rnnt = RNNT()
rnnt.load("rnnt_e_20")

#%%
res = rnnt.decode(Tensor(X[0]),Tensor(X[1]))

#%%

from reference_implementation.inference.speech_recognition.rnnt.pytorch import rnn