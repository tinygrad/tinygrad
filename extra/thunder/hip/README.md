You'll need the HipKitten headers to run this kernel:

```
git clone https://github.com/Qazalin/HipKittens.git ~/HipKittens
```

Then, add the directory in `THUNDERKITTENS_ROOT=`, eg:

```
DEBUG=2 THUNDERKITTENS_ROOT=~/HipKittens python extra/thunder/hip/matmul.py
```
