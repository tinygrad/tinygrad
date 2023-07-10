from extra.datasets.librispeech import iterate

if __name__ == "__main__":
    next(iterate(mode="val"))
