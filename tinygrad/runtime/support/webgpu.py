import os, platform, sysconfig
from tinygrad.helpers import findlib, OSX, WIN

if WIN: WEBGPU_PATH = findlib('webgpu_dawn', [os.path.join(sysconfig.get_paths()["purelib"], "pydawn")], "install it with `pip install dawn-python`")
elif OSX: WEBGPU_PATH = findlib('webgpu_dawn', [], "install it with `brew install wpmed92/dawn/dawn`")
else: WEBGPU_PATH = findlib('webgpu_dawn', [], "install it with `sudo curl -L https://github.com/wpmed92/pydawn/releases/download/v0.3.0/"+
                            f"libwebgpu_dawn_{platform.machine()}.so -o /usr/lib/libwebgpu_dawn.so`")
