from tinygrad.helpers import findlib, WIN, OSX

if WIN: LLVM_PATH = findlib('LLVM-C', [r'C:\Program Files\LLVM\bin'], "install it with `winget install LLVM.LLVM`")
elif OSX: LLVM_PATH = findlib('LLVM', ['/opt/homebrew/opt/llvm@20/lib', '/usr/local/opt/llvm@20'], "install it with `brew install llvm@20`")
else: LLVM_PATH = findlib('LLVM', [], "install it via your distro's package manager and ensure it's findable as 'LLVM'")
