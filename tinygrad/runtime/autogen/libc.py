# mypy: ignore-errors
# -*- coding: utf-8 -*-
#
# TARGET arch is: []
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
import ctypes, ctypes.util, os


c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

class FunctionFactoryStub:
    def __getattr__(self, _):
      return ctypes.CFUNCTYPE(lambda y:y)

# libraries['libc'] explanation
# As you did not list (-l libraryname.so) a library that exports this function
# This is a non-working stub instead.
# You can either re-run clan2py with -l /path/to/library.so
# Or manually fix this by comment the ctypes.CDLL loading
_libraries = {}
_libraries['libc'] = ctypes.CDLL(ctypes.util.find_library('c')) #  ctypes.CDLL('libc')
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))





off_t = ctypes.c_int64
mode_t = ctypes.c_uint32
size_t = ctypes.c_uint64
__off_t = ctypes.c_int64
try:
    mmap = _libraries['libc'].mmap
    mmap.restype = ctypes.POINTER(None)
    mmap.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    munmap = _libraries['libc'].munmap
    munmap.restype = ctypes.c_int32
    munmap.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mprotect = _libraries['libc'].mprotect
    mprotect.restype = ctypes.c_int32
    mprotect.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    msync = _libraries['libc'].msync
    msync.restype = ctypes.c_int32
    msync.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    madvise = _libraries['libc'].madvise
    madvise.restype = ctypes.c_int32
    madvise.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    posix_madvise = _libraries['libc'].posix_madvise
    posix_madvise.restype = ctypes.c_int32
    posix_madvise.argtypes = [ctypes.POINTER(None), size_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    mlock = _libraries['libc'].mlock
    mlock.restype = ctypes.c_int32
    mlock.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    munlock = _libraries['libc'].munlock
    munlock.restype = ctypes.c_int32
    munlock.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    mlockall = _libraries['libc'].mlockall
    mlockall.restype = ctypes.c_int32
    mlockall.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    munlockall = _libraries['libc'].munlockall
    munlockall.restype = ctypes.c_int32
    munlockall.argtypes = []
except AttributeError:
    pass
try:
    mincore = _libraries['libc'].mincore
    mincore.restype = ctypes.c_int32
    mincore.argtypes = [ctypes.POINTER(None), size_t, ctypes.POINTER(ctypes.c_ubyte)]
except AttributeError:
    pass
try:
    shm_open = _libraries['libc'].shm_open
    shm_open.restype = ctypes.c_int32
    shm_open.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, mode_t]
except AttributeError:
    pass
try:
    shm_unlink = _libraries['libc'].shm_unlink
    shm_unlink.restype = ctypes.c_int32
    shm_unlink.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
ssize_t = ctypes.c_int64
gid_t = ctypes.c_uint32
uid_t = ctypes.c_uint32
useconds_t = ctypes.c_uint32
pid_t = ctypes.c_int32
intptr_t = ctypes.c_int64
socklen_t = ctypes.c_uint32
try:
    access = _libraries['libc'].access
    access.restype = ctypes.c_int32
    access.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    faccessat = _libraries['libc'].faccessat
    faccessat.restype = ctypes.c_int32
    faccessat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    lseek = _libraries['libc'].lseek
    lseek.restype = __off_t
    lseek.argtypes = [ctypes.c_int32, __off_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    close = _libraries['libc'].close
    close.restype = ctypes.c_int32
    close.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    closefrom = _libraries['libc'].closefrom
    closefrom.restype = None
    closefrom.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    read = _libraries['libc'].read
    read.restype = ssize_t
    read.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    write = _libraries['libc'].write
    write.restype = ssize_t
    write.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t]
except AttributeError:
    pass
try:
    pread = _libraries['libc'].pread
    pread.restype = ssize_t
    pread.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t, __off_t]
except AttributeError:
    pass
try:
    pwrite = _libraries['libc'].pwrite
    pwrite.restype = ssize_t
    pwrite.argtypes = [ctypes.c_int32, ctypes.POINTER(None), size_t, __off_t]
except AttributeError:
    pass
try:
    pipe = _libraries['libc'].pipe
    pipe.restype = ctypes.c_int32
    pipe.argtypes = [ctypes.c_int32 * 2]
except AttributeError:
    pass
try:
    alarm = _libraries['libc'].alarm
    alarm.restype = ctypes.c_uint32
    alarm.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
try:
    sleep = _libraries['libc'].sleep
    sleep.restype = ctypes.c_uint32
    sleep.argtypes = [ctypes.c_uint32]
except AttributeError:
    pass
__useconds_t = ctypes.c_uint32
try:
    ualarm = _libraries['libc'].ualarm
    ualarm.restype = __useconds_t
    ualarm.argtypes = [__useconds_t, __useconds_t]
except AttributeError:
    pass
try:
    usleep = _libraries['libc'].usleep
    usleep.restype = ctypes.c_int32
    usleep.argtypes = [__useconds_t]
except AttributeError:
    pass
try:
    pause = _libraries['libc'].pause
    pause.restype = ctypes.c_int32
    pause.argtypes = []
except AttributeError:
    pass
__uid_t = ctypes.c_uint32
__gid_t = ctypes.c_uint32
try:
    chown = _libraries['libc'].chown
    chown.restype = ctypes.c_int32
    chown.argtypes = [ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t]
except AttributeError:
    pass
try:
    fchown = _libraries['libc'].fchown
    fchown.restype = ctypes.c_int32
    fchown.argtypes = [ctypes.c_int32, __uid_t, __gid_t]
except AttributeError:
    pass
try:
    lchown = _libraries['libc'].lchown
    lchown.restype = ctypes.c_int32
    lchown.argtypes = [ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t]
except AttributeError:
    pass
try:
    fchownat = _libraries['libc'].fchownat
    fchownat.restype = ctypes.c_int32
    fchownat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), __uid_t, __gid_t, ctypes.c_int32]
except AttributeError:
    pass
try:
    chdir = _libraries['libc'].chdir
    chdir.restype = ctypes.c_int32
    chdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    fchdir = _libraries['libc'].fchdir
    fchdir.restype = ctypes.c_int32
    fchdir.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    getcwd = _libraries['libc'].getcwd
    getcwd.restype = ctypes.POINTER(ctypes.c_char)
    getcwd.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    getwd = _libraries['libc'].getwd
    getwd.restype = ctypes.POINTER(ctypes.c_char)
    getwd.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    dup = _libraries['libc'].dup
    dup.restype = ctypes.c_int32
    dup.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    dup2 = _libraries['libc'].dup2
    dup2.restype = ctypes.c_int32
    dup2.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
__environ = ctypes.POINTER(ctypes.POINTER(ctypes.c_char))() # Variable ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
try:
    execve = _libraries['libc'].execve
    execve.restype = ctypes.c_int32
    execve.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0, ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    fexecve = _libraries['libc'].fexecve
    fexecve.restype = ctypes.c_int32
    fexecve.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char) * 0, ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execv = _libraries['libc'].execv
    execv.restype = ctypes.c_int32
    execv.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execle = _libraries['libc'].execle
    execle.restype = ctypes.c_int32
    execle.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    execl = _libraries['libc'].execl
    execl.restype = ctypes.c_int32
    execl.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    execvp = _libraries['libc'].execvp
    execvp.restype = ctypes.c_int32
    execvp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char) * 0]
except AttributeError:
    pass
try:
    execlp = _libraries['libc'].execlp
    execlp.restype = ctypes.c_int32
    execlp.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    nice = _libraries['libc'].nice
    nice.restype = ctypes.c_int32
    nice.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    _exit = _libraries['libc']._exit
    _exit.restype = None
    _exit.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    pathconf = _libraries['libc'].pathconf
    pathconf.restype = ctypes.c_int64
    pathconf.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    fpathconf = _libraries['libc'].fpathconf
    fpathconf.restype = ctypes.c_int64
    fpathconf.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    sysconf = _libraries['libc'].sysconf
    sysconf.restype = ctypes.c_int64
    sysconf.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    confstr = _libraries['libc'].confstr
    confstr.restype = size_t
    confstr.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
__pid_t = ctypes.c_int32
try:
    getpid = _libraries['libc'].getpid
    getpid.restype = __pid_t
    getpid.argtypes = []
except AttributeError:
    pass
try:
    getppid = _libraries['libc'].getppid
    getppid.restype = __pid_t
    getppid.argtypes = []
except AttributeError:
    pass
try:
    getpgrp = _libraries['libc'].getpgrp
    getpgrp.restype = __pid_t
    getpgrp.argtypes = []
except AttributeError:
    pass
try:
    __getpgid = _libraries['libc'].__getpgid
    __getpgid.restype = __pid_t
    __getpgid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    getpgid = _libraries['libc'].getpgid
    getpgid.restype = __pid_t
    getpgid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    setpgid = _libraries['libc'].setpgid
    setpgid.restype = ctypes.c_int32
    setpgid.argtypes = [__pid_t, __pid_t]
except AttributeError:
    pass
try:
    setpgrp = _libraries['libc'].setpgrp
    setpgrp.restype = ctypes.c_int32
    setpgrp.argtypes = []
except AttributeError:
    pass
try:
    setsid = _libraries['libc'].setsid
    setsid.restype = __pid_t
    setsid.argtypes = []
except AttributeError:
    pass
try:
    getsid = _libraries['libc'].getsid
    getsid.restype = __pid_t
    getsid.argtypes = [__pid_t]
except AttributeError:
    pass
try:
    getuid = _libraries['libc'].getuid
    getuid.restype = __uid_t
    getuid.argtypes = []
except AttributeError:
    pass
try:
    geteuid = _libraries['libc'].geteuid
    geteuid.restype = __uid_t
    geteuid.argtypes = []
except AttributeError:
    pass
try:
    getgid = _libraries['libc'].getgid
    getgid.restype = __gid_t
    getgid.argtypes = []
except AttributeError:
    pass
try:
    getegid = _libraries['libc'].getegid
    getegid.restype = __gid_t
    getegid.argtypes = []
except AttributeError:
    pass
try:
    getgroups = _libraries['libc'].getgroups
    getgroups.restype = ctypes.c_int32
    getgroups.argtypes = [ctypes.c_int32, ctypes.c_uint32 * 0]
except AttributeError:
    pass
try:
    setuid = _libraries['libc'].setuid
    setuid.restype = ctypes.c_int32
    setuid.argtypes = [__uid_t]
except AttributeError:
    pass
try:
    setreuid = _libraries['libc'].setreuid
    setreuid.restype = ctypes.c_int32
    setreuid.argtypes = [__uid_t, __uid_t]
except AttributeError:
    pass
try:
    seteuid = _libraries['libc'].seteuid
    seteuid.restype = ctypes.c_int32
    seteuid.argtypes = [__uid_t]
except AttributeError:
    pass
try:
    setgid = _libraries['libc'].setgid
    setgid.restype = ctypes.c_int32
    setgid.argtypes = [__gid_t]
except AttributeError:
    pass
try:
    setregid = _libraries['libc'].setregid
    setregid.restype = ctypes.c_int32
    setregid.argtypes = [__gid_t, __gid_t]
except AttributeError:
    pass
try:
    setegid = _libraries['libc'].setegid
    setegid.restype = ctypes.c_int32
    setegid.argtypes = [__gid_t]
except AttributeError:
    pass
try:
    fork = _libraries['libc'].fork
    fork.restype = __pid_t
    fork.argtypes = []
except AttributeError:
    pass
try:
    vfork = _libraries['libc'].vfork
    vfork.restype = ctypes.c_int32
    vfork.argtypes = []
except AttributeError:
    pass
try:
    ttyname = _libraries['libc'].ttyname
    ttyname.restype = ctypes.POINTER(ctypes.c_char)
    ttyname.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ttyname_r = _libraries['libc'].ttyname_r
    ttyname_r.restype = ctypes.c_int32
    ttyname_r.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    isatty = _libraries['libc'].isatty
    isatty.restype = ctypes.c_int32
    isatty.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    ttyslot = _libraries['libc'].ttyslot
    ttyslot.restype = ctypes.c_int32
    ttyslot.argtypes = []
except AttributeError:
    pass
try:
    link = _libraries['libc'].link
    link.restype = ctypes.c_int32
    link.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    linkat = _libraries['libc'].linkat
    linkat.restype = ctypes.c_int32
    linkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    symlink = _libraries['libc'].symlink
    symlink.restype = ctypes.c_int32
    symlink.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    readlink = _libraries['libc'].readlink
    readlink.restype = ssize_t
    readlink.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    symlinkat = _libraries['libc'].symlinkat
    symlinkat.restype = ctypes.c_int32
    symlinkat.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    readlinkat = _libraries['libc'].readlinkat
    readlinkat.restype = ssize_t
    readlinkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    unlink = _libraries['libc'].unlink
    unlink.restype = ctypes.c_int32
    unlink.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    unlinkat = _libraries['libc'].unlinkat
    unlinkat.restype = ctypes.c_int32
    unlinkat.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), ctypes.c_int32]
except AttributeError:
    pass
try:
    rmdir = _libraries['libc'].rmdir
    rmdir.restype = ctypes.c_int32
    rmdir.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    tcgetpgrp = _libraries['libc'].tcgetpgrp
    tcgetpgrp.restype = __pid_t
    tcgetpgrp.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    tcsetpgrp = _libraries['libc'].tcsetpgrp
    tcsetpgrp.restype = ctypes.c_int32
    tcsetpgrp.argtypes = [ctypes.c_int32, __pid_t]
except AttributeError:
    pass
try:
    getlogin = _libraries['libc'].getlogin
    getlogin.restype = ctypes.POINTER(ctypes.c_char)
    getlogin.argtypes = []
except AttributeError:
    pass
try:
    getlogin_r = _libraries['libc'].getlogin_r
    getlogin_r.restype = ctypes.c_int32
    getlogin_r.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    setlogin = _libraries['libc'].setlogin
    setlogin.restype = ctypes.c_int32
    setlogin.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    gethostname = _libraries['libc'].gethostname
    gethostname.restype = ctypes.c_int32
    gethostname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    sethostname = _libraries['libc'].sethostname
    sethostname.restype = ctypes.c_int32
    sethostname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    sethostid = _libraries['libc'].sethostid
    sethostid.restype = ctypes.c_int32
    sethostid.argtypes = [ctypes.c_int64]
except AttributeError:
    pass
try:
    getdomainname = _libraries['libc'].getdomainname
    getdomainname.restype = ctypes.c_int32
    getdomainname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    setdomainname = _libraries['libc'].setdomainname
    setdomainname.restype = ctypes.c_int32
    setdomainname.argtypes = [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError:
    pass
try:
    vhangup = _libraries['libc'].vhangup
    vhangup.restype = ctypes.c_int32
    vhangup.argtypes = []
except AttributeError:
    pass
try:
    revoke = _libraries['libc'].revoke
    revoke.restype = ctypes.c_int32
    revoke.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    profil = _libraries['libc'].profil
    profil.restype = ctypes.c_int32
    profil.argtypes = [ctypes.POINTER(ctypes.c_uint16), size_t, size_t, ctypes.c_uint32]
except AttributeError:
    pass
try:
    acct = _libraries['libc'].acct
    acct.restype = ctypes.c_int32
    acct.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getusershell = _libraries['libc'].getusershell
    getusershell.restype = ctypes.POINTER(ctypes.c_char)
    getusershell.argtypes = []
except AttributeError:
    pass
try:
    endusershell = _libraries['libc'].endusershell
    endusershell.restype = None
    endusershell.argtypes = []
except AttributeError:
    pass
try:
    setusershell = _libraries['libc'].setusershell
    setusershell.restype = None
    setusershell.argtypes = []
except AttributeError:
    pass
try:
    daemon = _libraries['libc'].daemon
    daemon.restype = ctypes.c_int32
    daemon.argtypes = [ctypes.c_int32, ctypes.c_int32]
except AttributeError:
    pass
try:
    chroot = _libraries['libc'].chroot
    chroot.restype = ctypes.c_int32
    chroot.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getpass = _libraries['libc'].getpass
    getpass.restype = ctypes.POINTER(ctypes.c_char)
    getpass.argtypes = [ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    fsync = _libraries['libc'].fsync
    fsync.restype = ctypes.c_int32
    fsync.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    gethostid = _libraries['libc'].gethostid
    gethostid.restype = ctypes.c_int64
    gethostid.argtypes = []
except AttributeError:
    pass
try:
    sync = _libraries['libc'].sync
    sync.restype = None
    sync.argtypes = []
except AttributeError:
    pass
try:
    getpagesize = _libraries['libc'].getpagesize
    getpagesize.restype = ctypes.c_int32
    getpagesize.argtypes = []
except AttributeError:
    pass
try:
    getdtablesize = _libraries['libc'].getdtablesize
    getdtablesize.restype = ctypes.c_int32
    getdtablesize.argtypes = []
except AttributeError:
    pass
try:
    truncate = _libraries['libc'].truncate
    truncate.restype = ctypes.c_int32
    truncate.argtypes = [ctypes.POINTER(ctypes.c_char), __off_t]
except AttributeError:
    pass
try:
    ftruncate = _libraries['libc'].ftruncate
    ftruncate.restype = ctypes.c_int32
    ftruncate.argtypes = [ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    brk = _libraries['libc'].brk
    brk.restype = ctypes.c_int32
    brk.argtypes = [ctypes.POINTER(None)]
except AttributeError:
    pass
try:
    sbrk = _libraries['libc'].sbrk
    sbrk.restype = ctypes.POINTER(None)
    sbrk.argtypes = [intptr_t]
except AttributeError:
    pass
try:
    syscall = _libraries['libc'].syscall
    syscall.restype = ctypes.c_int64
    syscall.argtypes = [ctypes.c_int64]
except AttributeError:
    pass
try:
    lockf = _libraries['libc'].lockf
    lockf.restype = ctypes.c_int32
    lockf.argtypes = [ctypes.c_int32, ctypes.c_int32, __off_t]
except AttributeError:
    pass
try:
    fdatasync = _libraries['libc'].fdatasync
    fdatasync.restype = ctypes.c_int32
    fdatasync.argtypes = [ctypes.c_int32]
except AttributeError:
    pass
try:
    crypt = _libraries['libc'].crypt
    crypt.restype = ctypes.POINTER(ctypes.c_char)
    crypt.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError:
    pass
try:
    getentropy = _libraries['libc'].getentropy
    getentropy.restype = ctypes.c_int32
    getentropy.argtypes = [ctypes.POINTER(None), size_t]
except AttributeError:
    pass
__all__ = \
    ['__environ', '__getpgid', '__gid_t', '__off_t', '__pid_t',
    '__uid_t', '__useconds_t', '_exit', 'access', 'acct', 'alarm',
    'brk', 'chdir', 'chown', 'chroot', 'close', 'closefrom',
    'confstr', 'crypt', 'daemon', 'dup', 'dup2', 'endusershell',
    'execl', 'execle', 'execlp', 'execv', 'execve', 'execvp',
    'faccessat', 'fchdir', 'fchown', 'fchownat', 'fdatasync',
    'fexecve', 'fork', 'fpathconf', 'fsync', 'ftruncate', 'getcwd',
    'getdomainname', 'getdtablesize', 'getegid', 'getentropy',
    'geteuid', 'getgid', 'getgroups', 'gethostid', 'gethostname',
    'getlogin', 'getlogin_r', 'getpagesize', 'getpass', 'getpgid',
    'getpgrp', 'getpid', 'getppid', 'getsid', 'getuid',
    'getusershell', 'getwd', 'gid_t', 'intptr_t', 'isatty', 'lchown',
    'link', 'linkat', 'lockf', 'lseek', 'madvise', 'mincore', 'mlock',
    'mlockall', 'mmap', 'mode_t', 'mprotect', 'msync', 'munlock',
    'munlockall', 'munmap', 'nice', 'off_t', 'pathconf', 'pause',
    'pid_t', 'pipe', 'posix_madvise', 'pread', 'profil', 'pwrite',
    'read', 'readlink', 'readlinkat', 'revoke', 'rmdir', 'sbrk',
    'setdomainname', 'setegid', 'seteuid', 'setgid', 'sethostid',
    'sethostname', 'setlogin', 'setpgid', 'setpgrp', 'setregid',
    'setreuid', 'setsid', 'setuid', 'setusershell', 'shm_open',
    'shm_unlink', 'size_t', 'sleep', 'socklen_t', 'ssize_t',
    'symlink', 'symlinkat', 'sync', 'syscall', 'sysconf', 'tcgetpgrp',
    'tcsetpgrp', 'truncate', 'ttyname', 'ttyname_r', 'ttyslot',
    'ualarm', 'uid_t', 'unlink', 'unlinkat', 'useconds_t', 'usleep',
    'vfork', 'vhangup', 'write']
