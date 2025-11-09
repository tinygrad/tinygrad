# mypy: ignore-errors
import ctypes
from tinygrad.helpers import Struct, CEnum, _IO, _IOW, _IOR, _IOWR, unwrap
from ctypes.util import find_library
from tinygrad.runtime.support import objc
def dll():
  try: return ctypes.CDLL(unwrap(find_library('System')))
  except: pass
  return None
dll = dll()

# int __darwin_check_fd_set_overflow(int, const void *, int) __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0)))
try: (__darwin_check_fd_set_overflow:=dll.__darwin_check_fd_set_overflow).restype, __darwin_check_fd_set_overflow.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
except AttributeError: pass

# __attribute__((always_inline)) inline int __darwin_check_fd_set(int _a, const void *_b)
try: (__darwin_check_fd_set:=dll.__darwin_check_fd_set).restype, __darwin_check_fd_set.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p]
except AttributeError: pass

class struct_fd_set(Struct): pass
__int32_t = ctypes.c_int
struct_fd_set._fields_ = [
  ('fds_bits', (ctypes.c_int * 32)),
]
# __attribute__((always_inline)) inline int __darwin_fd_isset(int _fd, const struct fd_set *_p)
try: (__darwin_fd_isset:=dll.__darwin_fd_isset).restype, __darwin_fd_isset.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(struct_fd_set)]
except AttributeError: pass

# __attribute__((always_inline)) inline void __darwin_fd_set(int _fd, struct fd_set *const _p)
try: (__darwin_fd_set:=dll.__darwin_fd_set).restype, __darwin_fd_set.argtypes = None, [ctypes.c_int, ctypes.POINTER(struct_fd_set)]
except AttributeError: pass

# __attribute__((always_inline)) inline void __darwin_fd_clr(int _fd, struct fd_set *const _p)
try: (__darwin_fd_clr:=dll.__darwin_fd_clr).restype, __darwin_fd_clr.argtypes = None, [ctypes.c_int, ctypes.POINTER(struct_fd_set)]
except AttributeError: pass

size_t = ctypes.c_ulong
# void *memchr(const void *__s, int __c, size_t __n)
try: (memchr:=dll.memchr).restype, memchr.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int, size_t]
except AttributeError: pass

# int memcmp(const void *__s1, const void *__s2, size_t __n)
try: (memcmp:=dll.memcmp).restype, memcmp.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void *memcpy(void *__dst, const void *__src, size_t __n)
try: (memcpy:=dll.memcpy).restype, memcpy.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void *memmove(void *__dst, const void *__src, size_t __len)
try: (memmove:=dll.memmove).restype, memmove.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void *memset(void *__b, int __c, size_t __len)
try: (memset:=dll.memset).restype, memset.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_int, size_t]
except AttributeError: pass

# char *strcat(char *__s1, const char *__s2)
try: (strcat:=dll.strcat).restype, strcat.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strchr(const char *__s, int __c)
try: (strchr:=dll.strchr).restype, strchr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int strcmp(const char *__s1, const char *__s2)
try: (strcmp:=dll.strcmp).restype, strcmp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int strcoll(const char *__s1, const char *__s2)
try: (strcoll:=dll.strcoll).restype, strcoll.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strcpy(char *__dst, const char *__src)
try: (strcpy:=dll.strcpy).restype, strcpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# unsigned long strcspn(const char *__s, const char *__charset)
try: (strcspn:=dll.strcspn).restype, strcspn.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strerror(int __errnum) asm("_strerror")
try: (strerror:=dll.strerror).restype, strerror.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int]
except AttributeError: pass

# unsigned long strlen(const char *__s)
try: (strlen:=dll.strlen).restype, strlen.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strncat(char *__s1, const char *__s2, size_t __n)
try: (strncat:=dll.strncat).restype, strncat.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int strncmp(const char *__s1, const char *__s2, size_t __n)
try: (strncmp:=dll.strncmp).restype, strncmp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# char *strncpy(char *__dst, const char *__src, size_t __n)
try: (strncpy:=dll.strncpy).restype, strncpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# char *strpbrk(const char *__s, const char *__charset)
try: (strpbrk:=dll.strpbrk).restype, strpbrk.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strrchr(const char *__s, int __c)
try: (strrchr:=dll.strrchr).restype, strrchr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# unsigned long strspn(const char *__s, const char *__charset)
try: (strspn:=dll.strspn).restype, strspn.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strstr(const char *__big, const char *__little)
try: (strstr:=dll.strstr).restype, strstr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strtok(char *__str, const char *__sep)
try: (strtok:=dll.strtok).restype, strtok.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# unsigned long strxfrm(char *__s1, const char *__s2, size_t __n)
try: (strxfrm:=dll.strxfrm).restype, strxfrm.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# char *strtok_r(char *__str, const char *__sep, char **__lasts)
try: (strtok_r:=dll.strtok_r).restype, strtok_r.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int strerror_r(int __errnum, char *__strerrbuf, size_t __buflen)
try: (strerror_r:=dll.strerror_r).restype, strerror_r.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# char *strdup(const char *__s1)
try: (strdup:=dll.strdup).restype, strdup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void *memccpy(void *__dst, const void *__src, int __c, size_t __n)
try: (memccpy:=dll.memccpy).restype, memccpy.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, size_t]
except AttributeError: pass

# char *stpcpy(char *__dst, const char *__src)
try: (stpcpy:=dll.stpcpy).restype, stpcpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *stpncpy(char *__dst, const char *__src, size_t __n) __attribute__((availability(macos, introduced=10.7)))
try: (stpncpy:=dll.stpncpy).restype, stpncpy.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# char *strndup(const char *__s1, size_t __n) __attribute__((availability(macos, introduced=10.7)))
try: (strndup:=dll.strndup).restype, strndup.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# size_t strnlen(const char *__s1, size_t __n) __attribute__((availability(macos, introduced=10.7)))
try: (strnlen:=dll.strnlen).restype, strnlen.argtypes = size_t, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# char *strsignal(int __sig)
try: (strsignal:=dll.strsignal).restype, strsignal.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int]
except AttributeError: pass

errno_t = ctypes.c_int
rsize_t = ctypes.c_ulong
# errno_t memset_s(void *__s, rsize_t __smax, int __c, rsize_t __n) __attribute__((availability(macos, introduced=10.9)))
try: (memset_s:=dll.memset_s).restype, memset_s.argtypes = errno_t, [ctypes.c_void_p, rsize_t, ctypes.c_int, rsize_t]
except AttributeError: pass

# void *memmem(const void *__big, size_t __big_len, const void *__little, size_t __little_len) __attribute__((availability(macos, introduced=10.7)))
try: (memmem:=dll.memmem).restype, memmem.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, ctypes.c_void_p, size_t]
except AttributeError: pass

# void memset_pattern4(void *__b, const void *__pattern4, size_t __len) __attribute__((availability(macos, introduced=10.5)))
try: (memset_pattern4:=dll.memset_pattern4).restype, memset_pattern4.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void memset_pattern8(void *__b, const void *__pattern8, size_t __len) __attribute__((availability(macos, introduced=10.5)))
try: (memset_pattern8:=dll.memset_pattern8).restype, memset_pattern8.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void memset_pattern16(void *__b, const void *__pattern16, size_t __len) __attribute__((availability(macos, introduced=10.5)))
try: (memset_pattern16:=dll.memset_pattern16).restype, memset_pattern16.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# char *strcasestr(const char *__big, const char *__little)
try: (strcasestr:=dll.strcasestr).restype, strcasestr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strnstr(const char *__big, const char *__little, size_t __len)
try: (strnstr:=dll.strnstr).restype, strnstr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# unsigned long strlcat(char *__dst, const char *__source, size_t __size)
try: (strlcat:=dll.strlcat).restype, strlcat.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# unsigned long strlcpy(char *__dst, const char *__source, size_t __size)
try: (strlcpy:=dll.strlcpy).restype, strlcpy.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# void strmode(int __mode, char *__bp)
try: (strmode:=dll.strmode).restype, strmode.argtypes = None, [ctypes.c_int, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *strsep(char **__stringp, const char *__delim)
try: (strsep:=dll.strsep).restype, strsep.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

ssize_t = ctypes.c_long
# void swab(const void *restrict, void *restrict, ssize_t)
try: (swab:=dll.swab).restype, swab.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, ssize_t]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.12.1))) __attribute__((availability(ios, introduced=10.1))) __attribute__((availability(tvos, introduced=10.0.1))) __attribute__((availability(watchos, introduced=3.1))) int timingsafe_bcmp(const void *__b1, const void *__b2, size_t __len)
try: (timingsafe_bcmp:=dll.timingsafe_bcmp).restype, timingsafe_bcmp.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) int strsignal_r(int __sig, char *__strsignalbuf, size_t __buflen)
try: (strsignal_r:=dll.strsignal_r).restype, strsignal_r.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int bcmp(const void *, const void *, size_t)
try: (bcmp:=dll.bcmp).restype, bcmp.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void bcopy(const void *, void *, size_t)
try: (bcopy:=dll.bcopy).restype, bcopy.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, size_t]
except AttributeError: pass

# void bzero(void *, size_t)
try: (bzero:=dll.bzero).restype, bzero.argtypes = None, [ctypes.c_void_p, size_t]
except AttributeError: pass

# char *index(const char *, int)
try: (index:=dll.index).restype, index.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# char *rindex(const char *, int)
try: (rindex:=dll.rindex).restype, rindex.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int ffs(int)
try: (ffs:=dll.ffs).restype, ffs.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int strcasecmp(const char *, const char *)
try: (strcasecmp:=dll.strcasecmp).restype, strcasecmp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int strncasecmp(const char *, const char *, size_t)
try: (strncasecmp:=dll.strncasecmp).restype, strncasecmp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int ffsl(long) __attribute__((availability(macos, introduced=10.5)))
try: (ffsl:=dll.ffsl).restype, ffsl.argtypes = ctypes.c_int, [ctypes.c_long]
except AttributeError: pass

# int ffsll(long long) __attribute__((availability(macos, introduced=10.9)))
try: (ffsll:=dll.ffsll).restype, ffsll.argtypes = ctypes.c_int, [ctypes.c_longlong]
except AttributeError: pass

# int fls(int) __attribute__((availability(macos, introduced=10.5)))
try: (fls:=dll.fls).restype, fls.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int flsl(long) __attribute__((availability(macos, introduced=10.5)))
try: (flsl:=dll.flsl).restype, flsl.argtypes = ctypes.c_int, [ctypes.c_long]
except AttributeError: pass

# int flsll(long long) __attribute__((availability(macos, introduced=10.9)))
try: (flsll:=dll.flsll).restype, flsll.argtypes = ctypes.c_int, [ctypes.c_longlong]
except AttributeError: pass

uint64_t = ctypes.c_ulonglong
# int getattrlistbulk(int, void *, void *, size_t, uint64_t) __attribute__((availability(macos, introduced=10.10)))
try: (getattrlistbulk:=dll.getattrlistbulk).restype, getattrlistbulk.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, size_t, uint64_t]
except AttributeError: pass

# int getattrlistat(int, const char *, void *, void *, size_t, unsigned long) __attribute__((availability(macos, introduced=10.10)))
try: (getattrlistat:=dll.getattrlistat).restype, getattrlistat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_ulong]
except AttributeError: pass

uint32_t = ctypes.c_uint
# int setattrlistat(int, const char *, void *, void *, size_t, uint32_t) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, introduced=11.0))) __attribute__((availability(tvos, introduced=11.0))) __attribute__((availability(watchos, introduced=4.0)))
try: (setattrlistat:=dll.setattrlistat).restype, setattrlistat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_void_p, size_t, uint32_t]
except AttributeError: pass

# ssize_t freadlink(int, char *restrict, size_t) __attribute__((availability(macos, introduced=13.0))) __attribute__((availability(ios, introduced=16.0))) __attribute__((availability(tvos, introduced=16.0))) __attribute__((availability(watchos, introduced=9.0)))
try: (freadlink:=dll.freadlink).restype, freadlink.argtypes = ssize_t, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int faccessat(int, const char *, int, int) __attribute__((availability(macos, introduced=10.10)))
try: (faccessat:=dll.faccessat).restype, faccessat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

uid_t = ctypes.c_uint
gid_t = ctypes.c_uint
# int fchownat(int, const char *, uid_t, gid_t, int) __attribute__((availability(macos, introduced=10.10)))
try: (fchownat:=dll.fchownat).restype, fchownat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), uid_t, gid_t, ctypes.c_int]
except AttributeError: pass

# int linkat(int, const char *, int, const char *, int) __attribute__((availability(macos, introduced=10.10)))
try: (linkat:=dll.linkat).restype, linkat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# ssize_t readlinkat(int, const char *, char *, size_t) __attribute__((availability(macos, introduced=10.10)))
try: (readlinkat:=dll.readlinkat).restype, readlinkat.argtypes = ssize_t, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int symlinkat(const char *, int, const char *) __attribute__((availability(macos, introduced=10.10)))
try: (symlinkat:=dll.symlinkat).restype, symlinkat.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int unlinkat(int, const char *, int) __attribute__((availability(macos, introduced=10.10)))
try: (unlinkat:=dll.unlinkat).restype, unlinkat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# void _exit(int)
try: (_exit:=dll._exit).restype, _exit.argtypes = None, [ctypes.c_int]
except AttributeError: pass

# int access(const char *, int)
try: (access:=dll.access).restype, access.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# unsigned int alarm(unsigned int)
try: (alarm:=dll.alarm).restype, alarm.argtypes = ctypes.c_uint, [ctypes.c_uint]
except AttributeError: pass

# int chdir(const char *)
try: (chdir:=dll.chdir).restype, chdir.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int chown(const char *, uid_t, gid_t)
try: (chown:=dll.chown).restype, chown.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), uid_t, gid_t]
except AttributeError: pass

# int close(int) asm("_close")
try: (close:=dll.close).restype, close.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int dup(int)
try: (dup:=dll.dup).restype, dup.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int dup2(int, int)
try: (dup2:=dll.dup2).restype, dup2.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int execl(const char *__path, const char *__arg0, ...) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execl:=dll.execl).restype, execl.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int execle(const char *__path, const char *__arg0, ...) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execle:=dll.execle).restype, execle.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int execlp(const char *__file, const char *__arg0, ...) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execlp:=dll.execlp).restype, execlp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int execv(const char *__path, char *const *__argv) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execv:=dll.execv).restype, execv.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int execve(const char *__file, char *const *__argv, char *const *__envp) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execve:=dll.execve).restype, execve.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int execvp(const char *__file, char *const *__argv) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execvp:=dll.execvp).restype, execvp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

pid_t = ctypes.c_int
# pid_t fork(void) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (fork:=dll.fork).restype, fork.argtypes = pid_t, []
except AttributeError: pass

# long fpathconf(int, int)
try: (fpathconf:=dll.fpathconf).restype, fpathconf.argtypes = ctypes.c_long, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# char *getcwd(char *, size_t)
try: (getcwd:=dll.getcwd).restype, getcwd.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# gid_t getegid(void)
try: (getegid:=dll.getegid).restype, getegid.argtypes = gid_t, []
except AttributeError: pass

# uid_t geteuid(void)
try: (geteuid:=dll.geteuid).restype, geteuid.argtypes = uid_t, []
except AttributeError: pass

# gid_t getgid(void)
try: (getgid:=dll.getgid).restype, getgid.argtypes = gid_t, []
except AttributeError: pass

# int getgroups(int, gid_t[])
try: (getgroups:=dll.getgroups).restype, getgroups.argtypes = ctypes.c_int, [ctypes.c_int, (gid_t * 0)]
except AttributeError: pass

# char *getlogin(void)
try: (getlogin:=dll.getlogin).restype, getlogin.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# pid_t getpgrp(void)
try: (getpgrp:=dll.getpgrp).restype, getpgrp.argtypes = pid_t, []
except AttributeError: pass

# pid_t getpid(void)
try: (getpid:=dll.getpid).restype, getpid.argtypes = pid_t, []
except AttributeError: pass

# pid_t getppid(void)
try: (getppid:=dll.getppid).restype, getppid.argtypes = pid_t, []
except AttributeError: pass

# uid_t getuid(void)
try: (getuid:=dll.getuid).restype, getuid.argtypes = uid_t, []
except AttributeError: pass

# int isatty(int)
try: (isatty:=dll.isatty).restype, isatty.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int link(const char *, const char *)
try: (link:=dll.link).restype, link.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

off_t = ctypes.c_longlong
# off_t lseek(int, off_t, int)
try: (lseek:=dll.lseek).restype, lseek.argtypes = off_t, [ctypes.c_int, off_t, ctypes.c_int]
except AttributeError: pass

# long pathconf(const char *, int)
try: (pathconf:=dll.pathconf).restype, pathconf.argtypes = ctypes.c_long, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int pause(void) asm("_pause")
try: (pause:=dll.pause).restype, pause.argtypes = ctypes.c_int, []
except AttributeError: pass

# int pipe(int[2])
try: (pipe:=dll.pipe).restype, pipe.argtypes = ctypes.c_int, [(ctypes.c_int * 2)]
except AttributeError: pass

# ssize_t read(int, void *, size_t) asm("_read")
try: (read:=dll.read).restype, read.argtypes = ssize_t, [ctypes.c_int, ctypes.c_void_p, size_t]
except AttributeError: pass

# int rmdir(const char *)
try: (rmdir:=dll.rmdir).restype, rmdir.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int setgid(gid_t)
try: (setgid:=dll.setgid).restype, setgid.argtypes = ctypes.c_int, [gid_t]
except AttributeError: pass

# int setpgid(pid_t, pid_t)
try: (setpgid:=dll.setpgid).restype, setpgid.argtypes = ctypes.c_int, [pid_t, pid_t]
except AttributeError: pass

# pid_t setsid(void)
try: (setsid:=dll.setsid).restype, setsid.argtypes = pid_t, []
except AttributeError: pass

# int setuid(uid_t)
try: (setuid:=dll.setuid).restype, setuid.argtypes = ctypes.c_int, [uid_t]
except AttributeError: pass

# unsigned int sleep(unsigned int) asm("_sleep")
try: (sleep:=dll.sleep).restype, sleep.argtypes = ctypes.c_uint, [ctypes.c_uint]
except AttributeError: pass

# long sysconf(int)
try: (sysconf:=dll.sysconf).restype, sysconf.argtypes = ctypes.c_long, [ctypes.c_int]
except AttributeError: pass

# pid_t tcgetpgrp(int)
try: (tcgetpgrp:=dll.tcgetpgrp).restype, tcgetpgrp.argtypes = pid_t, [ctypes.c_int]
except AttributeError: pass

# int tcsetpgrp(int, pid_t)
try: (tcsetpgrp:=dll.tcsetpgrp).restype, tcsetpgrp.argtypes = ctypes.c_int, [ctypes.c_int, pid_t]
except AttributeError: pass

# char *ttyname(int)
try: (ttyname:=dll.ttyname).restype, ttyname.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int]
except AttributeError: pass

# int ttyname_r(int, char *, size_t) asm("_ttyname_r")
try: (ttyname_r:=dll.ttyname_r).restype, ttyname_r.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int unlink(const char *)
try: (unlink:=dll.unlink).restype, unlink.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# ssize_t write(int __fd, const void *__buf, size_t __nbyte) asm("_write")
try: (write:=dll.write).restype, write.argtypes = ssize_t, [ctypes.c_int, ctypes.c_void_p, size_t]
except AttributeError: pass

# size_t confstr(int, char *, size_t) asm("_confstr")
try: (confstr:=dll.confstr).restype, confstr.argtypes = size_t, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int getopt(int, char *const[], const char *) asm("_getopt")
try: (getopt:=dll.getopt).restype, getopt.argtypes = ctypes.c_int, [ctypes.c_int, (ctypes.POINTER(ctypes.c_char) * 0), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *ctermid(char *)
try: (ctermid:=dll.ctermid).restype, ctermid.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((deprecated(""))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) void *brk(const void *)
try: (brk:=dll.brk).restype, brk.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

# int chroot(const char *)
try: (chroot:=dll.chroot).restype, chroot.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *crypt(const char *, const char *)
try: (crypt:=dll.crypt).restype, crypt.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void encrypt(char *, int) asm("_encrypt")
try: (encrypt:=dll.encrypt).restype, encrypt.argtypes = None, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int fchdir(int)
try: (fchdir:=dll.fchdir).restype, fchdir.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# long gethostid(void)
try: (gethostid:=dll.gethostid).restype, gethostid.argtypes = ctypes.c_long, []
except AttributeError: pass

# pid_t getpgid(pid_t)
try: (getpgid:=dll.getpgid).restype, getpgid.argtypes = pid_t, [pid_t]
except AttributeError: pass

# pid_t getsid(pid_t)
try: (getsid:=dll.getsid).restype, getsid.argtypes = pid_t, [pid_t]
except AttributeError: pass

# int getdtablesize(void)
try: (getdtablesize:=dll.getdtablesize).restype, getdtablesize.argtypes = ctypes.c_int, []
except AttributeError: pass

# int getpagesize(void) __attribute__((const))
try: (getpagesize:=dll.getpagesize).restype, getpagesize.argtypes = ctypes.c_int, []
except AttributeError: pass

# char *getpass(const char *)
try: (getpass:=dll.getpass).restype, getpass.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *getwd(char *)
try: (getwd:=dll.getwd).restype, getwd.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int lchown(const char *, uid_t, gid_t) asm("_lchown")
try: (lchown:=dll.lchown).restype, lchown.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), uid_t, gid_t]
except AttributeError: pass

# int lockf(int, int, off_t) asm("_lockf")
try: (lockf:=dll.lockf).restype, lockf.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int, off_t]
except AttributeError: pass

# int nice(int) asm("_nice")
try: (nice:=dll.nice).restype, nice.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# ssize_t pread(int __fd, void *__buf, size_t __nbyte, off_t __offset) asm("_pread")
try: (pread:=dll.pread).restype, pread.argtypes = ssize_t, [ctypes.c_int, ctypes.c_void_p, size_t, off_t]
except AttributeError: pass

# ssize_t pwrite(int __fd, const void *__buf, size_t __nbyte, off_t __offset) asm("_pwrite")
try: (pwrite:=dll.pwrite).restype, pwrite.argtypes = ssize_t, [ctypes.c_int, ctypes.c_void_p, size_t, off_t]
except AttributeError: pass

# __attribute__((deprecated(""))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) void *sbrk(int)
try: (sbrk:=dll.sbrk).restype, sbrk.argtypes = ctypes.c_void_p, [ctypes.c_int]
except AttributeError: pass

# pid_t setpgrp(void) asm("_setpgrp")
try: (setpgrp:=dll.setpgrp).restype, setpgrp.argtypes = pid_t, []
except AttributeError: pass

# int setregid(gid_t, gid_t) asm("_setregid")
try: (setregid:=dll.setregid).restype, setregid.argtypes = ctypes.c_int, [gid_t, gid_t]
except AttributeError: pass

# int setreuid(uid_t, uid_t) asm("_setreuid")
try: (setreuid:=dll.setreuid).restype, setreuid.argtypes = ctypes.c_int, [uid_t, uid_t]
except AttributeError: pass

# void swab(const void *restrict, void *restrict, ssize_t)
try: (swab:=dll.swab).restype, swab.argtypes = None, [ctypes.c_void_p, ctypes.c_void_p, ssize_t]
except AttributeError: pass

# void sync(void)
try: (sync:=dll.sync).restype, sync.argtypes = None, []
except AttributeError: pass

# int truncate(const char *, off_t)
try: (truncate:=dll.truncate).restype, truncate.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), off_t]
except AttributeError: pass

useconds_t = ctypes.c_uint
# useconds_t ualarm(useconds_t, useconds_t)
try: (ualarm:=dll.ualarm).restype, ualarm.argtypes = useconds_t, [useconds_t, useconds_t]
except AttributeError: pass

# int usleep(useconds_t) asm("_usleep")
try: (usleep:=dll.usleep).restype, usleep.argtypes = ctypes.c_int, [useconds_t]
except AttributeError: pass

# __attribute__((deprecated("Use posix_spawn or fork"))) int vfork(void) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (vfork:=dll.vfork).restype, vfork.argtypes = ctypes.c_int, []
except AttributeError: pass

# int fsync(int) asm("_fsync")
try: (fsync:=dll.fsync).restype, fsync.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int ftruncate(int, off_t)
try: (ftruncate:=dll.ftruncate).restype, ftruncate.argtypes = ctypes.c_int, [ctypes.c_int, off_t]
except AttributeError: pass

# int getlogin_r(char *, size_t)
try: (getlogin_r:=dll.getlogin_r).restype, getlogin_r.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int fchown(int, uid_t, gid_t)
try: (fchown:=dll.fchown).restype, fchown.argtypes = ctypes.c_int, [ctypes.c_int, uid_t, gid_t]
except AttributeError: pass

# int gethostname(char *, size_t)
try: (gethostname:=dll.gethostname).restype, gethostname.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# ssize_t readlink(const char *restrict, char *restrict, size_t)
try: (readlink:=dll.readlink).restype, readlink.argtypes = ssize_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int setegid(gid_t)
try: (setegid:=dll.setegid).restype, setegid.argtypes = ctypes.c_int, [gid_t]
except AttributeError: pass

# int seteuid(uid_t)
try: (seteuid:=dll.seteuid).restype, seteuid.argtypes = ctypes.c_int, [uid_t]
except AttributeError: pass

# int symlink(const char *, const char *)
try: (symlink:=dll.symlink).restype, symlink.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

fd_set = struct_fd_set
class struct_timespec(Struct): pass
__darwin_time_t = ctypes.c_long
struct_timespec._fields_ = [
  ('tv_sec', ctypes.c_long),
  ('tv_nsec', ctypes.c_long),
]
sigset_t = ctypes.c_uint
# int pselect(int, fd_set *restrict, fd_set *restrict, fd_set *restrict, const struct timespec *restrict, const sigset_t *restrict) asm("_pselect")
try: (pselect:=dll.pselect).restype, pselect.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timespec), ctypes.POINTER(sigset_t)]
except AttributeError: pass

class struct_timeval(Struct): pass
__darwin_suseconds_t = ctypes.c_int
struct_timeval._fields_ = [
  ('tv_sec', ctypes.c_long),
  ('tv_usec', ctypes.c_int),
]
# int select(int, fd_set *restrict, fd_set *restrict, fd_set *restrict, struct timeval *restrict) asm("_select")
try: (select:=dll.select).restype, select.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(fd_set), ctypes.POINTER(struct_timeval)]
except AttributeError: pass

# void _Exit(int)
try: (_Exit:=dll._Exit).restype, _Exit.argtypes = None, [ctypes.c_int]
except AttributeError: pass

class struct_accessx_descriptor(Struct): pass
struct_accessx_descriptor._fields_ = [
  ('ad_name_offset', ctypes.c_uint),
  ('ad_flags', ctypes.c_int),
  ('ad_pad', (ctypes.c_int * 2)),
]
# int accessx_np(const struct accessx_descriptor *, size_t, int *, uid_t)
try: (accessx_np:=dll.accessx_np).restype, accessx_np.argtypes = ctypes.c_int, [ctypes.POINTER(struct_accessx_descriptor), size_t, ctypes.POINTER(ctypes.c_int), uid_t]
except AttributeError: pass

# int acct(const char *)
try: (acct:=dll.acct).restype, acct.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int add_profil(char *, size_t, unsigned long, unsigned int) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (add_profil:=dll.add_profil).restype, add_profil.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_ulong, ctypes.c_uint]
except AttributeError: pass

# void endusershell(void)
try: (endusershell:=dll.endusershell).restype, endusershell.argtypes = None, []
except AttributeError: pass

# int execvP(const char *__file, const char *__searchpath, char *const *__argv) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (execvP:=dll.execvP).restype, execvP.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# char *fflagstostr(unsigned long)
try: (fflagstostr:=dll.fflagstostr).restype, fflagstostr.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_ulong]
except AttributeError: pass

# int getdomainname(char *, int)
try: (getdomainname:=dll.getdomainname).restype, getdomainname.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int getgrouplist(const char *, int, int *, int *)
try: (getgrouplist:=dll.getgrouplist).restype, getgrouplist.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

uuid_t = (ctypes.c_ubyte * 16)
# int gethostuuid(uuid_t, const struct timespec *) __attribute__((availability(macos, introduced=10.5))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable)))
try: (gethostuuid:=dll.gethostuuid).restype, gethostuuid.argtypes = ctypes.c_int, [uuid_t, ctypes.POINTER(struct_timespec)]
except AttributeError: pass

mode_t = ctypes.c_ushort
# mode_t getmode(const void *, mode_t)
try: (getmode:=dll.getmode).restype, getmode.argtypes = mode_t, [ctypes.c_void_p, mode_t]
except AttributeError: pass

# int getpeereid(int, uid_t *, gid_t *)
try: (getpeereid:=dll.getpeereid).restype, getpeereid.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(uid_t), ctypes.POINTER(gid_t)]
except AttributeError: pass

# int getsgroups_np(int *, uuid_t)
try: (getsgroups_np:=dll.getsgroups_np).restype, getsgroups_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_int), uuid_t]
except AttributeError: pass

# char *getusershell(void)
try: (getusershell:=dll.getusershell).restype, getusershell.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# int getwgroups_np(int *, uuid_t)
try: (getwgroups_np:=dll.getwgroups_np).restype, getwgroups_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_int), uuid_t]
except AttributeError: pass

# int initgroups(const char *, int)
try: (initgroups:=dll.initgroups).restype, initgroups.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int issetugid(void)
try: (issetugid:=dll.issetugid).restype, issetugid.argtypes = ctypes.c_int, []
except AttributeError: pass

# char *mkdtemp(char *)
try: (mkdtemp:=dll.mkdtemp).restype, mkdtemp.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

dev_t = ctypes.c_int
# int mknod(const char *, mode_t, dev_t)
try: (mknod:=dll.mknod).restype, mknod.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), mode_t, dev_t]
except AttributeError: pass

# int mkpath_np(const char *path, mode_t omode) __attribute__((availability(macos, introduced=10.8)))
try: (mkpath_np:=dll.mkpath_np).restype, mkpath_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), mode_t]
except AttributeError: pass

# int mkpathat_np(int dfd, const char *path, mode_t omode) __attribute__((availability(macos, introduced=10.12))) __attribute__((availability(ios, introduced=10.0))) __attribute__((availability(tvos, introduced=10.0))) __attribute__((availability(watchos, introduced=3.0)))
try: (mkpathat_np:=dll.mkpathat_np).restype, mkpathat_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), mode_t]
except AttributeError: pass

# int mkstemp(char *)
try: (mkstemp:=dll.mkstemp).restype, mkstemp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int mkstemps(char *, int)
try: (mkstemps:=dll.mkstemps).restype, mkstemps.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# char *mktemp(char *)
try: (mktemp:=dll.mktemp).restype, mktemp.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int mkostemp(char *path, int oflags) __attribute__((availability(macos, introduced=10.12))) __attribute__((availability(ios, introduced=10.0))) __attribute__((availability(tvos, introduced=10.0))) __attribute__((availability(watchos, introduced=3.0)))
try: (mkostemp:=dll.mkostemp).restype, mkostemp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int mkostemps(char *path, int slen, int oflags) __attribute__((availability(macos, introduced=10.12))) __attribute__((availability(ios, introduced=10.0))) __attribute__((availability(tvos, introduced=10.0))) __attribute__((availability(watchos, introduced=3.0)))
try: (mkostemps:=dll.mkostemps).restype, mkostemps.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int mkstemp_dprotected_np(char *path, int dpclass, int dpflags) __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, introduced=10.0))) __attribute__((availability(tvos, introduced=10.0))) __attribute__((availability(watchos, introduced=3.0)))
try: (mkstemp_dprotected_np:=dll.mkstemp_dprotected_np).restype, mkstemp_dprotected_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# char *mkdtempat_np(int dfd, char *path) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, introduced=11.0))) __attribute__((availability(tvos, introduced=11.0))) __attribute__((availability(watchos, introduced=4.0)))
try: (mkdtempat_np:=dll.mkdtempat_np).restype, mkdtempat_np.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int mkstempsat_np(int dfd, char *path, int slen) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, introduced=11.0))) __attribute__((availability(tvos, introduced=11.0))) __attribute__((availability(watchos, introduced=4.0)))
try: (mkstempsat_np:=dll.mkstempsat_np).restype, mkstempsat_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int mkostempsat_np(int dfd, char *path, int slen, int oflags) __attribute__((availability(macos, introduced=10.13))) __attribute__((availability(ios, introduced=11.0))) __attribute__((availability(tvos, introduced=11.0))) __attribute__((availability(watchos, introduced=4.0)))
try: (mkostempsat_np:=dll.mkostempsat_np).restype, mkostempsat_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int nfssvc(int, void *)
try: (nfssvc:=dll.nfssvc).restype, nfssvc.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p]
except AttributeError: pass

# int profil(char *, size_t, unsigned long, unsigned int)
try: (profil:=dll.profil).restype, profil.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), size_t, ctypes.c_ulong, ctypes.c_uint]
except AttributeError: pass

# __attribute__((deprecated("Use of per-thread security contexts is error-prone and discouraged."))) int pthread_setugid_np(uid_t, gid_t)
try: (pthread_setugid_np:=dll.pthread_setugid_np).restype, pthread_setugid_np.argtypes = ctypes.c_int, [uid_t, gid_t]
except AttributeError: pass

# int pthread_getugid_np(uid_t *, gid_t *)
try: (pthread_getugid_np:=dll.pthread_getugid_np).restype, pthread_getugid_np.argtypes = ctypes.c_int, [ctypes.POINTER(uid_t), ctypes.POINTER(gid_t)]
except AttributeError: pass

# int reboot(int)
try: (reboot:=dll.reboot).restype, reboot.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int revoke(const char *)
try: (revoke:=dll.revoke).restype, revoke.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((deprecated(""))) int rcmd(char **, int, const char *, const char *, const char *, int *)
try: (rcmd:=dll.rcmd).restype, rcmd.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# __attribute__((deprecated(""))) int rcmd_af(char **, int, const char *, const char *, const char *, int *, int)
try: (rcmd_af:=dll.rcmd_af).restype, rcmd_af.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
except AttributeError: pass

# __attribute__((deprecated(""))) int rresvport(int *)
try: (rresvport:=dll.rresvport).restype, rresvport.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# __attribute__((deprecated(""))) int rresvport_af(int *, int)
try: (rresvport_af:=dll.rresvport_af).restype, rresvport_af.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
except AttributeError: pass

# __attribute__((deprecated(""))) int iruserok(unsigned long, int, const char *, const char *)
try: (iruserok:=dll.iruserok).restype, iruserok.argtypes = ctypes.c_int, [ctypes.c_ulong, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((deprecated(""))) int iruserok_sa(const void *, int, int, const char *, const char *)
try: (iruserok_sa:=dll.iruserok_sa).restype, iruserok_sa.argtypes = ctypes.c_int, [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((deprecated(""))) int ruserok(const char *, int, const char *, const char *)
try: (ruserok:=dll.ruserok).restype, ruserok.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int setdomainname(const char *, int)
try: (setdomainname:=dll.setdomainname).restype, setdomainname.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int setgroups(int, const gid_t *)
try: (setgroups:=dll.setgroups).restype, setgroups.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(gid_t)]
except AttributeError: pass

# void sethostid(long)
try: (sethostid:=dll.sethostid).restype, sethostid.argtypes = None, [ctypes.c_long]
except AttributeError: pass

# int sethostname(const char *, int)
try: (sethostname:=dll.sethostname).restype, sethostname.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# void setkey(const char *) asm("_setkey")
try: (setkey:=dll.setkey).restype, setkey.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int setlogin(const char *)
try: (setlogin:=dll.setlogin).restype, setlogin.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void *setmode(const char *) asm("_setmode")
try: (setmode:=dll.setmode).restype, setmode.argtypes = ctypes.c_void_p, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int setrgid(gid_t)
try: (setrgid:=dll.setrgid).restype, setrgid.argtypes = ctypes.c_int, [gid_t]
except AttributeError: pass

# int setruid(uid_t)
try: (setruid:=dll.setruid).restype, setruid.argtypes = ctypes.c_int, [uid_t]
except AttributeError: pass

# int setsgroups_np(int, const uuid_t)
try: (setsgroups_np:=dll.setsgroups_np).restype, setsgroups_np.argtypes = ctypes.c_int, [ctypes.c_int, uuid_t]
except AttributeError: pass

# void setusershell(void)
try: (setusershell:=dll.setusershell).restype, setusershell.argtypes = None, []
except AttributeError: pass

# int setwgroups_np(int, const uuid_t)
try: (setwgroups_np:=dll.setwgroups_np).restype, setwgroups_np.argtypes = ctypes.c_int, [ctypes.c_int, uuid_t]
except AttributeError: pass

# int strtofflags(char **, unsigned long *, unsigned long *)
try: (strtofflags:=dll.strtofflags).restype, strtofflags.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(ctypes.c_ulong)]
except AttributeError: pass

# int swapon(const char *)
try: (swapon:=dll.swapon).restype, swapon.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int ttyslot(void)
try: (ttyslot:=dll.ttyslot).restype, ttyslot.argtypes = ctypes.c_int, []
except AttributeError: pass

# int undelete(const char *)
try: (undelete:=dll.undelete).restype, undelete.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int unwhiteout(const char *)
try: (unwhiteout:=dll.unwhiteout).restype, unwhiteout.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void *valloc(size_t)
try: (valloc:=dll.valloc).restype, valloc.argtypes = ctypes.c_void_p, [size_t]
except AttributeError: pass

# __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(ios, deprecated=10.0))) __attribute__((availability(macos, deprecated=10.12))) int syscall(int, ...)
try: (syscall:=dll.syscall).restype, syscall.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int getsubopt(char **, char *const *, char **)
try: (getsubopt:=dll.getsubopt).restype, getsubopt.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int fgetattrlist(int, void *, void *, size_t, unsigned int) __attribute__((availability(macos, introduced=10.6)))
try: (fgetattrlist:=dll.fgetattrlist).restype, fgetattrlist.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# int fsetattrlist(int, void *, void *, size_t, unsigned int) __attribute__((availability(macos, introduced=10.6)))
try: (fsetattrlist:=dll.fsetattrlist).restype, fsetattrlist.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# int getattrlist(const char *, void *, void *, size_t, unsigned int) asm("_getattrlist")
try: (getattrlist:=dll.getattrlist).restype, getattrlist.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# int setattrlist(const char *, void *, void *, size_t, unsigned int) asm("_setattrlist")
try: (setattrlist:=dll.setattrlist).restype, setattrlist.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.c_uint]
except AttributeError: pass

# int exchangedata(const char *, const char *, unsigned int) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (exchangedata:=dll.exchangedata).restype, exchangedata.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_uint]
except AttributeError: pass

# int getdirentriesattr(int, void *, void *, size_t, unsigned int *, unsigned int *, unsigned int *, unsigned int) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (getdirentriesattr:=dll.getdirentriesattr).restype, getdirentriesattr.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, size_t, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint), ctypes.c_uint]
except AttributeError: pass

class struct_fssearchblock(Struct): pass
class struct_searchstate(Struct): pass
# int searchfs(const char *, struct fssearchblock *, unsigned long *, unsigned int, unsigned int, struct searchstate *) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (searchfs:=dll.searchfs).restype, searchfs.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_fssearchblock), ctypes.POINTER(ctypes.c_ulong), ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(struct_searchstate)]
except AttributeError: pass

# int fsctl(const char *, unsigned long, void *, unsigned int)
try: (fsctl:=dll.fsctl).restype, fsctl.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_ulong, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# int ffsctl(int, unsigned long, void *, unsigned int) __attribute__((availability(macos, introduced=10.6)))
try: (ffsctl:=dll.ffsctl).restype, ffsctl.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_ulong, ctypes.c_void_p, ctypes.c_uint]
except AttributeError: pass

# int fsync_volume_np(int, int) __attribute__((availability(macos, introduced=10.8)))
try: (fsync_volume_np:=dll.fsync_volume_np).restype, fsync_volume_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int sync_volume_np(const char *, int) __attribute__((availability(macos, introduced=10.8)))
try: (sync_volume_np:=dll.sync_volume_np).restype, sync_volume_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int open(const char *, int, ...) asm("_open")
try: (open:=dll.open).restype, open.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int openat(int, const char *, int, ...) asm("_openat") __attribute__((availability(macos, introduced=10.10)))
try: (openat:=dll.openat).restype, openat.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int creat(const char *, mode_t) asm("_creat")
try: (creat:=dll.creat).restype, creat.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), mode_t]
except AttributeError: pass

# int fcntl(int, int, ...) asm("_fcntl")
try: (fcntl:=dll.fcntl).restype, fcntl.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

class struct__filesec(Struct): pass
filesec_t = ctypes.POINTER(struct__filesec)
# int openx_np(const char *, int, filesec_t)
try: (openx_np:=dll.openx_np).restype, openx_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, filesec_t]
except AttributeError: pass

# int open_dprotected_np(const char *, int, int, int, ...)
try: (open_dprotected_np:=dll.open_dprotected_np).restype, open_dprotected_np.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int openat_dprotected_np(int, const char *, int, int, int, ...)
try: (openat_dprotected_np:=dll.openat_dprotected_np).restype, openat_dprotected_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int openat_authenticated_np(int, const char *, int, int)
try: (openat_authenticated_np:=dll.openat_authenticated_np).restype, openat_authenticated_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int flock(int, int)
try: (flock:=dll.flock).restype, flock.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# filesec_t filesec_init(void)
try: (filesec_init:=dll.filesec_init).restype, filesec_init.argtypes = filesec_t, []
except AttributeError: pass

# filesec_t filesec_dup(filesec_t)
try: (filesec_dup:=dll.filesec_dup).restype, filesec_dup.argtypes = filesec_t, [filesec_t]
except AttributeError: pass

# void filesec_free(filesec_t)
try: (filesec_free:=dll.filesec_free).restype, filesec_free.argtypes = None, [filesec_t]
except AttributeError: pass

filesec_property_t = CEnum(ctypes.c_uint)
FILESEC_OWNER = filesec_property_t.define('FILESEC_OWNER', 1)
FILESEC_GROUP = filesec_property_t.define('FILESEC_GROUP', 2)
FILESEC_UUID = filesec_property_t.define('FILESEC_UUID', 3)
FILESEC_MODE = filesec_property_t.define('FILESEC_MODE', 4)
FILESEC_ACL = filesec_property_t.define('FILESEC_ACL', 5)
FILESEC_GRPUUID = filesec_property_t.define('FILESEC_GRPUUID', 6)
FILESEC_ACL_RAW = filesec_property_t.define('FILESEC_ACL_RAW', 100)
FILESEC_ACL_ALLOCSIZE = filesec_property_t.define('FILESEC_ACL_ALLOCSIZE', 101)

# int filesec_get_property(filesec_t, filesec_property_t, void *)
try: (filesec_get_property:=dll.filesec_get_property).restype, filesec_get_property.argtypes = ctypes.c_int, [filesec_t, filesec_property_t, ctypes.c_void_p]
except AttributeError: pass

# int filesec_query_property(filesec_t, filesec_property_t, int *)
try: (filesec_query_property:=dll.filesec_query_property).restype, filesec_query_property.argtypes = ctypes.c_int, [filesec_t, filesec_property_t, ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# int filesec_set_property(filesec_t, filesec_property_t, const void *)
try: (filesec_set_property:=dll.filesec_set_property).restype, filesec_set_property.argtypes = ctypes.c_int, [filesec_t, filesec_property_t, ctypes.c_void_p]
except AttributeError: pass

# int filesec_unset_property(filesec_t, filesec_property_t) __attribute__((availability(macos, introduced=10.6)))
try: (filesec_unset_property:=dll.filesec_unset_property).restype, filesec_unset_property.argtypes = ctypes.c_int, [filesec_t, filesec_property_t]
except AttributeError: pass

# void (*signal(int, void (*)(int)))(int)
try: (signal:=dll.signal).restype, signal.argtypes = ctypes.CFUNCTYPE(None, ctypes.c_int), [ctypes.c_int, ctypes.CFUNCTYPE(None, ctypes.c_int)]
except AttributeError: pass

id_t = ctypes.c_uint
# int getpriority(int, id_t)
try: (getpriority:=dll.getpriority).restype, getpriority.argtypes = ctypes.c_int, [ctypes.c_int, id_t]
except AttributeError: pass

# int getiopolicy_np(int, int) __attribute__((availability(macos, introduced=10.5)))
try: (getiopolicy_np:=dll.getiopolicy_np).restype, getiopolicy_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

class struct_rlimit(Struct): pass
rlim_t = ctypes.c_ulonglong
struct_rlimit._fields_ = [
  ('rlim_cur', rlim_t),
  ('rlim_max', rlim_t),
]
# int getrlimit(int, struct rlimit *) asm("_getrlimit")
try: (getrlimit:=dll.getrlimit).restype, getrlimit.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(struct_rlimit)]
except AttributeError: pass

class struct_rusage(Struct): pass
struct_rusage._fields_ = [
  ('ru_utime', struct_timeval),
  ('ru_stime', struct_timeval),
  ('ru_maxrss', ctypes.c_long),
  ('ru_ixrss', ctypes.c_long),
  ('ru_idrss', ctypes.c_long),
  ('ru_isrss', ctypes.c_long),
  ('ru_minflt', ctypes.c_long),
  ('ru_majflt', ctypes.c_long),
  ('ru_nswap', ctypes.c_long),
  ('ru_inblock', ctypes.c_long),
  ('ru_oublock', ctypes.c_long),
  ('ru_msgsnd', ctypes.c_long),
  ('ru_msgrcv', ctypes.c_long),
  ('ru_nsignals', ctypes.c_long),
  ('ru_nvcsw', ctypes.c_long),
  ('ru_nivcsw', ctypes.c_long),
]
# int getrusage(int, struct rusage *)
try: (getrusage:=dll.getrusage).restype, getrusage.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(struct_rusage)]
except AttributeError: pass

# int setpriority(int, id_t, int)
try: (setpriority:=dll.setpriority).restype, setpriority.argtypes = ctypes.c_int, [ctypes.c_int, id_t, ctypes.c_int]
except AttributeError: pass

# int setiopolicy_np(int, int, int) __attribute__((availability(macos, introduced=10.5)))
try: (setiopolicy_np:=dll.setiopolicy_np).restype, setiopolicy_np.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# int setrlimit(int, const struct rlimit *) asm("_setrlimit")
try: (setrlimit:=dll.setrlimit).restype, setrlimit.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(struct_rlimit)]
except AttributeError: pass

# pid_t wait(int *) asm("_wait")
try: (wait:=dll.wait).restype, wait.argtypes = pid_t, [ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# pid_t waitpid(pid_t, int *, int) asm("_waitpid")
try: (waitpid:=dll.waitpid).restype, waitpid.argtypes = pid_t, [pid_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
except AttributeError: pass

idtype_t = CEnum(ctypes.c_uint)
P_ALL = idtype_t.define('P_ALL', 0)
P_PID = idtype_t.define('P_PID', 1)
P_PGID = idtype_t.define('P_PGID', 2)

class struct___siginfo(Struct): pass
siginfo_t = struct___siginfo
class union_sigval(ctypes.Union): pass
union_sigval._fields_ = [
  ('sival_int', ctypes.c_int),
  ('sival_ptr', ctypes.c_void_p),
]
struct___siginfo._fields_ = [
  ('si_signo', ctypes.c_int),
  ('si_errno', ctypes.c_int),
  ('si_code', ctypes.c_int),
  ('si_pid', pid_t),
  ('si_uid', uid_t),
  ('si_status', ctypes.c_int),
  ('si_addr', ctypes.c_void_p),
  ('si_value', union_sigval),
  ('si_band', ctypes.c_long),
  ('__pad', (ctypes.c_ulong * 7)),
]
# int waitid(idtype_t, id_t, siginfo_t *, int) asm("_waitid")
try: (waitid:=dll.waitid).restype, waitid.argtypes = ctypes.c_int, [idtype_t, id_t, ctypes.POINTER(siginfo_t), ctypes.c_int]
except AttributeError: pass

# pid_t wait3(int *, int, struct rusage *)
try: (wait3:=dll.wait3).restype, wait3.argtypes = pid_t, [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(struct_rusage)]
except AttributeError: pass

# pid_t wait4(pid_t, int *, int, struct rusage *)
try: (wait4:=dll.wait4).restype, wait4.argtypes = pid_t, [pid_t, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(struct_rusage)]
except AttributeError: pass

# void *alloca(size_t)
try: (alloca:=dll.alloca).restype, alloca.argtypes = ctypes.c_void_p, [size_t]
except AttributeError: pass

malloc_type_id_t = ctypes.c_ulonglong
# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_malloc(size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(1)))
try: (malloc_type_malloc:=dll.malloc_type_malloc).restype, malloc_type_malloc.argtypes = ctypes.c_void_p, [size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_calloc(size_t count, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(1, 2)))
try: (malloc_type_calloc:=dll.malloc_type_calloc).restype, malloc_type_calloc.argtypes = ctypes.c_void_p, [size_t, size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void malloc_type_free(void *ptr, malloc_type_id_t type_id)
try: (malloc_type_free:=dll.malloc_type_free).restype, malloc_type_free.argtypes = None, [ctypes.c_void_p, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_realloc(void *ptr, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2)))
try: (malloc_type_realloc:=dll.malloc_type_realloc).restype, malloc_type_realloc.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_valloc(size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(1)))
try: (malloc_type_valloc:=dll.malloc_type_valloc).restype, malloc_type_valloc.argtypes = ctypes.c_void_p, [size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_aligned_alloc(size_t alignment, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2)))
try: (malloc_type_aligned_alloc:=dll.malloc_type_aligned_alloc).restype, malloc_type_aligned_alloc.argtypes = ctypes.c_void_p, [size_t, size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) int malloc_type_posix_memalign(void **memptr, size_t alignment, size_t size, malloc_type_id_t type_id)
try: (malloc_type_posix_memalign:=dll.malloc_type_posix_memalign).restype, malloc_type_posix_memalign.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), size_t, size_t, malloc_type_id_t]
except AttributeError: pass

class struct__malloc_zone_t(Struct): pass
malloc_zone_t = struct__malloc_zone_t
# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_zone_malloc(malloc_zone_t *zone, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2)))
try: (malloc_type_zone_malloc:=dll.malloc_type_zone_malloc).restype, malloc_type_zone_malloc.argtypes = ctypes.c_void_p, [ctypes.POINTER(malloc_zone_t), size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_zone_calloc(malloc_zone_t *zone, size_t count, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2, 3)))
try: (malloc_type_zone_calloc:=dll.malloc_type_zone_calloc).restype, malloc_type_zone_calloc.argtypes = ctypes.c_void_p, [ctypes.POINTER(malloc_zone_t), size_t, size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void malloc_type_zone_free(malloc_zone_t *zone, void *ptr, malloc_type_id_t type_id)
try: (malloc_type_zone_free:=dll.malloc_type_zone_free).restype, malloc_type_zone_free.argtypes = None, [ctypes.POINTER(malloc_zone_t), ctypes.c_void_p, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_zone_realloc(malloc_zone_t *zone, void *ptr, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(3)))
try: (malloc_type_zone_realloc:=dll.malloc_type_zone_realloc).restype, malloc_type_zone_realloc.argtypes = ctypes.c_void_p, [ctypes.POINTER(malloc_zone_t), ctypes.c_void_p, size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_zone_valloc(malloc_zone_t *zone, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2)))
try: (malloc_type_zone_valloc:=dll.malloc_type_zone_valloc).restype, malloc_type_zone_valloc.argtypes = ctypes.c_void_p, [ctypes.POINTER(malloc_zone_t), size_t, malloc_type_id_t]
except AttributeError: pass

# __attribute__((availability(macos, unavailable))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(xros, unavailable))) void *malloc_type_zone_memalign(malloc_zone_t *zone, size_t alignment, size_t size, malloc_type_id_t type_id) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(3)))
try: (malloc_type_zone_memalign:=dll.malloc_type_zone_memalign).restype, malloc_type_zone_memalign.argtypes = ctypes.c_void_p, [ctypes.POINTER(malloc_zone_t), size_t, size_t, malloc_type_id_t]
except AttributeError: pass

# void *malloc(size_t __size) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(1)))
try: (malloc:=dll.malloc).restype, malloc.argtypes = ctypes.c_void_p, [size_t]
except AttributeError: pass

# void *calloc(size_t __count, size_t __size) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(1, 2)))
try: (calloc:=dll.calloc).restype, calloc.argtypes = ctypes.c_void_p, [size_t, size_t]
except AttributeError: pass

# void free(void *)
try: (free:=dll.free).restype, free.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

# void *realloc(void *__ptr, size_t __size) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2)))
try: (realloc:=dll.realloc).restype, realloc.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t]
except AttributeError: pass

# void *valloc(size_t) __attribute__((alloc_size(1)))
try: (valloc:=dll.valloc).restype, valloc.argtypes = ctypes.c_void_p, [size_t]
except AttributeError: pass

# void *aligned_alloc(size_t __alignment, size_t __size) __attribute__((warn_unused_result(""))) __attribute__((alloc_size(2))) __attribute__((availability(macos, introduced=10.15))) __attribute__((availability(ios, introduced=13.0))) __attribute__((availability(tvos, introduced=13.0))) __attribute__((availability(watchos, introduced=6.0)))
try: (aligned_alloc:=dll.aligned_alloc).restype, aligned_alloc.argtypes = ctypes.c_void_p, [size_t, size_t]
except AttributeError: pass

# int posix_memalign(void **__memptr, size_t __alignment, size_t __size) __attribute__((availability(macos, introduced=10.6)))
try: (posix_memalign:=dll.posix_memalign).restype, posix_memalign.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_void_p), size_t, size_t]
except AttributeError: pass

# void abort(void) __attribute__((cold))
try: (abort:=dll.abort).restype, abort.argtypes = None, []
except AttributeError: pass

# int abs(int) __attribute__((const))
try: (abs:=dll.abs).restype, abs.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int atexit(void (* _Nonnull)(void))
try: (atexit:=dll.atexit).restype, atexit.argtypes = ctypes.c_int, [ctypes.CFUNCTYPE(None, )]
except AttributeError: pass

# double atof(const char *)
try: (atof:=dll.atof).restype, atof.argtypes = ctypes.c_double, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int atoi(const char *)
try: (atoi:=dll.atoi).restype, atoi.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# long atol(const char *)
try: (atol:=dll.atol).restype, atol.argtypes = ctypes.c_long, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# long long atoll(const char *)
try: (atoll:=dll.atoll).restype, atoll.argtypes = ctypes.c_longlong, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void *bsearch(const void *__key, const void *__base, size_t __nel, size_t __width, int (* _Nonnull __compar)(const void *, const void *))
try: (bsearch:=dll.bsearch).restype, bsearch.argtypes = ctypes.c_void_p, [ctypes.c_void_p, ctypes.c_void_p, size_t, size_t, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

class div_t(Struct): pass
div_t._fields_ = [
  ('quot', ctypes.c_int),
  ('rem', ctypes.c_int),
]
# div_t div(int, int) __attribute__((const))
try: (div:=dll.div).restype, div.argtypes = div_t, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# void exit(int)
try: (exit:=dll.exit).restype, exit.argtypes = None, [ctypes.c_int]
except AttributeError: pass

# char *getenv(const char *)
try: (getenv:=dll.getenv).restype, getenv.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# long labs(long) __attribute__((const))
try: (labs:=dll.labs).restype, labs.argtypes = ctypes.c_long, [ctypes.c_long]
except AttributeError: pass

class ldiv_t(Struct): pass
ldiv_t._fields_ = [
  ('quot', ctypes.c_long),
  ('rem', ctypes.c_long),
]
# ldiv_t ldiv(long, long) __attribute__((const))
try: (ldiv:=dll.ldiv).restype, ldiv.argtypes = ldiv_t, [ctypes.c_long, ctypes.c_long]
except AttributeError: pass

# long long llabs(long long)
try: (llabs:=dll.llabs).restype, llabs.argtypes = ctypes.c_longlong, [ctypes.c_longlong]
except AttributeError: pass

class lldiv_t(Struct): pass
lldiv_t._fields_ = [
  ('quot', ctypes.c_longlong),
  ('rem', ctypes.c_longlong),
]
# lldiv_t lldiv(long long, long long)
try: (lldiv:=dll.lldiv).restype, lldiv.argtypes = lldiv_t, [ctypes.c_longlong, ctypes.c_longlong]
except AttributeError: pass

# int mblen(const char *__s, size_t __n)
try: (mblen:=dll.mblen).restype, mblen.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

wchar_t = ctypes.c_int
# size_t mbstowcs(wchar_t *restrict, const char *restrict, size_t)
try: (mbstowcs:=dll.mbstowcs).restype, mbstowcs.argtypes = size_t, [ctypes.POINTER(wchar_t), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int mbtowc(wchar_t *restrict, const char *restrict, size_t)
try: (mbtowc:=dll.mbtowc).restype, mbtowc.argtypes = ctypes.c_int, [ctypes.POINTER(wchar_t), ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# void qsort(void *__base, size_t __nel, size_t __width, int (* _Nonnull __compar)(const void *, const void *))
try: (qsort:=dll.qsort).restype, qsort.argtypes = None, [ctypes.c_void_p, size_t, size_t, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

# int rand(void) __attribute__((availability(swift, unavailable)))
try: (rand:=dll.rand).restype, rand.argtypes = ctypes.c_int, []
except AttributeError: pass

# void srand(unsigned int) __attribute__((availability(swift, unavailable)))
try: (srand:=dll.srand).restype, srand.argtypes = None, [ctypes.c_uint]
except AttributeError: pass

# double strtod(const char *, char **) asm("_strtod")
try: (strtod:=dll.strtod).restype, strtod.argtypes = ctypes.c_double, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# float strtof(const char *, char **) asm("_strtof")
try: (strtof:=dll.strtof).restype, strtof.argtypes = ctypes.c_float, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# long strtol(const char *__str, char **__endptr, int __base)
try: (strtol:=dll.strtol).restype, strtol.argtypes = ctypes.c_long, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int]
except AttributeError: pass

# long double strtold(const char *, char **)
try: (strtold:=dll.strtold).restype, strtold.argtypes = ctypes.c_longdouble, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# long long strtoll(const char *__str, char **__endptr, int __base)
try: (strtoll:=dll.strtoll).restype, strtoll.argtypes = ctypes.c_longlong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int]
except AttributeError: pass

# unsigned long strtoul(const char *__str, char **__endptr, int __base)
try: (strtoul:=dll.strtoul).restype, strtoul.argtypes = ctypes.c_ulong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int]
except AttributeError: pass

# unsigned long long strtoull(const char *__str, char **__endptr, int __base)
try: (strtoull:=dll.strtoull).restype, strtoull.argtypes = ctypes.c_ulonglong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int]
except AttributeError: pass

# __attribute__((availability(swift, unavailable))) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) int system(const char *) asm("_system")
try: (system:=dll.system).restype, system.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# size_t wcstombs(char *restrict, const wchar_t *restrict, size_t)
try: (wcstombs:=dll.wcstombs).restype, wcstombs.argtypes = size_t, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(wchar_t), size_t]
except AttributeError: pass

# int wctomb(char *, wchar_t)
try: (wctomb:=dll.wctomb).restype, wctomb.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), wchar_t]
except AttributeError: pass

# void _Exit(int)
try: (_Exit:=dll._Exit).restype, _Exit.argtypes = None, [ctypes.c_int]
except AttributeError: pass

# long a64l(const char *)
try: (a64l:=dll.a64l).restype, a64l.argtypes = ctypes.c_long, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# double drand48(void)
try: (drand48:=dll.drand48).restype, drand48.argtypes = ctypes.c_double, []
except AttributeError: pass

# char *ecvt(double, int, int *restrict, int *restrict)
try: (ecvt:=dll.ecvt).restype, ecvt.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# double erand48(unsigned short[3])
try: (erand48:=dll.erand48).restype, erand48.argtypes = ctypes.c_double, [(ctypes.c_ushort * 3)]
except AttributeError: pass

# char *fcvt(double, int, int *restrict, int *restrict)
try: (fcvt:=dll.fcvt).restype, fcvt.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
except AttributeError: pass

# char *gcvt(double, int, char *)
try: (gcvt:=dll.gcvt).restype, gcvt.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_double, ctypes.c_int, ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int getsubopt(char **, char *const *, char **)
try: (getsubopt:=dll.getsubopt).restype, getsubopt.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int grantpt(int)
try: (grantpt:=dll.grantpt).restype, grantpt.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# char *initstate(unsigned int, char *, size_t)
try: (initstate:=dll.initstate).restype, initstate.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_uint, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# long jrand48(unsigned short[3]) __attribute__((availability(swift, unavailable)))
try: (jrand48:=dll.jrand48).restype, jrand48.argtypes = ctypes.c_long, [(ctypes.c_ushort * 3)]
except AttributeError: pass

# char *l64a(long)
try: (l64a:=dll.l64a).restype, l64a.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_long]
except AttributeError: pass

# void lcong48(unsigned short[7])
try: (lcong48:=dll.lcong48).restype, lcong48.argtypes = None, [(ctypes.c_ushort * 7)]
except AttributeError: pass

# long lrand48(void) __attribute__((availability(swift, unavailable)))
try: (lrand48:=dll.lrand48).restype, lrand48.argtypes = ctypes.c_long, []
except AttributeError: pass

# __attribute__((deprecated("This function is provided for compatibility reasons only.  Due to security concerns inherent in the design of mktemp(3), it is highly recommended that you use mkstemp(3) instead."))) char *mktemp(char *)
try: (mktemp:=dll.mktemp).restype, mktemp.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int mkstemp(char *)
try: (mkstemp:=dll.mkstemp).restype, mkstemp.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# long mrand48(void) __attribute__((availability(swift, unavailable)))
try: (mrand48:=dll.mrand48).restype, mrand48.argtypes = ctypes.c_long, []
except AttributeError: pass

# long nrand48(unsigned short[3]) __attribute__((availability(swift, unavailable)))
try: (nrand48:=dll.nrand48).restype, nrand48.argtypes = ctypes.c_long, [(ctypes.c_ushort * 3)]
except AttributeError: pass

# int posix_openpt(int)
try: (posix_openpt:=dll.posix_openpt).restype, posix_openpt.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# char *ptsname(int)
try: (ptsname:=dll.ptsname).restype, ptsname.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.c_int]
except AttributeError: pass

# int ptsname_r(int fildes, char *buffer, size_t buflen) __attribute__((availability(macos, introduced=10.13.4))) __attribute__((availability(ios, introduced=11.3))) __attribute__((availability(tvos, introduced=11.3))) __attribute__((availability(watchos, introduced=4.3)))
try: (ptsname_r:=dll.ptsname_r).restype, ptsname_r.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.POINTER(ctypes.c_char), size_t]
except AttributeError: pass

# int putenv(char *) asm("_putenv")
try: (putenv:=dll.putenv).restype, putenv.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# long random(void) __attribute__((availability(swift, unavailable)))
try: (random:=dll.random).restype, random.argtypes = ctypes.c_long, []
except AttributeError: pass

# int rand_r(unsigned int *) __attribute__((availability(swift, unavailable)))
try: (rand_r:=dll.rand_r).restype, rand_r.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_uint)]
except AttributeError: pass

# char *realpath(const char *restrict, char *restrict) asm("_realpath$DARWIN_EXTSN")
try: (realpath:=dll.realpath).restype, realpath.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# unsigned short *seed48(unsigned short[3])
try: (seed48:=dll.seed48).restype, seed48.argtypes = ctypes.POINTER(ctypes.c_ushort), [(ctypes.c_ushort * 3)]
except AttributeError: pass

# int setenv(const char *__name, const char *__value, int __overwrite) asm("_setenv")
try: (setenv:=dll.setenv).restype, setenv.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# void setkey(const char *) asm("_setkey")
try: (setkey:=dll.setkey).restype, setkey.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# char *setstate(const char *)
try: (setstate:=dll.setstate).restype, setstate.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# void srand48(long)
try: (srand48:=dll.srand48).restype, srand48.argtypes = None, [ctypes.c_long]
except AttributeError: pass

# void srandom(unsigned int)
try: (srandom:=dll.srandom).restype, srandom.argtypes = None, [ctypes.c_uint]
except AttributeError: pass

# int unlockpt(int)
try: (unlockpt:=dll.unlockpt).restype, unlockpt.argtypes = ctypes.c_int, [ctypes.c_int]
except AttributeError: pass

# int unsetenv(const char *) asm("_unsetenv")
try: (unsetenv:=dll.unsetenv).restype, unsetenv.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# uint32_t arc4random(void)
try: (arc4random:=dll.arc4random).restype, arc4random.argtypes = uint32_t, []
except AttributeError: pass

# void arc4random_addrandom(unsigned char *, int) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(macos, deprecated=10.12))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(ios, deprecated=10.0))) __attribute__((availability(tvos, introduced=2.0))) __attribute__((availability(tvos, deprecated=10.0))) __attribute__((availability(watchos, introduced=1.0))) __attribute__((availability(watchos, deprecated=3.0)))
try: (arc4random_addrandom:=dll.arc4random_addrandom).restype, arc4random_addrandom.argtypes = None, [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int]
except AttributeError: pass

# void arc4random_buf(void *__buf, size_t __nbytes) __attribute__((availability(macos, introduced=10.7)))
try: (arc4random_buf:=dll.arc4random_buf).restype, arc4random_buf.argtypes = None, [ctypes.c_void_p, size_t]
except AttributeError: pass

# void arc4random_stir(void)
try: (arc4random_stir:=dll.arc4random_stir).restype, arc4random_stir.argtypes = None, []
except AttributeError: pass

# uint32_t arc4random_uniform(uint32_t __upper_bound) __attribute__((availability(macos, introduced=10.7)))
try: (arc4random_uniform:=dll.arc4random_uniform).restype, arc4random_uniform.argtypes = uint32_t, [uint32_t]
except AttributeError: pass

# char *cgetcap(char *, const char *, int)
try: (cgetcap:=dll.cgetcap).restype, cgetcap.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# int cgetclose(void)
try: (cgetclose:=dll.cgetclose).restype, cgetclose.argtypes = ctypes.c_int, []
except AttributeError: pass

# int cgetent(char **, char **, const char *)
try: (cgetent:=dll.cgetent).restype, cgetent.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int cgetfirst(char **, char **)
try: (cgetfirst:=dll.cgetfirst).restype, cgetfirst.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int cgetmatch(const char *, const char *)
try: (cgetmatch:=dll.cgetmatch).restype, cgetmatch.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int cgetnext(char **, char **)
try: (cgetnext:=dll.cgetnext).restype, cgetnext.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int cgetnum(char *, const char *, long *)
try: (cgetnum:=dll.cgetnum).restype, cgetnum.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_long)]
except AttributeError: pass

# int cgetset(const char *)
try: (cgetset:=dll.cgetset).restype, cgetset.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int cgetstr(char *, const char *, char **)
try: (cgetstr:=dll.cgetstr).restype, cgetstr.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int cgetustr(char *, const char *, char **)
try: (cgetustr:=dll.cgetustr).restype, cgetustr.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# int daemon(int, int) asm("_daemon") __attribute__((availability(macos, introduced=10.0, deprecated=10.5))) __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable)))
try: (daemon:=dll.daemon).restype, daemon.argtypes = ctypes.c_int, [ctypes.c_int, ctypes.c_int]
except AttributeError: pass

# char *devname(dev_t, mode_t)
try: (devname:=dll.devname).restype, devname.argtypes = ctypes.POINTER(ctypes.c_char), [dev_t, mode_t]
except AttributeError: pass

# char *devname_r(dev_t, mode_t, char *buf, int len)
try: (devname_r:=dll.devname_r).restype, devname_r.argtypes = ctypes.POINTER(ctypes.c_char), [dev_t, mode_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int]
except AttributeError: pass

# char *getbsize(int *, long *)
try: (getbsize:=dll.getbsize).restype, getbsize.argtypes = ctypes.POINTER(ctypes.c_char), [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_long)]
except AttributeError: pass

# int getloadavg(double[], int)
try: (getloadavg:=dll.getloadavg).restype, getloadavg.argtypes = ctypes.c_int, [(ctypes.c_double * 0), ctypes.c_int]
except AttributeError: pass

# const char *getprogname(void)
try: (getprogname:=dll.getprogname).restype, getprogname.argtypes = ctypes.POINTER(ctypes.c_char), []
except AttributeError: pass

# void setprogname(const char *)
try: (setprogname:=dll.setprogname).restype, setprogname.argtypes = None, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int heapsort(void *__base, size_t __nel, size_t __width, int (* _Nonnull __compar)(const void *, const void *))
try: (heapsort:=dll.heapsort).restype, heapsort.argtypes = ctypes.c_int, [ctypes.c_void_p, size_t, size_t, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

# int mergesort(void *__base, size_t __nel, size_t __width, int (* _Nonnull __compar)(const void *, const void *))
try: (mergesort:=dll.mergesort).restype, mergesort.argtypes = ctypes.c_int, [ctypes.c_void_p, size_t, size_t, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

# void psort(void *__base, size_t __nel, size_t __width, int (* _Nonnull __compar)(const void *, const void *)) __attribute__((availability(macos, introduced=10.6)))
try: (psort:=dll.psort).restype, psort.argtypes = None, [ctypes.c_void_p, size_t, size_t, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

# void psort_r(void *__base, size_t __nel, size_t __width, void *, int (* _Nonnull __compar)(void *, const void *, const void *)) __attribute__((availability(macos, introduced=10.6)))
try: (psort_r:=dll.psort_r).restype, psort_r.argtypes = None, [ctypes.c_void_p, size_t, size_t, ctypes.c_void_p, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

# void qsort_r(void *__base, size_t __nel, size_t __width, void *, int (* _Nonnull __compar)(void *, const void *, const void *))
try: (qsort_r:=dll.qsort_r).restype, qsort_r.argtypes = None, [ctypes.c_void_p, size_t, size_t, ctypes.c_void_p, ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)]
except AttributeError: pass

# int radixsort(const unsigned char **__base, int __nel, const unsigned char *__table, unsigned int __endbyte)
try: (radixsort:=dll.radixsort).restype, radixsort.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint]
except AttributeError: pass

# int rpmatch(const char *) __attribute__((availability(macos, introduced=10.15))) __attribute__((availability(ios, introduced=13.0))) __attribute__((availability(tvos, introduced=13.0))) __attribute__((availability(watchos, introduced=6.0)))
try: (rpmatch:=dll.rpmatch).restype, rpmatch.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# int sradixsort(const unsigned char **__base, int __nel, const unsigned char *__table, unsigned int __endbyte)
try: (sradixsort:=dll.sradixsort).restype, sradixsort.argtypes = ctypes.c_int, [ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)), ctypes.c_int, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_uint]
except AttributeError: pass

# void sranddev(void)
try: (sranddev:=dll.sranddev).restype, sranddev.argtypes = None, []
except AttributeError: pass

# void srandomdev(void)
try: (srandomdev:=dll.srandomdev).restype, srandomdev.argtypes = None, []
except AttributeError: pass

# void *reallocf(void *__ptr, size_t __size) __attribute__((alloc_size(2)))
try: (reallocf:=dll.reallocf).restype, reallocf.argtypes = ctypes.c_void_p, [ctypes.c_void_p, size_t]
except AttributeError: pass

# long long strtonum(const char *__numstr, long long __minval, long long __maxval, const char **__errstrp) __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0)))
try: (strtonum:=dll.strtonum).restype, strtonum.argtypes = ctypes.c_longlong, [ctypes.POINTER(ctypes.c_char), ctypes.c_longlong, ctypes.c_longlong, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))]
except AttributeError: pass

# long long strtoq(const char *__str, char **__endptr, int __base)
try: (strtoq:=dll.strtoq).restype, strtoq.argtypes = ctypes.c_longlong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int]
except AttributeError: pass

# unsigned long long strtouq(const char *__str, char **__endptr, int __base)
try: (strtouq:=dll.strtouq).restype, strtouq.argtypes = ctypes.c_ulonglong, [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), ctypes.c_int]
except AttributeError: pass

# __attribute__((visibility("default"))) extern const char * _Nonnull sel_getName(SEL  _Nonnull sel) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(tvos, introduced=9.0))) __attribute__((availability(watchos, introduced=1.0)))
try: (sel_getName:=dll.sel_getName).restype, sel_getName.argtypes = ctypes.POINTER(ctypes.c_char), [objc.id_]
except AttributeError: pass

# __attribute__((visibility("default"))) extern SEL  _Nonnull sel_registerName(const char * _Nonnull str) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(tvos, introduced=9.0))) __attribute__((availability(watchos, introduced=1.0)))
try: (sel_registerName:=dll.sel_registerName).restype, sel_registerName.argtypes = objc.id_, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

# __attribute__((visibility("default"))) extern const char * _Nonnull object_getClassName(id  _Nullable obj) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(tvos, introduced=9.0))) __attribute__((availability(watchos, introduced=1.0)))
try: (object_getClassName:=dll.object_getClassName).restype, object_getClassName.argtypes = ctypes.POINTER(ctypes.c_char), [objc.id_]
except AttributeError: pass

# __attribute__((visibility("default"))) extern void * _Nullable object_getIndexedIvars(id  _Nullable obj) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(tvos, introduced=9.0))) __attribute__((availability(watchos, introduced=1.0)))
try: (object_getIndexedIvars:=dll.object_getIndexedIvars).restype, object_getIndexedIvars.argtypes = ctypes.c_void_p, [objc.id_]
except AttributeError: pass

BOOL = ctypes.c_int
# __attribute__((visibility("default"))) extern BOOL sel_isMapped(SEL  _Nonnull sel) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(tvos, introduced=9.0))) __attribute__((availability(watchos, introduced=1.0)))
try: (sel_isMapped:=dll.sel_isMapped).restype, sel_isMapped.argtypes = BOOL, [objc.id_]
except AttributeError: pass

# __attribute__((visibility("default"))) extern SEL  _Nonnull sel_getUid(const char * _Nonnull str) __attribute__((availability(macos, introduced=10.0))) __attribute__((availability(ios, introduced=2.0))) __attribute__((availability(tvos, introduced=9.0))) __attribute__((availability(watchos, introduced=1.0)))
try: (sel_getUid:=dll.sel_getUid).restype, sel_getUid.argtypes = objc.id_, [ctypes.POINTER(ctypes.c_char)]
except AttributeError: pass

objc_objectptr_t = ctypes.c_void_p
# __attribute__((visibility("default"))) extern id  _Nullable objc_retainedObject(objc_objectptr_t  _Nullable obj) __attribute__((unavailable("use CFBridgingRelease() or a (__bridge_transfer id) cast instead")))
try: (objc_retainedObject:=dll.objc_retainedObject).restype, objc_retainedObject.argtypes = objc.id_, [objc_objectptr_t]
except AttributeError: pass

# __attribute__((visibility("default"))) extern id  _Nullable objc_unretainedObject(objc_objectptr_t  _Nullable obj) __attribute__((unavailable("use a (__bridge id) cast instead")))
try: (objc_unretainedObject:=dll.objc_unretainedObject).restype, objc_unretainedObject.argtypes = objc.id_, [objc_objectptr_t]
except AttributeError: pass

# __attribute__((visibility("default"))) extern objc_objectptr_t  _Nullable objc_unretainedPointer(id  _Nullable obj) __attribute__((unavailable("use a __bridge cast instead")))
try: (objc_unretainedPointer:=dll.objc_unretainedPointer).restype, objc_unretainedPointer.argtypes = objc_objectptr_t, [objc.id_]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((availability(swift, unavailable))) extern void *os_retain(void *object)
try: (os_retain:=dll.os_retain).restype, os_retain.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((availability(swift, unavailable))) extern void os_release(void *object)
try: (os_release:=dll.os_release).restype, os_release.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

class OS_os_workgroup(objc.Spec): pass
class OS_object(objc.Spec): pass
class NSObject(objc.Spec): pass
IMP = ctypes.CFUNCTYPE(None, )
class NSInvocation(objc.Spec): pass
class NSMethodSignature(objc.Spec): pass
class struct__NSZone(Struct): pass
class Protocol(objc.Spec): pass
NSUInteger = ctypes.c_ulong
class NSString(objc.Spec): pass
NSObject._methods_ = [
  ('init', 'instancetype', []),
  ('dealloc', None, []),
  ('finalize', None, []),
  ('copy', objc.id_, [], True),
  ('mutableCopy', objc.id_, [], True),
  ('methodForSelector:', IMP, [objc.id_]),
  ('doesNotRecognizeSelector:', None, [objc.id_]),
  ('forwardingTargetForSelector:', objc.id_, [objc.id_]),
  ('forwardInvocation:', None, [NSInvocation]),
  ('methodSignatureForSelector:', NSMethodSignature, [objc.id_]),
  ('allowsWeakReference', BOOL, []),
  ('retainWeakReference', BOOL, []),
]
NSObject._classmethods_ = [
  ('load', None, []),
  ('initialize', None, []),
  ('new', 'instancetype', [], True),
  ('allocWithZone:', 'instancetype', [ctypes.POINTER(struct__NSZone)], True),
  ('alloc', 'instancetype', [], True),
  ('copyWithZone:', objc.id_, [ctypes.POINTER(struct__NSZone)], True),
  ('mutableCopyWithZone:', objc.id_, [ctypes.POINTER(struct__NSZone)], True),
  ('instancesRespondToSelector:', BOOL, [objc.id_]),
  ('conformsToProtocol:', BOOL, [Protocol]),
  ('instanceMethodForSelector:', IMP, [objc.id_]),
  ('instanceMethodSignatureForSelector:', NSMethodSignature, [objc.id_]),
  ('resolveClassMethod:', BOOL, [objc.id_]),
  ('resolveInstanceMethod:', BOOL, [objc.id_]),
  ('hash', NSUInteger, []),
  ('description', NSString, []),
  ('debugDescription', NSString, []),
]
OS_object._bases_ = [NSObject]
OS_object._methods_ = [
  ('init', 'instancetype', []),
]
OS_os_workgroup._bases_ = [OS_object]
OS_os_workgroup._methods_ = [
  ('init', 'instancetype', []),
]
os_workgroup_t = OS_os_workgroup
mach_port_t = ctypes.c_uint
# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((swift_private)) __attribute__((visibility("default"))) __attribute__((warn_unused_result(""))) extern int os_workgroup_copy_port(os_workgroup_t  _Nonnull wg, mach_port_t * _Nonnull mach_port_out)
try: (os_workgroup_copy_port:=dll.os_workgroup_copy_port).restype, os_workgroup_copy_port.argtypes = ctypes.c_int, [os_workgroup_t, ctypes.POINTER(mach_port_t)]
except AttributeError: pass

# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, unavailable))) __attribute__((availability(tvos, unavailable))) __attribute__((availability(watchos, unavailable))) __attribute__((swift_name("WorkGroup.init(__name:port:)"))) __attribute__((visibility("default"))) __attribute__((ns_returns_retained)) __attribute__((ns_returns_retained)) extern os_workgroup_t  _Nullable os_workgroup_create_with_port(const char * _Nullable name, mach_port_t mach_port)
try: (os_workgroup_create_with_port:=dll.os_workgroup_create_with_port).restype, os_workgroup_create_with_port.argtypes = os_workgroup_t, [ctypes.POINTER(ctypes.c_char), mach_port_t]
except AttributeError: pass

os_workgroup_create_with_port = objc.returns_retained(os_workgroup_create_with_port)
# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) __attribute__((ns_returns_retained)) __attribute__((ns_returns_retained)) extern os_workgroup_t  _Nullable os_workgroup_create_with_workgroup(const char * _Nullable name, os_workgroup_t  _Nonnull wg)
try: (os_workgroup_create_with_workgroup:=dll.os_workgroup_create_with_workgroup).restype, os_workgroup_create_with_workgroup.argtypes = os_workgroup_t, [ctypes.POINTER(ctypes.c_char), os_workgroup_t]
except AttributeError: pass

os_workgroup_create_with_workgroup = objc.returns_retained(os_workgroup_create_with_workgroup)
class struct_os_workgroup_join_token_opaque_s(Struct): pass
struct_os_workgroup_join_token_opaque_s._fields_ = [
  ('sig', uint32_t),
  ('opaque', (ctypes.c_char * 36)),
]
os_workgroup_join_token_t = ctypes.POINTER(struct_os_workgroup_join_token_opaque_s)
# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) __attribute__((warn_unused_result(""))) extern int os_workgroup_join(os_workgroup_t  _Nonnull wg, os_workgroup_join_token_t  _Nonnull token_out)
try: (os_workgroup_join:=dll.os_workgroup_join).restype, os_workgroup_join.argtypes = ctypes.c_int, [os_workgroup_t, os_workgroup_join_token_t]
except AttributeError: pass

# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) extern void os_workgroup_leave(os_workgroup_t  _Nonnull wg, os_workgroup_join_token_t  _Nonnull token)
try: (os_workgroup_leave:=dll.os_workgroup_leave).restype, os_workgroup_leave.argtypes = None, [os_workgroup_t, os_workgroup_join_token_t]
except AttributeError: pass

os_workgroup_working_arena_destructor_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) __attribute__((warn_unused_result(""))) extern int os_workgroup_set_working_arena(os_workgroup_t  _Nonnull wg, void * _Nullable arena, uint32_t max_workers, os_workgroup_working_arena_destructor_t  _Nonnull destructor)
try: (os_workgroup_set_working_arena:=dll.os_workgroup_set_working_arena).restype, os_workgroup_set_working_arena.argtypes = ctypes.c_int, [os_workgroup_t, ctypes.c_void_p, uint32_t, os_workgroup_working_arena_destructor_t]
except AttributeError: pass

os_workgroup_index = ctypes.c_uint
# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) extern void * _Nullable os_workgroup_get_working_arena(os_workgroup_t  _Nonnull wg, os_workgroup_index * _Nullable index_out)
try: (os_workgroup_get_working_arena:=dll.os_workgroup_get_working_arena).restype, os_workgroup_get_working_arena.argtypes = ctypes.c_void_p, [os_workgroup_t, ctypes.POINTER(os_workgroup_index)]
except AttributeError: pass

# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) extern void os_workgroup_cancel(os_workgroup_t  _Nonnull wg)
try: (os_workgroup_cancel:=dll.os_workgroup_cancel).restype, os_workgroup_cancel.argtypes = None, [os_workgroup_t]
except AttributeError: pass

# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) extern int os_workgroup_testcancel(os_workgroup_t  _Nonnull wg)
try: (os_workgroup_testcancel:=dll.os_workgroup_testcancel).restype, os_workgroup_testcancel.argtypes = ctypes.c_int, [os_workgroup_t]
except AttributeError: pass

class struct_os_workgroup_max_parallel_threads_attr_s(Struct): pass
os_workgroup_mpt_attr_t = ctypes.POINTER(struct_os_workgroup_max_parallel_threads_attr_s)
# __attribute__((availability(macos, introduced=11.0))) __attribute__((availability(ios, introduced=14.0))) __attribute__((availability(tvos, introduced=14.0))) __attribute__((availability(watchos, introduced=7.0))) __attribute__((swift_private)) __attribute__((visibility("default"))) extern int os_workgroup_max_parallel_threads(os_workgroup_t  _Nonnull wg, os_workgroup_mpt_attr_t  _Nullable attr)
try: (os_workgroup_max_parallel_threads:=dll.os_workgroup_max_parallel_threads).restype, os_workgroup_max_parallel_threads.argtypes = ctypes.c_int, [os_workgroup_t, os_workgroup_mpt_attr_t]
except AttributeError: pass

dispatch_time_t = ctypes.c_ulonglong
int64_t = ctypes.c_longlong
# __attribute__((availability(macos, introduced=10.6))) __attribute__((availability(ios, introduced=4.0))) __attribute__((visibility("default"))) __attribute__((warn_unused_result(""))) __attribute__((nothrow)) __attribute__((swift_private)) extern dispatch_time_t dispatch_time(dispatch_time_t when, int64_t delta)
try: (dispatch_time:=dll.dispatch_time).restype, dispatch_time.argtypes = dispatch_time_t, [dispatch_time_t, int64_t]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.6))) __attribute__((availability(ios, introduced=4.0))) __attribute__((visibility("default"))) __attribute__((warn_unused_result(""))) __attribute__((nothrow)) __attribute__((swift_private)) extern dispatch_time_t dispatch_walltime(const struct timespec * _Nullable when, int64_t delta)
try: (dispatch_walltime:=dll.dispatch_walltime).restype, dispatch_walltime.argtypes = dispatch_time_t, [ctypes.POINTER(struct_timespec), int64_t]
except AttributeError: pass

qos_class_t = CEnum(ctypes.c_uint)
QOS_CLASS_USER_INTERACTIVE = qos_class_t.define('QOS_CLASS_USER_INTERACTIVE', 33)
QOS_CLASS_USER_INITIATED = qos_class_t.define('QOS_CLASS_USER_INITIATED', 25)
QOS_CLASS_DEFAULT = qos_class_t.define('QOS_CLASS_DEFAULT', 21)
QOS_CLASS_UTILITY = qos_class_t.define('QOS_CLASS_UTILITY', 17)
QOS_CLASS_BACKGROUND = qos_class_t.define('QOS_CLASS_BACKGROUND', 9)
QOS_CLASS_UNSPECIFIED = qos_class_t.define('QOS_CLASS_UNSPECIFIED', 0)

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) qos_class_t qos_class_self(void)
try: (qos_class_self:=dll.qos_class_self).restype, qos_class_self.argtypes = qos_class_t, []
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) qos_class_t qos_class_main(void)
try: (qos_class_main:=dll.qos_class_main).restype, qos_class_main.argtypes = qos_class_t, []
except AttributeError: pass

intptr_t = ctypes.c_long
# __attribute__((unavailable(""))) __attribute__((visibility("default"))) __attribute__((nonnull(1))) __attribute__((nothrow)) extern intptr_t dispatch_wait(void * _Nonnull object, dispatch_time_t timeout)
try: (dispatch_wait:=dll.dispatch_wait).restype, dispatch_wait.argtypes = intptr_t, [ctypes.c_void_p, dispatch_time_t]
except AttributeError: pass

# __attribute__((unavailable(""))) __attribute__((visibility("default"))) __attribute__((nonnull)) __attribute__((nothrow)) extern void dispatch_cancel(void * _Nonnull object)
try: (dispatch_cancel:=dll.dispatch_cancel).restype, dispatch_cancel.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

# __attribute__((unavailable(""))) __attribute__((visibility("default"))) __attribute__((nonnull)) __attribute__((warn_unused_result(""))) __attribute__((pure)) __attribute__((nothrow)) extern intptr_t dispatch_testcancel(void * _Nonnull object)
try: (dispatch_testcancel:=dll.dispatch_testcancel).restype, dispatch_testcancel.argtypes = intptr_t, [ctypes.c_void_p]
except AttributeError: pass

dispatch_qos_class_t = qos_class_t
# __attribute__((availability(macos, introduced=10.6))) __attribute__((availability(ios, introduced=4.0))) __attribute__((visibility("default"))) __attribute__((nothrow)) __attribute__((swift_name("dispatchMain()"))) extern void dispatch_main(void)
try: (dispatch_main:=dll.dispatch_main).restype, dispatch_main.argtypes = None, []
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.7))) __attribute__((availability(ios, introduced=5.0))) __attribute__((visibility("default"))) __attribute__((pure)) __attribute__((warn_unused_result(""))) __attribute__((nothrow)) __attribute__((swift_private)) extern void * _Nullable dispatch_get_specific(const void * _Nonnull key)
try: (dispatch_get_specific:=dll.dispatch_get_specific).restype, dispatch_get_specific.argtypes = ctypes.c_void_p, [ctypes.c_void_p]
except AttributeError: pass

dispatch_block_flags_t = CEnum(ctypes.c_ulong)
DISPATCH_BLOCK_BARRIER = dispatch_block_flags_t.define('DISPATCH_BLOCK_BARRIER', 1)
DISPATCH_BLOCK_DETACHED = dispatch_block_flags_t.define('DISPATCH_BLOCK_DETACHED', 2)
DISPATCH_BLOCK_ASSIGN_CURRENT = dispatch_block_flags_t.define('DISPATCH_BLOCK_ASSIGN_CURRENT', 4)
DISPATCH_BLOCK_NO_QOS_CLASS = dispatch_block_flags_t.define('DISPATCH_BLOCK_NO_QOS_CLASS', 8)
DISPATCH_BLOCK_INHERIT_QOS_CLASS = dispatch_block_flags_t.define('DISPATCH_BLOCK_INHERIT_QOS_CLASS', 16)
DISPATCH_BLOCK_ENFORCE_QOS_CLASS = dispatch_block_flags_t.define('DISPATCH_BLOCK_ENFORCE_QOS_CLASS', 32)

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((nonnull(2))) __attribute__((ns_returns_retained)) __attribute__((warn_unused_result(""))) __attribute__((nothrow)) __attribute__((availability(swift, unavailable))) __attribute__((ns_returns_retained)) extern dispatch_block_t  _Nonnull dispatch_block_create(dispatch_block_flags_t flags, dispatch_block_t  _Nonnull block)
try: (dispatch_block_create:=dll.dispatch_block_create).restype, dispatch_block_create.argtypes = ctypes.c_void_p, [dispatch_block_flags_t, ctypes.c_void_p]
except AttributeError: pass

dispatch_block_create = objc.returns_retained(dispatch_block_create)
# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((nonnull(4))) __attribute__((ns_returns_retained)) __attribute__((warn_unused_result(""))) __attribute__((nothrow)) __attribute__((availability(swift, unavailable))) __attribute__((ns_returns_retained)) extern dispatch_block_t  _Nonnull dispatch_block_create_with_qos_class(dispatch_block_flags_t flags, dispatch_qos_class_t qos_class, int relative_priority, dispatch_block_t  _Nonnull block)
try: (dispatch_block_create_with_qos_class:=dll.dispatch_block_create_with_qos_class).restype, dispatch_block_create_with_qos_class.argtypes = ctypes.c_void_p, [dispatch_block_flags_t, dispatch_qos_class_t, ctypes.c_int, ctypes.c_void_p]
except AttributeError: pass

dispatch_block_create_with_qos_class = objc.returns_retained(dispatch_block_create_with_qos_class)
# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((nonnull(2))) __attribute__((nothrow)) __attribute__((availability(swift, unavailable))) extern void dispatch_block_perform(dispatch_block_flags_t flags, __attribute__((noescape)) dispatch_block_t  _Nonnull block)
try: (dispatch_block_perform:=dll.dispatch_block_perform).restype, dispatch_block_perform.argtypes = None, [dispatch_block_flags_t, ctypes.c_void_p]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((nonnull(1))) __attribute__((nothrow)) __attribute__((availability(swift, unavailable))) extern intptr_t dispatch_block_wait(dispatch_block_t  _Nonnull block, dispatch_time_t timeout)
try: (dispatch_block_wait:=dll.dispatch_block_wait).restype, dispatch_block_wait.argtypes = intptr_t, [ctypes.c_void_p, dispatch_time_t]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((nonnull)) __attribute__((nothrow)) __attribute__((availability(swift, unavailable))) extern void dispatch_block_cancel(dispatch_block_t  _Nonnull block)
try: (dispatch_block_cancel:=dll.dispatch_block_cancel).restype, dispatch_block_cancel.argtypes = None, [ctypes.c_void_p]
except AttributeError: pass

# __attribute__((availability(macos, introduced=10.10))) __attribute__((availability(ios, introduced=8.0))) __attribute__((visibility("default"))) __attribute__((nonnull)) __attribute__((warn_unused_result(""))) __attribute__((pure)) __attribute__((nothrow)) __attribute__((availability(swift, unavailable))) extern intptr_t dispatch_block_testcancel(dispatch_block_t  _Nonnull block)
try: (dispatch_block_testcancel:=dll.dispatch_block_testcancel).restype, dispatch_block_testcancel.argtypes = intptr_t, [ctypes.c_void_p]
except AttributeError: pass

mach_msg_return_t = ctypes.c_int
class mach_msg_header_t(Struct): pass
mach_msg_bits_t = ctypes.c_uint
mach_msg_size_t = ctypes.c_uint
mach_port_name_t = ctypes.c_uint
mach_msg_id_t = ctypes.c_int
mach_msg_header_t._fields_ = [
  ('msgh_bits', mach_msg_bits_t),
  ('msgh_size', mach_msg_size_t),
  ('msgh_remote_port', mach_port_t),
  ('msgh_local_port', mach_port_t),
  ('msgh_voucher_port', mach_port_name_t),
  ('msgh_id', mach_msg_id_t),
]
mach_msg_option_t = ctypes.c_int
mach_msg_timeout_t = ctypes.c_uint
# __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) extern mach_msg_return_t mach_msg_overwrite(mach_msg_header_t *msg, mach_msg_option_t option, mach_msg_size_t send_size, mach_msg_size_t rcv_size, mach_port_name_t rcv_name, mach_msg_timeout_t timeout, mach_port_name_t notify, mach_msg_header_t *rcv_msg, mach_msg_size_t rcv_limit)
try: (mach_msg_overwrite:=dll.mach_msg_overwrite).restype, mach_msg_overwrite.argtypes = mach_msg_return_t, [ctypes.POINTER(mach_msg_header_t), mach_msg_option_t, mach_msg_size_t, mach_msg_size_t, mach_port_name_t, mach_msg_timeout_t, mach_port_name_t, ctypes.POINTER(mach_msg_header_t), mach_msg_size_t]
except AttributeError: pass

# __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) extern mach_msg_return_t mach_msg(mach_msg_header_t *msg, mach_msg_option_t option, mach_msg_size_t send_size, mach_msg_size_t rcv_size, mach_port_name_t rcv_name, mach_msg_timeout_t timeout, mach_port_name_t notify)
try: (mach_msg:=dll.mach_msg).restype, mach_msg.argtypes = mach_msg_return_t, [ctypes.POINTER(mach_msg_header_t), mach_msg_option_t, mach_msg_size_t, mach_msg_size_t, mach_port_name_t, mach_msg_timeout_t, mach_port_name_t]
except AttributeError: pass

kern_return_t = ctypes.c_int
# __attribute__((availability(watchos, unavailable))) __attribute__((availability(tvos, unavailable))) extern kern_return_t mach_voucher_deallocate(mach_port_name_t voucher)
try: (mach_voucher_deallocate:=dll.mach_voucher_deallocate).restype, mach_voucher_deallocate.argtypes = kern_return_t, [mach_port_name_t]
except AttributeError: pass

uintptr_t = ctypes.c_ulong
dispatch_once_t = ctypes.c_long
# __attribute__((availability(macos, introduced=10.6))) __attribute__((availability(ios, introduced=4.0))) __attribute__((visibility("default"))) __attribute__((nonnull)) __attribute__((nothrow)) extern void dispatch_once(dispatch_once_t * _Nonnull predicate, __attribute__((noescape)) dispatch_block_t  _Nonnull block)
try: (dispatch_once:=dll.dispatch_once).restype, dispatch_once.argtypes = None, [ctypes.POINTER(dispatch_once_t), ctypes.c_void_p]
except AttributeError: pass

dispatch_function_t = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
# __attribute__((availability(macos, introduced=10.6))) __attribute__((availability(ios, introduced=4.0))) __attribute__((visibility("default"))) __attribute__((nonnull(1))) __attribute__((nonnull(3))) __attribute__((nothrow)) extern void dispatch_once_f(dispatch_once_t * _Nonnull predicate, void * _Nullable context, dispatch_function_t  _Nonnull function)
try: (dispatch_once_f:=dll.dispatch_once_f).restype, dispatch_once_f.argtypes = None, [ctypes.POINTER(dispatch_once_t), ctypes.c_void_p, dispatch_function_t]
except AttributeError: pass

dispatch_fd_t = ctypes.c_int
DISPATCH_API_VERSION = 20181008
