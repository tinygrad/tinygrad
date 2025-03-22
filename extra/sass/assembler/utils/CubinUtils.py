# -*- coding: utf-8 -*-
from subprocess import STDOUT, check_output, CalledProcessError
from elftools.elf.elffile import ELFFile
from io import BytesIO, StringIO

import struct
import glob
import os
import re
import shutil
import traceback

from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from extra.sass.assembler.CuInsFeeder import CuInsFeeder
from extra.sass.assembler.CuAsmLogger import CuAsmLogger
from extra.sass.assembler.CuNVInfo import CuNVInfo
from extra.sass.assembler.common import getTempFileName

pShowDesc = 1<<101   # set this bit to show desc explicitly in SM8x

fname_pattern = re.compile(r'file *(?P<idx>\d+): *(?P<fname>.*)$')
arch_pattern = re.compile(r'\.(?P<arch>sm_\d+)\.(cubin|ptx)$')

# ins filter for rull out QNAN input for corrupting the repos
f_QNAN = lambda x: 'QNAN' not in x

def f_glob(fpattern):
    ''' Yield file names from glob pattern, skip directories.'''
    for fname in glob.iglob(fpattern):
        if os.path.isfile(fname):
            yield fname

def parseListLine(line):
    ''' Parse list lines from `cuobjdump -lelf ...` output.'''
    res = fname_pattern.search(line)
    if res is None:
        return None
    else:
        return res.group('fname').strip()

def transPTXVersion(ptxname, outname=None, arch='sm_75', version='7.3'):
    ''' Modify ptx version and target.
        CAUTION: this may cause some inconsistency!
        
        Example:
           .version 7.1
           .target sm_75
    '''

    sio = StringIO()
    with open(ptxname, 'r') as fin:
        for line in fin:
            if re.match(r'\s*\.target\s+', line):
                sio.write(f'.target {arch}\n')
            elif re.match(r'\s*\.version\s+', line):
                sio.write(f'.version {version}\n')
            else:
                sio.write(line+'\n')

    if outname is None: # default to modify ptx inplace
        outname = ptxname

    with open(outname, 'w+') as fout:
        fout.write(sio.getvalue())

def feedBinFromCubin(binname:str, outname:str=None, merge_all_kernels=False):
    ''' Generator to feed text section in cubin as list of binary files.
    
        Args:
            binname : input cubin file name
            outname : output bin file name, None for just replacing .cubin to .bin
            merge_all_kernels : write all kernels into one bin file. 
                NOTE: this may change disassembly of branching instructions.
        
        Yield: yield file names for bin files.
            NOTE: the file name maybe remain unchanged, but the contents will be different 
    '''
    if outname is None:
        if binname.endswith('.cubin'):
            outname = binname.replace('.cubin', '.bin')
        else:
            outname = binname + '.bin'

    with open(binname, 'rb') as fin:
        ef = ELFFile(fin)

        if merge_all_kernels:
            fout = open(outname, 'wb')
            is_empty = True
            for sec in ef.iter_sections():
                if sec.name.startswith('.text'):
                    # all instructions will be written into same bin file and disassembled
                    # NOTE: the function info will be lost, do not use this in program analysis
                    # NOTE 2: disassembly of branching instruction may be changed
                    fout.write(sec.data())
                    is_empty = False
            fout.close()
            if not is_empty: # skip cubin without .text sections
                yield outname
        else:
            for sec in ef.iter_sections():
                if sec.name.startswith('.text'):
                    fout = open(outname, 'wb')
                    fout.write(sec.data())
                    fout.close()
                    yield outname

def updateReposWithCubin(repos:CuInsAssemblerRepos, binname:str, savname=None, use_nvdisasm=True, do_desc_hack=True, merge_all_kernels=False):
    ''' Update input repos with given cubin.

        Args:
            repos : CuInsAssemblerRepos to be updated, should be with same arch as the input cubin
            binname : cubin file name
            savname : filename for saving repos, None(default) for no save.
            use_nvdisasm : usually cuobjdump will be utilized to obtain sass, but it's usually too slow. 
                           Thus the raw binary approach 'nvdisasm -b SM75 #.bin' is recommended.
            do_desc_hack : do desc hack for SM80/SM86 if needed.
            merge_all_kernels : merge all kernel into one bin file and then run nvdisasm, only used when use_nvdisasm=True.
    '''
    arch = repos.getArchString()           # such as SM_75
    b_arch = arch.replace('_', '').upper() # `nvdisasm -b` only accept SM60/SM75/SM86/...

    if do_desc_hack and repos.m_Arch.needsDescHack():
        bindir, basename = os.path.split(binname)
        hbinname = os.path.join(bindir, 'hack.' + basename)

        hackCubinDesc(binname, hbinname)
        binname = hbinname
        tfeeder = lambda x: transDescFeeder(x)
    else:
        tfeeder = lambda x: x

    try:
        if use_nvdisasm:
            # text sections will be written into a new file and disassembled as binary
            ncnt = 0
            for outname in feedBinFromCubin(binname, outname=None, merge_all_kernels=merge_all_kernels):
                # NOTE: disassemble from binary will not show fixups, just raw disassembly
                sass_b = check_output(['nvdisasm', '-hex', '-c', '-b', b_arch, outname], stderr=STDOUT)
                sass = sass_b.decode()
                sio = StringIO(sass)

                feeder = tfeeder(CuInsFeeder(sio, insfilter=f_QNAN)) # filter QNAN for build
                ncnt += repos.update(feeder)
                
            if ncnt>0 and savname is not None: # only save when changed
                repos.save2file(savname)
        else:
            sass_b = check_output(['cuobjdump', '-sass', binname], stderr=STDOUT)
            sass = sass_b.decode()
            sio = StringIO(sass)
            feeder = tfeeder(CuInsFeeder(sio, insfilter=f_QNAN)) # filter QNAN for build
            repos.update(feeder)
        
    except CalledProcessError as cpe:
        CuAsmLogger.logWarning(cpe.output.decode().strip())
        CuAsmLogger.resetIndent()
        return None
    except Exception as e:
        CuAsmLogger.logError('Unknown subprocess error (%s)! '% str(e))
        CuAsmLogger.resetIndent()
        return None

def updateUnknownNVInfoWithCubin(binname, unknown_nvinfo:dict):
    ''' Update the unknown nvinfo dict with input cubin.'''
    with open(binname, 'rb') as fin:
        ef = ELFFile(fin)

        for sec in ef.iter_sections():
            if sec.name.startswith('.nv.info.'):
                nvinfo = CuNVInfo(sec.data())
        
                for attr, val in nvinfo.m_AttrList:
                    if attr not in unknown_nvinfo:
                        CuAsmLogger.logWarning(f'Unknown NVInfo Attr ({attr}) : ({val})')
                        unknown_nvinfo[attr] = val

    return unknown_nvinfo

def updateReposWithPTX(repos:CuInsAssemblerRepos, ptxname, savname=None):
    binname = re.sub(r'\.ptx$', '.cubin', ptxname)
    arch = repos.getArchString()
    try:
        transPTXVersion(ptxname, arch=arch)
        check_output(['ptxas', '-arch', arch, '-o', binname, ptxname])
        updateReposWithCubin(repos, binname, savname=savname)
    except CalledProcessError as cpe:
        CuAsmLogger.logWarning(cpe.output.decode().strip())
        CuAsmLogger.resetIndent()
        return None
    except Exception as e:
        CuAsmLogger.logError('Unknown subprocess error (%s)! '% str(e))
        CuAsmLogger.resetIndent()
        return None
        
class CudaBinFile:
    def __init__(self, name) -> None:
        self.name = name

        self.TempBinName = f'cuasm_{os.getpid():x}.tmp.cubin'
        self.TempPTXName = f'cuasm_{os.getpid():x}.tmp.ptx'

    def resetFileName(self, name):
        self.name = name

    def dumpFile(self, fname, arch=None, ftype=None):
        ''' Dump cubin or ptx from fat binary.

            Args:
                fname : output cubin/ptx name
                arch  : arch type, may be auto determined from fname
                ftype : 'elf' or 'ptx', may be auto determined from fname
        '''
        if ftype is None:
            if fname.endswith('.cubin'):
                ftype = 'elf'
            elif fname.endswith('.ptx'):
                ftype = 'ptx'
            else:
                raise ValueError('Unknown file type for %s!!!'%fname)

        if arch is None:
            res = arch_pattern.search(fname)
            arch = res.group('arch')

        xopt = '-x' + ftype

        if ftype == 'elf':
            check_output(['cuobjdump', xopt, fname, '-arch', arch, self.name])
        elif ftype == 'ptx':
            check_output(['cuobjdump', xopt, fname, self.name])

    def updateUnknownNVInfo(self, arch, unknown_nvinfo=None):
        if unknown_nvinfo is None:
            unknown_nvinfo = {}

        elflist = self.listFile(arch=arch)
        nelf = len(elflist)
        for ielf, fname in enumerate(elflist):
            try:
                self.dumpFile(fname, arch=arch)
                shutil.move(fname, self.TempBinName)

                CuAsmLogger.logProcedure(f'#### {ielf:4d}/{nelf} : Updating with {fname} ...')
                updateUnknownNVInfoWithCubin(self.TempBinName, unknown_nvinfo)
            except CalledProcessError as cpe:
                CuAsmLogger.logError(cpe.output.decode())
                CuAsmLogger.logError(traceback.format_exc())
                continue
            except Exception as e:
                CuAsmLogger.logError(traceback.format_exc())
                continue        
        
        return unknown_nvinfo

    def iterELF(self, arch):
        elflist = self.listFile(ftype='elf', arch=arch)
        for elf in elflist:
            yield elf

    def iterPTX(self):
        ptxlist = self.listFile(ftype='ptx', arch=None)
        for ptx in ptxlist:
            yield ptx

    def updateRepos(self, repos:CuInsAssemblerRepos, savename:str, ftype={'elf'}):
        arch = repos.getArchString().lower()
        if 'elf' in ftype:
            elflist = self.listFile(arch=arch)
            nelf = len(elflist)
            for ielf, fname in enumerate(elflist):
                try:
                    self.dumpFile(fname, arch=arch)
                    shutil.move(fname, self.TempBinName)
                    CuAsmLogger.logProcedure(f'#### {ielf:4d}/{nelf} : Updating with {fname} ...')
                    updateReposWithCubin(repos, self.TempBinName, savename)
                except CalledProcessError as cpe:
                    CuAsmLogger.logError(cpe.output.decode())
                    CuAsmLogger.logError(traceback.format_exc())
                    continue
                except Exception as e:
                    CuAsmLogger.logError(traceback.format_exc())
                    continue
        
        if 'ptx' in ftype:
            ptxlist = self.listFile(ftype='ptx')
            nptx = len(ptxlist)
            for iptx, fname in enumerate(ptxlist):
                try:
                    self.dumpFile(fname, arch=arch)
                    shutil.move(fname, self.TempPTXName)

                    transPTXVersion(self.TempPTXName, arch=arch)

                    CuAsmLogger.logProcedure(f'#### {iptx:4d}/{nptx} : Updating with {fname} ...')
                    updateReposWithPTX(repos, self.TempPTXName, savename)
                except CalledProcessError as cpe:
                    CuAsmLogger.logError(cpe.output.decode())
                    CuAsmLogger.logError(traceback.format_exc())
                    continue
                except Exception as e:
                    CuAsmLogger.logError(traceback.format_exc())
                    continue
        
    def listFile(self, ftype='elf', arch='sm_75'):
        # PTX file    1: cublas64_11.1.sm_80.ptx
        # ELF file 1840: cublas64_11.1840.sm_61.cubin
        
        lopt = '-l' + ftype
        try:
            if ftype == 'elf':
                outlist = check_output(['cuobjdump', lopt, '-arch', arch, self.name])
            elif ftype == 'ptx': # no arch specifier for ptx, may be modified 
                outlist = check_output(['cuobjdump', lopt, self.name])

        except CalledProcessError as cpe:
            CuAsmLogger.logError(cpe.output.decode())
            CuAsmLogger.logError(traceback.format_exc())
            return []
        except Exception as e:
            CuAsmLogger.logError(traceback.format_exc())
            return []

        flist = []
        for line in outlist.decode().splitlines():
            fname = parseListLine(line)
            if fname is not None:
                flist.append(fname)

        return flist

    def iterProcess(self, callback, *, arch, ftype='elf'):
        ''' Iteratively process cubin/ptx inside a bin file, calling callback for every file.
            
            Args:
                callback : callable accept binname/ptxname as input
                arch     : arch
                ftype    : 'elf' or 'ptx'
            
            NOTE: the cubin will be dumped as is, do desc hack in callback if necessary.
        '''
        if ftype == 'elf':
            tname = self.TempBinName
        elif ftype == 'ptx':
            tname = self.TempPTXName
        else:
            raise ValueError(f'Unknown ftype {ftype}!!!')

        flist = self.listFile(ftype=ftype, arch=arch)
        nfile = len(flist)
        for i, fname in enumerate(flist):
            try:
                self.dumpFile(fname, arch=arch, ftype=ftype)
                shutil.move(fname, tname)

                CuAsmLogger.logProcedure(f'#### {i:4d}/{nfile} : Updating with {fname} ...')
                callback(tname)
            except CalledProcessError as cpe:
                CuAsmLogger.logError(cpe.output.decode())
                CuAsmLogger.logError(traceback.format_exc())
                continue
            except Exception as e:
                CuAsmLogger.logError(traceback.format_exc())
                continue

def transDescFeeder(feeder):
    ''' Transform input feeder for desc hack.
    
        If no 'desc[' in assembly, the desc bit will be removed
    '''
    for addr, code, asm, ctrl in feeder:
        if 'desc' not in asm:
            code = code ^ (code & pShowDesc) # set show desc bit to 0
        
        yield addr, code, asm, ctrl

def updateNVInfoForArch(fpattern_list, arch):
    unknown_nvinfo = {}

    for fpattern in fpattern_list:
        for fname in f_glob(fpattern):
            CuAsmLogger.logEntry(f'Processing dll file {fname}')
            try:
                cbf = CudaBinFile(fname)
                unknown_nvinfo = cbf.updateUnknownNVInfo(arch=arch, unknown_nvinfo=unknown_nvinfo)
            except Exception as e:
                CuAsmLogger.logError(traceback.format_exc())
                continue

def iterProcessFilesFromBinFiles(fpattern_list, arch, callback, *, ftype='elf', callback2=None, callback3=None):
    ''' Iterate all elf or ptx files from glob patterns, calling callback for every cubin/ptx.
    
        Args:
            fpattern_list : list of glob patterns, NOTE: glob will not go into directives recursively
            arch          : arch string such as 'sm_75'
            callback      : callable which will be called for every cubin/ptx with fname as only input
            callback2     : will be called for every bin/ptx file (can be none), no args;
            callback3     : will be called for every file pattern (can be none), no args; 
    '''
    for fpattern in fpattern_list:
        for fname in f_glob(fpattern):
            CuAsmLogger.logEntry(f'#### Processing file {fname}')
            try:
                cbf = CudaBinFile(fname)
                cbf.iterProcess(callback, arch=arch, ftype=ftype)
                if callback2 is not None:
                    callback2()
            except Exception as e:
                CuAsmLogger.logError(traceback.format_exc())
                CuAsmLogger.logError(str(e))
                continue
        
        if callback3 is not None:
            callback3()

@CuAsmLogger.logTimeIt
def hackCubinDesc(fin, fout, always_output=True):
    ''' Hack sm_8x cubin with desc bit set, which makes the desc show explicitly.

        For Ampere(SM80/86) or maybe future SM, the default Cache-policy desc is not displayed.
        This may lead to some problems during assembling.
    
        Args:
            fin  : input file name
            fout : output file name
            always_output : copy the src to dst even if the hack is not needed.
                            set this to False to skip unnecessary copy.
        Return:
            True for hacked, False for not hacked.
    '''
    with open(fin, 'rb') as fin_s:
        fin_s.seek(0)
        bs = fin_s.read()
        
        bio = BytesIO(bs)
        ef = ELFFile(fin_s)
        smv = ef.header['e_flags'] & 0xff
        if smv // 10 < 8: # SM major version < 8 means pre-Ampere
            if always_output:
                shutil.copy(fin, fout)
            return False
        
        slist = []
        for i, s in enumerate(ef.iter_sections()):
            if s.name.startswith('.text.'):
                slist.append((s.name, s.header.sh_offset, s.header.sh_size))
        
        for sname, offset, size in slist:
            fin_s.seek(offset)
            bio.seek(offset)
            
            for i in range(0, size, 16):
                q1, q2 = struct.unpack('QQ', fin_s.read(16))
                q2 = q2 | (1<<37)
                
                vs = struct.pack('QQ', q1, q2)
                bio.write(vs)
    
    with open(fout, 'wb') as fout_s:
        bv = bio.getvalue()
        fout_s.write(bv)
    
    return True

# @CuAsmLogger.logTimeIt
def fixCubinDesc(fin, fout):
    ''' Make a cubin with only valid desc bit set. 
        
        Similar to hackCubinDesc, but no desc bit set for irrelevant instructions.
    
        NOTE: currently the rules when desc will be used is not quite clear. 
              Thus this function will dump the sass first, and set the bit according
              to disassembly, hence significantly slower than hackCubinDesc. 
    '''
    
    # first set all desc bit to 1
    with open(fin, 'rb') as fin_s:
        fin_s.seek(0)
        fbytes = fin_s.read()  # bs is read-only, will be reused later
        
        bio = BytesIO(fbytes)
        ef = ELFFile(fin_s)
        smv = (ef.header['e_flags'] & 0xff) // 10
        if smv < 8: # SM major version < 8 means pre-Ampere, no need to hack/fix
            CuAsmLogger.logProcedure(f'Cubin ({fin}) with SM major version {smv} does not need desc hack! Skipping...')
            return False
        
        CuAsmLogger.logProcedure(f'Hacking cubin bytes from {fin} ...')
        sec_dict = {}
        for i, s in enumerate(ef.iter_sections()):
            if s.name.startswith('.text.'):
                sec_dict[s.name] = s.header.sh_offset, s.header.sh_size
        
        for _, (offset, size) in sec_dict.items():
            fin_s.seek(offset)
            bio.seek(offset)
            
            for i in range(0, size, 16):
                q1, q2 = struct.unpack('QQ', fin_s.read(16))
                q2 = q2 | (1<<37)
                
                vs = struct.pack('QQ', q1, q2)
                bio.write(vs)
    
    # write to tmp cubin
    tmpname = getTempFileName(suffix='cubin')
    CuAsmLogger.logProcedure(f'Writing hacked cubin to {tmpname} ...')
    with open(tmpname, 'wb') as fout_s:
        bv = bio.getvalue()
        fout_s.write(bv)

    # dump sass with all desc bit set to 1
    # CuAsmLogger.logProcedure(f'Dumping hacked sass from {tmpname} ...')
    sass_b = check_output(['nvdisasm', '-hex', '-c', tmpname], stderr=STDOUT)
    sass = sass_b.decode()
    os.remove(tmpname)
    
    sio = StringIO(sass)
    feeder = CuInsFeeder(sio)

    desc_dict = {} # key = kernel name, value = addr list of desc instructions
    # find all instructions with valid desc
    CuAsmLogger.logProcedure(f'Locating valid desc instructions ...')
    for addr, _, asm, _ in feeder:
        if 'desc[' in asm:
            kname = feeder.CurrFuncName
            if kname not in desc_dict:
                desc_dict[kname] = [addr]
            else:
                desc_dict[kname].append(addr)

    bio = BytesIO(fbytes)
    for kname in desc_dict:
        addr_list = desc_dict[kname]
        secname = '.text.' + kname
        sec_off, sec_size = sec_dict[secname]
        for addr in addr_list:
            offset = sec_off + addr
            bio.seek(offset)

            q1, q2 = struct.unpack('QQ', bio.read(16))
            q2 = q2 | (1<<37)
            vs = struct.pack('QQ', q1, q2)

            bio.seek(offset) # bio.read has advanced 16B, go back to overwrite it
            bio.write(vs)
    
    CuAsmLogger.logProcedure(f'Writing fixed cubin to {fout} ...')
    with open(fout, 'wb') as fout_s:
        bv = bio.getvalue()
        fout_s.write(bv)

    return True

if __name__ == '__main__':
    pass
    