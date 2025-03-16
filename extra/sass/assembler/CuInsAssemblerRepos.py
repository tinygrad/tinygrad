# -*- coding: utf-8 -*-

import re
import time
import struct
import traceback
from extra.sass.assembler.common import reprDict

from extra.sass.assembler.CuInsParser import CuInsParser
from extra.sass.assembler.CuInsAssembler import CuInsAssembler
from extra.sass.assembler.CuSMVersion import CuSMVersion
from extra.sass.assembler.CuAsmLogger import CuAsmLogger
from extra.sass.assembler.config import Config

from io import StringIO
from sympy import Matrix
import os

class CuInsAssemblerRepos():
    ''' A repository consists of a set of instruction assemblers.

        TODO: Version control? Should work with CuInsParser/CuInsFeeder.
    '''
    StaticRepos = {}

    def __init__(self, InsAsmDict=None, arch=None):
        self.resetArch(arch)

        if InsAsmDict is None:
            self.reset(None)
        elif isinstance(InsAsmDict, str):
            self.initFromFile(InsAsmDict)
        elif isinstance(InsAsmDict, dict):
            self.reset(InsAsmDict)
        else:
            raise ValueError('Unknown input type of InsAsmDict!')
    
    def resetArch(self, arch):
        if arch is not None:
            self.m_Arch = CuSMVersion(arch)
            self.m_InsParser = CuInsParser(arch)
        else:
            self.m_Arch = None
            self.m_InsParser = None

    def convertArch(self, arch):
        dst_arch = CuSMVersion(arch)
        if dst_arch == self.m_Arch:
            return
        
        self.resetArch(dst_arch)
        for k, v in self.m_InsAsmDict.items():
            v.m_Arch = dst_arch

    def setToDefaultInsAsmDict(self):
        vnum = self.m_Arch.getVersionNumber()
        fname = Config.getDefaultInsAsmReposFile(vnum)
        if os.path.isfile(fname):
            self.initFromFile(fname)
        else:
            # No default InsAsmRepos, but the encoding can be copied from another version
            if vnum in CuSMVersion.InsAsmReposAliasDict:
                anum = CuSMVersion.InsAsmReposAliasDict[vnum]
                aname = Config.getDefaultInsAsmReposFile(anum)
                if os.path.isfile(aname):
                    CuAsmLogger.logWarning(f'No default InsAsmRepos for SM_{vnum} found! Use SM_{anum} instead...')
                    self.initFromFile(aname)
                    self.convertArch(anum)
                    return
            
            CuAsmLogger.logError(f'No default or alias InsAsmRepos for SM_{vnum} found! Use empty repos ...')
            self.reset()

    @staticmethod
    def getDefaultRepos(arch) -> 'CuInsAssemblerRepos':
        repos = CuInsAssemblerRepos(arch=arch)
        repos.setToDefaultInsAsmDict()
        return repos

    @staticmethod
    def getStaticRepos(arch) -> 'CuInsAssemblerRepos':
        ''' Get a static repos for arch.
        
            NOTE: The purpose of this method is to avoid multiple instantiation.
                  Usually static repos will be read-only.
                  If it's read/write, be cautious for alias.  
        '''
        if arch not in CuInsAssemblerRepos.StaticRepos:
            CuInsAssemblerRepos.StaticRepos[arch] = CuInsAssemblerRepos.getDefaultRepos(arch)
        
        return CuInsAssemblerRepos.StaticRepos[arch]

    def reset(self, InsAsmDict=None):
        if InsAsmDict is None:
            self.m_InsAsmDict = {}
        else:
            self.m_InsAsmDict = InsAsmDict

    def __getitem__(self, k):
        return self.m_InsAsmDict[k]
    
    def __setitem__(self, k, v):
        self.m_InsAsmDict[k] = v
    
    def __delitem__(self, k):
        del self.m_InsAsmDict[k]

    def __constains__(self, k):
        return k in self.m_InsAsmDict
    
    def __len__(self):
        return len(self.m_InsAsmDict)

    def __iter__(self):
        return iter(self.m_InsAsmDict)

    def items(self):
        return self.m_InsAsmDict.items()

    def initFromFile(self, fname):
        ''' Load repos from file. '''
        with open(fname,'r') as fin:
            fconts = fin.read()
            asm_repos = eval(fconts)
            self.m_InsAsmDict = asm_repos.m_InsAsmDict
        
        for k, v in self.m_InsAsmDict.items():
            if self.m_Arch is None:
                self.resetArch(v.m_Arch)
            elif v.m_Arch != self.m_Arch:
                CuAsmLogger.logWarning(f'InsAsm arch {v.m_Arch} of {k} does not match with repos {self.m_Arch}!!! Resetting...')
                self.resetArch(v.m_Arch)
            
            # only check the first insasm
            break
        
    def assemble(self, addr, s, precheck=True, showCandidates=True):
        ''' Try to assemble the input instruction string. 
        
            Raise KeyError when the ins_key is not found.
            if precheck is true, a ValueError will be raised if it cannot be assembled.
        '''
        ins_key, ins_vals, ins_modi = self.m_InsParser.parse(s, addr, 0)
        if ins_key not in self.m_InsAsmDict:
            msg = 'Unknown InsKey(%s) in Repos!' % ins_key
            if showCandidates:
                ckeys = self.getInsKeyCandidates(ins_key)
                msg += '\n    Available InsKeys: \n' + ckeys
                # print(msg)
            raise ValueError(msg)

        insAsm = self.m_InsAsmDict[ins_key]
        if precheck:
            brief, info = insAsm.canAssemble(ins_vals, ins_modi)
            if brief is not None:
                msg = 'Assembling failed (%s): %s'%(brief, info)
                if showCandidates:
                    msg += '\n    Known Records:\n'
                    for _, _, asm in insAsm.iterRecords():
                        msg += ' '*8 + asm + '\n'
                    # print(msg)
                raise ValueError(msg)

        code = insAsm.buildCode(ins_vals, ins_modi)
        return code

    @CuAsmLogger.logTimeIt
    def verify(self, feeder):
        ''' Verify current repos. 
        
            The feeder should yield (addr, code, asm, ctrl), but "ctrl" is not used.
        '''
        res = True
        t0 = time.time()
        cnt = 0
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            try:
                casm = self.assemble(addr, s)
                if code != casm:
                    CuAsmLogger.logError('Error when verifying :')
                    CuAsmLogger.logError('  ' + s)
                    CuAsmLogger.logError('  CodeOrg: %s'%self.m_Arch.formatCode(code))
                    CuAsmLogger.logError('  CodeAsm: %s'%self.m_Arch.formatCode(casm))
                    # raise Exception('Assembled code not match!')
            except Exception as e:
                CuAsmLogger.logError(str(e))
                CuAsmLogger.logError('Error when assembling :')
                CuAsmLogger.logError('  ' + s)
                CuAsmLogger.logError(traceback.format_exc())
                res = False

        t1 = time.time()

        if res:
            msg = "Verified %d ins in %8.3f secs." % (cnt, t1-t0)
            if t0!=t1:
                msg += "  ~%8.2f ins/s." % (cnt/(t1-t0))
            
            CuAsmLogger.logProcedure(msg)
        else:
            CuAsmLogger.logError("Verifying failed in %8.3f secs!!!" % (t1-t0))

        return res

    @CuAsmLogger.logTimeIt
    def update(self, feeder, ins_asm_dict=None):
        ''' Update the input instruction assembler dict with input from feeder.

            Args:
                feeder : yield (addr, code, asm, ctrl), but "ctrl" is not used.
                ins_asm_dict : destination dict
                    For ins_asm_dict=None(default), use the internal self.m_InsAsmDict as dst.
            Return:
                ncnt : number of new records, 0 for unchanged
        '''
        if ins_asm_dict is None:
            ins_asm_dict = self.m_InsAsmDict

        t0 = time.time()
        cnt = 0
        ncnt = 0
        
        for addr, code, s, ctrlcodes in feeder:
            cnt += 1
            # print('%#6x : %s'%(addr, s))
            ins_key, ins_vals, ins_modi = self.m_InsParser.parse(s, addr, code)
            # 

            if ins_key not in ins_asm_dict:
                ins_asm_dict[ins_key] = CuInsAssembler(ins_key, arch=self.m_Arch)

            ins_info = (addr, code, s)
            res_flag, res_info = ins_asm_dict[ins_key].push(ins_vals, ins_modi, code, ins_info)

            # if not res_flag:
            #     CuAsmLogger.logError("CuInsAsmRepos update error!!! Unmatched codes!")
            #     CuAsmLogger.logError('    Str : ' + s)
            #     CuAsmLogger.logError('    Addr: %#6x'%addr)
            #     CuAsmLogger.logInfo(repr(ins_asm_dict[ins_key]))
            if res_info in {'NewModi', 'NewVals', 'NewConflict'}:
                ncnt += 1
                
        t1 = time.time()
        msg = "Updated %d ins (%d new) in %8.3f secs ." % (cnt, ncnt, t1-t0)
        if (t0!=t1):
            msg += "  ~%8.2f ins/s." % (cnt/(t1-t0))
        
        CuAsmLogger.logProcedure(msg)
        
        return ncnt

    @CuAsmLogger.logTimeIt
    def save2file(self, fname):
        CuAsmLogger.logEntry('Saving to %s...'%fname)
        with open(fname, 'w') as fout:
            fout.write(self.__repr__())

    @CuAsmLogger.logTimeIt
    def rebuild(self):
        ''' When the CuInsParser is updated, the meaning of ins value/modifier may have changed.
        
            Thus CuInsAsmRepos should be rebuilt from original input (saved in ins records)

            TODO: We may store some redundant records?
        '''

        tmp_ins_asm_dict = {}
        feeder = self.recordsFeeder()

        self.update(feeder, tmp_ins_asm_dict)
        self.m_InsAsmDict = tmp_ins_asm_dict
    
    @CuAsmLogger.logTimeIt
    def merge(self, merge_source):
        ''' Merge instruction assembler from another source.

            TODO: Check version?
        '''
        if isinstance(merge_source, (str,dict)):
            repos = CuInsAssemblerRepos(merge_source)
        elif isinstance(merge_source, CuInsAssemblerRepos):
            repos = merge_source
        else:
            raise TypeError('Unknown merge source type!')
        
        feeder = repos.recordsFeeder()
        self.update(feeder)

    def iterRecords(self, key_filter = None):
        ''' Iterate over all records from CuInsAssembler with keys filtered by key_filter.

            Return (addr, code, asm)
            key_filter can be:
                1. list/tuple/dict of keys
                2. string of re pattern
                3. re pattern
                4. None for all keys
                
        '''

        if key_filter is None:
            keys = self.m_InsAsmDict.keys()
        elif isinstance(key_filter, str):
            keys = filter(lambda x: re.match(key_filter, x), self.m_InsAsmDict.keys())
        elif isinstance(key_filter, re.Pattern):
            keys = filter(lambda x: key_filter.match(x), self.m_InsAsmDict.keys())
        elif isinstance(key_filter, (list, tuple, dict)): # any iteratable
            keys = key_filter
        elif callable(key_filter):
            keys = filter(key_filter, self.m_InsAsmDict.keys())
        else:
            raise TypeError(f'Unknown key_filter: {key_filter} with type {type(key_filter)}!!!')

        for ins_key in keys:
            for r in self.m_InsAsmDict[ins_key].iterRecords():
                yield r

    def recordsFeeder(self, key_filter=None):
        ''' A generator as internal instruction feeder, pulling from instruction records.
        
            Return (addr, code, asm, ctrl=0), same as CuInsFeeder.
            key_filter can be:
                1. list/tuple/dict of keys
                2. string of re pattern
                3. re pattern
                4. None for all keys
        '''

        # Records is a list of ins_info => (addr, code, s), control codes is not stored, thus return 0.
        for r in self.iterRecords(key_filter):
            yield r[0], r[1], r[2], 0

    def showErrRecords(self):
        for ins_key, ins_asm in self.items():
            # print(ins_key)
            if len(ins_asm.m_ErrRecords) > 0:
                print(f'#### ErrRecords for {ins_key}:')

            for k, (addr, code, s) in ins_asm.m_ErrRecords.items():
                print(f'  {addr:#08x} : {s}')
                print('      Org: %s' % self.m_Arch.formatCode(code))
                acode = self.assemble(addr, s)
                print('      Asm: %s' % self.m_Arch.formatCode(acode))
                diff = abs(acode - code)
                diffs = self.m_Arch.formatCode(diff)[2:].replace('0', ' ')

                print('     Diff:   %s' % diffs)

    def completePredCodes(self):
        ''' Some instructions seem very rarely appear with guard predicates.

            Thus when the instruction assemblers are gathered from ptxas output, 
            many of them will not be able to encode predicates.

            This may give some useful information as performance guidelines.
            However, there will be certainly some occasions predicates will be needed.
        '''

        feeder = self.genPredRecords()
        self.update(feeder)

    def clearErrRecords(self):
        for k in self.m_InsAsmDict:
            self.m_InsAsmDict[k].m_ErrRecords = {}

    def genPredRecords(self):
        ''' A generator that yields modified instruction info with predicates. '''
        for ins_key, ins_asm in self.m_InsAsmDict.items():
            ins_info = ins_asm.m_InsRecords[0]
            pred_ins_info = self.m_Arch.genPredCode(ins_info)
            if pred_ins_info is not None:
                yield pred_ins_info[0], pred_ins_info[1], pred_ins_info[2], 0

    def genUndefRecords(self):
        key = 'UNDEF'
        for v in [0x1, 0x2, 0x3]:
            yield 0x0, v, f'{key} {v:#x};', 0

    def getArchString(self):
        return self.m_Arch.getVersionString().lower()

    def getSMVersion(self):
        return self.m_Arch

    def __repr__(self):
        sio = StringIO()

        sio.write('CuInsAssemblerRepos(')
        reprDict(sio, self.m_InsAsmDict)
        sio.write(', arch=%s)'%self.m_Arch)

        return sio.getvalue()

    def __str__(self):
        return "CuInsAssemblerRepos(%d keys)" % len(self.m_InsAsmDict)

    def getInsKeyCandidates(self, key, n=5):
        from difflib import get_close_matches 
        keys = self.m_InsAsmDict.keys()
        cs = get_close_matches(key, keys, n=n)
        if len(cs)==0:
            return 'None'
        else:
            return '\n'.join([' '*8+s for s in cs])
    