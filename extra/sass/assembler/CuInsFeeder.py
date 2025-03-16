# -*- coding: utf-8 -*-

from io import StringIO
import re
from enum import Enum, auto
from extra.sass.assembler.CuSMVersion import CuSMVersion
from extra.sass.assembler.CuAsmLogger import CuAsmLogger
from extra.sass.assembler.CuControlCode import c_ControlStringLen, CuControlCode

class SassLineType(Enum):
    ''' Six types of lines in a dumped sass file:

        1. function name line:
            Function : _Z5ktestPmPi
        2. headerflags line:
            .headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
        3. ins with code (first code line of SM7x/8x, normal code line of SM5x/6x)
            /*0030*/                   UMOV UR11, 0x14b00000 ;     /* 0x14b00000000b7882 */
        4. ins without code (second ins line of dual-issue)
            /*5348*/                   LDG.E.U16 R28, [R28]  }
        5. code without ins (control code of SM5x/6x, second ins code of dual-issue, second code line of SM7x/8x)
            /* 0xeed2200000071c1c */
        6. others
            Fatbin elf code:
            ================
            arch = sm_35
            code version = [1,7]
            producer = <unknown>
            host = windows
            compile_size = 64bit
            ............
            (blank lines)

    '''
    # Pattern that contains an instruction string (including address and code)
    # NOTE: For maxwell/pascal, there may be braces "{}" for dual-issued instructions.
    InsCode    = 0, re.compile(r'^\s*\/\*(?P<addr>\w+)\*\/\s*\{?\s*(?P<asm>.*;)\s*\/\* (?P<code>.*) \*\/')

    # Pattern for the dual-issued instruction, the code is in next line.
    InsOnly    = 1, re.compile(r'^\s*\/\*(?P<addr>\w+)\*\/\s*(?P<asm>.*\})')

    #    
    CodeOnly   = 2, re.compile(r'^\s*\/\* (?P<code>0x[0-9a-f]{16}) \*\/')   

    # Function : _Z5ktestPmPi
    FuncName   = 3, re.compile(r'^\s*Function\s*:\s*(?P<func>.*)') 

    # .section	.text._ZN8xmma_trt13implicit_gemm24cuda_reorder_hmma_filterENS0_26Reorder_imma_filter_paramsE,"ax",@progbits
    SectionName = 4, re.compile(r'^\s*\.section\s*\.text\.(?P<sec>[^,\s]*)') 

    # .headerflags    @"EF_CUDA_SM86 EF_CUDA_PTX_SM(EF_CUDA_SM86)"
    # .headerflags	@"EF_CUDA_TEXMODE_UNIFIED EF_CUDA_64BIT_ADDRESS EF_CUDA_SM75 EF_CUDA_VIRTUAL_SM(EF_CUDA_SM75)"
    HeaderFlag = 5, re.compile(r'^\s*\.headerflags.*EF_CUDA_SM(?P<arch>\d+)')

    # all other lines	
    Others     = 6, re.compile('.*')

    def __init__(self, idx, pattern):
        self.idx = idx
        self.pattern = pattern

    def match(self, s):
        return self.pattern.match(s)
    
    def search(self, s):
        return self.pattern.search(s)

    @classmethod
    def getLineType(cls, line):
        for t in cls:
            r = t.match(line)
            if r:
                return t, r
        
        raise Exception(f'Unknown line type for line:\n  {line}')

    @classmethod
    def getCallbackArgs(cls, op, res):
        ''' Get call back args for current linetype.
        
            NOTE 1: return tuples
            NOTE 2: all ints are converted
        '''
        if op == cls.InsCode:
            return int(res.group('addr'), 16), res.group('asm'), int(res.group('code'), 16) 
        elif op == cls.InsOnly:
            return int(res.group('addr'), 16), res.group('asm')
        elif op == cls.CodeOnly:
            return int(res.group('code'), 16),
        elif op == cls.FuncName:
            return res.group('func'),
        elif op == cls.SectionName:
            return res.group('sec'),
        elif op == cls.HeaderFlag:
            return res.group('arch'),
        elif op == cls.Others:
            return res.group(),
        else:
            raise Exception(f'Unknown line type {op}!')

class ParserState(Enum):
    ''' State of the Parser.

    '''

    Ready        = auto() # may accept others, instructions, codes, funcnames
    WaitForFunc  = auto() # ???  not needed?
    WaitForArch  = auto() # Deprecated, the arch will be switched inline, thus no need to transfer to this state
    
    WaitForCode4 = auto()
    WaitForCode3 = auto()
    WaitForCode2 = auto()
    WaitForCode1 = auto()
    
    WaitForIns7  = auto()
    WaitForIns6  = auto()
    WaitForIns5  = auto()
    WaitForIns4  = auto()
    WaitForIns3  = auto()
    WaitForIns2  = auto()
    WaitForIns1  = auto()
    Invalid      = auto()

SLT = SassLineType
PS = ParserState

def IterNone():
    while True:
        yield None

class StateTransferMatrix:
    ''' The state transfer matrix is a dict of dict:
        
            tm = { s0 : {op0 : s1, op1 : s2, ...}, 
                   s1 : {op0 : s0, op1 : s0, ...}
                 }
        
        If s not in tm, s must be an invalid state of current parser.
        If op not in tm[s], op is an invalid op for current state.
    '''
    def __init__(self):
        self.TM = {}
    
    def __contains__(self, key):
        return key in self.TM
    
    def __getitem__(self, key):
        return self.TM[key]

    def __setitem__(self, key, value):
        self.TM[key] = value

    def addop(self, s, op, ts, callback=None):
        ''' Add a single link op(s) -> ts '''
        if s not in self.TM:
            self.TM[s] = {}
        
        self.TM[s][op] = ts, callback

    def addops(self, s, ops, tss, callbacks=IterNone):
        ''' Add a list/tuple of ops and transferred states.'''
        if s not in self.TM:
            self.TM[s] = {}
        
        for op, ts in zip(ops, tss):
            self.TM[s][op] = ts

    def addop_dict(self, s, opd):
        ''' Add ops and transferred states as a dict'''
        if s not in self.TM:
            self.TM[s] = {}
        
        # NOTE: keep current entries!!! Direct assign may overwrite them.
        for op, vs in opd.items():
            if isinstance(vs, tuple):
                self.TM[s][op] = vs # vs = (ts, callback)
            else:
                self.TM[s][op] = vs, None # with default callback to None

    def __str__(self):
        return 'StateTransferMatrix: ' + repr(self.TM)

class ParserStateMachine:
    def __init__(self, init_state, tr_matrix):
        self.state = init_state
        self.tr_matrix = tr_matrix

    def reset(self, init_state, tr_matrix):
        self.state = init_state
        self.tr_matrix = tr_matrix

    def feed(self, op):
        ''' Feed in an op and transfer the state.'''

        # current state don't accept the input op
        if op not in self.tr_matrix[self.state]:
            raise Exception(f'Invalid input {op} for state {self.state} (Unacceptable op)!!!')
        
        self.state = self.tr_matrix[self.state][op]
        
        # current state transferred to an invalid state
        if self.state not in self.tr_matrix or self.state == PS.Invalid:
            raise Exception(f'Invalid input {op} for state {self.state} (Invalid transferred state)!!!')
        
        return self.state

class CuInsFeeder():
    def __init__(self, fstream, archfilter=None, insfilter=None):
        """ Construct a instruction feeder, yield (addr, code, asm, ctrl).

        Args:
            fstream (str or file stream): file name or the file object
            arch (optional): should be a valid input for CuSMVersion. 
                 Defaults to None, means all arches should be processed.
            insfilter (optional): filter for lines. Usually used for feeding a perticular instruction.
                insfilter can be:
                1. regex pattern string
                2. regex pattern object
                3. a callable function that accepts a line string input
                Defaults to None (empty string also means None).
        """

        if isinstance(fstream, str):
            self.__mFileName = fstream
            self.__mFStream = open(self.__mFileName, 'r')
                
        else:
            self.__mFileName = None
            self.__mFStream = fstream

        # compile ins filter
        if insfilter is None or insfilter=='':
            self.__mInsFilterFun = lambda x: True
        elif isinstance(insfilter, str):
            p = re.compile(insfilter)
            self.__mInsFilterFun = lambda x: p.search(x)
        elif isinstance(insfilter, re.Pattern):
            self.__mInsFilterFun = lambda x: insfilter.search(x)
        elif callable(insfilter):
            self.__mInsFilterFun = insfilter
        else:
            raise TypeError(f'Unknown type of insfilter {insfilter}!')

        if archfilter is None or archfilter == '':
            self.__mArchFilterFun = lambda x: True
        else:
            arch = CuSMVersion(archfilter)
            self.__mArchFilterFun = lambda x: x==arch 

        self.__mLineNo = 0

        self.CurrFuncName = ''
        self.CurrArch = ''

        self.__SplitCodeList = lambda x: (x, x)
        self.__mAddrList = []
        self.__mAsmList  = []
        self.__mCodeList = []

        self.__TMs = {'default': self.__getTrMatrixDefault(), 
                      '3x'     : self.__getTrMatrixForSM_3x(),
                      '5x6x'   : self.__getTrMatrixForSM_5x6x(),
                      '7x8x'   : self.__getTrMatrixForSM_7x8x()}
        
        self.__CurrTM = self.__TMs['default']
        self.__mPState = PS.Ready

    @staticmethod
    def parseInsFilter(insfilter):
        if insfilter is None or insfilter=='':
            InsFilterFun = lambda x: True
        elif isinstance(insfilter, str):
            p = re.compile(insfilter)
            InsFilterFun = lambda x: p.search(x)
        elif isinstance(insfilter, re.Pattern):
            InsFilterFun = lambda x: insfilter.search(x)
        elif callable(insfilter):
            InsFilterFun = insfilter
        else:
            raise TypeError(f'Unknown type of insfilter {insfilter}!')
        
        return InsFilterFun

    def nextParseLine(self):
        ''' Parse next line.
        
            Return tuple(linetype, line, res)
                linetype : type of next line;
                line     : contents of next line;
                res      : re match object for linetype.
            
            Return (None, None, None) if lines are exhausted.
        '''
        line = self.readline()
        if len(line) == 0:
            return None, None, None

        linetype, res = SassLineType.getLineType(line)
        return linetype, line, res

    def __feedLineOp(self, op, *args):
        ''' Feed in an linetype op and transfer the state.'''
        
        # CuAsmLogger.logDebug(f'Line{self.__mLineNo:04d}: feedOp op:{op} with args:{args}')
        # current state don't accept the input op
        if op not in self.__CurrTM[self.__mPState]:
            raise Exception(f'Invalid input {op} for state {self.__mPState} (Unacceptable op) @Line{self.__mLineNo:04d}!!!')
        
        _, callback = self.__CurrTM[self.__mPState][op]
        if callback is not None:
            callback(*args)
        
        # callback may change __CurrTM (such as switchArch) !!!
        # thus the state should base on the new __CurrTM
        self.__mPState, _ = self.__CurrTM[self.__mPState][op]

        # current state transferred to an invalid state
        if self.__mPState not in self.__CurrTM or self.__mPState == PS.Invalid:
            raise Exception(f'Invalid input {op} for state {self.__mPState} (Invalid transferred state) @Line{self.__mLineNo:04d}!!!')
        
        return self.__mPState

    def __iter__(self):
        ''' yield (addr, code, asm, ctrl).
        
            NOTE: feeder will be re-initialized when the iterator is used again.
        '''
        self.restart()
        
        doFeed = False
        while True:
            linetype, line, res = self.nextParseLine()
            if linetype is None: # ended
                break
            elif linetype is SassLineType.HeaderFlag:
                # Skip filtered arches
                args = SassLineType.getCallbackArgs(linetype, res)
                ns = self.__feedLineOp(linetype, *args)
                doFeed = self.__mArchFilterFun(self.CurrArch)
                continue
            elif linetype in {SassLineType.FuncName, SassLineType.SectionName}:
                # func name line appears before arch line, thus need to be processed 
                args = SassLineType.getCallbackArgs(linetype, res)
                ns = self.__feedLineOp(linetype, *args)
                continue

            if not doFeed:
                continue

            args = SassLineType.getCallbackArgs(linetype, res)
            ns = self.__feedLineOp(linetype, *args)

            if ns == ParserState.Ready:
                for addr, code, asm, ctrl in self.__iterPopIns():
                    if self.__filterIns(asm):
                        yield addr, code, asm, ctrl

    @CuAsmLogger.logTimeIt
    def trans(self, fout, codeonly_line_mode='none'):
        ''' Translate an input sass to sass with control codes. 
            The sass input is usually obtained by `cuobjdump -sass fname > a.sass`.

            fout : output filename or stream object
            codeonly_line_mode :  
                whether to keep lines with only codes
                such as the control code line for sm5x/6x, and 2nd line for sm7x/8x

                'keep' : keep unchanged
                'none' : skipped (default)
            
            NOTE: the filter does not work for this function.
            NOTE 2: this function is not quite robust, not recommended for any hand-written sass.
        '''

        if isinstance(fout, str):
            fout_stream = open(fout, 'w+')
            need_close = True
        else:
            fout_stream = fout
        
        if codeonly_line_mode == 'keep':
            pCodeLine = lambda ctrl_str, l: f'{ctrl_str}  {l}\n'
        elif codeonly_line_mode == 'none':
            pCodeLine = lambda ctrl_str, l: None
        else:
            pass

        self.restart()

        line_buffers = []
        out_buffers = []
        while True:
            linetype, line, res = self.nextParseLine()
            if linetype is None:
                break
            
            line = line.rstrip()
            line_buffers.append( (line, linetype))
            
            args = SassLineType.getCallbackArgs(linetype, res)
            ns = self.__feedLineOp(linetype, *args)

            pre_lt = None
            line_len = -1
            if ns == ParserState.Ready:
                inslist = [ins for ins in self.__iterPopIns()]

                for l, lt in line_buffers:
                    if lt == SLT.InsCode:
                        __, __, __, ctrl = inslist.pop(0)
                        ctrl_str = self.formatCtrlCodeString(ctrl)
                        line = f'{ctrl_str}  {l}'
                        out_buffers.append(line + '\n')
                        if line_len == -1:
                            line_len = len(line)
                    elif lt == SLT.InsOnly:
                        __, __, __, ctrl = inslist.pop(0)
                        ctrl_str = self.formatCtrlCodeString(ctrl)
                        out_buffers.append(f'{ctrl_str}  {l.rstrip()}')
                    elif lt == SLT.CodeOnly:
                        if pre_lt == SLT.InsOnly:
                            lcode = l.strip()
                            out_buffers[-1] = out_buffers[-1], lcode                            
                        else:
                            ctrl_str = self.formatCtrlCodeString(0, phantom_mode=True)
                            oline = pCodeLine(ctrl_str, l)
                            if oline is not None:
                                out_buffers.append(oline)
                    else:
                        out_buffers.append(l+'\n')
                    
                    pre_lt = lt
                
                for oline in out_buffers:
                    if isinstance(oline, tuple):
                        slen = line_len - len(oline[0]) - len(oline[1])
                        nline = oline[0] + (slen*' ') + oline[1] + '\n'
                        fout_stream.write(nline)
                    else:
                        fout_stream.write(oline)

                line_buffers.clear()
                out_buffers.clear()
                
        if need_close:
            fout_stream.close()

    def extract(self, fout, *, func_filter=None, ins_filter=None):
        ''' Extracting kernel matching the filter to fout.

        Sometimes whole kernel sass is needed to check the context of an instruction, 
        this will help to identify some rules of instruction correlations.

            fout: output filename
            func_filter: filter for the function name, may be string/re.Pattern/callable
            ins_filter: filter for the instruction

        Match rules:
            1. when func_filter matched the name, output first matched kernel;
            2. when ins_filter matched an instruction, output the first kernel containing the instruction;
        '''
        buf = StringIO()
        do_dump = False

        InsFilterFun = CuInsFeeder.parseInsFilter(ins_filter)
        FuncFilterFun = CuInsFeeder.parseInsFilter(func_filter)

        def tryDump():
            if do_dump:
                if buf.tell() == 0:
                    print('Empty buffer! Nothing to dump...')
                    return False

                print('================================')
                print(buf.getvalue())
                with open(fout, 'w') as fout_stream:
                    print(f'Dump to file {fout}...')
                    fout_stream.write(buf.getvalue())
                return True
            else:
                return False
        
        while True:
            linetype, line, res = self.nextParseLine()

            if linetype is None:
                tryDump()
                break

            if linetype == SLT.FuncName:
                if tryDump():                    
                    break
                else:
                    if func_filter is not None:
                        if FuncFilterFun(res.group('func')):
                            do_dump = True
                    buf = StringIO()
            elif linetype in {SLT.InsCode, SLT.InsOnly, SLT.CodeOnly}:
                if InsFilterFun(line):
                    do_dump = True
            else:
                pass

            buf.write(line.rstrip() + '\n')
        
        if not do_dump:
            print('Nothing to dump...')

    def formatCtrlCodeString(self, ccode, phantom_mode=False):
        if self.CurrArch.getMajor()<5:
            return ''
        else:
            if phantom_mode:
                return ' '*(c_ControlStringLen+2) # +2 for "[]"
            else:
                return '[' + CuControlCode.decode(ccode) + ']'

    def __del__(self):
        '''Close the stream if the handler is owned by this feeder.'''

        if self.__mFileName is not None and not self.__mFStream.closed:
            self.__mFStream.close()

    def close(self):
        if not self.__mFStream.closed:
            self.__mFStream.close()
            self.__mLineNo = 0
            return True
        else:
            return False

    def restart(self):
        if self.__mFStream.seekable:
            self.__mFStream.seek(0)
            self.__mLineNo = 0
        else:
            raise Exception("This feeder cannot be restarted!")

    def readline(self):
        ''' A helper function for reading lines, with line number recorded.'''
        self.__mLineNo += 1
        return self.__mFStream.readline()

    def lines(self):
        ''' Iterator for reading the stream line by line. '''
        while True:
            line = self.readline()
            if len(line)>0:
                yield line
            else:
                break

    def tell(self):
        '''Report the progress of file or stream.'''

        return self.__mFStream.tell()
    
    def tellLine(self):
        '''Report current line number.'''

        return self.__mLineNo

#### subroutines for operation ins queue
    def __pushAddr(self, addr):
        self.__mAddrList.append(addr)
    
    def __pushAsm(self, asm):
        self.__mAsmList.append(asm)

    def __pushCode(self, code):
        self.__mCodeList.append(code)

    def __pushInsCode_3x(self, addr, asm, code):
        self.__pushAddr(addr)
        self.__pushAsm(asm)
        self.__pushCode(code)

    def __pushInsCode(self, addr, asm, code):
        self.__pushAddr(addr)
        self.__pushAsm(asm)
        self.__pushCode(code)

    def __pushInsOnly_5x6x(self, addr, asm):
        self.__pushAddr(addr)
        self.__pushAsm(asm)
    
    def __filterIns(self, asm):
        ''' Check whether current instruction can pass the filter.
        
            True for pass, False for filterred.
        '''
        return self.__mInsFilterFun(asm)

    def __SplitCodeList_3x(self, int_list):
        ''' Split code list to (ctrl_list, code_list).'''
        return [0 for _ in int_list], int_list
    
    def __SplitCodeList_5x6x(self, int_list):
        ''' Split code list to (ctrl_list, code_list).'''
        return CuSMVersion.splitCtrlCodeFromIntList_5x_6x(int_list)
    
    def __SplitCodeList_7x8x(self, int_list):
        ''' Split code list to (ctrl_list, code_list).'''

        cs = [int_list[i] + (int_list[i+1]<<64) for i in range(0, len(int_list), 2)]
        return CuSMVersion.splitCtrlCodeFromIntList_7x_8x(cs)

    def __iterPopIns(self):
        ''' Pop (addr, code, asm, ctrl) iteratively.'''
        clist, ilist = self.__SplitCodeList(self.__mCodeList)
        for addr, code, asm, ctrl in zip(self.__mAddrList,
                                         ilist,
                                         self.__mAsmList,
                                         clist):
            yield addr, code, asm, ctrl
        
        # clear current buffer
        self.__mAddrList = []
        self.__mCodeList = []
        self.__mAsmList = []
        
    def __setFuncName(self, func):
        self.CurrFuncName = func

    def __setSectionName(self, sec):
        ''' section name is .text.funcname. '''
        self.CurrFuncName = sec

    def __switchArch(self, arch):
        smversion = CuSMVersion(arch)
        self.CurrArch = smversion

        if smversion.getMajor() == 3:
            self.__CurrTM = self.__TMs['3x']
            self.__SplitCodeList = self.__SplitCodeList_3x
        elif smversion.getMajor() in {5,6}:
            self.__CurrTM = self.__TMs['5x6x']
            self.__SplitCodeList = self.__SplitCodeList_5x6x
        elif smversion.getMajor() in {7,8}:
            self.__CurrTM = self.__TMs['7x8x']
            self.__SplitCodeList = self.__SplitCodeList_7x8x
        else:
            raise NotImplementedError(f'ERROR! No implemented state machine for arch {smversion}!!!')

    def __emitMessage(self, msg):
        CuAsmLogger.logWarning(f'CuInsFeeder Message: {msg} @Line{self.__mLineNo-1:04d}')

#### Subroutines for constructing StateTransferMatrix
    def __getTrMatrixDefault(self):
        stm = StateTransferMatrix()
        stm.addop_dict(PS.Ready, 
                       {SLT.FuncName    : (PS.Ready, self.__setFuncName),
                        SLT.SectionName : (PS.Ready, self.__setSectionName),
                        SLT.HeaderFlag  : (PS.Ready, self.__switchArch),
                        SLT.Others      : (PS.Ready, None),
                        })
        return stm
    
    def __getTrMatrixForSM_3x(self):
        stm = StateTransferMatrix()

        stm.addop_dict(PS.Ready, 
                        { SLT.FuncName    : (PS.Ready, self.__setFuncName), 
                          SLT.SectionName : (PS.Ready, self.__setSectionName),
                          SLT.HeaderFlag  : (PS.Ready, self.__switchArch),
                          SLT.CodeOnly    : (PS.Ready, None),
                          SLT.InsCode     : (PS.Ready, self.__pushInsCode_3x), # sometimes the padded ins may just follow normal ins
                                                                               # no extra ctrl line.
                          SLT.Others      : (PS.Ready, None),
                        })

        return stm
    
    def __getTrMatrixForSM_3x_dep(self):
        stm = StateTransferMatrix()

        stm.addop_dict(PS.Ready, 
                        { SLT.FuncName : (PS.Ready, self.__setFuncName), 
                          SLT.CodeOnly : (PS.WaitForIns7, None),
                          SLT.InsCode  : (PS.Ready, None), # sometimes the padded ins may just follow normal ins, no ctrl line.
                          SLT.Others   : (PS.Ready, None),
                        })
        
        # Arch always follows FuncName, then wait for control code
        # stm.addop(PS.WaitForArch, SLT.HeaderFlag, PS.WaitForCode1, self.__switchArch)

        # 1 CodeOnly(control code) + 7 InsCode
        stm.addop(PS.WaitForCode1, SLT.CodeOnly, PS.WaitForIns7, None)
        
        # 7 InsCode as a chain, no explicit dual issue
        stm.addop(PS.WaitForIns7, SLT.InsCode, PS.WaitForIns6, self.__pushInsCode_3x)
        stm.addop(PS.WaitForIns6, SLT.InsCode, PS.WaitForIns5, self.__pushInsCode_3x)
        stm.addop(PS.WaitForIns5, SLT.InsCode, PS.WaitForIns4, self.__pushInsCode_3x)
        stm.addop(PS.WaitForIns4, SLT.InsCode, PS.WaitForIns3, self.__pushInsCode_3x)
        stm.addop(PS.WaitForIns3, SLT.InsCode, PS.WaitForIns2, self.__pushInsCode_3x)
        stm.addop(PS.WaitForIns2, SLT.InsCode, PS.WaitForIns1, self.__pushInsCode_3x)
        stm.addop(PS.WaitForIns1, SLT.InsCode, PS.Ready      , self.__pushInsCode_3x)

        return stm

    def __getTrMatrixForSM_5x6x(self):
        stm = StateTransferMatrix()

        stm.addop_dict(PS.Ready, 
                        { SLT.FuncName    : (PS.Ready, self.__setFuncName), 
                          SLT.SectionName : (PS.Ready, self.__setSectionName),
                          SLT.HeaderFlag  : (PS.Ready, self.__switchArch),
                          SLT.CodeOnly    : (PS.WaitForIns3, self.__pushCode),
                          SLT.Others      : (PS.Ready, None),
                        })
        

        # 1 CodeOnly(Control code) + 3 InsCode
        # stm.addop(PS.WaitForCode4, SLT.CodeOnly, PS.WaitForIns3, self.__pushCtrl_5x6x)
        
        # 3->2 
        stm.addop(PS.WaitForIns3, SLT.InsCode , PS.WaitForIns2, self.__pushInsCode)
        # wait for code of the ins (dual-issued: 3/3 of last pack4 + 1/3 of current pack4)
        stm.addop(PS.WaitForIns3, SLT.InsOnly,  PS.WaitForCode3, self.__pushInsOnly_5x6x)
        stm.addop(PS.WaitForCode3, SLT.CodeOnly,  PS.WaitForIns2, self.__pushCode)

        # 2->1
        stm.addop(PS.WaitForIns2, SLT.InsCode, PS.WaitForIns1, self.__pushInsCode)
        # wait for code of the ins (dual-issued: 1/3 + 2/3 of current pack4)
        stm.addop(PS.WaitForIns2, SLT.InsOnly,  PS.WaitForCode2, self.__pushInsOnly_5x6x)
        stm.addop(PS.WaitForCode2, SLT.CodeOnly,  PS.WaitForIns1, self.__pushCode)
        
        # 1-> ready
        stm.addop(PS.WaitForIns1, SLT.InsCode, PS.Ready, self.__pushInsCode)
        # wait for code of the ins (dual-issued: 2/3 + 3/3 of current pack4)
        stm.addop(PS.WaitForIns1, SLT.InsOnly,  PS.WaitForCode1, self.__pushInsOnly_5x6x)
        stm.addop(PS.WaitForCode1, SLT.CodeOnly,  PS.Ready, self.__pushCode)

        return stm

    def __getTrMatrixForSM_7x8x(self):
        stm = StateTransferMatrix()

        stm.addop_dict(PS.Ready, 
                        { SLT.FuncName    : (PS.Ready, self.__setFuncName),
                          SLT.SectionName : (PS.Ready, self.__setSectionName), 
                          SLT.HeaderFlag  : (PS.Ready, self.__switchArch),
                          SLT.InsCode     : (PS.WaitForCode1, self.__pushInsCode),
                          SLT.CodeOnly    : (PS.Ready, lambda x: self.__emitMessage('Missing Instruction')),
                          SLT.Others      : (PS.Ready, None),
                        })

        # 1 InsCode + 1 CodeOnly
        stm.addop(PS.WaitForIns1, SLT.InsCode, PS.WaitForCode1, self.__pushInsCode)
        stm.addop(PS.WaitForCode1, SLT.CodeOnly, PS.Ready, self.__pushCode)

        return stm

if __name__ == '__main__':
    pass
