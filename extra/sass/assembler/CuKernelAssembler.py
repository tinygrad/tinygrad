# -*- coding: utf-8 -*-

from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from extra.sass.assembler.CuInsAssembler import CuInsAssembler
from extra.sass.assembler.CuSMVersion import CuSMVersion
from extra.sass.assembler.CuInsParser import CuInsParser
from extra.sass.assembler.CuAsmLogger import CuAsmLogger
from extra.sass.assembler.CuControlCode import CuControlCode

class CuKernelAssembler():
    # Opcodes that may have some associated attributes (NVInfo)
    # Standard form of the function: __AutoAttr_{$Opcode}({info, addr, ins_parser)

    def __init__(self, ins_asm_repos=None, version='sm_75'):
        if ins_asm_repos is None:
            self.m_InsAsmRepos = None
        elif isinstance(ins_asm_repos, str):
            self.initInsAsmRepos(ins_asm_repos)
        elif isinstance(ins_asm_repos, CuInsAssemblerRepos):
            self.m_InsAsmRepos = ins_asm_repos
        else:
            raise Exception("Unknown input for CuKernelAssembler!")
        
        self.m_Arch = CuSMVersion(version)
        self.reset()

    def reset(self):
        self.m_CCodeList = []
        self.m_ICodeList = []
        self.m_ExtraInfo = {}       # Extra info, such as reg/bar count, auto-detected nvinfo
                                    # TODO: Not implement yet

        self.m_InsIdx = 0
        self.m_CodeBytes = None

        self.AutoAttrOpcodeCallback = {
            'EXIT' : CuKernelAssembler.__AutoAttr_EXIT,
            'S2R'  : CuKernelAssembler.__AutoAttr_S2R,
            'BAR'  : CuKernelAssembler.__AutoAttr_BAR,
            'SHFL' : CuKernelAssembler.__AutoAttr_SHFL,
            'VOTE' : CuKernelAssembler.__AutoAttr_VOTE,
            'DMMA' : CuKernelAssembler.__AutoAttr_MMA, # all MMA instructions share the same callback
            'HMMA' : CuKernelAssembler.__AutoAttr_MMA,
            'IMMA' : CuKernelAssembler.__AutoAttr_MMA,
            'BMMA' : CuKernelAssembler.__AutoAttr_MMA
            }

    def initInsAsmRepos(self, fname):
        self.m_InsAsmRepos = CuInsAssemblerRepos(fname)
    
    def push(self, addr, icode_s, ccode_s):
        ''' Push in a new instruction.

            Note fixups should be filled before pushing, including relocations.
        '''
        
        # offset = self.m_Arch.getInsOffsetFromIndex(self.m_InsIdx)

        ccode = CuControlCode.encode(ccode_s)
        icode = self.m_InsAsmRepos.assemble(addr, icode_s)

        # Generate some attributes for special set of opcodes
        ins_op = self.m_InsAsmRepos.m_InsParser.m_InsOp
        if ins_op in self.AutoAttrOpcodeCallback:
            auto_attr_fun = self.AutoAttrOpcodeCallback[ins_op]

            # extra info will be updated in the function
            auto_attr_fun(self.m_ExtraInfo, addr, self.m_InsAsmRepos.m_InsParser)
            
        self.m_InsIdx += 1

        self.m_CCodeList.append(ccode)
        self.m_ICodeList.append(icode)

    def genCode(self):
        self.m_CodeBytes = self.m_Arch.mergeCtrlCodes(self.m_ICodeList, self.m_CCodeList)
        return self.m_CodeBytes
        
    def getCodeBytes(self):
        return self.m_CodeBytes

    @staticmethod
    def __AutoAttr_EXIT(info, addr, ins_parser):
        '''EIATTR_EXIT_INSTR_OFFSETS'''

        attr = 'EIATTR_EXIT_INSTR_OFFSETS'
        if attr not in info:
            info[attr] = [addr]
        else:
            info[attr].append(addr)

    @staticmethod
    def __AutoAttr_S2R(info, addr, ins_parser:CuInsParser):
        ''' EIATTR_S2RCTAID_INSTR_OFFSETS
            EIATTR_CTAIDZ_USED '''

        if ins_parser.m_InsKey == 'S2R_R_L':
            sreg = None
            for modi in ins_parser.m_InsModifier:
                if modi.startswith('2_'):
                    sreg = modi[2:]
            
            if sreg is None:
                CuAsmLogger.logWarning('Unknown SREG for S2R_R_L!!!')
                return

            if not sreg.startswith('SR_CTAID'):
                return

            attr = 'EIATTR_S2RCTAID_INSTR_OFFSETS'
            if attr not in info:
                info[attr] = [addr]
            else:
                info[attr].append(addr)

            if sreg == 'SR_CTAID.Z':
                zattr = 'EIATTR_CTAIDZ_USED'
                if zattr not in info:
                    info[zattr] = 0

    @staticmethod
    def __AutoAttr_BAR(info, addr, ins_parser):
        ''' Set barrier number ${BARNUM}

            TODO: EIATTR_COOP_GROUP_INSTR_OFFSETS ???
        '''

        attr = 'BARNUM'
        bar_idx = ins_parser.m_InsVals[CuInsParser.OPERAND_VAL_IDX]

        if attr in info:
            info[attr] = max(bar_idx+1, info[attr])
        else:
            info[attr] = bar_idx + 1

    @staticmethod
    def __AutoAttr_SHFL(info, addr, ins_parser):
        ''' EIATTR_INT_WARP_WIDE_INSTR_OFFSETS 
        '''
        pass

    @staticmethod
    def __AutoAttr_VOTE(info, addr, ins_parser):
        ''' EIATTR_INT_WARP_WIDE_INSTR_OFFSETS '''
        pass

    @staticmethod
    def __AutoAttr_MMA(info, addr, ins_parser):
        ''' EIATTR_WMMA_USED '''

        attr = 'EIATTR_WMMA_USED'
        if attr not in info:
            info[attr] = 0
