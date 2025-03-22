# -*- coding: utf-8 -*-
from io import BytesIO
import struct
import re

p_QNAN = re.compile(r'(\+|-)QNAN\b')

def makeVersionDict(vlist):
    d = {}
    for v in vlist:
        d[f'SM{v:d}'] = v
        d[f'SM_{v:d}'] = v
        d[f'{v:d}'] = v
        d[v] = v
    return d

class CuSMVersion(object):
    ''' CuSMVersion will handle most of sm version related features, thus it's used everywhere.

        Note the same version will share the same instance, since there is no private member needed.

        TODO: Use a better form of version related attributes, rather than defined seperately.
              A class with default values?
    '''

    __InstanceRepos = {}

    SMVersionDict = makeVersionDict([35, 37, 
                                     50, 52, 53, 
                                     60, 61, 62, 
                                     70, 72, 75, 
                                     80, 86, 87, 89, 
                                     90])

    # Some versions do not have pre-gathered InsAsmRepos, but since the encoding may be almost identical
    # we may just copy the InsAsmRepos from another version     
    InsAsmReposAliasDict = {62:61, 72:75, 87:86}

    SMCodeNameDict = { 35:'Kepler',  37:'Kepler',
                       50:'Maxwell', 52:'Maxwell', 53:'Maxwell',
                       60:'Pascal',  61:'Pascal',  62:'Pascal',
                       70:'Volta',   72:'Turing',  75:'Turing',
                       80:'Ampere',  86:'Ampere',  87:'Ampere',
                       89:'Adalovelace', 90:'Hopper'}

    PadBytes_5x_6x  = bytes.fromhex('e00700fc00801f00 000f07000000b050 000f07000000b050 000f07000000b050')
    Pad_CCode_5x_6x = 0x7e0               # [----:B------:R-:W-:Y:S00]
    Pad_ICode_5x_6x = 0x50b0000000070f00  # NOP

    PadBytes_7x_8x  = bytes.fromhex('1879000000000000 0000000000c00f00')
    Pad_CCode_7x_8x = 0x7e0               # [----:B------:R-:W-:Y:S00]
    Pad_ICode_7x_8x = 0x7918              # NOP

    PredCode_5x_6x =  0xf0000
    PredCode_7x_8x =  0xf000

    B64Mask = (1<<64) - 1

    # Bit mask of control codes for sm5x/6x
    # NOTE: reuse bits not included 
    CCMask0_5x_6x = 0x000000000001ffff
    CCMask1_5x_6x = 0x0000003fffe00000
    CCMask2_5x_6x = 0x07fffc0000000000 

    # Bit mask of reuse bits
    CCReuse0_5x_6x = 0x00000000001e0000
    CCReuse1_5x_6x = 0x000003c000000000
    CCReuse2_5x_6x = 0x7800000000000000 

    # Bit position for control codes of sm5x/6x
    CCPos0_5x_6x = 0
    CCPos1_5x_6x = 21
    CCPos2_5x_6x = 42 

    # Bit mask of control codes for sm7x/8x
    # NOTE: reuse bits not included
    CCMask_7x_8x = 0x1ffff<<(64+41)
    
    # Bit position for control codes of sm7x/8x
    CCPos_7x_8x = 64+41

    RelMaps_7x_8x = {'32@hi' : 'R_CUDA_ABS32_HI_32',
                     '32@lo' : 'R_CUDA_ABS32_LO_32',
                     'target': 'R_CUDA_ABS47_34'}
    
    RelMaps_5x_6x = {'32@hi' : 'R_CUDA_ABS32_HI_20',
                     '32@lo' : 'R_CUDA_ABS32_LO_20',
                     'target': 'R_CUDA_ABS32_20'}

    # keep 20bits, but the sign bit is moved to neg modifier
    FloatImmeFormat_5x_6x = {'H':('e','H', 16, 16), 'F':('f','I', 32, 20), 'D':('d','Q', 64, 20)}
    FloatImmeFormat_7x_8x = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}

    # EIATTR_AutoGen is the EIATTR set can be handled by the assembler automatically
    # NOTE: REGCOUNT/PARAM* will be handled seperately

    EIATTR_AutoGen_7x_8x = set(['EIATTR_CTAIDZ_USED', 
                             'EIATTR_WMMA_USED',
                             'EIATTR_EXIT_INSTR_OFFSETS'])

    EIATTR_AutoGen_5x_6x = set(['EIATTR_CTAIDZ_USED', 
                             'EIATTR_WMMA_USED',
                             'EIATTR_EXIT_INSTR_OFFSETS',
                             'EIATTR_S2RCTAID_INSTR_OFFSETS'])
    
    EIATTR_ManualGen_7x_8x = set(['EIATTR_COOP_GROUP_INSTR_OFFSETS'])
    EIATTR_ManualGen_5x_6x = set()

    c_PosDepFuncs = set(['I2I', 'F2F', 'IDP', 'HMMA', 'IMMA', 'XMAD', 'IMAD', 'IMADSP',
                     'VADD', 'VMAD', 'VSHL', 'VSHR', 'VSET', 'VSETP', 'VMNMX',
                     'VABSDIFF', 'VABSDIFF4', 'TLD4', 'PSET', 'PSETP'])

    POSDEP_Opcodes_Common = set(['I2I', 'F2F', 'IDP', 'TLD4', 
                                 'VADD', 'VMAD', 'VSHL', 'VSHR', 
                                 'VSET', 'VSETP', 'VMNMX',
                                 'VABSDIFF', 'VABSDIFF4'])
    POSDEP_Opcodes_SM5x6x = POSDEP_Opcodes_Common.union(set(['XMAD', 'IMAD', 'IMAD32I', 'IMADSP', 
                                                             'IMUL', 'IMUL32I', 'PSET', 'PSETP']))
    POSDEP_Opcodes_SM7x = POSDEP_Opcodes_Common.union(set(['HMMA', 'IMMA']))
    POSDEP_Opcodes_SM8x = POSDEP_Opcodes_Common.union(set(['HMMA', 'IMMA', 'I2IP', 'F2FP']))
    
    def __init__(self, version):
        self.__mVersion = CuSMVersion.parseVersionNumber(version)
    
        self.__mMajor = self.__mVersion // 10
        self.__mMinor = self.__mVersion % 10
        if self.__mMajor<=6:
            self.__mFloatImmeFormat = self.FloatImmeFormat_5x_6x
            self.m_PosDepOpcodes = self.POSDEP_Opcodes_SM5x6x
            self.splitCtrlCodeFromBytes = self.splitCtrlCodeFromBytes_5x_6x
            self.splitCtrlCodeFromIntList = self.splitCtrlCodeFromIntList_5x_6x
            self.mergeCtrlCodes = self.mergeCtrlCodes_5x_6x
        else:
            self.__mFloatImmeFormat = self.FloatImmeFormat_7x_8x
            self.splitCtrlCodeFromBytes = self.splitCtrlCodeFromBytes_7x_8x
            self.splitCtrlCodeFromIntList = self.splitCtrlCodeFromIntList_7x_8x
            self.mergeCtrlCodes = self.mergeCtrlCodes_7x_8x

            if self.__mMajor == 7:
                self.m_PosDepOpcodes = self.POSDEP_Opcodes_SM7x
            elif self.__mMajor == 8:
                self.m_PosDepOpcodes = self.POSDEP_Opcodes_SM8x
            else:
                self.m_PosDepOpcodes = self.POSDEP_Opcodes_Common

    def __new__(cls, version, *args, **kwargs):
        ''' Create new instance if the version is not in repos.

            Otherwise return current corresponding instance.
        '''
        vnum = CuSMVersion.parseVersionNumber(version)
        if vnum not in CuSMVersion.__InstanceRepos:
            instance = super().__new__(cls)
            CuSMVersion.__InstanceRepos[vnum] = instance
        else:
            instance = CuSMVersion.__InstanceRepos[vnum]
        
        return instance

    def getMajor(self):
        return self.__mMajor
    
    def getMinor(self):
        return self.__mMinor

    def getVersionNumber(self):
        return self.__mVersion

    def getVersionString(self):
        return 'SM_%d'%self.__mVersion

    def getNOP(self):
        ''' Get NOP instruction code (no control codes).'''
        if self.__mMajor<=6:
            return self.Pad_ICode_5x_6x
        else:
            return self.Pad_ICode_7x_8x
    
    def getPadBytes(self):
        ''' Get padding bytes.

            NOTE: For sm_5x/6x, the padding byte length is 32B (1+3 group);
                  For sm_7x/8x, the padding byte length is 16B.
        '''

        if self.__mMajor <= 6:
            return CuSMVersion.PadBytes_5x_6x
        else:
            return CuSMVersion.PadBytes_7x_8x

    def getInstructionLength(self):
        ''' (At least) Since Kepler, SASS becomes a constant length ISA.

            5.x 6.x :  64bit =  8 bytes (1 control codes + 3 normal instructions)
            7.x 8.x : 128bit = 16 bytes
        '''
        if self.__mMajor<=6:
            return 8
        else:
            return 16

    def getInsOffsetFromIndex(self, idx):
        ''' Get instruction offset according to the instruction index.
        '''
        if self.__mMajor<=6:
            return (idx//3 + 1)*8 + idx*8
        else:
            return idx * 16

    def getInsIndexFromOffset(self, offset):
        ''' Get Instruction index according to the instruction offset.

            For SM_5x, SM_6x, offset should be multiple of  8
            For SM_7x, SM_8x, offset should be multiple of 16
        '''

        if self.__mMajor<=6:
            ridx = offset>>3
            if (ridx & 0x3) == 0: # Input is the control codes offset
                return -1
            v = (ridx>>2)*3 + (ridx & 0x3) - 1
            return v
        else:
            return offset >> 4

    def getNextInsAddr(self, addr):
        idx = self.getInsIndexFromOffset(addr)
        offset = self.getInsOffsetFromIndex(idx+1)
        return offset

    def getPrevInsAddr(self, addr):
        idx = self.getInsIndexFromOffset(addr)
        offset = self.getInsOffsetFromIndex(idx-1)
        return offset

    def getInsRelocationType(self, key):
        ''' Get Instruction relocation type from keys.

            Available keys: ["32@hi", "32@lo", "target"]
        '''
        if self.__mMajor<=6:
            return self.RelMaps_5x_6x[key]
        else:
            return self.RelMaps_7x_8x[key]
 
    def getTextSectionSizeUnit(self):
        ''' The text section should be padded to integer multiple of this unit.

            NOTE: This is different from the section align, which is applied to offset, not size.
        '''
        if self.__mMajor <= 6:
            return 64
        else:
            return 128

    def setRegCountInNVInfo(self, nvinfo, reg_count_dict):
        ''' Update NVInfo for regcount, only for SM_70 and above.

            reg_count_dict = {kernelname_symidx:regnum, ...}
            Return: flag for whether found and updated.
        '''
        if self.__mMajor<=6: # No this nvinfo for SM<=6x
            return nvinfo.setRegCount(reg_count_dict)
            # return True
        else:
            return nvinfo.setRegCount(reg_count_dict)

    def extractFloatImme(self, bs):
        ''' Not implemented yet. '''
        pass

    def convertFloatImme(self, fval, prec, nbits=-1):
        ''' Convert float immediate to value (and modifiers if needed).

            Input:
                fval : float in string
                prec : string, 'H':half / 'F':float / 'D':double
                nbits: int, how many bits to keep, -1 means default values of given precision, 
                       only for opcodes end with "32I" in sm5x/sm6x
            Return:
                value, [modi]

        '''

        fval = fval.lower().strip() #

        if self.__mMajor<=6:
            if fval.startswith('-'):
                val = fval[1:]
                modi = ['FINeg']
            elif fval.endswith('.neg'): # Only for maxwell/pascal ?
                val = fval[:-4]
                modi = ['ExplicitFINeg']
            else:
                val = fval
                modi = []
        else:
            val = fval
            modi = []

        if val.startswith('0f'):
            v = int(val[2:], 16)
            return v, modi
        else:
            fv = float(val)
            ifmt, ofmt, fullbits, keepbits = self.__mFloatImmeFormat[prec]
            fb = struct.pack(ifmt, fv)
            ival = struct.unpack(ofmt, fb)[0]

            trunc_bits = fullbits - max(nbits, keepbits)
            if trunc_bits>0:
                ival = ival >> trunc_bits
            
            return ival, modi

    def splitIntImmeModifier(self, ins_parser, int_val):
        if self.__mMajor<=6 and (not ins_parser.m_InsOp.endswith('32I')) and ((int_val & 0x80000) != 0):
            new_val = int_val - (int_val & 0x80000)
            modi = ['ImplicitNegIntImme']
            return [new_val], modi
        else:
            return [int_val], []

    def formatCode(self, code):
        if self.__mMajor<=6:  # 64 + 23bit
            return '0x%022x'%code
        else:                 # 128 bit
            return '0x%032x'%code

    def getHighestCodeBit(self):
        ''' Get a constant with highest code bit set to 1.'''
        if self.__mMajor<=6:
            return 2**63
        else:
            return 2**127

    def genPredCode(self, ins_info):
        ''' Generate instruction string with modified predicates.

            If the instruction already has predicates, return None.
        '''

        addr, code, s = ins_info
        if s.startswith('@'):
            # print(s)
            return None
        
        # CHECK: currently seems all uniform path opcode with uniform predicate starts with U
        #
        if s.startswith('U'):
            if s.startswith('UNDEF'): # UNDEF is reserved for un-disassembled instructions
                return None
            else:
                s2 = '@UP0 ' + s
        else:
            s2 = '@P0 ' + s

        if self.__mMajor<=6:
            pred = CuSMVersion.PredCode_5x_6x
        else:
            pred = CuSMVersion.PredCode_7x_8x

        # @P0 will set the predicate bit to zero
        code2 = code ^ (code & pred)

        return addr, code2, s2

    def getNVInfoAttrAutoGenSet(self):
        ''' Get NVInfo attribute set can be automatically generated by kernel assembler.

            TODO: Current list is not complete, check the implementation in class CuNVInfo.
        '''
        if self.__mMajor <= 6:
            return CuSMVersion.EIATTR_AutoGen_5x_6x
        else:
            return CuSMVersion.EIATTR_AutoGen_7x_8x

    def getNVInfoAttrManualGenSet(self):
        ''' Get NVInfo attribute set should be generated MANUALLY by kernel assembler.

            TODO: Current list is not complete, check the implementation in class CuNVInfo.
        '''
        if self.__mMajor <= 6:
            return CuSMVersion.EIATTR_ManualGen_5x_6x
        else:
            return CuSMVersion.EIATTR_ManualGen_7x_8x

    def needsDescHack(self):
        return self.__mMajor >= 8

    def hackDisassembly(self, code, asm):
        if self.__mMajor<=6:
            return CuSMVersion.hackDisassembly_5x_6x(code, asm)
        else:
            return CuSMVersion.hackDisassembly_7x_8x(code, asm)

    def __str__(self):
        return 'CuSMVersion(%d)'%self.__mVersion
    
    def __repr__(self):
        return 'CuSMVersion(%d)'%self.__mVersion

    @staticmethod
    def splitCtrlCodeFromBytes_5x_6x(codebytes:bytes):
        ''' Split Control codes and normal codes from bytes object.
        
            For 5.x~6.x, 1 64bit control codes + 3*64bit asm instructions.
            NOTE: Storing too many big int in python may be very memory consuming.
                  So this may be called segment by segment.

            Args:
                codebytes 

            Return:
                (ctrl_list, ins_list)
        '''
        # 32B for 1+3 group
        assert (len(codebytes) & 0x1f) == 0

        int_list = []

        bio = BytesIO(codebytes)
        bs = bio.read(8)
        while len(bs)==8:
            int_list.append(int.from_bytes(bs, 'little'))
            bs = bio.read(8)

        return CuSMVersion.splitCtrlCodeFromIntList_5x_6x(int_list)

    @staticmethod
    def splitCtrlCodeFromIntList_5x_6x(int_list:list):
        ''' Split Control codes and normal codes from a list of int.
        
            Args:
                int_list   a list of python ints.

            Return:
                (ins_list, ctrl_list)
        '''

        assert (len(int_list) & 0x3) == 0

        ctrl_code_list = []
        ins_code_list = []
        
        for i in range(0, len(int_list), 4):
            ccode, c0, c1, c2 = tuple(int_list[i:i+4])
            cc = [(ccode & CuSMVersion.CCMask0_5x_6x) >> CuSMVersion.CCPos0_5x_6x,
                  (ccode & CuSMVersion.CCMask1_5x_6x) >> CuSMVersion.CCPos1_5x_6x,
                  (ccode & CuSMVersion.CCMask2_5x_6x) >> CuSMVersion.CCPos2_5x_6x]

            c0 += ((ccode & CuSMVersion.CCReuse0_5x_6x) >> CuSMVersion.CCPos0_5x_6x ) << 64
            c1 += ((ccode & CuSMVersion.CCReuse1_5x_6x) >> CuSMVersion.CCPos1_5x_6x ) << 64
            c2 += ((ccode & CuSMVersion.CCReuse2_5x_6x) >> CuSMVersion.CCPos2_5x_6x ) << 64

            ctrl_code_list.extend(cc)
            ins_code_list.extend([c0, c1, c2])

        return ctrl_code_list, ins_code_list
    
    @staticmethod
    def splitCtrlCodeFromBytes_7x_8x(codebytes:bytes):
        ''' Split Control codes and normal codes from bytes object.

            Args:
                codebytes 

            Return:
                (ctrl_list, ins_list)
        '''

        # 16B for every instruction, should be aligned
        assert (len(codebytes) & 0xf) == 0

        int_list = []
        
        bio = BytesIO(codebytes)
        bs = bio.read(16) # 128bit
        while len(bs)==16:
            int_list.append(int.from_bytes(bs, 'little'))
            bs = bio.read(16)

        return CuSMVersion.splitCtrlCodeFromIntList_7x_8x(int_list)
    
    @staticmethod
    def splitCtrlCodeFromIntList_7x_8x(int_list:list):
        ''' Split Control codes and normal codes from a list of int.
        
            Args:
                int_list   a list of python ints.

            Return:
                (ins_list, ctrl_list)
        '''
        ctrl_code_list = []
        ins_code_list = []
        
        for c in int_list:
            cc = c & CuSMVersion.CCMask_7x_8x
            ic = c ^ cc
            cc = cc >> CuSMVersion.CCPos_7x_8x

            ctrl_code_list.append(cc)
            ins_code_list.append(ic)

        return ctrl_code_list, ins_code_list

    @staticmethod
    def remixCode_5x_6x(i0, i1, i2, c0, c1, c2):
        ''' Remix the group of control codes and normal codes for sm5x/6x.

            Args:
                i0, i1, i2   normal instruction code sequence
                c0, c1, c2   control code sequence
            Return:
                cc, mi0, mi1, mi2   int code list
        '''

        mc0 = c0 + (i0 >> 64)
        mc1 = c1 + (i1 >> 64)
        mc2 = c2 + (i2 >> 64)

        cc  = mc0<<CuSMVersion.CCPos0_5x_6x
        cc += mc1<<CuSMVersion.CCPos1_5x_6x
        cc += mc2<<CuSMVersion.CCPos2_5x_6x

        mi0 = i0 & CuSMVersion.B64Mask
        mi1 = i1 & CuSMVersion.B64Mask
        mi2 = i2 & CuSMVersion.B64Mask

        return cc, mi0, mi1, mi2

    @staticmethod
    def mergeCtrlCodes_5x_6x(ins_code_list, ctrl_code_list):
        n_ins = len(ins_code_list)
        if len(ctrl_code_list) != n_ins:
            raise Exception('Length of control codes(%d) != length of instruction(%d)!'
                            %(len(ctrl_code_list), n_ins))
        
        bio = BytesIO()
        nccode_intact =  n_ins // 3  # intact part of control code groups (1+3)
        for i in range(nccode_intact):
            cc, i0, i1, i2 = CuSMVersion.remixCode_5x_6x(*ins_code_list[3*i:3*(i+1)], *ctrl_code_list[3*i:3*(i+1)])
            bio.write(cc.to_bytes(8, 'little'))
            bio.write(i0.to_bytes(8, 'little'))
            bio.write(i1.to_bytes(8, 'little'))
            bio.write(i2.to_bytes(8, 'little'))

        if nccode_intact * 3 != n_ins:
            
            ntail = n_ins - nccode_intact*3
            t_ctrl_code_list = ctrl_code_list[3*nccode_intact:]
            t_ins_code_list  = ins_code_list[3*nccode_intact:]
            
            npad = 3 - ntail
            for i in range(npad):
                t_ctrl_code_list.append(CuSMVersion.Pad_CCode_5x_6x)
                t_ins_code_list.append(CuSMVersion.Pad_ICode_5x_6x)
            
            cc, i0, i1, i2 = CuSMVersion.remixCode_5x_6x(*t_ins_code_list, *t_ctrl_code_list)
            bio.write(cc.to_bytes(8, 'little'))
            bio.write(i0.to_bytes(8, 'little'))
            bio.write(i1.to_bytes(8, 'little'))
            bio.write(i2.to_bytes(8, 'little'))
        
        return bio.getvalue()

    @staticmethod
    def mergeCtrlCodes_7x_8x(ins_code_list, ctrl_code_list):
        n_ins = len(ins_code_list)
        if len(ctrl_code_list) != n_ins:
            raise Exception('Length of control codes(%d) != length of instruction(%d)!'
                            %(len(ctrl_code_list), n_ins))
        
        bio = BytesIO()
        for i in range(n_ins):
            code =  (ctrl_code_list[i]<<CuSMVersion.CCPos_7x_8x) + ins_code_list[i]
            bio.write(code.to_bytes(16, 'little'))

        return bio.getvalue()

    @staticmethod
    def parseVersionNumber(version):
        if isinstance(version, str):
            version = version.upper()
        
        if isinstance(version, CuSMVersion):
            version = version.__mVersion
        elif version in CuSMVersion.SMVersionDict:
            version = CuSMVersion.SMVersionDict[version]
        else:
            raise ValueError('Invalid SM version %s!!!' % version)
    
        return version

    @staticmethod
    def hackDisassembly_7x_8x(code, asm, blen=32):
        ''' Hack the disassembly so that it can be reassembled exactly.
        
            Currently only QNAN is handled.

            For sm7x/8x, FP imme is always 32bit.

            Return hacked(or original) asm.
        '''

        if p_QNAN.search(asm): # 
            fimm = (code & (0xffffffff<<32))>>32
            asm = p_QNAN.sub(f'0f{fimm:08x}', asm)

        return asm

    @staticmethod
    def hackDisassembly_5x_6x(code, asm, blen=32):
        ''' For disassembly of sm5x/6x, float imme can be 32bit or 20bit? 

            This may depend on the opcode.

            Example 1 (blen=32):
                0x080 3f8ccccd 7  0a 04
                      fimme    pg ra rd 
                FADD32I R4, R10, 1.1000000238418579102 ;

            Example 2(blen=19?):
                0x338 000 40000 7  03 0e
                          fimme pg ra rd
                FFMA R14, R3, -2, R0 ;

                The negtive sign "-" of float imme is moved...
        '''

        if p_QNAN.search(asm): # 
            bmask = (1<<blen) - 1
            fimm = (code & (bmask<<20))>>20 # 
            if '-QNAN' in asm:
                fstr = f'-0f{fimm:05x}'
            else:
                fstr = f'0f{fimm:05x}'

            asm = p_QNAN.sub(fstr, asm) # FIXME: neg sign moved?

        return asm

def testOffset():
    v5 = CuSMVersion(52)
    v7 = CuSMVersion(75)

    for i in range(32):
        v5_offset = v5.getInsOffsetFromIndex(i)
        v7_offset = v7.getInsOffsetFromIndex(i)
        print('%2d %04x %04x'%(i, v5_offset, v7_offset))
    
    for v in range(0, 32*8, 8):
        v5_idx = v5.getInsIndexFromOffset(v)
        v7_idx = v7.getInsIndexFromOffset(v)
        print('%04x %4d %4d'%(v, v5_idx, v7_idx))

def testInstance():

    print('Checking sm61...')
    v61 = CuSMVersion(61)
    v61_id = id(v61)
    assert id(CuSMVersion('61')) == v61_id
    assert id(CuSMVersion('sm_61')) == v61_id
    assert id(CuSMVersion('SM61')) == v61_id
    assert id(CuSMVersion(v61)) == v61_id
    print('  Passed!')

    print('Checking sm75...')
    v75 = CuSMVersion(75)
    v75_id = id(v75)
    assert id(CuSMVersion('75')) == v75_id
    assert id(CuSMVersion('sm_75')) == v75_id
    assert id(CuSMVersion('SM75')) == v75_id
    assert id(CuSMVersion(v75)) == v75_id
    print('  Passed!')

if __name__ == '__main__':
    testInstance()
