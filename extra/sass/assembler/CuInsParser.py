# -*- coding: utf-8 -*-

import re
import struct
from extra.sass.assembler.common import *
from extra.sass.assembler.CuSMVersion import CuSMVersion

# Pattern that matches an instruction string
p_InsPattern = re.compile(r'(?P<Pred>@!?U?P\w\s+)?\s*(?P<Op>[\w\.\?]+)(?P<Operands>.*)')

# Pattern that matches scoreboard sets, such as {1}, {4,2}
# Seems only appear after opcode DEPBAR
p_SBSet = re.compile(r'\{(\d,)*\d\}')

# NOTE: about constants translate dict
# 1) +/-QNAN is not recognized by python float(), use +/-NAN
#    +/-INF seems OK,
#    QNAN for FSEL may not work properly, needs special treatment
# 2) (.reuse will be treated seperately for control codes, hence ignored here.)
#    Bugfix: reuse will be treated as normal modifier
# 3) RZ may also appear in FADD/FMUL/FFMA.RZ ...
# 4) UPT is not found, may be just PT?
p_ConstTrDict = {r'(?<!\.)\bRZ\b' : 'R255', r'\bURZ\b' : 'UR63',
                r'\bPT\b' : 'P7', r'\bUPT\b' : 'UP7', r'\bQNAN\b' : 'NAN'} #, r'\.reuse\b':''

# Pattern for striping modifiers from an operand
# .*? for non-greedy match, needed for [R0.X4].A
p_ModifierPattern = re.compile(r'^(?P<PreModi>[~\-\|!]*)(?P<Main>.*?)\|?(?P<PostModi>(\.\w+)*)\|?$')

# Match Label+Index (including translated RZ/URZ/PT)
# SBSet is the score board set for DEPBAR, translated before parsing
p_IndexedPattern = re.compile(r'\b(?P<Label>R|UR|P|UP|B|SB|SBSET|SR)(?P<Index>\d+)$')

# Immediate floating point numbers, (NOTE: add 0f0000000 to skip conversion)
# NOTE: QNAN is tranlated into NAN in preprocessing, since python only recognizes NAN
p_FIType = re.compile(r'^(?P<Value>((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)|([+-]?INF)|([+-]NAN)|-?(0[fF][0-9a-fA-F]+))(?P<ModiSet>(\.[a-zA-Z]\w*)*)$')

# ???
p_IIType = re.compile(r'^(?P<Value>0x[0-9a-f]+)(?P<ModiSet>(\.[a-zA-Z]\w*)*)$')

# Pattern for splitting immediate modifiers, such as 0x1.INV, 1.5.NEG.H1
#         here, only modifiers not precedented with number is supported...
p_ImmeModi = re.compile(r'^(?P<Value>.*?)(?P<ModiSet>(\.[a-zA-Z]\w*)*)$')

# Pattern for constant memory, some instructions have a mysterious space between two square brackets...
p_ConstMemType = re.compile(r'c\[(?P<Bank>0x\w+)\]\[(?P<Addr>[+-?\w\.]+)\]')

# Pattern for constant memory, some instructions have a mysterious space between two square brackets...
p_URConstMemType = re.compile(r'cx\[(?P<URBank>UR\w+)\]\[(?P<Addr>[+-?\w\.]+)\]')

# Pattern for memory address with a cache-policy description, such as:
p_DescAddressType = re.compile(r'desc\[(?P<URIndex>UR\d+)\](?P<Addr>\[.*\])$')

# Pattern for matching white spaces
p_WhiteSpace = re.compile(r'\s+')

# Pattern for insignificant spaces, they will be collapsed first, and removed finally
# Spaces between words([0-9A-Za-z_]) will be kept, others will be removed
p_InsignificantSpace = re.compile(r'((?<=[\w\?]) (?![\w\?]))|((?<![\w\?]) (?=[\w\?]))|((?<![\w\?]) (?![\w\?]))')

# RImmeAddr
p_RImmeAddr = re.compile(r'(?P<R>R\d+)\s*(?P<II>-?0x[0-9a-fA-F]+)')

# modifiers (1 char) that may appear before operands
# NOTE: NOT/NEG/ABS/INV may be used by explicit modifiers for immediates, such as 1.5.neg
#       thus here we use cNOT/cNEG/cABS/cINV
c_OpPreModifierChar = {'!':'cNOT', '-':'cNEG', '|':'cABS', '~':'cINV'}

# Jump functions that may use the instruction address
# TODO: Some instruction already have neg sign before address, will it still work?
c_AddrFuncs = set(['BRA', 'BRX', 'BRXU', 'CALL', 'JMP',
                   'JMX', 'JMXU', 'RET', 'BSSY',
                   'SSY', 'CAL', 'PRET', 'PBK'])

# Functions that have position dependent modifiers, such as F2F.F16.F32 != F2F.F32.F16
c_PosDepFuncs = set(['I2I', 'F2F', 'IDP', 'HMMA', 'IMMA', 'XMAD', 'IMAD', 'IMADSP',
                     'VADD', 'VMAD', 'VSHL', 'VSHR', 'VSET', 'VSETP', 'VMNMX',
                     'VABSDIFF', 'VABSDIFF4', 'TLD4', 'PSET', 'PSETP'])

c_ModiDTypes = set(['S4', 'S8', 'S16', 'S32', 'S64', 'U4', 'U8', 'U16', 'U32', 'U64', 'F16', 'F32', 'F64'])
c_ModiDTypesExt = c_ModiDTypes.union(set(['S24', 'U24', 'S16H0', 'S16H1', 'U16H0', 'U16H1'])) # IMAD/IMADSP/IMUL(32I)* of sm_6x
c_ModiRGBA = set(['R', 'G', 'B', 'A']) # For TLD4
c_ModiLOP = set(['AND', 'OR', 'XOR', 'NOT']) # PSET/PSETP for sm_6x

# NOTE: position dependent opcode list is arch dependent, 
c_PosDepModis = {
                'I2I'      : c_ModiDTypes,
                'F2F'      : c_ModiDTypes,
                'I2IP'     : c_ModiDTypes,
                'F2FP'     : c_ModiDTypes,

                'VADD'     : c_ModiDTypes,
                'VMAD'     : c_ModiDTypes,
                'VSHL'     : c_ModiDTypes,
                'VSHR'     : c_ModiDTypes,
                'VSET'     : c_ModiDTypes,
                'VSETP'    : c_ModiDTypes,
                'VMNMX'    : c_ModiDTypes,
                'VABSDIFF' : c_ModiDTypes,
                'VABSDIFF4': c_ModiDTypes,

                'XMAD'     : c_ModiDTypesExt,
                'IMAD'     : c_ModiDTypesExt,
                'IMAD32I'  : c_ModiDTypesExt,
                'IMADSP'   : c_ModiDTypesExt,
                'IMUL'     : c_ModiDTypesExt,
                'IMUL32I'  : c_ModiDTypesExt,

                'PSET'     : c_ModiLOP,
                'PSETP'    : c_ModiLOP,

                'IDP'      : c_ModiDTypes,
                'HMMA'     : c_ModiDTypes,
                'IMMA'     : c_ModiDTypes,

                'TLD4'     : c_ModiRGBA,
                }
# I2F/F2I/F2F has different OpCode for 32/64,
# but 32bit modifier may not be displayed
# this will be a problem only for multiple 64bit modifiers 
# FRND may not need this?
c_CvtOpcodes = set(['I2F', 'I2I', 'F2I', 'F2F', 'I2FP', 'I2IP', 'F2IP', 'F2FP', 'FRND'])

class CuInsParser():
    ''' CuInsParser will parse the instruction string to inskey, values, and modifiers.

        Which could be then assembled by CuInsAssembler.

        Since the parser will consume considerable amount of memory, the "parse" should be
        called with limited instances, which will update the members accordingly.

        We don't make the "parse" a static function, since we frequently need to check some
        internal variables of parsing results, especially during debugging.
    '''

    # predicate value is the first element in value vector
    PRED_VAL_IDX = 0

    #
    OPERAND_VAL_IDX = 1

    #
    StaticParserRepos = {}

    def __init__(self, arch='sm_75'):
        self.m_Arch = CuSMVersion(arch)
        self.reset()

    def reset(self):
        self.m_InsAddr = 0             # ins address, needed by branch type of ins
        self.m_InsString = ''          # original asm string
        self.m_CTrString = ''          # constants translated asm string
        self.m_InsCode = 0             # instruction code

        self.m_InsKey = ''             # key for current type of ins, eg: FFMA_R_R_R_R
        self.m_InsOp = ''              # function name, such as FFMA, MOV, ...
        self.m_InsOpFull = ''          # function name with modifiers
        self.m_InsPredVal = 0          # predicate value (0b****)
        self.m_InsPredStr = ''         # predicate string
        self.m_InsModifier = []        # modifier dict
        self.m_InsVals = []            # array of operand values
        self.m_InsTags = []            # tag (R/P/UR/UP/Imme/...) of every elem in InsVals

        self.m_RPList = []             # 

    def dumpInfo(self):
        print('#### CuInsParser @ 0x%016x ####' % id(self))
        print('  InsString : ' + self.m_InsString)
        print('  CTrString : ' + self.m_CTrString)
        print('  InsAddr   : %#x' % self.m_InsAddr)
        print('  InsPred   : %s (%#x)' % (self.m_InsPredStr, self.m_InsPredVal) )
        print('  InsCode   : %s' % self.m_Arch.formatCode(self.m_InsCode))
        print('  InsOp     : %s' % self.m_InsOp)
        print('  InsOpFull : %s' % self.m_InsOpFull)
        print('  InsKey    : ' + self.m_InsKey)
        print('  InsVals   : ' + intList2Str(self.m_InsVals))
        print('  InsTags   : ' + str(self.m_InsTags))
        for t, v in zip(self.m_InsTags, self.m_InsVals):
            t0 = t.split('.')[0]
            if t0 in ['R', 'P', 'UP', 'UR']:
                if t == 'P.Guard':
                    v = v & 0x7
                
                print(f'      {t:10s} = {t0}{v}')
            else:
                print(f'      {t:10s} = {v:#x}')

        print('  InsModi   : ' + str(self.m_InsModifier))
        # rps = []
        # for lbl, idx in self.m_RPList:
        #     if (lbl,idx) in {('P',7), ('UP', 7), ('R', 255), ('UR', 63)}:
        #         continue
        #     else:
        #         rps.append(f'{lbl}{idx}')
        
        rps = [f'{lbl}{idx}' for lbl, idx in self.m_RPList]
        print('  RPList    : [' + (', '.join(rps)) + ']' )
        print('\n')

    def dumpInfoAsDict(self):
        d = {}
        d['InsString'] =  self.m_InsString
        d['CTrString'] =  self.m_CTrString
        d['InsAddr'] =    self.m_InsAddr
        d['InsPredStr'] =  self.m_InsPredStr
        d['InsPredVal'] = self.m_InsPredVal
        d['InsCode'] =    self.m_InsCode
        d['InsOp'] =      self.m_InsOp
        d['InsOpFull'] =  self.m_InsOpFull
        d['InsKey'] =     self.m_InsKey
        d['InsVals'] =    self.m_InsVals
        d['InsTags'] =    self.m_InsTags
        d['InsModi'] = self.m_InsModifier
        d['RPList'] = self.m_RPList

        return d

    def parse(self, s, addr=0, code=0):
        ''' Parse input string as instruction. 
        
            Return: (InsKey, InsVals, InsModifier)
        '''

        self.reset()
        self.m_InsString = s.strip()
        self.m_CTrString = self.__constTr(self.m_InsString)
        
        r = p_InsPattern.match(self.m_CTrString)
        if r is None:
            raise ValueError(f'Unrecognized asm "{s}"')

        self.m_InsAddr = addr
        self.m_InsCode = code
        self.m_InsPredStr = r.group('Pred')

        self.m_InsOpFull = r.group('Op')
        op_tokens = self.m_InsOpFull.split('.') # Op and Op modifiers
        self.m_InsKey = op_tokens[0]
        self.m_InsOp = op_tokens[0]

        # Currently guard pred is treated as known format operand
        # The value will be directly computed.
        # For predicate as operand, the '!' will be treated as modifier.
        self.m_InsPredVal = self.__parsePred(self.m_InsPredStr)
        self.m_InsVals = [self.m_InsPredVal]
        self.m_InsTags = ['P.Guard']
        self.m_InsModifier = ['0_' + m for m in op_tokens]
                
        operands = self.__preprocessOperands(r.group('Operands'))
        if len(operands) > 0:
            # Splitting operands
            # usually ',' will be sufficient to split the operands
            # Two Exceptions:
            #    1: "RET.REL.NODEC R10 0x0 ;"
            #       In operand preprocess, "R10 0x0" will be replace with "R10, 0x0"
            #    2: "DEPBAR {4,3,2,1} ;"
            #       {4,3,2,1} will be translated into "SB#", thus no ","
            # Thus, splitting with ',' here is safe
            operands = re.split(',', operands)  
            
            for i, operand in enumerate(operands):
                optype, opval, opmodi, optag = self.__parseOperand(operand)
                self.m_InsKey += '_' + optype
                self.m_InsVals.extend(opval)
                self.m_InsTags.extend(optag)
                
                self.m_InsModifier.extend([('%d_'%(i+1))+m for m in opmodi])

        self.__specialTreatment() #
        return self.m_InsKey, self.m_InsVals, self.m_InsModifier

    def __constTr(self, s):
        ''' Translate pre-defined constants (RZ/URZ/PT/...) to known or indexed values.

            Translate scoreboard sets {4,2} to SBSet
        '''
        # strip all comments
        s = stripComments(s)
        
        for cm in p_ConstTrDict:
            s = re.sub(cm, p_ConstTrDict[cm], s)

        res = p_SBSet.search(s)
        if res is not None:
            SB_valstr = self.__transScoreboardSet(res.group())
            s = p_SBSet.sub(SB_valstr, s)
        
        s = p_WhiteSpace.sub(' ', s)
        s = p_InsignificantSpace.sub('', s)
        
        return s.strip(' {};')
            
    def __preprocessOperands(self, s):
        s = s.strip()
        if self.m_InsOp in c_AddrFuncs:
            # Usually "R#-0x####" only appears in branch/jump instructions
            #   thus "R2-0x10" will can be always treated as "R2, -0x10"
            # However, we want support a grammar sugar for addressing with neg imme , such as [R2-0x10]
            #   thus we only do this for branch/jump instructions
            res = p_RImmeAddr.search(s)
            if res is not None:
                s = s.replace(res.group(), res.group('R')+','+res.group('II'))     #
        
        # s = p_WhiteSpace.sub('', s)     # collapse sequence of spaces
        # s = s.strip('{};')
        
        return s

    def __parseOperand(self, operand):
        '''Parse operand to (type, val, modi).

        Every operand should return with:
            type:str, val:list, modi:list'''

        #print('Parsing operand: ' + operand)

        # Every operand may have one or more modifiers
        op, modi = self.stripModifier(operand)

        if p_IndexedPattern.match(op) is not None:
            optype, opval, tmodi = self.__parseIndexedToken(op)
            opmodi = modi
            opmodi.extend(tmodi)
            optag = [optype]
        elif op[0] == '[': # address
            optype, opval, opmodi, optag = self.__parseAddress(op)
        # elif op[0] == '{': # BarSet such as {3,4}, only for DEPBAR (deprecated? could set in control codes)
        #                    # DEPBAR may wait a certain number of counts for one scoreboard,
        #     optype, opval, opmodi = self.__parseBarSet(op)
        # NOTE: the scoreboard set is translated to indexed type in preprocess, thus no treatment here.
        elif op.startswith('c['):
            optype, opval, opmodi, optag = self.__parseConstMemory(op)
            opmodi.extend(modi)
        elif op.startswith('0x'):
            optype = 'II'
            op, modi = self.stripImmeModifier(operand)
            opval, opmodi = self.__parseIntImme(op)
            opmodi.extend(modi)
            optag = [optype]
        elif p_FIType.match(operand) is not None:
            optype = 'FI'
            op, modi = self.stripImmeModifier(operand)
            opval, opmodi = self.__parseFloatImme(op)
            opmodi.extend(modi)
            optag = [optype]
        elif op.startswith('desc'):
            optype, opval, opmodi, optag = self.__parseDescAddress(op)
            opmodi.extend(modi)
        elif op.startswith('cx['):
            optype, opval, opmodi, optag = self.__parseURConstMemory(op)
            opmodi.extend(modi)
        else: # label type, keep as is
              # label will always work together with opcodes, thus no value here
            optype = 'L'
            opval = []
            opmodi = [operand]
            optag = []

        return optype, opval, opmodi, optag

    def __parseIndexedToken(self, s):
        '''Parse index token such as R0, UR1, P2, UP3, B4, SB5, ...

         (RZ, URZ, PT should be translated In advance)'''

        tmain, modi = self.stripModifier(s)
        r = p_IndexedPattern.match(tmain)
        if r is None:
            raise ValueError(f'Unknown indexedToken "{s}" in "{self.m_InsString}"')
            
        t = r.group('Label')
        v = [int(r.group('Index'))]
        
        # currently all indexed tokens go to RPList
        self.m_RPList.append((t, v[0]))
        
        return t, v, modi

    def __parsePred(self, s):
        r''' Parse predicates (@!?U?P[\dT]) to values.

        '''

        if s is None or len(s)==0:
            if self.m_InsOp[0] == 'U':
                self.m_RPList.append(('UP', 7)) # a placeholder for True Predicate input
            else:
                self.m_RPList.append(('P', 7))
                
            return 7

        t, v, modi = self.__parseIndexedToken(s.strip('@! '))
        if '!' in s:
            return v[0] + 8
        else:
            return v[0]

    def __parseFloatImme(self, s):
        ''' Parse float point immediates to binary, according to the instruction precision.

            precision is the opcode precision, currently D/F/H for double/single(float)/half.
            NOTE: currently, +/-QNAN will be always translated to a UNIQUE binary,
              but sometimes nan could represent a set of values.
              But since it's not showed in the assembly string, there's no way to recover this value.

        '''
        p = self.m_InsOp[0]

        if p in set(['H', 'F', 'D']): # FIXME: F2F.F64.F32 vs F2F.F32.F64 may need special treatment
                                      #        although they are not likely to be used practically.
            prec = p
        elif self.m_InsOp in {"MUFU", "RRO"}: # It's rather wield that MUFU will have an imme input, any side effect?
                                              # maybe just due to completeness of ISA
            if '64' in self.m_InsOpFull:
                prec = 'D'
            elif '16' in self.m_InsOpFull:
                prec = 'H'
            else:
                prec = 'F'
        else:
            self.dumpInfo()
            raise ValueError('Unknown float precision (%s)!' % self.m_InsOp)

        if self.m_InsOp.endswith('32I'):
            nbits = 32
        else:
            nbits = -1

        v, modi = self.m_Arch.convertFloatImme(s, prec, nbits)
        return [v], modi

    def __parseIntImme(self, s):
        ''' Parse interger immediates.

            Positive int immediates are always kept as is,
            but negtive ints may depend on the type.
            Currently we try to let the coefficient determined by the code, not predetermined.

            TODO(Done):
                Some ALU instructions such as IADD3 in sm5x/6x, the sign bit will be moved to the modifier.
                If the sign bit is explicitly show (such as -0x1), it can be handled by 'NegIntImme'.
                But if it's implicitly defined (such as 0xfffff, 20bit used, but int imme has only 19bit),
                we need to handle it seperately.
        '''

        i = int(s, 16)

        if i>=0:
            return self.m_Arch.splitIntImmeModifier(self, i)
        else:
            return [i], ['NegIntImme']

    def __parseConstMemory(self, s):
        opmain, opmodi = self.stripModifier(s)

        r = p_ConstMemType.match(opmain)
        if r is None:
            raise ValueError("Invalid constant memory operand: %s" %s)

        opval = [int(r.group('Bank'), 16)]
        optag = ['Imme.CBank']

        atype, aval, amodi, atag = self.__parseAddress(r.group('Addr'))

        optype = 'c' + atype
        opval.extend(aval)
        opmodi.extend(amodi)
        optag.extend(atag)

        return optype, opval, opmodi, optag

    def __parseURConstMemory(self, s):
        opmain, opmodi = self.stripModifier(s)

        r = p_URConstMemType.match(opmain)
        
        if r is None:
            raise ValueError("Invalid UR constant memory operand: %s" %s)

        btype, opval, opmodi = self.__parseIndexedToken(r.group('URBank'))
        opmodi = [btype + '_' + m for m in opmodi]
        optag = ['UR.CBank']
        
        atype, aval, amodi, atag = self.__parseAddress(r.group('Addr'))

        optype = 'cx' + atype
        opval.extend(aval)
        opmodi.extend(amodi)
        optag.extend(atag)

        return optype, opval, opmodi, optag
    
    def __parseDescAddress(self, s):
        r = p_DescAddressType.match(s)
        if r is None:
            raise ValueError('Invalid desc address operand: %s' % s)
        
        _, opval, _ = self.__parseIndexedToken(r.group('URIndex'))
        optag = ['UR.Desc']

        atype, aval, amodi, atag = self.__parseAddress(r.group('Addr'))

        optype = 'd' + atype
        opval.extend(aval)
        optag.extend(atag)
        
        return optype, opval, amodi, optag

    def __transScoreboardSet(self, s):
        ''' Translate scoreboard set such as {3,4} to int values.

            This is done during preprocessing, since the comma(',') will be used to split the operands.
        '''

        ss = s.strip('{}').split(',')
        v = 0
        for bs in ss: # ???
            v += 1<<(int(bs))

        return 'SBSET%d'%v

    def __parseAddress(self, s):
        ''' Parse operand type Address [R0.X8+UR4+-0x8]

            Zero immediate will be appended if not present.
            It's harmless if there is no such field, since the value will always be 0.

            TODO(Done): what for [R0.U32+UR4.U64] ?? Could in another order?
                  May need extra tag in modifiers?
        '''

        ss = re.sub(r'(?<![\[\+])-0x', '+-0x', s) # NOTE: [R0-0x100] is illegal! should be [R0+-0x100]
        ss = ss.strip('[]').split('+') 

        pdict = {}
        for ts in ss:
            if '0x' in ts:
                i_opval, i_opmodi = self.__parseIntImme(ts)
                pdict['I'] = ('I', i_opval, i_opmodi)
            elif len(ts) == 0:
                continue
            else:
                ttype, tval, tmodi = self.__parseIndexedToken(ts)

                # The modifier is prefixed by type
                # Thus     [R0.U32+UR4.U64] => ['R.U32', 'UR.U64']
                # (If any) [R0.U64+UR4.U32] => ['R.U64', 'UR.U32']
                tmodi = [ (ttype+'.'+m) for m in tmodi]
                pdict[ttype] = (ttype, tval, tmodi)

        optype = 'A'
        opval = []
        opmodi = []
        optag = []

        # rearrange to R+UR+I order
        if 'R' in pdict:
            optype += 'R'
            opval.extend(pdict['R'][1])
            opmodi.extend(pdict['R'][2])
            optag.append('R.Addr')
        
        if 'UR' in pdict:
            optype += 'UR'
            opval.extend(pdict['UR'][1])
            opmodi.extend(pdict['UR'][2])
            optag.append('UR.Addr')

        if 'I' in pdict:
            optype += 'I'
            opval.extend(pdict['I'][1])
            opmodi.extend(pdict['I'][2])
            optag.append('Imme.Addr')
        else:
            optype += 'I'  # Pad with zero immediate if not present
                           # Harmless even if it does not support immediates
            opval.append(0)
            optag.append('Imme.Addr')

        return optype, opval, opmodi, optag

    def __specialTreatment(self):
        ''' Special treatments after parsing.

            Handle exceptions that cannot processed with current approach.

            TODO: Use dict mapping to subroutines, rather than if/else
                How??? F2F may need several special treatments...
        '''

        if self.m_InsOp in {'PLOP3', 'UPLOP3'}: 
                                      # immLut for PLOP3 is encoded with seperating 5+3 bits
                                      # e.g.: 0x2a = 0b00101010 => 00101 xxxxx 010
                                      # LOP3 seems fine
            v = self.m_InsVals[-2]    # TODO: locate the immLUT adaptively?
            self.m_InsVals[-2] = (v&7) + ((v&0xf8)<<5)

        elif self.m_InsOp in c_CvtOpcodes:
            if '64' in self.m_InsOpFull:
                self.m_InsModifier.append('0_CVT64')

        elif self.m_InsOp in c_AddrFuncs: # Functions that use address of current instruction
                                          # CHECK: what if the address is not the last operand?
            if self.m_InsKey.endswith('_II'):
                if 'ABS' not in self.m_InsOpFull: # CHECK: Other absolute address?
                    addr = self.m_InsVals[-1] - self.m_InsAddr - self.m_Arch.getInstructionLength()
                    if addr<0:
                        self.m_InsModifier.append('0_NegAddrOffset')

                    # The value length of same key should be kept the same
                    self.m_InsVals[-1] = addr

        if self.m_InsOp in self.m_Arch.m_PosDepOpcodes:
            # the modifier of I2I/F2F is position dependent
            # eg: F2F.F32.F64 vs F2F.F64.F32
            # TODO: find all instructions with position dependent modifiers
            counter = 0
            for i,m in enumerate(self.m_InsModifier):
                if m.startswith('0_') and m[2:] in c_PosDepModis[self.m_InsOp]:
                    self.m_InsModifier[i] += '@%d'%counter
                    counter += 1

    def stripModifier(self, s):
        '''Split the token to three parts

        preModifier([~-|!]), opmain, postModifier(.FTZ, .X16, ...) '''

        r = p_ModifierPattern.match(s)  # split token to three parts

        if r is None:
            raise ValueError("Unknown token %s" % s)
        else:
            pre = r.group('PreModi')
            post = r.group('PostModi')

            opmain = r.group('Main')
            opmodi = []

            for c in pre:
                opmodi.append(c_OpPreModifierChar[c])

            for c in post.split('.'):
                if len(c)==0:
                    continue
                opmodi.append(c)

            return opmain, opmodi

    def stripImmeModifier(self, s):
        r = p_ImmeModi.match(s)
        if r is None:
            raise ValueError(f'Unknown immediate string: {s}')
        else:
            sval = r.group('Value')
            modis = r.group('ModiSet')
            if modis is None or len(modis)==0:
                modis = []
            else:
                modis = modis.lstrip('.').split('.')
            return sval, modis

    @staticmethod
    def getStaticParser(arch) -> 'CuInsParser':
        ''' Get a static CuInsParser for arch.

            Although CuInsParser is stateful, all states will be overwritten when calling cip.parse(...) again.
            Thus usually we don't need duplicate CuInsParser instances for same arch.

            Be cautious when using SAME instance in different places, the states may be entangled.  
        '''
        if arch not in CuInsParser.StaticParserRepos:
            CuInsParser.StaticParserRepos[arch] = CuInsParser(arch)
        
        return CuInsParser.StaticParserRepos[arch]

