# -*- coding: utf-8 -*-

from io import BytesIO
import struct
from extra.sass.assembler.CuSMVersion import CuSMVersion

class CuNVInfo(object):
    EIFMT = {1 : 'EIFMT_NVAL', 2 : 'EIFMT_BVAL', 3 : 'EIFMT_HVAL', 4 : 'EIFMT_SVAL'}
    EIATTR = {0x0401 : 'EIATTR_CTAIDZ_USED',
              0x0504 : 'EIATTR_MAX_THREADS',
              0x0a04 : 'EIATTR_PARAM_CBANK',
              0x0f04 : 'EIATTR_EXTERNS',
              0x1004 : 'EIATTR_REQNTID',
              0x1104 : 'EIATTR_FRAME_SIZE',
              0x1204 : 'EIATTR_MIN_STACK_SIZE',
              0x1502 : 'EIATTR_BINDLESS_TEXTURE_BANK',
              0x1602 : 'EIATTR_BINDLESS_SURFACE_BANK',
              0x1704 : 'EIATTR_KPARAM_INFO',
              0x1903 : 'EIATTR_CBANK_PARAM_SIZE',
              0x1b03 : 'EIATTR_MAXREG_COUNT',
              0x1c04 : 'EIATTR_EXIT_INSTR_OFFSETS',
              0x1d04 : 'EIATTR_S2RCTAID_INSTR_OFFSETS',
              0x1e04 : 'EIATTR_CRS_STACK_SIZE',
              0x1f01 : 'EIATTR_NEED_CNP_WRAPPER',
              0x2001 : 'EIATTR_NEED_CNP_PATCH',
              0x2101 : 'EIATTR_EXPLICIT_CACHING',
              0x2304 : 'EIATTR_MAX_STACK_SIZE',
              0x2504 : 'EIATTR_LD_CACHEMOD_INSTR_OFFSETS',
              0x2704 : 'EIATTR_ATOM_SYS_INSTR_OFFSETS',
              0x2804 : 'EIATTR_COOP_GROUP_INSTR_OFFSETS',
              0x2a01 : 'EIATTR_SW1850030_WAR',
              0x2b01 : 'EIATTR_WMMA_USED',
              0x2e04 : 'EIATTR_ATOM16_EMUL_INSTR_REG_MAP',
              0x2f04 : 'EIATTR_REGCOUNT',
              0x3001 : 'EIATTR_SW2393858_WAR',
              0x3104 : 'EIATTR_INT_WARP_WIDE_INSTR_OFFSETS',
              0x3404 : 'EIATTR_INDIRECT_BRANCH_TARGETS',
              0x3501 : 'EIATTR_SW2861232_WAR',
              0x3604 : 'EIATTR_SW_WAR',
              0x3704 : 'EIATTR_CUDA_API_VERSION'}
    
    # EIATTRVAL is just the reciprocal mapping of EIATTR
    EIATTRVAL = { v:k for k,v in EIATTR.items()}

    def __init__(self, bytecodes, arch='sm_75'):

        # (attr, val)
        self.m_AttrList = CuNVInfo.decode(bytecodes)
        self.__mEIATTR_AutoGen = CuSMVersion(arch).getNVInfoAttrAutoGenSet().copy()
        self.__mEIATTR_MaunalGen = CuSMVersion(arch).getNVInfoAttrManualGenSet().copy()
        
    def serialize(self):
        return CuNVInfo.encode(self.m_AttrList)
    
    def getOffsetLabelDict(self, secname):
        label_dict = {}

        for attr, val in self.m_AttrList:
            if attr.endswith('OFFSETS') and attr not in self.__mEIATTR_AutoGen:
                # CHECK: Could there be two nvinfo attributes with same offset?
                # TODO: Some OFFSETS attributes seem have special structure?
                #       May add some assembler directive in kernel to handle them?
                for offset in val:
                    label_dict[offset] = ('.CUASM_OFFSET_LABEL.%s.'%secname) + attr + ('.#')
        
        return label_dict

    def updateNVInfoFromDict(self, nvinfo_dict):
        ''' Update nvinfo attributes according to kernel extra info.
        
            NOTE: This should only be called for .nv.info.{kernelsection}, not '.nv.info'
        '''

        d = nvinfo_dict.copy()
        
        new_attr_list = []
        for attr, val in self.m_AttrList:
            if attr not in self.__mEIATTR_AutoGen and attr not in self.__mEIATTR_MaunalGen: # kept as is
                new_attr_list.append((attr, val))

            elif attr in d: # autogen/manual gen attributes, updated and remove from appended list
                new_attr_list.append((attr, val))
                del d[attr] 

            else: # attr in AutoGen but not in d, which means it's deleted
                pass
        
        # Append new autogen attributes not previously defined ()
        for k, v in d.items():
            # print('Appending new ', k, '=', v)
            # new_attr_list.append((k, v))
            if k in self.__mEIATTR_AutoGen or k in self.__mEIATTR_MaunalGen: # some item in d may be not valid attributes, skipped
                new_attr_list.append((k, v))
            # else:
            #     CuAsmLogger.logWarning(f'Ignoring unknown NVInfo attribute: {k}')

        self.m_AttrList = new_attr_list

    def setRegCount(self, reg_count_dict):
        ''' Update NVInfo for regcount, only for SM_70 and above.
            
            (???) : Seems for latest CUDA(at least 11.2?), this attr is set for every SM version?

            NOTE: this should only be called by section '.nv.info', not '.nv.info.{kernelname}'

            reg_count_dict = {kernelname_symidx:regnum, ...}
            Return: flag for whether found and updated.
        '''

        flag = True

        d = {}
        for k, v in reg_count_dict.items():
            #d[k.to_bytes(4, 'little')] = struct.pack('II', k, v)
            d[k] = [k, v]
            
        for i, (attr, val) in enumerate(self.m_AttrList):
            if attr == 'EIATTR_REGCOUNT' :
                if val[0] in d:
                    self.m_AttrList[i] = attr, d[val[0]]
                else:
                    flag = False
                    # raise Exception('No RegCount attribute found for symbol(%d)'%symidx)

        return flag
    
    def specialAttrTreatment(self, attr, val):
        ''' Not implementated yet... '''
        pass

    def __iter__(self):
        # (attr, val)
        for v in self.m_AttrList:
            yield v

    def getUnknownAttrList(self):
        # return [('Test', 0)]
        l = []
        for attr, val in self.m_AttrList:
            if attr not in CuNVInfo.EIATTRVAL:
                l.append((attr, val))    
        return l
        
    @staticmethod
    def decode(bytescodes):
        attrlist = [] # cannot use dict here, since some attributes
                      # may have more than 1 entry

        bio = BytesIO(bytescodes)
        fmt = bio.read(1)
        while len(fmt)>0:
            vfmt = struct.unpack('B', fmt)[0]
            attr_type = struct.unpack('B', bio.read(1))[0]
            attr_key = (attr_type<<8) + vfmt
            
            val = None
            if vfmt==1: # EIFMT_NVAL
                bio.read(2)  # usually zeros, thus skipped
                val = None
            elif vfmt==2: # EIFMT_BVAL ??
                val = int.from_bytes(bio.read(2), 'little')
            elif vfmt==3: # EIFMT_HVAL
                val = int.from_bytes(bio.read(2), 'little')
            elif vfmt==4: # EIFMT_SVAL, length should be multiple of 4B
                nelem = int.from_bytes(bio.read(2), 'little') >> 2
                val = []
                for i in range(nelem):
                    val.append(int.from_bytes(bio.read(4), 'little'))
            else:
                raise Exception("Unknwon EIFMT(%#x) in nv.info!" % vfmt)
            
            attr_name = CuNVInfo.getAttrName(attr_key)
            if attr_key not in CuNVInfo.EIATTR:
                print('WARNING!!! Unknown EIATTR 0x%04x! Some offsets may not work properly!'%attr_key)

            attrlist.append((attr_name, val))

            # print('0x%04x  %-24s : %s' % (attrkey, attr, val.hex()))
            fmt = bio.read(1)

        return attrlist
    
    @staticmethod
    def encode(attrlist):
        bio = BytesIO()
        for attr_name, val in attrlist:
            attr_key = CuNVInfo.getAttrKey(attr_name)
            
            vfmt = attr_key & 0xff
            
            bio.write(attr_key.to_bytes(2, 'little'))
            bval = CuNVInfo.packValue(vfmt, val)
            bio.write(bval)
        
        return bio.getvalue()

    @staticmethod
    def packValue(vfmt, val):
        ''' Pack values into bytes.

            vfmt:
              1: val not used, just 2 zero bytes
              2: val in 2Bytes, int
              3: val in 2Bytes, int
              4: val in 4Bytes lists
        
        '''

        if isinstance(val, bytes):
            return val

        if vfmt==1: # EIFMT_NVAL
            bval = b'\x00\x00'
        elif vfmt==2: # EIFMT_BVAL ??
            bval = int.to_bytes(val, 2, 'little')
        elif vfmt==3: # EIFMT_HVAL
            bval = int.to_bytes(val, 2, 'little')
        elif vfmt==4: # EIFMT_SVAL
            bio = BytesIO()
            length = len(val)*4
            bio.write(length.to_bytes(2, 'little'))

            for v in val:
                bio.write(v.to_bytes(4, 'little')) # 4B value
            bval = bio.getvalue()
        else:
            raise Exception("Unknwon EIFMT(%d) in nv.info!" % vfmt)
        
        return bval
    
    @staticmethod
    def getAttrKey(attr_name):
        if attr_name in CuNVInfo.EIATTRVAL:
            attr_key = CuNVInfo.EIATTRVAL[attr_name]
        else:
            # EIATTR_UNKNOWN_0x????
            attr_key = int(attr_name[-6:], 16)
        
        return attr_key
    
    @staticmethod
    def getAttrName(attr_key):
        if attr_key in CuNVInfo.EIATTR:
            attr_name = CuNVInfo.EIATTR[attr_key]
        else:
            attr_name = 'EIATTR_UNKNOWN_0x%04x'%attr_key

        return attr_name

def doTest(bs1):
    info = CuNVInfo(bs1)
    print('Attributes list:')
    for name, val in info:
        print("  0x%04x : %-32s  %s"%(CuNVInfo.getAttrKey(name), name, val))

    bs1_cmp = info.serialize()
    
    idx = 0
    for b1, b2 in zip(bs1, bs1_cmp):
        idx += 1
        if b1 != b2:
            print('!!! %d byte not match! (%s vs %s)'%(idx, b1, b2))

    label_dict = info.getOffsetLabelDict('_Ztestkernel')

    print('Offset Label dicts: ')
    if len(label_dict) == 0:
        print('  {}')
    else:
        for k, v in label_dict.items():
            print('  0x%08x : %s'%(k, v))
    
def testCase1():
    bs = bytes.fromhex(''.join(['042f0800130000002100000004230800',
                                '13000000000000000412080013000000',
                                '00000000041108001300000000000000',
                                '042f0800120000002300000004230800',
                                '12000000000000000412080012000000',
                                '00000000041108001200000000000000',
                                '042f0800110000001800000004230800',
                                '11000000000000000412080011000000',
                                '00000000041108001100000000000000',
                                '042f0800100000001a00000004230800',
                                '10000000000000000412080010000000',
                                '00000000041108001000000000000000',
                                '042f08000f0000001800000004230800',
                                '0f00000000000000041208000f000000',
                                '00000000041108000f00000000000000',
                                '042f08000e0000001c00000004230800',
                                '0e00000000000000041208000e000000',
                                '00000000045108000e00000000000000'])) # Error injection 0x5104
    doTest(bs)

def testCase2():
    bs = bytes.fromhex(''.join(['0436040001000000043704006f000000',
                                '040a0800150000006001180003191800',
                                '04170c00000000000200100000f02100',
                                '04170c00000000000100080000f02100',
                                '04170c00000000000000000000f02100',
                                '04310c00000100000002000000030000',
                                '031bff000216000002150000040f0400',
                                '26000000041c08005008000030090000',
                                '041e040000000000']))
    doTest(bs)

if __name__ == '__main__':

    testCase1()
    testCase2()
    