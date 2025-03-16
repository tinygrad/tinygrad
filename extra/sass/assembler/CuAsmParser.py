# -*- coding: utf-8 -*-

import re
import os
from io import BytesIO
from collections import OrderedDict, defaultdict

from elftools.elf.elffile import ELFFile

from extra.sass.assembler.CuKernelAssembler import CuKernelAssembler
from extra.sass.assembler.CuInsAssemblerRepos import CuInsAssemblerRepos
from extra.sass.assembler.CuSMVersion import CuSMVersion
from extra.sass.assembler.CuNVInfo import CuNVInfo
from extra.sass.assembler.CuAsmLogger import CuAsmLogger
from extra.sass.assembler.CubinFile import PROGRAM_HEADER_TAG

from extra.sass.assembler.config import Config
from extra.sass.assembler.common import splitAsmSection, alignTo, bytes2Asm
from extra.sass.assembler.CuControlCode import c_ControlCodesPattern

m_hex    = re.compile(r'\b0x[a-fA-F0-9]+\b')
m_int    = re.compile(r'\b[0-9]+\b')
m_intval = re.compile(r'\b(0x[a-fA-F0-9]+)|([0-9]+)\b') 

def updateDictWithInput(din, dout, label='', kprefix=''):
    ''' Update a dict with input from another dict.

        The key will be prefixed with kprefix.
        the value will be converted to int if possible (for hex or dec int).
        
        label is only used for error tracing.
    '''
    for k,v in din.items():
        kp = kprefix + k
        if kp not in dout:
            # CuAsmLogger.logWarning('Unknown header attribute (%s) for %s!!!'%(k,label))
            pass

        if isinstance(v, str):
            if m_hex.match(v):
                vv = int(v, 16)
            elif m_int.match(v):
                vv = int(v)
            else:
                vv = v
        else:
            vv = v

        dout[kp] = vv

def buildStringDict(bytelist):
    ''' build strings dict from b'\x00' joined byte list.

        The dict key/value is just the offset/value of the string.
    '''
    p = 0
    counter = 0

    sdict = OrderedDict()
    while True:
        counter += 1
        pnext = bytelist.find(b'\x00', p)
        if pnext<0:
            break

        s = bytelist[p:pnext] # not include the ending b'\x00'
        sdict[p] = s.decode()
        p = pnext+1

    return sdict

class CuAsmSymbol(object):
    '''
        typedef struct
        {
            Elf64_Word    st_name;  /* Symbol name */
            unsigned char st_info;  /* Type and Binding attributes */
            unsigned char st_other; /* Reserved */
            Elf64_Half    st_shndx; /* Section table index */
            Elf64_Addr    st_value; /* Symbol value */
            Elf64_Xword   st_size;  /* Size of object (e.g., common) */
        } Elf64_Sym;

        //
        typedef uint64_t	Elf64_Addr;
        typedef uint16_t	Elf64_Half;
        typedef uint64_t	Elf64_Off;
        typedef int32_t		Elf64_Sword;
        typedef int64_t		Elf64_Sxword;
        typedef uint32_t	Elf64_Word;
        typedef uint64_t	Elf64_Lword;
        typedef uint64_t	Elf64_Xword;


        All internal symbols should also be defined as labels.
        The label offset is just the symbol value, and the section where the label
        is defined will affect the behavior of jump/branch instructions.

        FIXME: Currently some attributes in st_other (such as "STO_CUDA_ENTRY") cannot be 
               recognized by pyelftools, thus may be lost if parsed and built again.
    '''

    # TODO: Not implemented yet, just copied from cubin
    SymbolTypes = {'@function'          :0,
                   '@object'            :1,
                   '@"STT_CUDA_TEXTURE"':2,
                   '@"STT_CUDA_SURFACE"':3}

    def __init__(self, name):
        self.name = name
        self.type = None
        self.value = None
        self.size = None
        self.sizeval = None
        self.other = None
        self.index = None #

        self.entry = Config.defaultSymbol.copy()

    def __str__(self):
        s = 'name=%s, type=%s, value=%s, size(%s)=%s'%(
                self.name, self.type, self.value, self.sizeval, self.size)
        return s

    def build(self):
        ''' Build symbol entry.

            TODO: not implemented, symtab entries are copied from cubin
                  but value/size may be updated
        '''
        return Config.CubinELFStructs.Elf_Sym.build(self.entry)

    @staticmethod
    def buildSymbolDict(strtab, symbytes):
        symdict = OrderedDict()
        symsize = Config.CubinELFStructs.Elf_Sym.sizeof()
        index = 0
        for p in range(0, len(symbytes), symsize):
            sym = Config.CubinELFStructs.Elf_Sym.parse(symbytes[p:p+symsize])
            nameidx = sym['st_name']
            if nameidx not in strtab:
                raise Exception('Unknown symbol @%#x with name string index 0x%x!'%(p, nameidx))

            name = strtab[nameidx]
            if name in symdict:
                raise Exception('Duplicate symbol @%#x with name %s!', p, name)
            symdict[name] = index, sym
            index += 1

        return symdict

    @staticmethod
    def resetSymtabEntryValueSize(bio, base_offset, value, size):
        ''' reset Symbol entry value/size in symtab byte stream. 

            bio: BytesIO stream
            base_offset: base offset of current entry
            value/size: symbol value/size to be set
        '''

        p = bio.tell() # save current pos
        bio.seek(base_offset + 8) # +8 is offset for the value
        bio.write(int.to_bytes(value, 8, 'little')) 
        bio.write(int.to_bytes(size, 8, 'little'))
        bio.seek(p) # restore pos

class CuAsmLabel(object):
    ''' A label is defined by "label:"

        Every symbol (non-external) is also a label, the symbol value is just the label offset.
    '''
    def __init__(self, name, section, offset, lineno):
        self.name = name
        self.section = section
        self.offset = offset
        self.lineno = lineno
        CuAsmLogger.logSubroutine('Line %6d: New Label "%s" at section "%s":%#x'%(lineno, name, section.name, offset))

    def __str__(self):
        s = 'Label @Line %4d in section %s : %-#7x(%6d)  %s'%(self.lineno, self.section.name, self.offset, self.offset, 
                    self.name)
        return s

class CuAsmFixup(object):
    ''' Fixups are a set of undetermined values during the first scan.

        Some fixups can be evaluated after first scan. Then the true values will be filled.
        There are also some fixups cannot be determined during compile time, thus they will
        go to relocations and the true values will be filled by the program loader.
    '''

    def __init__(self, section, offset, expr, dtype, lineno):
        self.section = section
        self.offset = offset
        self.lineno = lineno
        self.dtype = dtype
        self.expr = expr
        self.value = None

        CuAsmLogger.logSubroutine('Line %6d: New Fixup "%s" at section "%s":%#x'%(lineno, expr, section.name, offset))

    def __str__(self):
        s = 'section=%s, offset=%d, lineno=%d, dtype=%s, expr=%s, value=%s'%(
              self.section.name, self.offset, self.lineno, self.dtype, self.expr, self.value)
        return s

class CuAsmSection(object):
    '''
        Section header struct (Only ELF64 supported):

        typedef struct
        {
            Elf64_Word  sh_name;      /* Section name               */
            Elf64_Word  sh_type;      /* Section type               */
            Elf64_Xword sh_flags;     /* Section attributes         */
            Elf64_Addr  sh_addr;      /* Virtual address in memory  */
            Elf64_Off   sh_offset;    /* Offset in file             */
            Elf64_Xword sh_size;      /* Size of section            */
            Elf64_Word  sh_link;      /* Link to other section      */
            Elf64_Word  sh_info;      /* Miscellaneous information  */
            Elf64_Xword sh_addralign; /* Address alignment boundary */
            Elf64_Xword sh_entsize;   /* Size of entries, if section has table */
        } Elf64_Shdr;
    '''

    def __init__(self, sname, stype, sflags):
        '''Construct an ELF section.

        Currently there are 3 systems for section headers.
            1. self.name/type/flags/... work for predefined directives, such as .section/.sectioninfo
            2. self.header['name']...   work for supplementary directives, namely .section_*
            3. self.__mSectionHeader    is the struct form for building header bytes

        Only 1 and 2 can be set in assembly, 1 has higher priority.
        Information from 1 and 2 will be combined to form the final header.

        Surely there are redundencies here, but it's the safest way to keep some attributes
        set by ptxas, yet still give user enough flexibility to modify them.

        '''
        self.name = sname
        self.type = stype       #  “A” stands for SHF_ALLOC
                                #  “W” for SHF_WRITE
                                #  “X” for SHF_EXECINSTR
        self.flags = [sflags]   #  some extra flags may be appended later

        self.info = []
        self.offset = None
        self.size   = None
        self.addralign = None
        self.entsize = 0

        self.header = {}
        self.extra  = {}    # barnum/regnum, only for update nvinfo

        # 
        self.padsize = 0
        self.padbytes = b''

        self.__isTextSection = sname.startswith('.text')
        self.__mSectionHeader = Config.defaultSectionHeader.copy()
        self.__mData = BytesIO()

    def updateHeader(self):
        '''Update section header with user inputs.

        TODO: currently only offset/size will be updated.
        '''

        updateDictWithInput(self.header, self.__mSectionHeader,
                            label='section %s'%self.name, kprefix = 'sh_')

        # maybe we can just update self.header?
        if self.header['type'] == 'SHT_NULL':
            self.__mSectionHeader['sh_offset'] = 0
        else:
            self.__mSectionHeader['sh_offset'] = self.offset
        self.__mSectionHeader['sh_size']   = self.getDataSize() #self.size

    def getHeaderStruct(self):
        return self.__mSectionHeader

    def updateResourceInfo(self):
        '''Update register/barrier number. 
        
            Examples:
          	.sectionflags	@"SHF_BARRIERS=1"
  	        .sectioninfo	@"SHI_REGISTERS=12"
        '''

        #
        p_regnum = re.compile(r'@"SHI_REGISTERS=(\d+)"')
        p_barnum = re.compile(r'@"SHF_BARRIERS=(\d+)"')
        
        regnum = None
        barnum = 0  # There may be no barrier used in a kernel

        for info in self.info:
            res = p_regnum.match(info)
            if res is not None:
                regnum = int(res.groups()[0])
        
        for flag in self.flags:
            res = p_barnum.match(flag)
            if res is not None:
                barnum = int(res.groups()[0])
        
        if regnum is None:
            raise Exception("Unknown register number for section %s!"%self.name)
        elif  regnum > 255 or regnum<0: # TODO: use MAX_REG_COUNT instead?
            raise Exception("Invalid register number %d for section %s!"%(regnum, self.name))
        else:
            rinfo = self.header['info']
            self.header['info'] = (rinfo & 0x00ffffff) + (regnum<<24)
            self.extra['regnum'] = regnum
        
        if barnum>15: # always rewrite bar number~
            raise Exception("Invalid barrier number %d for section %s!"%(barnum, self.name))
        else:
            rflag = self.header['flags']
            self.header['flags'] = (rflag&0xff0fffff) + (barnum<<20)
            self.extra['barnum'] = barnum

    def buildHeader(self):
        ''' Build section header bytes with current header struct. '''
        
        self.updateHeader()
        # print(self.__mSectionHeader)
        return Config.CubinELFStructs.Elf_Shdr.build(self.__mSectionHeader)

    def emitBytes(self, bs):
        self.__mData.write(bs)

    def updateForFixup(self, offset, bs):
        ''' Update corresponding bytes for fixup.

            Input: 
                offset       the absolute w.r.t the beginning of the section
                bs           bytes to be updated
        '''
        blen = len(bs)

        if (offset+blen) > self.getDataSize():
            raise Exception('Fixup out of boundary!')

        # save original pos
        opos = self.tell()
        self.__mData.seek(offset)

        # value is guaranteed within range during fixup evaluation.
        self.__mData.write(bs)
        self.__mData.seek(opos)
        
    def emitAlign(self, align):
        ''' Set alignment of next bytes.

            Note: When current position is section start, the alignment is the addralign of current section.
            Then the padding is done to previous section.
        '''

        pos = self.tell()
        if pos == 0:
            self.addralign = align
            self.header['addralign'] = align
        else:
            ppos, padsize = alignTo(pos, align)
            if ppos > pos:  # do padding with required 0-bytes/nops
                self.emitBytes(b'\x00' * (ppos-pos))

    def emitPadding(self, bs):
        ''' This is only for .text sections.

            Emitting padding here will change the size of current text section.
            For non-text sections, the padding should be done without changing the size.
        '''
        pos = self.tell()
        self.seek(0, 2) # seek to end
        self.emitBytes(bs)
        self.seek(pos) # restore original position

    def seek(self, pos, whence=0):
        return self.__mData.seek(pos, whence)

    def tell(self):
        return self.__mData.tell()

    def getData(self):
        return self.__mData.getvalue()
    
    def writePaddedData(self, stream):
        if self.header['type'] == 'SHT_NOBITS': # nobits sections will not write to file.
            return
        else:
            stream.write(self.__mData.getvalue())
            stream.write(self.padbytes)

    def setData(self, bs):
        ''' Update section data with given bytes. '''

        self.__mData = BytesIO(bs)
        self.size = len(bs)

    def getDataSize(self):
        ''' Get memory size of current section.

            For section of type nobits, no actual file contents.
        '''
        return len(self.__mData.getvalue())
    
    def getPaddedDataSize(self):
        return self.getDataSize() + self.padsize

    def getRegNum(self):
        return self.extra['regnum']

    def __str__(self):
        s = 'Section:\n'
        s += '    name      : %s\n' % self.name
        s += '    type      : %s\n' % self.type
        s += '    flags     : %s\n' % str(self.flags)
        s += '    info      : %s\n' % self.info
        s += '    offset    : %s\n' % self.offset
        s += '    addralign : %s\n' % self.addralign

        return s

class CuAsmSegment(object):
    def __init__(self, p_type, p_flags):
        self.header = {'type':p_type, 'flags':p_flags}
        self.__mSegmentHeader = Config.defaultSegmentHeader.copy()

    def updateHeader(self):
        ''' Update header with inputs'''

        updateDictWithInput(self.header, self.__mSegmentHeader,
                            label='segment', kprefix = 'p_')
    
    def getHeaderStruct(self):
        return self.__mSegmentHeader

    def build(self):
        return Config.CubinELFStructs.Elf_Phdr.build(self.__mSegmentHeader)

class CuAsmRelocation(object):
    ''' Relocation class.

        Relocation is a special section that may modify some contents of its linked section.
        This procedure is generally done during loading, the modified contents are typically
        the real memory address of some symbols.

        typedef struct
        {
            Elf64_Addr   r_offset; /* Address of reference */
            Elf64_Xword  r_info;   /* Symbol index and type of relocation */
        } Elf64_Rel;

        typedef struct
        {
            Elf64_Addr   r_offset; /* Address of reference */
            Elf64_Xword  r_info;   /* Symbol index and type of relocation */
            Elf64_Sxword r_addend; /* Constant part of expression */
        } Elf64_Rela;


        Relocations are typically for some dynamic variables (symbols).
        Sources of relocations:
        1. .dword/.word defined values in normal sections
        2. 32lo@* or 32hi@* kind of operands in text sections

            such as :
            /*0040*/                   MOV R2, 32@lo(flist) ;
            /*0060*/                   MOV R3, 32@hi(flist) ;

            RELA is a relocation section with extra offsets, such as:
            /*00f0*/                   MOV R20, 32@lo((_Z4testPiS_S_ + .L_6@srel)) ;
            /*0100*/                   MOV R21, 32@hi((_Z4testPiS_S_ + .L_6@srel)) ;

        3. `(symbol) in text sections (for symbols not defined in current section)
           
    '''

    REL_TYPES = {
        'R_CUDA_32'               : 1,
        'R_CUDA_64'               : 2,
        'R_CUDA_G64'              : 4,
        'R_CUDA_TEX_HEADER_INDEX' : 6,
        'R_CUDA_SURF_HEADER_INDEX': 52,
        'R_CUDA_ABS32_20'         : 42,
        'R_CUDA_ABS32_LO_20'      : 43,
        'R_CUDA_ABS32_HI_20'      : 44,
        'R_CUDA_ABS32_LO_32'      : 56,
        'R_CUDA_ABS32_HI_32'      : 57,
        'R_CUDA_ABS47_34'         : 58}

    def __init__(self, section, offset, relsymname, relsymid, reltype, reladd=None):
        self.section    = section
        self.offset     = offset
        self.relsymname = relsymname
        self.relsymid   = relsymid
        self.reltype    = reltype
        self.reladd     = reladd      # reladd=None means rel, otherwise rela

        CuAsmLogger.logSubroutine('New Relocation "%s" at section "%s":%#x'%(relsymname, section.name, offset))

    def isRELA(self):
        return self.reladd is not None

    def buildEntry(self):
        ''' Build binary entry of current relocation.

            Examples:
              _Z4testPiS_S_, Container({'r_offset': 528, 'r_info': 124554051586, 'r_info_sym': 29, 'r_info_type': 2})
              _Z4testPiS_S_, Container({'r_offset': 2288, 'r_info': 124554051641, 'r_info_sym': 29, 'r_info_type': 57, 'r_addend': 2352})
        '''
        if self.isRELA():  # RELA
            rela = Config.defaultRela.copy()
            rela['r_offset']    = self.offset
            rela['r_info_sym']  = self.relsymid
            rela['r_info_type'] = self.REL_TYPES[self.reltype]
            rela['r_info']      = (rela['r_info_sym']<<32) + rela['r_info_type']
            rela['r_addend']    = self.reladd
            # print(rela)
            return Config.CubinELFStructs.Elf_Rela.build(rela)

        else: # REL
            rel = Config.defaultRel.copy()
            rel['r_offset']    = self.offset
            rel['r_info_sym']  = self.relsymid
            rel['r_info_type'] = self.REL_TYPES[self.reltype]
            rel['r_info']      = (rel['r_info_sym']<<32) + rel['r_info_type']
            # print(rel)
            return Config.CubinELFStructs.Elf_Rel.build(rel)

    def __str__(self):
        s = '@section %s: offset=%s, relsym=%d(%s), reltype=%s, reladd=%s'%(
            self.section.name,
            self.offset,
            self.relsymid,
            self.relsymname,
            self.reltype,
            self.reladd)
        return s

class CuAsmFile(object):

    def __init__(self):
        self.mSMVersion = None           # sm version

        self.headerflags = None
        self.elftype = None

        self.fileHeader = {}  # unprocessed elf file header
        self.__mFileHeader = Config.defaultCubinFileHeader.copy()

        self.__mSectionList = OrderedDict()
        self.__mSegmentList = []

        self.__mLastSection = None
        self.__mCurrSection = None

        self.__mBuf = BytesIO() # global buffer for whole elf file, but without current section

    def buildFileHeader(self):

        self.__mFileHeader['e_ident']['EI_OSABI']      = self.fileHeader['ident_osabi']
        self.__mFileHeader['e_ident']['EI_ABIVERSION'] = self.fileHeader['ident_abiversion']
        self.__mFileHeader['e_type']                   = self.fileHeader['type']
        self.__mFileHeader['e_machine']                = self.fileHeader['machine']
        self.__mFileHeader['e_version']                = self.fileHeader['version']
        self.__mFileHeader['e_entry']                  = self.fileHeader['entry']
        self.__mFileHeader['e_phoff']                  = self.fileHeader['phoff']
        self.__mFileHeader['e_shoff']                  = self.fileHeader['shoff']
        self.__mFileHeader['e_flags']                  = self.fileHeader['flags']
        self.__mFileHeader['e_ehsize']                 = self.fileHeader['ehsize']
        self.__mFileHeader['e_phentsize']              = self.fileHeader['phentsize']
        self.__mFileHeader['e_phnum']                  = self.fileHeader['phnum']
        self.__mFileHeader['e_shentsize']              = self.fileHeader['shentsize']
        self.__mFileHeader['e_shnum']                  = self.fileHeader['shnum']
        self.__mFileHeader['e_shstrndx']               = self.fileHeader['shstrndx']

        return Config.CubinELFStructs.Elf_Ehdr.build(self.__mFileHeader)

    def getFileHeaderStruct(self):
        return self.__mFileHeader

    def emitAlign(self, align):
        ''' padding last section to required alignments.
        
            Return the padded length.
        '''

        pos = self.tell()
        ppos = align * ((pos+align-1) // align)
        if ppos > pos:  # do padding with required 0-bytes/nops
            if self.__mLastSection is not None:
                padbytes = self.__mLastSection.genSectionPaddingBytes(ppos - pos)
            else:
                padbytes = b'\x00' * (ppos - pos)
            self.__mBuf.write(padbytes)
        
        return ppos-pos

    def seek(self, offset):
        self.__mBuf.seek(offset)

    def tell(self):
        return self.__mBuf.tell()

    def saveAsCubin(self, cubinname):
        with open(cubinname, 'wb') as fout:
            fout.write(self.__mBuf.getvalue())

class CuAsmParser(object):
    ''' Parser for cuasm file.'''

#### static variables, mostly re patterns
    m_cppcomment = re.compile(r'//.*$')      # cpp style line comments
    m_ccomment = re.compile(r'\/\*.*?\*\/')  # c   style line
    m_bracomment = re.compile(r'\(\*.*\*\)') # notes for bra targets in sm_5x/6x
                                             # such as (*"INDIRECT_CALL"*)

    m_directive = re.compile(r'(\.[a-zA-Z0-9_]+)\s*(.*)')
    m_label     = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')  # "#" for offset label auto rename
    m_symbol    = re.compile(r'[a-zA-Z0-9._$@]+') #???

    m_byte  = re.compile(r'\b0x[a-fA-F0-9]{2}\b')
    m_short = re.compile(r'\b0x[a-fA-F0-9]{4}\b')
    m_word  = re.compile(r'\b0x[a-fA-F0-9]{8}\b')
    m_dword = re.compile(r'\b0x[a-fA-F0-9]{16}\b') # arch dependent?
    m_zero  = re.compile(r'\b[1-9][0-9]*\b')

    m_sufrel = re.compile(r'\[20@lo\(0x0\)=fun@R_CUDA_SURF_HEADER_INDEX\((\w+)\)\]')
    m_texrel = re.compile(r'\[20@lo\(0x0\)=(\w+)\]')

    # dtype that may take relocation arguments.
    rel_dtypes = {'dword':0, 'word' :1}

    dtype_pattern = {'byte'   : (m_byte , 1),
                     'short'  : (m_short, 2),
                     'word'   : (m_word , 4),
                     'dword'  : (m_dword, 8)}

#### constructors, and parsing entries
    def __init__(self):

        self.__mCuInsAsmRepos = None

        # directive dict
        self.__dirDict = {
            # predefined directives in nvdisasm
            '.headerflags'       : self.__dir_headerflags,           # set ELF header
            '.elftype'           : self.__dir_elftype,               # set ELF type
            '.section'           : self.__dir_section,               # declare a section
            '.sectioninfo'       : self.__dir_sectioninfo,           # set section info
            '.sectionflags'      : self.__dir_sectionflags,          # set section flags
            '.sectionentsize'    : self.__dir_sectionentsize,        # set section entsize
            '.align'             : self.__dir_align,                 # set alignment
            '.byte'              : self.__dir_byte,                  # emit bytes
            '.short'             : self.__dir_short,                 # emit shorts
            '.word'              : self.__dir_word,                  # emit word (4B?)
            '.dword'             : self.__dir_dword,                 # emit dword (8B?)
            '.type'              : self.__dir_type,                  # set symbol type
            '.size'              : self.__dir_size,                  # set symbol size
            '.global'            : self.__dir_global,                # declare a global symbol
            '.weak'              : self.__dir_weak,                  # declare a weak symbol
            '.zero'              : self.__dir_zero,                  # emit zero bytes
            '.other'             : self.__dir_other,                 # set symbol other 
            # supplementary directives defined by cuasm
            # all for setting some ELF/Section/Segment header attributes
            # some may with same funtionality as predefined directives
            # predefined directives of nvdisasm have higher priority
            '.__elf_ident_osabi'      : (lambda args: self.__dir_elfheader('ident_osabi'     , args)),
            '.__elf_ident_abiversion' : (lambda args: self.__dir_elfheader('ident_abiversion', args)),
            '.__elf_type'             : (lambda args: self.__dir_elfheader('type'            , args)),
            '.__elf_machine'          : (lambda args: self.__dir_elfheader('machine'         , args)),
            '.__elf_version'          : (lambda args: self.__dir_elfheader('version'         , args)),
            '.__elf_entry'            : (lambda args: self.__dir_elfheader('entry'           , args)),
            '.__elf_phoff'            : (lambda args: self.__dir_elfheader('phoff'           , args)),
            '.__elf_shoff'            : (lambda args: self.__dir_elfheader('shoff'           , args)),
            '.__elf_flags'            : (lambda args: self.__dir_elfheader('flags'           , args)),
            '.__elf_ehsize'           : (lambda args: self.__dir_elfheader('ehsize'          , args)),
            '.__elf_phentsize'        : (lambda args: self.__dir_elfheader('phentsize'       , args)),
            '.__elf_phnum'            : (lambda args: self.__dir_elfheader('phnum'           , args)),
            '.__elf_shentsize'        : (lambda args: self.__dir_elfheader('shentsize'       , args)),
            '.__elf_shnum'            : (lambda args: self.__dir_elfheader('shnum'           , args)),
            '.__elf_shstrndx'         : (lambda args: self.__dir_elfheader('shstrndx'        , args)),
            #
            '.__section_name'         : (lambda args: self.__dir_sectionheader('name'    , args)),
            '.__section_type'         : (lambda args: self.__dir_sectionheader('type'    , args)),
            '.__section_flags'        : (lambda args: self.__dir_sectionheader('flags'   , args)),
            '.__section_addr'         : (lambda args: self.__dir_sectionheader('addr'    , args)),
            '.__section_offset'       : (lambda args: self.__dir_sectionheader('offset'  , args)),
            '.__section_size'         : (lambda args: self.__dir_sectionheader('size'    , args)),
            '.__section_link'         : (lambda args: self.__dir_sectionheader('link'    , args)),
            '.__section_info'         : (lambda args: self.__dir_sectionheader('info'    , args)),
            '.__section_entsize'      : (lambda args: self.__dir_sectionheader('entsize' , args)),
            #
            '.__segment'              : self.__dir_segment,
            '.__segment_offset'       : (lambda args: self.__dir_segmentheader('offset' , args)),
            '.__segment_vaddr'        : (lambda args: self.__dir_segmentheader('vaddr'  , args)),
            '.__segment_paddr'        : (lambda args: self.__dir_segmentheader('paddr'  , args)),
            '.__segment_filesz'       : (lambda args: self.__dir_segmentheader('filesz' , args)),
            '.__segment_memsz'        : (lambda args: self.__dir_segmentheader('memsz'  , args)),
            '.__segment_align'        : (lambda args: self.__dir_segmentheader('align'  , args)),
            '.__segment_startsection' : (lambda args: self.__dir_segmentheader('startsection'  , args)),
            '.__segment_endsection'   : (lambda args: self.__dir_segmentheader('endsection'    , args))}

    def reset(self):
        self.__mLineNo        = 0
        self.__mInTextSection = False

        self.__mCurrSection   = None
        self.__mCurrSegment   = None
        self.__mCuAsmFile     = CuAsmFile()

        self.__mSectionDict   = OrderedDict()
        self.__mSymbolDict    = OrderedDict()
        self.__mSegmentList   = []
        self.__mFixupList     = []  # Fixup values that should be modified

        self.__mLabelDict     = OrderedDict()  # labels
        self.__mSecSizeLabel  = OrderedDict()  # labels that defined at last of one section
        self.__mRelList       = []  # relocations

        self.__mNVInfoOffsetLabels  = {} # key:sectionname, value: tuple(NVInfo_Attr, prefix)
        self.__mInsIndex      = 0   # Current instruction index
        self.m_Arch   = None

        self.__mPadSizeBeforeSecHeader = 0 # number of padding bytes before section header

        # TODO: not implemented yet
        # current all the entries are copied from cubin
        # self.__mStrList       = []  # string may have identical entries
        # self.__mShstrDict     = OrderedDict() # entries

    @CuAsmLogger.logTimeIt
    def parse(self, fname):
        ''' Parsing input file

            General parsing work flow:
            - scan whole file, gathering file headers, section headers/contents, segment headers
              build fixup lists, split kernel text sections for kernel assembler.
            - build internal tables, such as .shstrtab, .strtab. .symtab (Currently just copied except symbol size)
            - build kernel text sections, update .nv.info sections if necessary.
              update relocations if there are any.
            - evaluate fixups, patching the bytes of corresponding section data.
            - build relocation sections
            - layout sections, update file header, section header, segment header accordingly
            - write to file/stream
        '''
        self.reset()

        CuAsmLogger.logEntry('Parsing file %s'%fname)

        self.__mFilename = fname
        if not os.path.isfile(fname):
            raise self.__assert(False, "Cannot find input cuasm file %s!!!"%fname)
        else:
            with open(fname, 'r') as fin:
                self.__mLines = fin.readlines()

        self.__preScan()
        self.__gatherTextSectionSizeLabel()

        self.__buildInternalTables()
        self.__evalFixups() # 
        self.__parseKernels()
        #
        self.__buildRelocationSections()

        # Section layouting should be called when all sizes of sections are determined.
        # But section contents can be modified (but not resized)
        #
        # The layout will also determine the size label of text sections
        # which may affect the symbol size in symtab
        self.__layoutSections() 

        self.__updateSymtab()

    @CuAsmLogger.logTimeIt
    def saveAsCubin(self, fstream):
        if isinstance(fstream, str):
            fout = open(fstream, 'wb')
            needClose = True
            CuAsmLogger.logEntry('Saving as cubin file %s...'%fstream)
        else:
            fout = fstream
            needClose = False
            CuAsmLogger.logEntry('Saving as cubin file to stream...')

        disppos = lambda s: CuAsmLogger.logSubroutine("%#08x(%08d) : %s"%(fout.tell(), fout.tell(), s))
        # write ELF file header
        disppos('FileHeader')
        fout.write(self.__mCuAsmFile.buildFileHeader())

        # write section data
        for sname,sec in self.__mSectionDict.items():
            disppos('SectionData %s'%sname)
            sec.writePaddedData(fout)

        # write padding bytes before section header
        if self.__mPadSizeBeforeSecHeader > 0:
            disppos('Padding %d bytes before section header' % self.__mPadSizeBeforeSecHeader)
            fout.write(b'\x00' * self.__mPadSizeBeforeSecHeader)

        # write section headers
        for sname,sec in self.__mSectionDict.items():
            disppos('SectionHeader %s'%sname)
            fout.write(sec.buildHeader())

        # write segment headers
        for seg in self.__mSegmentList:
            disppos('Segment')
            fout.write(seg.build())

        if needClose:
            fout.close()

    def setInsAsmRepos(self, fname, arch):
        self.__mCuInsAsmRepos = CuInsAssemblerRepos(fname, arch=arch)

#### Procedures, every function is a seperate parsing step.
    @CuAsmLogger.logTraceIt
    def __preScan(self):
        ''' first scan to gather sections/symbol
            
            build all entries for labels.
        '''

        for line in self.__mLines:
            nline = CuAsmParser.stripComments(line).strip()
            self.__mLineNo += 1

            if len(nline)==0:  # skip blank/all-comments lines
                continue

            ltype = self.__getLineType(nline)
            if ltype is None:
                self.__assert(False, "Unreconized line contents:\n     %s"%line)

            elif ltype == 'label':
                res = self.m_label.match(nline)
                rlabel =  res.groups()[0]
                pos = self.__tellLocal()

                label =  self.__checkNVInfoOffsetLabels(self.__mCurrSection, rlabel, pos)
                
                if label not in self.__mLabelDict:
                    self.__mLabelDict[label] = CuAsmLabel(label, self.__mCurrSection,
                                                          pos, self.__mLineNo)
                else:
                    v = self.__mLabelDict[label]
                    self.__assert(False, 'Redefinition of label %s! First occurrence in Line%d!'%
                                  (v.name, v.lineno))

            elif ltype == 'directive':
                
                res = self.m_directive.match(nline)
                cmd = res.groups()[0]
                
                # print('Run directive %s @line %d.'%(cmd, self.__mLineNo))
                
                self.__assert(cmd in self.__dirDict, 'Unknown directive %s!!!' %cmd)

                farg = res.groups()[1].strip()
                if len(farg) == 0:
                    args = []
                else:
                    args = re.split(r'\s*,\s*', farg)

                # run the directive
                self.__dirDict[cmd](args)
            elif ltype == 'code':
                # During prescan, write all zeros for placeholder
                pos = self.m_Arch.getInsOffsetFromIndex(self.__mInsIndex)
                self.__mCurrSection.seek(pos)
                
                # all contents of .text section will be re-written 
                self.__emitBytes(b'\x00'*self.m_Arch.getInstructionLength())
                self.__mInsIndex += 1

            elif ltype == 'blank':
                continue
    
    @CuAsmLogger.logTraceIt
    def __gatherTextSectionSizeLabel(self):
        self.__mSecSizeLabel = OrderedDict()
        for label, labelobj in self.__mLabelDict.items():
            secname = labelobj.section.name
            if not secname.startswith('.text'):
                continue

            if labelobj.offset == self.__mSectionDict[secname].getDataSize():
                # print(f'Size label {label} for {secname}!')
                self.__mSecSizeLabel[secname] = labelobj

    @CuAsmLogger.logTraceIt
    def __parseKernels(self):
        # scan text sections to assemble kernels
        section_markers = splitAsmSection(self.__mLines)
        regnumdict = {}
        for secname in section_markers:
            if secname.startswith('.text.'):
                section = self.__mSectionDict[secname]
                m0, m1 = section_markers[secname]
                self.__mCurrSection = section
                self.__parseKernelText(section, m0, m1)
                section.updateResourceInfo()
                kname = secname[6:] # strip ".text."
                symidx = self.__getSymbolIdx(kname)
                regnumdict[symidx] = section.extra['regnum']

        sec = self.__mSectionDict['.nv.info']

        # print(sec.getData().hex())
        nvinfo = CuNVInfo(sec.getData(), self.m_Arch)
        self.m_Arch.setRegCountInNVInfo(nvinfo, regnumdict)
        sec.setData(nvinfo.serialize())

    @CuAsmLogger.logTraceIt    
    def __buildInternalTables(self):
        ''' Build .shstrtab/.strtab/.symtab entries.

        '''
        self.__mShstrtabDict = buildStringDict(self.__mSectionDict['.shstrtab'].getData())
        self.__mStrtabDict = buildStringDict(self.__mSectionDict['.strtab'].getData())
        self.__mSymtabDict = CuAsmSymbol.buildSymbolDict(self.__mStrtabDict,
                                                         self.__mSectionDict['.symtab'].getData())

    # @CuAsmLogger.logTraceIt
    def __parseKernelText(self, section, line_start, line_end):
        CuAsmLogger.logProcedure('Parsing kernel text of "%s"...'%section.name)

        kasm = CuKernelAssembler(ins_asm_repos=self.__mCuInsAsmRepos, version=self.m_Arch)

        p_textline = re.compile(r'\[([\w:-]+)\](.*)')

        ins_idx = 0
        for lineidx in range(line_start, line_end):
            line = self.__mLines[lineidx]

            nline = CuAsmParser.stripComments(line).strip()
            self.__mLineNo = lineidx + 1

            if len(nline)==0 or (self.m_label.match(nline) is not None) or (self.m_directive.match(nline) is not None):
                continue
            
            res = p_textline.match(nline)
            if res is None:
                self.__assert(False, 'Unrecognized code text!')
            
            ccode_s = res.groups()[0]
            icode_s = res.groups()[1]

            if c_ControlCodesPattern.match(ccode_s) is None:
                self.__assert(False, f'Illegal control code text "{ccode_s}"!')

            addr = self.m_Arch.getInsOffsetFromIndex(ins_idx)
            c_icode_s = self.__evalInstructionFixup(section, addr, icode_s)
            
            print("Parsing %s : %s "%(ccode_s, c_icode_s))
            try:
                kasm.push(addr, c_icode_s, ccode_s)
            except Exception as e:
                self.__assert(False, 'Error when assembling instruction "%s":\n        %s'%(nline, e))
            
            ins_idx += 1
        
        # rewrite text sections
        codebytes = kasm.genCode()
        section.seek(0)
        section.emitBytes(codebytes)

        # update offsets in NVInfo
        kname = section.name[6:] # strip '.text.'
        info_sec = self.__mSectionDict['.nv.info.' + kname]

        if kname in self.__mNVInfoOffsetLabels:
            offset_label_dict = self.__mNVInfoOffsetLabels[kname]
            offset_label_dict.update(kasm.m_ExtraInfo)
        else:
            offset_label_dict = kasm.m_ExtraInfo.copy()
        
        nvinfo = CuNVInfo(info_sec.getData(), self.m_Arch)
        nvinfo.updateNVInfoFromDict(offset_label_dict)
        info_sec.setData(nvinfo.serialize())

    @CuAsmLogger.logTraceIt
    def __sortSections(self):
        ''' Sort the sections. (TODO: Not implemented yet, all sections are kept as is.)

        Some section orders may do not matter, but the ELF segments may have some requirements ??? (TODO: checkit.)
        This is a sample layout of sections:

        Index Offset   Size ES Align   Type   Flags Link     Info Name
            1     40    2d9  0  1    STRTAB       0    0        0 .shstrtab
            2    319    416  0  1    STRTAB       0    0        0 .strtab
            3    730    2e8 18  8    SYMTAB       0    2       10 .symtab
            4    a18    2a0  0  1  PROGBITS       0    0        0 .debug_frame
            5    cb8     b4  0  4 CUDA_INFO       0    3        0 .nv.info
            6    d6c     6c  0  4 CUDA_INFO       0    3       17 .nv.info._Z4testPiS_S_
            7    dd8     40  0  4 CUDA_INFO       0    3       1b .nv.info._Z5childPii
            8    e18     40  0  4 CUDA_INFO       0    3       1c .nv.info._Z5stestfPf
            9    e58      4  0  4 CUDA_INFO       0    3       1a .nv.info._Z2f3ii
            a    e5c      4  0  4 CUDA_INFO       0    3       18 .nv.info._Z2f1ii
            b    e60      4  0  4 CUDA_INFO       0    3       19 .nv.info._Z2f2ii
            c    e68     40 10  8       REL       0    3       14 .rel.nv.constant0._Z4testPiS_S_
            d    ea8     50 10  8       REL       0    3       17 .rel.text._Z4testPiS_S_
            e    ef8     60 18  8      RELA       0    3       17 .rela.text._Z4testPiS_S_
            f    f58     20 10  8       REL       0    3       1b .rel.text._Z5childPii
           10    f78     30 10  8       REL       0    3       1d .rel.nv.global.init
           11    fa8     60 10  8       REL       0    3        4 .rel.debug_frame
           12   1008    118  0  4  PROGBITS       2    0        0 .nv.constant3
           13   1120      8  0  8  PROGBITS       2    0       17 .nv.constant2._Z4testPiS_S_
           14   1128    188  0  4  PROGBITS       2    0       17 .nv.constant0._Z4testPiS_S_
           15   12b0    16c  0  4  PROGBITS       2    0       1b .nv.constant0._Z5childPii
           16   141c    170  0  4  PROGBITS       2    0       1c .nv.constant0._Z5stestfPf
           17   1600    900  0 80  PROGBITS       6    3 18000011 .text._Z4testPiS_S_
           18   1f00     80  0 80  PROGBITS       6    3 18000012 .text._Z2f1ii
           19   1f80    200  0 80  PROGBITS       6    3 18000013 .text._Z2f2ii
           1a   2180    200  0 80  PROGBITS       6    3 18000014 .text._Z2f3ii
           1b   2380    180  0 80  PROGBITS       6    3  a000016 .text._Z5childPii
           1c   2500    100  0 80  PROGBITS       6    3  8000017 .text._Z5stestfPf
           1d   2600     24  0  8  PROGBITS       3    0        0 .nv.global.init
           1e   2624     40  0  4    NOBITS       3    0        0 .nv.global
        '''

        # TODO:
        # section_weights = ['.shstrtab', '.strtab', '.symtab', '.debug_frame', '.nv.info']

        pass

    @CuAsmLogger.logTraceIt
    def __buildRelocationSections(self):

        relSecDict = defaultdict(lambda : [])

        for rel in self.__mRelList:
            if rel.isRELA():
                sname = '.rela' + rel.section.name
            else:
                sname = '.rel' + rel.section.name
            
            # FIXME: insert REL/RELA sections if necessary
            relSecDict[sname].append(rel)
        
        # CHECK: The order of rel entries probably does not matter
        #        But to reduce unmatchness w.r.t. original cubin
        #        The order is reversed as the official toolkit does.
        for sname in relSecDict:
            section = self.__mSectionDict[sname]
            rellist = relSecDict[sname]
            nrel = len(rellist)
            for i in range(nrel):
                rel = rellist.pop() # FIFO of list
                section.emitBytes(rel.buildEntry())

    @CuAsmLogger.logTraceIt
    def __evalFixups(self):
        for i,fixup in enumerate(self.__mFixupList):
            try:
                # check relocation
                # Relocation rules for fixups (NOT include the text section):
                #   1. dtype in dword/word
                #   2. expr is non-literal (0x**)
                #   3. expr not started with index@, no @srel present
                #
                # CHECK: what if "Symbol + label@srel ? "
                #        seems still a relocation, but the value is the label value instead of zero.

                expr = fixup.expr

                if fixup.dtype not in self.rel_dtypes or expr.startswith('index@'):
                    val, _ = self.__evalExpr(expr)
                    fixup.value = val
                    self.__updateSectionForFixup(fixup)
                else: # 
                    # TODO: check other types of relocations

                    # Check relocations for texture/surface references
                    if fixup.dtype == 'word':
                        res = self.m_texrel.match(expr)
                        if res is not None:
                            symname = res.groups()[0]
                            relsymid = self.__getSymbolIdx(symname)
                            reltype = 'R_CUDA_TEX_HEADER_INDEX'
                            
                            rel = CuAsmRelocation(fixup.section, fixup.offset, symname, relsymid, reltype=reltype, reladd=None)
                            self.__mRelList.append(rel)
                            continue # go process next fixup
                        
                        res2 = self.m_sufrel.match(expr)
                        if res2 is not None:
                            symname = res2.groups()[0]
                            relsymid = self.__getSymbolIdx(symname)
                            reltype = 'R_CUDA_SURF_HEADER_INDEX'
                            
                            rel = CuAsmRelocation(fixup.section, fixup.offset, symname, relsymid, reltype=reltype, reladd=None)
                            self.__mRelList.append(rel)
                            continue # go process next fixup

                    # check explicit types of relocations
                    # Example : fun@R_CUDA_G64(C1)
                    # Seems only appear in debug version?
                    p_rel = re.compile(r'fun@(\w+)\(([^\)])\)')
                    res_rel = p_rel.match(expr)
                    if res_rel:
                        reltype = res_rel.groups()[0]
                        symname = res_rel.groups()[1]
                        symidx = self.__getSymbolIdx(symname)

                        rel = CuAsmRelocation(fixup.section, fixup.offset, symname, symidx, reltype=reltype, reladd=None)
                        self.__mRelList.append(rel)

                        continue

                    # check other types of relocations
                    val, vs = self.__evalExpr(expr)
                    if isinstance(vs[0], str): # symbol name in vs[0]
                        symname = vs[0]
                        relsymid = self.__getSymbolIdx(symname) # index of symbol
                        if fixup.dtype=='word':
                            reltype='R_CUDA_32'
                        elif fixup.dtype=='dword':
                            reltype='R_CUDA_64'
                        else:
                            self.__assert(False, 'Unknown data type for relocation: %s'%fixup.dtype)

                        rel = CuAsmRelocation(fixup.section, fixup.offset, symname, relsymid, reltype=reltype, reladd=None)
                        self.__mRelList.append(rel)

                    if val is not None: # symbol + label@srel, seems the label value is filled.
                        fixup.value = val 
                        self.__updateSectionForFixup(fixup)

            except Exception as e:
                self.__assert(False, 'Error when evaluating fixup @line%d: expr=%s, msg=%s'
                                %(fixup.lineno, fixup.expr, e))

    @CuAsmLogger.logTraceIt
    def __updateSymtab(self):
        
        bio = BytesIO(self.__mSectionDict['.symtab'].getData())
        symsize = Config.CubinELFStructs.Elf_Sym.sizeof()

        for i, s in enumerate(self.__mSymtabDict):
            symid, syment = self.__mSymtabDict[s]

            if s in self.__mLabelDict:
                syment['st_value'] = self.__mLabelDict[s].offset
                
                if s in self.__mSymbolDict: # symbols explicitly defined in assembly
                    symobj = self.__mSymbolDict[s]
                    symobj.value = self.__mLabelDict[s].offset
                    symobj.sizeval, _ = self.__evalExpr(symobj.size)
                
                    syment['st_size'] = symobj.sizeval

                    # print(syment)
                    CuAsmSymbol.resetSymtabEntryValueSize(bio, i*symsize, symobj.value, symobj.sizeval)

            else: # some symbols does not have corresponding labels, such as vprintf
                pass        
        self.__mSectionDict['.symtab'].setData(bio.getvalue())

    @CuAsmLogger.logTraceIt
    def __layoutSections(self):
        ''' Layout section data, do section padding if needed. Update section header.offset/size.
        
            Update segment range accordingly.
            Update ELF file header accordingly.
        '''

        # initialize the offset as the ELF header size
        elfheadersize = Config.CubinELFStructs.Elf_Ehdr.sizeof()
        file_offset = elfheadersize
        mem_offset = elfheadersize
        prev_sec = None

        sh_edges = {} # key=secname, value = (file_start, file_end, mem_start, mem_end)
        # First pass to get the size of every section
        # NOTE: the size of current section depends the padding, which is determined by next section
        #       Seems only for text section? For other sections, padding will not count in size?
        for secname, sec in self.__mSectionDict.items():
            if secname == '':
                continue

            # print(secname)
            align = sec.addralign
            if prev_sec is not None and prev_sec.name.startswith('.text'):
                align = 128
            file_offset, mem_offset = self.__updateSectionPadding(prev_sec, file_offset, mem_offset, align)

            sec.size = sec.getDataSize()
            sec.offset = file_offset

            sec.header['size'] = sec.size
            sec.header['offset'] = sec.offset
            
            prev_sec = sec
            sh_edges[secname] = (file_offset, 0, mem_offset, 0)

            mem_offset += sec.size
            if sec.header['type'] != 'SHT_NOBITS':
                file_offset += sec.size
        
        # ???
        if prev_sec is not None and prev_sec.name.startswith('.text'):
            file_offset, mem_offset = self.__updateSectionPadding(prev_sec, file_offset, mem_offset, 128)
        
        # Section pass to build the section edges, for locating segment range
        for secname, sec in self.__mSectionDict.items():
            if secname == '':
                continue
            
            sec.size = sec.getDataSize()
            sec.header['size'] = sec.size

            if sec.header['type'] != 'SHT_NOBITS':
                fsize = sec.size
                msize = fsize
            else:
                fsize = 0
                msize = sec.size

            file_pos, _, mem_pos, _ =  sh_edges[secname]
            sh_edges[secname] = (file_pos, file_pos + fsize, mem_pos, mem_pos + msize)

        # FIXME: better alignment for headers ?
        file_offset, self.__mPadSizeBeforeSecHeader = alignTo(file_offset, 8)
        
        # Current only the normal order is support:
        #      ELFHeader -> SectionData -> SectionHeader -> SegmentHeader
        # Other orders may be possible, but not supported yet.

        SecHeaderLen = len(self.__mSectionDict) * Config.CubinELFStructs.Elf_Shdr.sizeof()

        self.__mCuAsmFile.fileHeader['shoff'] = file_offset

        phoff = file_offset + SecHeaderLen
        phlen = self.__mCuAsmFile.fileHeader['phentsize'] * self.__mCuAsmFile.fileHeader['phnum']
        self.__mCuAsmFile.fileHeader['phoff'] = phoff

        sh_edges[PROGRAM_HEADER_TAG] = phoff, phoff+phlen, phoff, phoff+phlen

        for seg in self.__mSegmentList:
            if seg.header['type'] == 'PT_PHDR':
                seg.header['offset'] = file_offset + SecHeaderLen
                seg.header['filesz'] = Config.CubinELFStructs.Elf_Phdr.sizeof() * len(self.__mSegmentList)
                seg.header['memsz'] = seg.header['filesz']

            elif seg.header['type'] == 'PT_LOAD':
                # if startsection is empty, this segment is empty
                # Seems a convention of compiler?
                if seg.header['startsection'] != '' and seg.header['endsection'] != '':
                    file_start0, file_end0, mem_start0, mem_end0 = sh_edges[seg.header['startsection']]
                    file_start1, file_end1, mem_start1, mem_end1 = sh_edges[seg.header['endsection']]

                    seg.header['offset'] = file_start0
                    seg.header['filesz'] = file_end1 - file_start0
                    seg.header['memsz'] = mem_end1 - mem_start0

            else:
                msg = 'Unknown segment type %s!'%seg.header['type']
                CuAsmLogger.logError(msg)
                raise Exception(msg)

            # update header
            seg.updateHeader()
            
#### Directives
    def __dir_headerflags(self, args):
        self.__assertArgc('.headerflags', args, 1, allowMore=False)
        self.__mCuAsmFile.headerflags = args[0]

    def __dir_elftype(self, args):
        self.__assertArgc('.elftype', args, 1, allowMore=False)
        self.__mCuAsmFile.elftype = args[0]

    def __dir_section(self, args):
        self.__assertArgc('.section', args, 3, allowMore=False)

        # for implict sections, quotes are used for embracing the section name
        # mainly for the NULL section with empty name ""
        # thus the quotes will be stripped
        secname = args[0].strip('"')

        self.__assert(secname not in self.__mSectionDict, 'Redefinition of section "%s"!'%secname)
        self.__mCurrSection = CuAsmSection(secname, args[1], args[2])

        CuAsmLogger.logSubroutine('Line %6d: New section "%s"'%(self.__mLineNo, secname))

        self.__mSectionDict[secname] = self.__mCurrSection

        if args[0].startswith('.text.'):
            self.__mInTextSection = True
            self.__mInsIndex = 0
        else:
            self.__mInTextSection = False

    def __dir_sectionflags(self, args):
        self.__assertArgc('.sectionflags', args, 1, allowMore=False)
        self.__mCurrSection.flags.append(args[0])

    def __dir_sectionentsize(self, args):
        self.__assertArgc('.sectionentsize', args, 1, allowMore=False)
        self.__mCurrSection.entsize = int(args[0])

    def __dir_sectioninfo(self, args):
        self.__assertArgc('.sectioninfo', args, 1, allowMore=False)
        self.__assert(self.__mCurrSection is not None, "No active section!")

        # TODO: parse info, check correctness
        self.__mCurrSection.info.append(args[0])

    def __dir_byte(self, args):
        self.__assertArgc('.word', args, 1, allowMore=True)
        self.__emitTypedBytes('byte', args)

    def __dir_dword(self, args):
        ''' currently 1 dword = 8 bytes

        NOTE: .dword may reference a relocation symbol.
        '''

        self.__assertArgc('.dword', args, 1, allowMore=True)
        self.__emitTypedBytes('dword', args)

    def __dir_align(self, args):
        ''' .align directive may have different operations, depending on the context.

            Usually .align will pad current buffer with zeros/nops to required alignment.
            But for the first .align directive of a section, it also sets the alignment
            requirement of current section, which means the padding is done to last
            section, thus will not affect the local offset of current section.

            For `.align` inside a section, the padding counts to the local offset,
            thus will affect all the local fixup values.
        '''

        self.__assertArgc('.align', args, 1, allowMore=False)
        try:
            align = int(args[0])
        except:
            self.__assert(False, ' unknown alignment (%s)!' % args[0])

        self.__assert(align &(align-1) == 0, ' alignment(%d) should be power of 2!' % align)
        self.__mCurrSection.emitAlign(align)

    def __dir_short(self, args):
        self.__assertArgc('.short', args, 1, allowMore=True)
        self.__emitTypedBytes('short', args)

    def __dir_word(self, args):
        self.__assertArgc('.word', args, 1, allowMore=True)
        self.__emitTypedBytes('word', args)

    def __dir_type(self, args):
        ''' .type will define the symbol type.

        Example: .type     flist  ,@object
                 .type     $str   ,@object
                 .type     vprintf,@function
        '''

        self.__assertArgc('.type', args, 2, allowMore=False)
        symbol = args[0]
        if symbol not in self.__mSymbolDict:
            self.__mSymbolDict[symbol] = CuAsmSymbol(symbol)

        stype = args[1]
        self.__assert(stype in CuAsmSymbol.SymbolTypes,
                      'Unknown symbol type %s! Available: %s.'%(stype, str(CuAsmSymbol.SymbolTypes)))
        self.__mSymbolDict[symbol].type = stype

    def __dir_size(self, args):
        self.__assertArgc('.size', args, 2, allowMore=False)
        symbol = args[0]
        if symbol not in self.__mSymbolDict:
            self.__mSymbolDict[symbol] = CuAsmSymbol(symbol)

        # NOTE: the size of a symbol is probably an expression
        #       this will be evaluted when generating symbol tables
        self.__mSymbolDict[symbol].size = args[1]

    def __dir_global(self, args):
        '''.global defines a global symbol.

        A global symbol is visible to linker. For a cubin, it can be accessed by
        the driver api function `cuModuleGetGlobal`.
        '''

        self.__assertArgc('.global', args, 1, allowMore=False)

        symbol = args[0]
        if symbol not in self.__mSymbolDict:
            self.__mSymbolDict[symbol] = CuAsmSymbol(symbol)

        CuAsmLogger.logSubroutine('Line %6d global symbol %s'%(self.__mLineNo, symbol))

        self.__mSymbolDict[symbol].isGlobal = True

    def __dir_weak(self, args):
        '''.weak defines a weak symbol.

            A weak symbol is declared in current module, but may be overwritten by strong symbols.

            Currently no scope is implemented, thus
        '''

        self.__assertArgc('.weak', args, 1, allowMore=False)

        symbol = args[0]
        if symbol not in self.__mSymbolDict:
            self.__mSymbolDict[symbol] = CuAsmSymbol(symbol)
        
        CuAsmLogger.logWarning('Line %d: Weak symbol found! The implementation is not complete, please be cautious...'%self.__mLineNo)
        CuAsmLogger.logSubroutine('Line %6d: New weak symbol "%s"'%(self.__mLineNo, symbol))

        self.__mSymbolDict[symbol].isGlobal = True

    def __dir_zero(self, args):
        '''.zero emit zeros of specified length (in bytes).'''

        self.__assertArgc('.zero', args, 1, allowMore=False)
        try:
            # .zero only accepts a literal, no fixup allowed
            size = int(args[0])
            self.__emitBytes(b'\x00'*size)
        except:
            self.__assert(False, 'Unknown arg (%s) for .zero!'% args[0])

    def __dir_other(self, args):
        '''.other defines some properties of a symbol.

        Examples:
            .other    _Z4testPiS_S_, @"STO_CUDA_ENTRY STV_DEFAULT"
            .other    _Z5childPii  , @"STO_CUDA_ENTRY STV_DEFAULT"
            .other    _Z5stestfPf  , @"STO_CUDA_ENTRY STV_DEFAULT"
        '''
        self.__assertArgc('.other', args, 2, allowMore=False)

        symbol = args[0]
        if symbol not in self.__mSymbolDict:
            #self.__mSymbolDict[symbol] = CuAsmSymbol()
            self.__assert(False, 'Undefined symbol %s!!!'%symbol)

        self.__mSymbolDict[symbol].other = args[1]

    def __dir_elfheader(self, attrname, args):
        self.__assertArgc('.__elf_'+attrname, args, 1, allowMore=False)
        self.__mCuAsmFile.fileHeader[attrname] = self.__cvtValue(args[0])
        if attrname == 'flags':
            flags = int(args[0], 16)
            smversion = flags & 0xff
            self.m_Arch = CuSMVersion(smversion)

            if (not hasattr(self, '__mCuInsAsmRepos') 
                or self.__mCuInsAsmRepos is None 
                or (self.__mCuInsAsmRepos.getSMVersion() != self.m_Arch) ):

                CuAsmLogger.logSubroutine('Setting CuInsAsmRepos to default dict...')
                
                self.__mCuInsAsmRepos = CuInsAssemblerRepos(arch=self.m_Arch)
                self.__mCuInsAsmRepos.setToDefaultInsAsmDict()

    def __dir_sectionheader(self, attrname, args):
        self.__assertArgc('.__section_'+attrname, args, 1, allowMore=False)
        self.__mCurrSection.header[attrname] = self.__cvtValue(args[0])

    def __dir_segment(self, args):
        self.__assertArgc('.__segment', args, 2, allowMore=False)
        segment = CuAsmSegment(args[0].strip('"'), args[1])
        self.__mSegmentList.append(segment)
        self.__mCurrSegment = segment
        self.__mCurrSection = None

    def __dir_segmentheader(self, attrname, args):
        self.__assertArgc('.__segment_'+attrname, args, 1, allowMore=False)
        self.__mCurrSegment.header[attrname] = self.__cvtValue(args[0])

#### Subroutines
    def __assert(self, flag, msg=''):
        if not flag:
            full_msg  =  'Assertion failed in:\n'
            full_msg += f'    File {self.__mFilename}:{self.__mLineNo} :\n'
            full_msg += f'        {self.__mLines[self.__mLineNo-1].strip()}\n'
            full_msg += f'    {msg}'
            CuAsmLogger.logError(full_msg)
            raise Exception(full_msg)

    def __assertArgc(self, cmd, args, argc, allowMore=True):
        ''' Check the number of arguments.'''
        if allowMore:
            flag = len(args)>=argc
            es = 'at least '
        else:
            flag = len(args)==argc
            es = ''

        self.__assert(flag, '%s requires %s%d args! %d given: %s.'
                      %(cmd, es, argc, len(args), str(args))  )

    def __tellLocal(self):
        ''' tell current pos inside current active section.'''

        if self.__mCurrSection is not None:
            return self.__mCurrSection.tell()
        else:
            raise Exception("Cannot tell local pos without active section!")

    def __evalVar(self, var):
        """Evaluate a single variable

        Args:
            var ([string]): the variable expression

        Returns:
            (value, is_sym)
        """

        # symbol
        if var in self.__mSymtabDict:
            is_sym = True
        else:
            is_sym = False

        # int literal
        if m_intval.match(var):
            return eval(var), is_sym
        
        if var.endswith('@srel'):
            label = var.replace('@srel', '')
            if label not in self.__mLabelDict:
                raise Exception('Unknown expression %s'%var)

            return self.__mLabelDict[label].offset, is_sym
        
        if var in self.__mLabelDict:
            return self.__mLabelDict[var].offset, is_sym
        
        raise Exception('Unknown expression %s'%var)

    def __evalExpr(self, expr):
        ''' Evaluate the expression.

            value = value_a ((+|-) value_b)?
            Return: Tuple(value, Tuple(value_a, op, value_b) )

            For symbol at position a, the original symbol string will be returned as value a.

            Examples:
               Expr                       Value                 Section
               index@(symbol)             symbol index          non-text
               (.Label)                   label offset
               (.L0-.L1)                  
            
            NOTE: This subroutine has no context info, making it hard to interprete
                  thus all exceptions should be captured in __evalFixups, showing the full context
        '''

        # For expr: index@(symbol)
        if expr.startswith('index@'): # index of symbol
            symname = expr[6:].strip(' ()')
            index = self.__getSymbolIdx(symname)
            if index is None:
                raise Exception('Unknown symbol "%s"!!!'%symname)
            return index, (index, None, None)
        
        rexpr = expr.strip('`() ')
        res = re.match(r'([.\w$@]+)\s*(\+|-)*\s*([.\w$@]+)*', rexpr) # FIXME: what if the imme is negative???

        if res is None:
            raise Exception('Unknown expr %s !!!'%expr)
        else:
            a   = res.groups()[0]
            op  = res.groups()[1]
            b   = res.groups()[2]

            aval, a_issym = self.__evalVar(a)

            if op is None: # only one var
                if a_issym: # one symbol, definitely a relocation
                    return aval, (a   , None, None) 
                else:            # one label
                    return aval, (aval, None, None)
            else: # 
                bval, b_issym = self.__evalVar(b)  # in general context, the second var should not be symbol?
                                                   # but it's possible in size expression
                
                if a_issym:
                    a_realval = a
                else:
                    a_realval = aval

                if op == '+':
                    return aval + bval, (a_realval, '+', bval)
                elif op == '-':
                    return aval - bval, (a_realval, '-', bval)
                else: # never reach here, only +/- can be matched by re pattern.
                    raise Exception('Unknown expr.op "%s"'%op)

    def __getSymbolIdx(self, symname):
        ''' Get symbol index in symtab. '''
        if symname in self.__mSymtabDict:
            return self.__mSymtabDict[symname][0]
        else:
            return None

    def __evalInstructionFixup(self, section, offset, s):
        ''' Check fixups inside an instruction.

            Examples:
                RET.REL.NODEC R20 `(_Z4testPiS_S_);
                BRA `(.L_14);
            Relocations:    
                32@hi($str)                         => REL
                32@lo((_Z4testPiS_S_ + .L_8@srel))  => RELA
                `(vprintf)                          => REL
            
            TODO: How to determine the type of `(.LABEL) ???
              For symbol or label defined in the same section, it's a fixup
              Otherwise, it seems a relocation. (To be checked...)        
        '''
        p_ins_rel32 = re.compile(r'(32@hi|32@lo)\(([^\)]+)\)+')
        r1 = p_ins_rel32.search(s)
        if r1:
            expr = r1.groups()[1]
            val, val_sep = self.__evalExpr(expr)
            symname = val_sep[0]
            symidx = self.__getSymbolIdx(val_sep[0])
            relkey = r1.groups()[0]
            reltype = self.m_Arch.getInsRelocationType(relkey)

            if val_sep[1] is not None:
                rela = CuAsmRelocation(section, offset, symname, symidx, reltype=reltype, reladd=val_sep[2])
                self.__mRelList.append(rela)
            else:
                rel = CuAsmRelocation(section, offset, symname, symidx, reltype=reltype, reladd=None)
                self.__mRelList.append(rel)

            ns = p_ins_rel32.sub('0x0', s)
            return ns

        p_ins_label = re.compile(r'`\(([^\)]+)\)')
        r2 = p_ins_label.search(s)
        if r2:
            # print(s)
            label = r2.groups()[0]
            self.__assert((label in self.__mLabelDict) or (label in self.__mSymtabDict),
                          'Unknown label (%s) !!!'%label)
            
            # global symbols, no corresponding label (such as vprintf)
            if (label not in self.__mLabelDict) and (label in self.__mSymtabDict):
                # print(s)
                symname = label
                symidx = self.__getSymbolIdx(symname)
                reltype = self.m_Arch.getInsRelocationType('target')
                rel = CuAsmRelocation(section, offset, symname, symidx, reltype=reltype, reladd=None)
                self.__mRelList.append(rel)
                ns = p_ins_label.sub('0x0', s)
                return ns
            
            clabel = self.__mLabelDict[label]
            if section.name == clabel.section.name: # hardcoded target in current section
                val = clabel.offset
                ns = p_ins_label.sub('%#x'%val, s)
                return ns
            else: # relocations, since the target is in another section
                symname = label
                symidx = self.__getSymbolIdx(symname)
                reltype = self.m_Arch.getInsRelocationType('target')
                rel = CuAsmRelocation(section, offset, symname, symidx, reltype=reltype, reladd=None)
                self.__mRelList.append(rel)
                ns = p_ins_label.sub('0x0', s)
                return ns
        
        # No fixup patterns found
        return s

    def __updateSectionForFixup(self, fixup):
        ''' Update the corresponding section location for fixup.'''

        _, blen = self.dtype_pattern[fixup.dtype]
        bs = int.to_bytes(fixup.value, blen, 'little')
        fixup.section.updateForFixup(fixup.offset, bs)

        CuAsmLogger.logSubroutine('Eval fixup "%s" @line%d to %#x'%(fixup.expr, fixup.lineno, fixup.value))
        # print(fixup)

    def __emitBytes(self, bs):
        '''emit bytes to current section.'''
        self.__mCurrSection.emitBytes(bs)

    def __getLineType(self, line):
        '''There can be three line types:

            1. Directive: starts with ".\w+", but no following ":"
            2. Label: label name followed by ":"
            3. Instruction text: only in section with name prefix ".text",
               and not a label line
            (4. Blank lines, skipped)

        **NOTE**: usually all blanks lines will be skipped by the parser
        '''

        if len(line)==0:
            return 'blank'
        elif self.m_label.match(line) is not None:
            return 'label'
        elif self.m_directive.match(line) is not None:
            return 'directive'
        elif self.__mInTextSection:
            return 'code'
        else:
            return None
            #raise Exception("Unrecognized line contents!")

    def __emitTypedBytes(self, dtype, args):
        dp, dsize = self.dtype_pattern[dtype]

        for arg in args:
            # TODO: check contents of arg is really a fixup/relocation(may not defined yet!) ?
            #if dp.match(arg):
            #    self.__emitBytes(bytes.fromhex(arg[2:]))
            if arg.startswith('0x'):
                argv = int(arg, 16)
                arg_byte = argv.to_bytes(dsize, 'little')
                self.__emitBytes(arg_byte)
            else:
                # NOTE: currently all unknowns go to fixup list, 
                #       fixup will handle the relocations if needed.

                # all fixup values will be updated by the assembler
                fixup = CuAsmFixup(self.__mCurrSection, self.__tellLocal(),
                                   arg, dtype, self.__mLineNo)

                self.__mFixupList.append(fixup)

                # emit zeros as placeholder
                self.__emitBytes(b'\x00'*dsize)

    def __cvtValue(self, s):
        ''' Convert input string to int if possible.'''
        if m_intval.match(s):
            return eval(s)
        elif s.startswith('"') and s.endswith('"'):
            return s.strip('"')
        else:
            return s
    
    def __pushSectionSizeLabel(self):
        '''Identify the last label that marks the end of a text section. 
        
            DEPRECATED !!!

            The text section size label will be gathered in the procedure __gatherTextSectionSizeLabel()
        '''
        if self.__mCurrSection is not None and self.__mCurrSection.name.startswith('.text') and self.__mLabelDict is not None:
            key, lastlabel = self.__mLabelDict.popitem()
            if self.__mCurrSection.name == lastlabel.section.name and lastlabel.offset == self.__mCurrSection.tell():
                self.__mSecSizeLabel[self.__mCurrSection.name] = lastlabel
                self.__mLabelDict[key] = lastlabel # push it back
            else:
                self.__mLabelDict[key] = lastlabel # push it back

    def __genSectionPaddingBytes(self, sec, size):
        '''Generate padding bytes for section with given size.'''
        if sec.name.startswith('.text'):
            padbytes = self.m_Arch.getPadBytes()
        else:
            padbytes = b'\x00'

        if size % len(padbytes) != 0:
            raise Exception('Invalid padding size for section %s'%sec.name)

        npad = size // len(padbytes)
        return npad * padbytes

    def __updateSectionPadding(self, sec, file_offset, mem_offset, align):
        ''' Update section padding with size.
        
            For text sections: padding to the original section data, update size
            For other sections: padding to seperate padbytes, keep size unchanged
            For nobits sections: do nothing.
        '''
        if sec is None:
            return file_offset, mem_offset
        
        if sec.name.startswith('.text'):
            align = max(align, sec.addralign)
            file_offset, fpadsize = alignTo(file_offset, align)
            mem_offset, mpadsize = alignTo(mem_offset, align)

            sec.emitPadding(self.__genSectionPaddingBytes(sec, fpadsize))            
            
            # FIXME: This treatment is weird, but the text sections seems always aligned
            #        and last label of .text section seems to be the padded offset. 
            # 
            # Update size label offset, it will be used in symbol size evaluation.
            # I don't quite understand why it's this way, but let's just keep it as is.
            if sec.name in self.__mSecSizeLabel:
                sizelabel = self.__mSecSizeLabel[sec.name]
                # NOTE: donot use sec.size here
                sizelabel.offset = sec.getDataSize()
                CuAsmLogger.logSubroutine(f'Reset size label "{sizelabel.name}" of {sec.name} to {sec.getDataSize()}!')
            
        elif sec.header['type'] == 'SHT_NOBITS':
            mem_offset, mpadsize = alignTo(mem_offset, align)
            sec.padsize = mpadsize
            sec.padbytes = mpadsize * b'\x00'
        else:
            file_offset, fpadsize = alignTo(file_offset, align)
            mem_offset, mpadsize = alignTo(mem_offset, align)

            sec.padsize = fpadsize
            sec.padbytes = fpadsize * b'\x00'

        sec.updateHeader()

        return file_offset, mem_offset

    def __calcSegmentRange(self, sec_start, sec_end):

        inRange = False
        seg_off = 0
        filesz = 0
        memsz = 0

        for sname, sec in self.__mSectionDict.items():
            if sname == sec_start:
                inRange = True
                seg_off = sec.offset
                f_off = seg_off
                m_off = seg_off

            if inRange:
                psize = sec.getPaddedDataSize()
                m_off += psize
                if sec.header['type'] != 'SHT_NOBITS':
                    f_off += psize

                if sname == sec_end:
                    inRange = False
                    break
        
        filesz = f_off - seg_off
        memsz = m_off - seg_off

        return seg_off, filesz, memsz

    def __checkNVInfoOffsetLabels(self, section, labelname, offset):
        ''' Check whether the label is a NVInfoOffsetLabel, push to label offset dict if necessary.

            Valid offset label should be in form:
                .CUASM_OFFSET_LABEL.{SectionName}.{NVInfoAttributeName}.{Identifier}

            Identifier should be unique for every offset label (label cannot be defined twice).
            (A grammar sugar is to use "#", which will be replaced by "L+{LineNo}" such as "L000002f8"

            Example:
                .CUASM_OFFSET_LABEL._Z4testPiS_S_.EIATTR_EXIT_INSTR_OFFSETS.0:
                .CUASM_OFFSET_LABEL._Z4testPiS_S_.EIATTR_EXIT_INSTR_OFFSETS.#:
            
            Return: real label name

        '''

        # TODO: some offset labels (such as EXIT, CTAID.Z) may be detected automatically

        if not labelname.startswith('.CUASM_OFFSET_LABEL'):
            return labelname
        
        self.__assert(section.name.startswith('.text'), 'CUASM_OFFSET_LABEL should be defined in a text section!')

        kname = section.name[6:]
        vs = labelname[1:].split('.')
        self.__assert(len(vs)==4, 'Offset label should be in form: .CUASM_OFFSET_LABEL.{SectionName}.{NVInfoAttributeName}.{Identifier}')
        self.__assert(vs[1] == kname, 'CUASM_OFFSET_LABEL should include kernel name in second dot part!')

        if kname not in self.__mNVInfoOffsetLabels:
            self.__mNVInfoOffsetLabels[kname] = {}

        # .CUASM_OFFSET_LABEL._Z4testPiS_S_.EIATTR_EXIT_INSTR_OFFSETS.0:
        attr = vs[2]
        if attr in self.__mNVInfoOffsetLabels[kname]:
            self.__mNVInfoOffsetLabels[kname][attr].append(offset)
        else:
            self.__mNVInfoOffsetLabels[kname][attr] = [offset]
        
        if vs[3] == '#':
            lstr = 'L%08x'%self.__mLineNo
            return labelname[:-1] + lstr
        else:
            return labelname
      
#### Help functions to display some internal states.

    def dispFixupList(self):
        print('Fixup list:')
        if self.__mFixupList is None or len(self.__mFixupList)==0:
            print('    ' + str(self.__mFixupList))

        for i,f in enumerate(self.__mFixupList):
            print("Fixup %3d: %s"%(i, str(f)))

        print()

    def dispRelocationList(self):
        print('Relocation list:')
        if self.__mRelList is None or len(self.__mRelList)==0:
            print('    No relocations.')

        for i,r in enumerate(self.__mRelList):
            print('Relocation %3d: %s'%(i, r))
        print()

    def dispSectionList(self):
        print('Section list:')
        sdict = self.__mSectionDict
        if sdict is None or len(sdict) == 0:
            print('    No sections found.')
            return

        print(' Idx  Offset    Size    ES   AL  Type           Flags    Link      Info  Name')
        i = 0
        for s in sdict:
            sec = sdict[s]
            ss = '%4x' % i
            ss += '  {offset:6x}  {size:6x}  {entsize:4x}'.format(**sec.header)
            ss += '  {:3x}'.format(sec.addralign)
            if isinstance(sec.header['type'], str):
                ss += '  {type:12s}'.format(**sec.header)
            else:
                ss += '  {type:<12x}'.format(**sec.header)

            ss += '  {flags:6x}'.format(**sec.header)
            ss += '  {link:6x}  {info:8x}'.format(**sec.header)
            ss += '  ' + sec.name
            print(ss)
            
            i += 1
            
        print()

    def dispSymbolDict(self):
        print('\nSymbols:')
        for i,s in enumerate(self.__mSymbolDict):
            symbol = self.__mSymbolDict[s]
            print('Symbol %3d: %s'%(i,symbol))
        print()

    def dispSymtabDict(self):
        print('\nSymtab:')
        for s in self.__mSymtabDict:
            symid, syment = self.__mSymtabDict[s]
            print('Symbol %3d (%s): %s'%(symid, s, syment))
            if s in self.__mSymbolDict:
                print('    %s'%self.__mSymbolDict[s])
        print()
    
    def dispLabelDict(self):
        print('\nLabels: ')
        for i,l in enumerate(self.__mLabelDict):
            v = self.__mLabelDict[l]
            print('Label %3d: %s'%(i, str(v)))
        print()

    def dispSegmentHeader(self):
        print('Segment headers:')
        for seg in self.__mSegmentList:
            print(seg.header)

    def dispFileHeader(self):
        print('File header:')
        print(self.__mCuAsmFile.fileHeader)

    def dispTables(self):
        # self.buildInternalTables()
        print('.shstrtab:')
        for i, idx in enumerate(self.__mShstrtabDict):
            print('%3d  \t0x%x  \t%s'%(i, idx, self.__mShstrtabDict[idx]))

        print('.strtab:')
        for i, idx in enumerate(self.__mStrtabDict):
            print('%3d  \t0x%x  \t%s'%(i, idx, self.__mStrtabDict[idx]))

        print('.symtab')
        for i, s in enumerate(self.__mSymtabDict):
            print('%3d  \t%s'%(i, s))

    @CuAsmLogger.logTimeIt
    def saveCubinCmp(self, cubinname, sav_prefix):
        ''' A simple helper function to display current contents vs cubin in bytes. '''

        fasm = open(sav_prefix+'_asm.txt', 'w')
        fbin = open(sav_prefix+'_bin.txt', 'w')

        felf = open(cubinname, 'rb')
        ef = ELFFile(felf)

        fasm.write('FileHeader:\n' + str(self.__mCuAsmFile.getFileHeaderStruct()) + '\n')
        fbin.write('FileHeader:\n' + str(ef.header) + '\n' )

        # write section headers+data
        for sname,sec in self.__mSectionDict.items():
            fasm.write('# Section %s\n'%sname)
            fasm.write(str(sec.getHeaderStruct()) + '\n')
            if sec.getHeaderStruct()['sh_type'] != 'SHT_NOBITS':
                fasm.write(bytes2Asm(sec.getData()) +'\n\n')
            else:
                fasm.write('\n')

        # write segment headers
        for seg in self.__mSegmentList:
            fasm.write(str(seg.getHeaderStruct())+'\n')

        # write section headers+data
        for sec in ef.iter_sections():
            fbin.write('# Section %s\n'%sec.name)
            fbin.write(str(sec.header) + '\n')
            if sec.header['sh_type'] != 'SHT_NOBITS':
                fbin.write(bytes2Asm(sec.data()) + '\n\n')
            else:
                fbin.write('\n')

        # write segment headers
        for seg in ef.iter_segments():
            fbin.write(str(seg.header) + '\n')

        fasm.close()
        fbin.close()
        felf.close()

    @staticmethod
    def stripComments(s):
        ''' Strip comments of a line.

        NOTE: cross line comments are not supported yet.
        '''

        s = CuAsmParser.m_cppcomment.subn(' ', s)[0] # replace comments as a single space, avoid unwanted concatination
        s = CuAsmParser.m_ccomment.subn(' ', s)[0]
        s = CuAsmParser.m_bracomment.subn(' ', s)[0]
        s = re.subn(r'\s+', ' ', s)[0]       # replace one or more spaces/tabs into one single space

        return s.strip()

if __name__ == '__main__':
    pass
