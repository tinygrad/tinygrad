# -*- coding: utf-8 -*-

from elftools.elf.elffile import ELFFile
from elftools.elf.structs import ELFStructs

from subprocess import check_output

from extra.sass.assembler.CuSMVersion import CuSMVersion, p_QNAN
from extra.sass.assembler.CuNVInfo import CuNVInfo
from extra.sass.assembler.CuAsmLogger import CuAsmLogger
from extra.sass.assembler.CuControlCode import CuControlCode
from extra.sass.assembler.common import splitAsmSection, bytes2Asm, stringBytes2Asm, getTempFileName
from extra.sass.assembler.config import Config
from extra.sass.assembler.utils.CubinUtils import hackCubinDesc

from io import StringIO, BytesIO
from collections import OrderedDict

import re
import struct
import os

# tags for determining the range of segments containing program headers
PROGRAM_HEADER_TAG = '@PROGRAM_HEADER'

class CubinFile():
    ''' CubinFile class for cubin files, mainly used for saving as cuasm.

    '''

    def __init__(self, cubinname):
        self.__mCubinName = cubinname
        self.loadCubin(cubinname)

    def __reset(self):
        self.__mELFFileHeader = None
        self.__mELFSections = OrderedDict()
        self.__mELFSegments = []
        self.__mELFSegmentRange = []

        self.__mAsmLines = None
        self.__mAsmSectionMarkers = {}
        self.__mCubinBytes = None       #

        self.m_Arch = None
        self.m_VirtualSMVersion = None
        self.m_ToolKitVersion = None

    @CuAsmLogger.logTimeIt
    def loadCubin(self, binname):

        CuAsmLogger.logEntry('Loading cubin file %s...' % binname)

        self.__reset()

        sec_start_dict = {}
        sec_end_dict = {}

        with open(binname, 'rb') as fin:
            self.__mCubinBytes = fin.read()

        with BytesIO(self.__mCubinBytes) as bio:
            ef = ELFFile(bio)
            self.__mELFFileHeader = ef.header

            if ef.header['e_type'] != 'ET_EXEC':
                msg = 'Currently only ET_EXEC type of elf is supported! %s given...' % ef.header['e_type']
                CuAsmLogger.logWarning(msg)
                #raise Exception(msg)
            elif ef.header['e_shoff'] == ef.header['e_ehsize']:
                msg = 'Abnormal elf layout detected! Section headers directly follow elf header.'
                CuAsmLogger.logWarning(msg)
                raise Exception(msg)
            elif ef.header['e_phoff'] == 0 or ef.header['e_phnum'] == 0:
                msg = 'Abnormal elf layout detected! No program header found!'
                CuAsmLogger.logWarning(msg)
                # raise Exception(msg)

            # Example: type=2, abi=7, sm=86, toolkit=111, flags = 0x500556
            # flags>>16 = 0x50 = 80, means virtual arch compute_80
            # flags&0xff = 0x56 = 86, means sm_86
            vsm_version = (self.__mELFFileHeader['e_flags']>>16)&0xff
            sm_version = self.__mELFFileHeader['e_flags']&0xff
            self.m_Arch = CuSMVersion(sm_version)
            self.m_VirtualSMVersion = vsm_version
            self.m_ToolKitVersion = self.__mELFFileHeader['e_version']

            sh_index = self.__mELFFileHeader['e_ehsize']
            sh_edgelist = []
            for isec, sec in enumerate(ef.iter_sections()):
                self.__mELFSections[sec.name]= sec.header, sec.data()

                # build section start/end offset dict
                # only for determine the coverage of segments
                sh_align = sec.header['sh_addralign']
                sh_size = sec.header['sh_size']
                if sh_align==0:
                    sh_edgelist.append((0, 0, sec.name))
                    continue
                elif (sh_align>0) and (sh_index % sh_align != 0):
                    sh_index = ((sh_index + sh_align -1 ) // sh_align) * sh_align
                
                sh_start = sh_index
                sh_end = sh_index + sh_size
                sh_index += sh_size
                sh_edgelist.append((sh_start, sh_end, sec.name))
                # print('%8d(0x%4x)  %8d(0x%4x)  %s'%(sh_start, sh_start, sh_end, sh_end, sec.name))

            for sh_start, sh_end, sname in sh_edgelist:
                sec_start_dict[sh_start] = sname
                sec_end_dict[sh_end] = sname

            # Add segment header to start/end dict
            if self.__mELFFileHeader['e_phnum'] > 0:
                poff = self.__mELFFileHeader['e_phoff']
                sec_start_dict[poff] = PROGRAM_HEADER_TAG
                pend = poff + self.__mELFFileHeader['e_phnum'] * self.__mELFFileHeader['e_phentsize']
                sec_end_dict[pend] = PROGRAM_HEADER_TAG

            for seg in ef.iter_segments():
                self.__mELFSegments.append(seg.header)
                
        for iseg, segh in enumerate(self.__mELFSegments):
            if segh['p_type'] == 'PT_LOAD': # only P_LOAD type needs range
                p0 = segh['p_offset']
                p1 = p0 + segh['p_memsz'] # filesz will not count NOBITS sections

                if p0 not in sec_start_dict:
                    CuAsmLogger.logWarning(f'The segment start ({p0:#x}, {p1:#x}) doesnot align with sections!')
                    CuAsmLogger.logWarning('Try to seek the nearest one...')

                    max_d = max([k if k<p0 else -10 for k in sec_start_dict])  # find last section start before current bound
                    if max_d == -10:
                        msg = f'Cannot locate start position for segment {iseg} with range ({p0:#x}, {p1:#x})!'
                        CuAsmLogger.logCritical(msg)
                        raise Exception(msg)

                    sec_start = sec_start_dict[max_d]
                else:
                    sec_start = sec_start_dict[p0]

                if p1 not in sec_end_dict:
                    CuAsmLogger.logWarning(f'The segment end ({p0:#x}, {p1:#x}) doesnot align with sections!')
                    CuAsmLogger.logWarning('Try to seek the nearest one...')
                    
                    min_d = min([k if k>p1 else 2**32 for k in sec_end_dict]) # find first section end after current bound
                    if min_d == 2**32:
                        msg = f'Cannot locate end position for segment {iseg} with range ({p0:#x}, {p1:#x})!'
                        CuAsmLogger.logCritical(msg)
                        raise Exception(msg)

                    sec_end = sec_end_dict[min_d]
                else:
                    sec_end = sec_end_dict[p1]

                self.__mELFSegmentRange.append((sec_start, sec_end))
            else:
                self.__mELFSegmentRange.append((None,None))

        # get disassembly from nvdisasm
        # TODO: check availablity of nvdisasm
        if self.m_Arch.needsDescHack():
            tmpname = getTempFileName(suffix='cubin')
            CuAsmLogger.logWarning(f'This Cubin({self.m_Arch}) needs desc hack!')
            hackCubinDesc(binname, tmpname)
            asmtext = CubinFile.disassembleCubin(tmpname)
            os.remove(tmpname)
        else:
            asmtext = CubinFile.disassembleCubin(binname)

        self.__mAsmLines = asmtext.splitlines()

        # split asm text into sections, according to .section directive
        # the file header line range is in key "$FileHeader"
        self.__mAsmSectionMarkers = splitAsmSection(self.__mAsmLines)

    def __writeFileHeaderAsm(self, stream, ident='\t'):
        ''' generate file header asm.

        In cuasm parser, most of the fields will be loaded from a default header.
        But some of them can be set or modified by user in assembly.

        typedef struct
        {
            unsigned char e_ident[16]; /* ELF identification */
            Elf64_Half e_type;         /* Object file type */
            Elf64_Half e_machine;      /* Machine type */
            Elf64_Word e_version;      /* Object file version */
            Elf64_Addr e_entry;        /* Entry point address */
            Elf64_Off e_phoff;         /* Program header offset */
            Elf64_Off e_shoff;         /* Section header offset */
            Elf64_Word e_flags;        /* Processor-specific flags */
            Elf64_Half e_ehsize;       /* ELF header size */
            Elf64_Half e_phentsize;    /* Size of program header entry */
            Elf64_Half e_phnum;        /* Number of program header entries */
            Elf64_Half e_shentsize;    /* Size of section header entry */
            Elf64_Half e_shnum;        /* Number of section header entries */
            Elf64_Half e_shstrndx;     /* Section name string table index */
        } Elf64_Ehdr;

        A file header sample:
        Container({'e_ident': Container({'EI_MAG': [127, 69, 76, 70],
            'EI_CLASS': 'ELFCLASS64', 'EI_DATA': 'ELFDATA2LSB',
            'EI_VERSION': 'EV_CURRENT', 'EI_OSABI': 51, 'EI_ABIVERSION': 7}),
            'e_type': 'ET_EXEC', 'e_machine': 'EM_CUDA',
            'e_version': 111, 'e_entry': 0, 'e_phoff': 12224,
            'e_shoff': 10176, 'e_flags': 4916555, 'e_ehsize': 64,
            'e_phentsize': 56, 'e_phnum': 3, 'e_shentsize': 64,
            'e_shnum': 32, 'e_shstrndx': 1})
        '''

        CuAsmLogger.logSubroutine('Writing CuAsm file header...')

        fheader = self.__mELFFileHeader
        m0, m1 = self.__mAsmSectionMarkers['$FileHeader']
        # stream.writelines('\n'.join(self.__mAsmLines[m0:m1])) # usually only header flags and elftype
        stream.write(ident + '// All file header info is kept as is (unless offset/size attributes)\n')
        stream.write(ident + '// The original header flags is not complete, thus discarded. \n')
        for line in self.__mAsmLines[m0:m1]:
            stream.write(ident + '// ' + line + '\n')

        stream.write(ident + '.__elf_ident_osabi      %d\n'%fheader['e_ident']['EI_OSABI'])
        stream.write(ident + '.__elf_ident_abiversion %d\n'%fheader['e_ident']['EI_ABIVERSION'])
        stream.write(ident + '.__elf_type             %s\n'%fheader['e_type'])
        stream.write(ident + '.__elf_machine          %s\n'%fheader['e_machine'])
        stream.write(ident + '.__elf_version          %d \t\t// CUDA toolkit version \n'%fheader['e_version'])
        stream.write(ident + '.__elf_entry            %d \t\t// entry point address \n'%fheader['e_entry'])
        stream.write(ident + '.__elf_phoff            0x%x \t\t// program header offset, maybe updated by assembler\n'%fheader['e_phoff'])
        stream.write(ident + '.__elf_shoff            0x%x \t\t// section header offset, maybe updated by assembler\n'%fheader['e_shoff'])

        vsmversion = (fheader['e_flags']>>16)&0xff
        smversion = fheader['e_flags']&0xff
        stream.write(ident + '.__elf_flags            0x%x \t\t// Flags, SM_%d(0x%x), COMPUTE_%d(0x%x) \n'%(fheader['e_flags'], smversion, smversion, vsmversion, vsmversion))
        stream.write(ident + '.__elf_ehsize           %d \t\t// elf header size \n'%fheader['e_ehsize'])
        stream.write(ident + '.__elf_phentsize        %d \t\t// program entry size\n'%fheader['e_phentsize'])
        stream.write(ident + '.__elf_phnum            %d \t\t// number of program entries\n'%fheader['e_phnum'])
        stream.write(ident + '.__elf_shentsize        %d \t\t// section entry size\n'%fheader['e_shentsize'])
        stream.write(ident + '.__elf_shnum            %d \t\t// number of sections, currently no sections can be appended/removed\n'%fheader['e_shnum'])
        stream.write(ident + '.__elf_shstrndx         %d \t\t// Section name string table index \n'%fheader['e_shstrndx'])
        stream.write('\n')

    def __writeSectionHeaderAsm(self, stream, secname, header, ident='\t'):
        ''' Generate section header assembly according to header.

            (Only ELF64 is supported here)
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

            A sample input:
            {'sh_name': 11, 'sh_type': 'SHT_STRTAB', 'sh_flags': 0, 'sh_addr': 0,
            'sh_offset': 808, 'sh_size': 1061, 'sh_link': 0, 'sh_info': 0,
            'sh_addralign': 1, 'sh_entsize': 0})

            Fields required: name, type, flags, link, info, addralign, entsize
            Fields filled by assembler: addr(?), offset, size
        '''

        # stream.write('\t.section  "%s", %s, %s\n'%(secname, header['sh_flags'], header['sh_type']))
        stream.write(ident + '.__section_name         0x%x \t// offset in .shstrtab\n' % header['sh_name'])
        stream.write(ident + '.__section_type         %s\n'%header['sh_type'])
        stream.write(ident + '.__section_flags        0x%x\n'%header['sh_flags'])
        stream.write(ident + '.__section_addr         0x%x\n'%header['sh_addr'])
        stream.write(ident + '.__section_offset       0x%x \t// maybe updated by assembler\n'%header['sh_offset'])
        stream.write(ident + '.__section_size         0x%x \t// maybe updated by assembler\n'%header['sh_size'])
        stream.write(ident + '.__section_link         %d\n'%header['sh_link'])
        stream.write(ident + '.__section_info         0x%x\n'%header['sh_info'])
        stream.write(ident + '.__section_entsize      %d\n'%header['sh_entsize'])
        stream.write(ident + '.align                %d \t// equivalent to set sh_addralign\n'%header['sh_addralign'])

    def __writeCodeSectionAsm(self, stream, secname):
        ''' Rewrite the code sections in assembly

        Tasks:
            1. add control codes
            2. Add some offset labels
                such as EIATTR_COOP_GROUP_INSTR_OFFSETS, no way to recover from assembly.
            3. (TODO) Special treatment of some instructions.
                such as FSEL with NAN operand
                and some instructions missing assembly text (Maybe a bug of nvdisasm?)
        '''
        CuAsmLogger.logSubroutine('Writing code section %s...'%secname)

        # get assembly lines according to current text section
        mstart, mend = self.__mAsmSectionMarkers[secname]
        asmlines = self.__mAsmLines[mstart:mend]

        # extract nvinfo, get offset label dict
        # some offset nvinfo cannot recover from assembly
        # thus we need this label to keep them unaffected
        kname = re.sub(r'^\.text\.', '', secname)
        nvinfo_secname = '.nv.info.' + kname
        if nvinfo_secname not in self.__mELFSections:
            raise KeyError('Info section (%s) not found!'%nvinfo_secname)

        nvinfo_data = self.__mELFSections[nvinfo_secname][1]
        nvinfo = CuNVInfo(nvinfo_data, self.m_Arch)
        offset_labels = nvinfo.getOffsetLabelDict(kname)

        # get code bytes of current text section
        codeheader = self.__mELFSections[secname][0]
        codebytes = self.__mELFSections[secname][1]

        ctrl_code_list, ins_code_list = self.m_Arch.splitCtrlCodeFromBytes(codebytes)

        # Example : "/*0300*/                   IMAD.U32 R5, RZ, RZ, UR6 ;"
        m_ins = re.compile(r'^\s*\/\*([0-9a-f]+)\*\/\s+.*')

        pidx = -1
        stream.write(asmlines[0] + '\n') # asmline[0] contains .section declaration
        self.__writeSectionHeaderAsm(stream, secname, codeheader)

        for line in asmlines[1:]: # first line with .section already written
            res = m_ins.match(line)
            if res is not None:
                addr = int(res.groups()[0], 16) # instruction address(offset)

                # add offset label
                if addr in offset_labels:
                    stream.write('  ' + offset_labels[addr] + ':\n')

                # generate control code strings
                # NOTE: the addr comments do not matter, it will be ignored by assembler.
                # TODO (DONE!): check missing instructions, idx should be with stride 1
                idx = self.m_Arch.getInsIndexFromOffset(addr)
                if idx-pidx != 1:
                    CuAsmLogger.logWarning("!!! Missing instruction before %s:0x%x"%(secname, addr))

                for iIns in range(pidx+1, idx):
                    ccode = ctrl_code_list[iIns]
                    cstr  = CuControlCode.decode(ccode)

                    icode = ins_code_list[iIns]
                    istr  = '    UNDEF 0x%x; // Missing instructions, not disassembled' % icode

                    stream.write('      [%s] %s\n'%(cstr, istr) )

                pidx = idx

                ccode = ctrl_code_list[idx]
                cstr  = CuControlCode.decode(ccode)

                # rewrite QNAN as float binary 0fxxxx
                if p_QNAN.search(line):
                    hline = self.m_Arch.hackDisassembly(ins_code_list[idx], line)
                    CuAsmLogger.logWarning(f'QNAN rewritten in {secname} : {line}') # addr already in line
                    stream.write('      [%s] %s // QNAN rewritten: %s\n'%(cstr, hline, line) )
                else:
                    stream.write('      [%s] %s\n'%(cstr, line) )
            else:
                # 2 spaces for code folding
                stream.write('  ' + line+'\n')

    def __writeExplicitSectionAsm(self, stream, secname):
        ''' Write sections explicitly defined in nvdisasm output.

        Most of those assembly texts are copied, just add some section header info.
        '''
        CuAsmLogger.logSubroutine('Writing explicit section %s...'%secname)

        m0, m1 = self.__mAsmSectionMarkers[secname]
        stream.write(self.__mAsmLines[m0]+'\n')  # declaration first

        header, _ = self.__mELFSections[secname]
        self.__writeSectionHeaderAsm(stream, secname, header) # followed by section info

        # 2 spaces is for identation, thus all section contents can be collapsed
        stream.write('  ' + '\n  '.join(self.__mAsmLines[m0+1:m1])) # followed by section data

    def __writeImplicitSectionAsm(self, stream, secname):
        ''' Write implicit sections not shown in nvdisasm output.
            Such as .shstrtab, .strtab, .symtab, .rel*, etc.

            Ideally they can be generated from assembly inputs, but some entries in
            .shstrtab/.strtab/.symtab seem not referenced in nvdisasm output.
            And it's difficult to check the correctness and dig the hidden correlations.
            Thus we just keep it as is, since in most cases we do not need to modify them.

            All contents of relocation sections (.rel.*, .rela.*) will be generated by assembler.
            Size of symbols may be updated if necessary.
        '''

        CuAsmLogger.logSubroutine('Writing implicit section %s...'%secname)

        header, data = self.__mELFSections[secname]

        bio = BytesIO(self.__mCubinBytes)
        ef = ELFFile(bio)

        if secname == '.shstrtab' or secname == '.strtab':
            stream.write('\t.section  "%s", %s, %s\n'%(secname, header['sh_flags'], header['sh_type']))
            stream.write('\t// all strings in %s section will be kept as is.\n'%secname)
            self.__writeSectionHeaderAsm(stream, secname, header)
            stream.write(stringBytes2Asm(data, label=secname))
        elif secname == '.symtab':
            stream.write('\t.section  "%s", %s, %s\n'%(secname, header['sh_flags'], header['sh_type']))
            stream.write('\t// all symbols in .symtab sections will be kept\n')
            stream.write('\t// but the symbol size may be changed accordingly\n')
            self.__writeSectionHeaderAsm(stream, secname, header)

            sym_entsize = header['sh_entsize']
            nsym = header['sh_size'] // sym_entsize
            sym_section = ef.get_section_by_name('.symtab')

            isym = 0
            for sym in sym_section.iter_symbols():
                stream.write('    // Symbol[%d] "%s": %s\n'%(isym, sym.name, sym.entry))
                stream.write(bytes2Asm(data[isym*sym_entsize:(isym+1)*sym_entsize], addr_offset=isym*sym_entsize))
                stream.write('\n')
                isym += 1

        elif secname == '':
            stream.write('\t// there will always be an empty section at index 0\n')
            stream.write('\t.section  "%s", %s, %s\n'%(secname, header['sh_flags'], header['sh_type']))
            self.__writeSectionHeaderAsm(stream, secname, header)

        elif secname.startswith('.rel'):
            stream.write('\t.section  "%s", %s, %s\n'%(secname, header['sh_flags'], header['sh_type']))
            stream.write('\t// all relocation sections will be dynamically generated by assembler \n')
            stream.write('\t// but most of the section header will be kept as is.\n')
            self.__writeSectionHeaderAsm(stream, secname, header)

            rel_section = ef.get_section_by_name(secname)
            sym_section = ef.get_section_by_name('.symtab')
            rel_entsize = header['sh_entsize']
            nrel = rel_section.num_relocations()

            irel = 0
            for rel in rel_section.iter_relocations():
                symname = sym_section.get_symbol(rel.entry['r_info_sym']).name
                stream.write('    // Relocation[%d] : %s, %s\n'%(irel, symname, rel.entry))
                irel += 1

        else:
            raise Exception('Unknown implicit section %s !'%secname)

    def __writeSegmentHeaderAsm(self, stream, segheader, segrange):
        '''
        typedef struct
        {
            Elf64_Word  p_type;    /* Type of segment */
            Elf64_Word  p_flags;   /* Segment attributes */
            Elf64_Off   p_offset;  /* Offset in file */
            Elf64_Addr  p_vaddr;   /* Virtual address in memory */
            Elf64_Addr  p_paddr;   /* Reserved */
            Elf64_Xword p_filesz;  /* Size of segment in file */
            Elf64_Xword p_memsz;   /* Size of segment in memory */
            Elf64_Xword p_align;   /* Alignment of segment */
        } Elf64_Phdr;
        '''
        CuAsmLogger.logSubroutine('Writing segment header...')

        stream.write('// Program segment %s, %s \n' % (segheader['p_type'], segheader['p_flags']))
        stream.write('  .__segment  "%s", %s \n' % (segheader['p_type'], segheader['p_flags']))
        stream.write('  .__segment_offset  0x%x   \t\t// maybe updated by assembler \n' % (segheader['p_offset']))
        stream.write('  .__segment_vaddr   0x%x   \t\t// Seems always 0? \n' % (segheader['p_vaddr']))
        stream.write('  .__segment_paddr   0x%x   \t\t// ??? \n' % (segheader['p_paddr']))
        stream.write('  .__segment_filesz  0x%x   \t\t// file size, maybe updated by assembler \n' % (segheader['p_filesz']))
        stream.write('  .__segment_memsz   0x%x   \t\t// file size + nobits sections, maybe updated by assembler \n' % (segheader['p_memsz']))
        stream.write('  .__segment_align     %d   \t\t//  \n' % (segheader['p_align']))
        if segrange[0] is not None:
            stream.write('  .__segment_startsection    "%s"  \t\t// first section in this segment \n' % (segrange[0]))
            stream.write('  .__segment_endsection      "%s"  \t\t// last  section in this segment \n' % (segrange[1]))
        stream.write('\n')

    @CuAsmLogger.logTimeIt
    def saveAsCuAsm(self, asmname):
        ''' Saving current cubin as cuasm file.

        section entry tables : kept as is, offset/size may change
        segment entry tables : kept as is, offset/size may change

        .fileheader   : kept as is, offset for section/program header may change
        .shstrtab     : kept as is
        .strtab       : kept as is
        .symtab       : kept as is, size may change

        .debug_frame  : currently kept as is

        .nv.info      : usually kept as is, contents user modifiable
        .nv.info.*    : updated by assembler, contents user modifiable

        .nv.constant* : usually kept as is, contents user modifiable

        .text.*       : code sections, user modifiable

        Some optional sections:
        .rel.*        : relocation, dynamically generated by assembler
        .rela.*       : relocation with add, dynamically generated by assembler
        .nv.global.init: contents user modifiable
        .nv.global : Nobits, kept as is.

        '''
        CuAsmLogger.logEntry('Saving to cuasm file %s...' % asmname)

        with open(asmname, 'w+') as fout:
            # output file header
            fout.write('// --------------------- FileHeader --------------------------\n')
            self.__writeFileHeaderAsm(fout)

            fout.write('\n')
            fout.write('  //-------------------------------------------------\n')
            fout.write('  //------------ END of FileHeader ------------------\n')
            fout.write('  //-------------------------------------------------\n\n\n')

            # output sections
            for secname in self.__mELFSections:
                fout.write('\n// --------------------- %-32s --------------------------\n'%secname)
                # for sections in assembly, write the assembly
                if secname in self.__mAsmSectionMarkers:
                    if secname.startswith('.text.'):
                        self.__writeCodeSectionAsm(fout, secname)
                    else:
                        self.__writeExplicitSectionAsm(fout, secname)
                else: # for implicit sections, write raw strings/bytes
                    self.__writeImplicitSectionAsm(fout, secname)

            fout.write('\n')
            fout.write('  //-------------------------------------------------\n')
            fout.write('  //---------------- END of sections ----------------\n')
            fout.write('  //-------------------------------------------------\n\n\n')

            for segheader,segrange in zip(self.__mELFSegments, self.__mELFSegmentRange):
                self.__writeSegmentHeaderAsm(fout, segheader, segrange)

            fout.write('\n')
            fout.write('  //-------------------------------------------------\n')
            fout.write('  //---------------- END of segments ----------------\n')
            fout.write('  //-------------------------------------------------\n\n\n')

    @staticmethod
    @CuAsmLogger.logTimeIt
    def disassembleCubin(binname):
        ''' Get disassembly of input file from nvdisasm.

            TODO: check availablity of nvdisasm?
            TODO2: Cache disassembly output?
        '''
        CuAsmLogger.logProcedure('Disassembling %s...'%binname)

        asmtext = check_output([Config.NVDISASM_PATH, binname]).decode()

        return asmtext



if __name__ == '__main__':
    pass
