# -*- coding: utf-8 -*-

from elftools.elf.structs import ELFStructs
import os

# TODO: log system

def getDefaultStruct(st):
    return st.parse(b'\x00'*st.sizeof())

class Config(object):

    # Default path to nvdisasm
    NVDISASM_PATH = 'nvdisasm'

    # Currently only little_endian and ELF64 is supported
    # NOTE: There are quite a lot of hardcodes for endianness and elfclass
    #       thus just modifying the value here will not work
    CubinELFStructs = ELFStructs(little_endian=True, elfclass=64)
    CubinELFStructs.create_basic_structs()
    CubinELFStructs.create_advanced_structs()

    defaultCubinFileHeader = CubinELFStructs.Elf_Ehdr.parse(bytes.fromhex(''.join([
                            '7f454c460201013307000000000000000200be00650000000000000000000000',
                            'c09000000000000000890000000000004b054b0040003800030040001f000100'])))

    # 'e_phentsize': 56, 'e_shentsize': 64
    defaultSectionHeader = getDefaultStruct(CubinELFStructs.Elf_Shdr)
    defaultSegmentHeader = getDefaultStruct(CubinELFStructs.Elf_Phdr)
    
    # 24 B
    defaultSymbol = getDefaultStruct(CubinELFStructs.Elf_Sym)

    # rel/rela
    defaultRel    = getDefaultStruct(CubinELFStructs.Elf_Rel)
    defaultRela   = getDefaultStruct(CubinELFStructs.Elf_Rela)

    # TODO: load from / save to file?
    def load(self):
        pass

    def save(self):
        pass

    @staticmethod
    def getDefaultInsAsmReposFile(version_number):
        module_dir = os.path.split(__file__)
        repos_dir = os.path.join(module_dir[0], 'InsAsmRepos')
        repos_name = 'DefaultInsAsmRepos.sm_%d.txt' % version_number
        repos_path = os.path.join(repos_dir, repos_name)
        return repos_path

    @staticmethod
    def getDefaultIOInfoFile(version_number):
        module_dir = os.path.split(__file__)
        fdir = os.path.join(module_dir[0], 'InsAsmRepos')
        
        fname = 'IOInfo.sm_%d.txt' % version_number
        fpath = os.path.join(fdir, fname)
        
        if not os.path.isfile(fpath):
            fpath = os.path.join(fdir, 'IOInfo.all.json')
        
        return fpath
        
