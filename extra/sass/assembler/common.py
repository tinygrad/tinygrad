# -*- coding: utf-8 -*-

from io import StringIO
import re
import time
import os
import tempfile
import random

# Patterns for assembly comments
p_cppcomment = re.compile(r'//.*$')      # cpp style line comments
p_ccomment = re.compile(r'\/\*.*?\*\/')  # c   style line
p_bracomment = re.compile(r'\(\*.*\*\)') # notes for bra targets
                                         # such as (*"INDIRECT_CALL"*)
                                         # or (*"BRANCH_TARGETS .L_10"*)
                                         # or (*\"BRANCH_TARGETS .L_x_3722,.L_x_3723,.L_x_782\"*)

def alignTo(pos, align):
    ''' Padding current position to given alignment.
    
        Return: tuple(newpos, padsize)
    '''

    if align==0 or align==1:
        return pos, 0

    npos = ((pos + align -1 ) // align) * align
    return npos, npos-pos

def intList2Str(vlist, l=None):
    if l:
        fmt = f'%#{l+2}x'
    else:
        fmt = '%#x'
    return '['+ (', '.join([fmt%v for v in vlist])) +']'

def binstr(v, bitlen=128, width=4, sp=' '):
    bv = bin(v)[2:]
    lb = len(bv)
    if lb<bitlen:
        bv = '0' * (bitlen-lb) + bv

    return sp.join([bv[i:i+width] for i in range(0, bitlen, width)])

def hexstr(v, bitlen=128, width=4, sp=' '):
    hv = '%x'%v
    lhex = bitlen//4
    lb = len(hv)

    if lb<lhex:
        hv = '0' * (lhex-lb) + hv

    return sp.join([hv[i:i+width] for i in range(0, lhex, width)])

def splitAsmSection(lines):
    ''' Split assembly text line list into a set of sections.
        NOTE: the split is done according to ".section" directive

        Return: section_markers, a dict of section markers (a tuple of start line and end line).
                The header markers is stored in entry "$FileHeader".
        An example: section_markers = {'$FileHeader':(0,4), '.shstrtab':(4,82), '.strtab':(82,140),...}
    '''
    m_secdirective = re.compile(r'^\s*\.section\s+([\.\w]+),')

    secnames = []
    markers = [0]

    for iline, line in enumerate(lines):
        res = m_secdirective.match(line)
        if  res is not None:
            secname = res.groups()[0]
            # print("Line%4d (%s): %s"%(iline, secname, line.strip()))
            secnames.append(secname)

            # usually the previous line of .section will be a comment line
            # when splitting sections, we may want to skip this line
            has_prev_comment = False
            if iline>0:
                prev_line = lines[iline-1].strip()
                if prev_line.startswith('//') and secname in prev_line:
                    has_prev_comment = True
            
            if has_prev_comment:
                markers.append(iline-1)
            else:
                markers.append(iline)  
            markers.append(iline)

    # 
    markers.append(len(lines))

    section_markers = {}
    section_markers['$FileHeader'] = (markers[0], markers[1])  # File header parts

    for isec, secname in enumerate(secnames):
        section_markers[secname] = (markers[2*isec+2], markers[2*isec+3])
        
    return section_markers

def stringBytes2Asm(ss, label='', width=8):
    ''' Convert b'\x00' seperated string bytes into assembly bytes.
        label is the name of this string list, only for displaying the entry lists in comments.
        width is number of bytes to display per line.
    '''
    p = 0
    counter = 0
    
    sio = StringIO()
    while True:
        pnext = ss.find(b'\x00', p)
        if pnext<0:
            break
        
        s = ss[p:pnext+1]
        sio.write('    // %s[%d] = %s \n'%(label, counter, repr(s)))
        
        p0 = p
        while p0<pnext+1:
            sio.write('    /*%04x*/ .byte '%p0 + ', '.join(['0x%02x'%b for b in s[p0-p:p0-p+width]]))
            sio.write('\n')
            p0 += width
        
        sio.write('\n')
        p = pnext+1

        counter += 1

    return sio.getvalue()

def bytes2Asm(bs, width=8, addr_offset=0, ident='    '):
    ''' Convert bytes into assembly bytes.
        width is the max display length of one line.
    '''
    sio = StringIO()

    p = 0
    while p<len(bs):
        blist = ', '.join(['0x%02x'%b for b in bs[p:p+width]])
        sio.write('%s/*%04x*/ .byte '%(ident, p+addr_offset) + blist)
        sio.write('\n')
        p += width
        
    return sio.getvalue()

def bytesdump(inname, outname):
    with open(inname, 'rb') as fin:
        bs = fin.read()

    bstr = bytes2Asm(bs)

    with open(outname, 'w') as fout:
        fout.write(bstr)

def reprDict(sio, d):
    sio.write('{')
    n = len(d)
    cnt = 0
    for k, v in d.items():
        sio.write(repr(k) + ':' + repr(v))
        if cnt<n-1:
            sio.write(',\n')
        cnt += 1
    sio.write('}')

def reprList(sio, l):
    sio.write('[')
    n = len(l)
    cnt = 0
    for v in l:
        sio.write(repr(v))
        if cnt< n-1:
            sio.write(',\n')
        cnt += 1
    sio.write(']')

def reprHexMat(mat):
    smat = mat.tolist()

    colw = [0 for i in range(mat.cols)]
    
    for j in range(mat.cols):
        for i in range(mat.rows):
            if mat[i, j].is_integer:
                s = '%#x' % mat[i, j]
            else:
                s = str(mat[i, j])
                
            smat[i][j] = s
            colw[j] = max(colw[j], len(s))
    
    # print(smat)
    
    for i in range(mat.rows):
        for j in range(mat.cols):
            if len(smat[i][j]) < colw[j]:
                d = colw[j] - len(smat[i][j])
                smat[i][j] = (' '*d) + smat[i][j]
    
    # print(smat)
    
    sio = StringIO()
    sio.write('Matrix([\n')
    for i in range(mat.rows):
        sio.write('[')
        sio.write(', '.join(smat[i]))
        sio.write('],\n')
    
    sio.write('])')
    
    return sio.getvalue()

def getTempFileName(name='', *, prefix='cuasm', suffix=''):
    ''' Get temporary filename in temp dir.'''
    fpath = tempfile.gettempdir()
    if len(prefix)>0 and not prefix.endswith('.'):
        prefix += '.'
    if len(suffix)>0 and not suffix.startswith('.'):
        suffix = '.' + suffix
    
    if len(name)>0:
        return os.path.join(fpath, prefix+name+suffix)

    while True:
        ttag = time.strftime('%m%d-%H%M%S', time.localtime())
        tmpname = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k = 8))
        fname = os.path.join(fpath, prefix + ttag + tmpname + suffix)
        if not os.path.exists(fname):
            break

    return fname

def stripComments(s):
    ''' Strip comments of a line.

    NOTE: cross line comments are not supported yet.
    '''

    s = p_cppcomment.subn(' ', s)[0] # replace comments as a single space, avoid unwanted concatination
    s = p_ccomment.subn(' ', s)[0]
    s = p_bracomment.subn(' ', s)[0]
    s = re.subn(r'\s+', ' ', s)[0]       # replace one or more spaces/tabs into one single space

    return s.strip()

if __name__ == '__main__':
    pass