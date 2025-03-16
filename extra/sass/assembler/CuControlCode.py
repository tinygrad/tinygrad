# -*- coding: utf-8 -*-
import re

# Pattern for control codes string
# ChangeLog 20220915: remove reuse field
c_ControlCodesPattern = re.compile(r'B(0|-)(1|-)(2|-)(3|-)(4|-)(5|-):R[0-5\-]:W[0-5\-]:(Y|-):S\d{2}')
c_ControlStringLen = 19

class CuControlCode:
    def __init__(self, code):
        c_waitbar, c_readbar, c_writebar, c_yield, c_stall = CuControlCode.splitCode(code)

        self.Barrier = c_waitbar
        self.Read  = c_readbar
        self.Write = c_writebar
        self.Yield = c_yield
        self.Stall = c_stall
    
    def isYield(self):
        ''' If yield flag is set(code=0).'''
        return self.Yield == 0
    
    def getStallCount(self):
        ''' Get stall count.'''
        return self.Stall
    
    def getReadSB(self):
        ''' Get read scoreboard id, return None if not set.'''
        return None if self.Read == 7 else self.Read
    
    def getWriteSB(self):
        ''' Get write scoreboard id, return None if not set.'''
        return None if self.Write == 7 else self.Write
    
    def getBarrierSet(self):
        ''' Get a set of waiting scoreboards, return empty set if waiting on none.'''
        return {i for i in range(6) if (self.Barrier & (1<<i)>0)}

    @staticmethod
    def splitCode(code):
        ''' Split control codes into parts.

        # c.f. : https://github.com/NervanaSystems/maxas/wiki/Control-Codes
        #      : https://arxiv.org/abs/1903.07486
        # reuse  waitbar  rbar  wbar  yield   stall
        #  0000   000000   000   000      0    0000
        #
        # NOTE : It is known that for some instructions(HMMA.SP), reuse may use some other bits.
        #        And for some other instructions(TEXS/BRA/...), the reuse bits may be used for other encoding fields.
        #        Since reuses are displayed as explicit modifiers, we will not split the reuse field any more.
        #        Other fields will be extracted and encoded as control codes.
        # TODO : Maybe we can treat those control fields as normal modifier?        
        '''

        c_stall    = (code & 0x0000f) >> 0
        c_yield    = (code & 0x00010) >> 4
        c_writebar = (code & 0x000e0) >> 5  # write dependency barrier 
        c_readbar  = (code & 0x00700) >> 8  # read  dependency barrier 
        c_waitbar  = (code & 0x1f800) >> 11 # wait on dependency barrier

        return c_waitbar, c_readbar, c_writebar, c_yield, c_stall    
    
    @staticmethod
    def splitCode2(code):
        ''' Split control codes into parts.

            Mostly same as splitCode, but with yield/stall combined.     
        '''

        c_ystall   = (code & 0x0001f) >> 0
        c_writebar = (code & 0x000e0) >> 5  # write dependency barrier
        c_readbar  = (code & 0x00700) >> 8  # read  dependency barrier
        c_waitbar  = (code & 0x1f800) >> 11 # wait on dependency barrier

        return c_waitbar, c_readbar, c_writebar, c_ystall

    @staticmethod
    def mergeCode(c_waitbar, c_readbar, c_writebar, c_yield, c_stall):
        code = c_waitbar<<11
        code += c_readbar<<8
        code += c_writebar<<5
        code += c_yield<<4
        code += c_stall
        return code

    @staticmethod
    def decode(code):
        c_waitbar, c_readbar, c_writebar, c_yield, c_stall = CuControlCode.splitCode(code)

        s_yield = '-' if c_yield !=0 else 'Y'
        s_writebar = '-' if c_writebar == 7 else '%d'%c_writebar
        s_readbar = '-' if c_readbar == 7 else '%d'%c_readbar
        s_waitbar = ''.join(['-' if (c_waitbar & (2**i)) == 0 else '%d'%i for i in range(6)])
        s_stall = '%02d' % c_stall

        return 'B%s:R%s:W%s:%s:S%s' % (s_waitbar, s_readbar, s_writebar, s_yield, s_stall)

    @staticmethod
    def encode(s):
        if not c_ControlCodesPattern.match(s):
            raise ValueError('Invalid control code strings: %s !!!'%s)

        s_waitbar, s_readbar, s_writebar, s_yield, s_stall = tuple(s.split(':'))

        waitbar_tr = str.maketrans('012345-','1111110')

        c_waitbar = int(s_waitbar[:0:-1].translate(waitbar_tr), 2)
        c_readbar = int(s_readbar[1].replace('-', '7'))
        c_writebar = int(s_writebar[1].replace('-','7'))
        c_yield = int(s_yield!='Y')
        c_stall = int(s_stall[1:])

        code = CuControlCode.mergeCode(c_waitbar, c_readbar, c_writebar, c_yield, c_stall)

        return code

if __name__ == '__main__':
    cs = ['B--2---:R0:W1:-:S07',
        'B01--4-:R-:W-:-:S05',
        'B------:R-:W0:-:S01',
        'B------:R-:W-:-:S01',
        'B------:R2:W1:-:S01',
        'B0-----:R-:W-:Y:S04',
        'B------:R-:W-:-:S01',
        'B0----5:R0:W5:Y:S05',
        'B------:R-:W0:-:S02',
        'B0-----:R-:W0:-:S02',
        'B0-----:R-:W-:Y:S04']
    
    passed = True
    for s in cs:
        c = CuControlCode.encode(s)
        s2 = CuControlCode.decode(c)

        print('0x%06x:'%c)
        print('    %s'%s)
        print('    %s'%s2)
        if s != s2:
            print('!!! Unmatched !')
            passed = False
    
    if passed:
        print("Test passed!!!")
    else:
        print("Test failed!!!")
        