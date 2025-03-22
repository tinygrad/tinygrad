# -*- coding: utf-8 -*-

import sympy
from sympy import Matrix # Needed by repr
from sympy.core.numbers import Rational
from io import StringIO
from extra.sass.assembler.CuSMVersion import CuSMVersion
from extra.sass.assembler.common import reprList, reprHexMat
from extra.sass.assembler.CuAsmLogger import CuAsmLogger

class CuInsAssembler():
    '''CuInsAssembler is the assembler handles the values and weights of one type of instruction.'''

    def __init__(self, inskey, d=None, arch='sm_75'):
        ''' Initializer.

            inskey is mandatory, d is for initialization from saved repr.
        '''

        self.m_InsKey = inskey
        if d is not None:
            self.initFromDict(d)
        else:
            self.m_InsRepos = []
            self.m_InsModiSet = {}

            self.m_ValMatrix = None
            self.m_PSol = None
            self.m_PSolFac = None
            self.m_ValNullMat = []
            self.m_Rhs = None
            self.m_InsRecords = []
            self.m_ErrRecords = {}

            self.m_Arch = CuSMVersion(arch)

    def iterRecords(self):
        ''' Iterate over all records, including normal records and error records.'''
        
        # m_InsRecords is a list of ins_info => (addr, code, s)
        for r in self.m_InsRecords:
            yield r
        
        # m_ErrRecords is a dict of {code_diff : ins_info => (addr, code, s) }
        for _, r in self.m_ErrRecords.items():
            yield r
    
    def recordsFeeder(self):
        for r in self.iterRecords():
            yield r[0], r[1], r[2], 0
            
    def initFromDict(self, d):
        self.m_InsKey     = d['InsKey']
        self.m_InsRepos   = d['InsRepos']
        self.m_InsModiSet = d['InsModiSet']

        self.m_ValMatrix  = d['ValMatrix']
        self.m_PSol       = d['PSol']
        self.m_PSolFac    = d['PSolFac']
        self.m_ValNullMat = d['ValNullMat']
        self.m_Rhs        = d['Rhs']

        self.m_InsRecords = d['InsRecords']
        self.m_ErrRecords = d['ErrRecords'] if 'ErrRecords' in d else {} 

        self.m_Arch       = d['Arch']

    def initFromJsonDict(self, d):
        pass

    def expandModiSet(self, modi):
        ''' Push in new modifiers.
        
            NOTE: the order matters, since every modi has its own value.
        '''

        updated = False
        for m in modi:
            if m not in self.m_InsModiSet:
                self.m_InsModiSet[m] = len(self.m_InsModiSet)
                updated = True

        return updated

    def buildInsValVec(self, vals, modi, outRawList=False):
        ''' Convert instruction value vector from vals and modifiers.

            NOTE: Due to performance reason of Matrix.nullspace(), vals are placed after modis.
                  Usually vals are dense, but modifiers are sparse. 
                  This arrangement will make the valMatrix more like upper trangular, 
                  and this will usually make carrying out the nullspace much much faster.

            TODO: currently opcode is also a modifer, maybe placed after modis? 
        '''

        insval = [1 if m in modi else 0 for m in self.m_InsModiSet]    # first comes modi
        insval.extend(vals)                                            # then follows vals
        if outRawList:
            return insval
        else:
            insvec = sympy.Matrix(insval)
            return insvec

    def canAssemble(self, vals, modi):
        """ Check whether the input code can be assembled with current info.

        Args:
            vals ([int list]): value list in integers
            modi ([str list]): modifier list in strings

        Returns:
            (None , None) : input can be assembled
            (brief, info) : brief in ['NewModi', 'NewVals'], info gives the detailed info
        """

        if not all([m in self.m_InsModiSet for m in modi]):
            brief = 'NewModi'
            info = 'Unknown modifiers: (%s)' % (set(modi) - set(self.m_InsModiSet.keys()))
            return brief, info
        else:
            insvec = self.buildInsValVec(vals, modi)

            if self.m_ValNullMat is not None:
                insrhs = self.m_ValNullMat * insvec
                if not all([v==0 for v in insrhs]):
                    return 'NewVals', 'Insufficient basis, try CuAsming more instructions!'

        return None, None

    def push(self, vals, modi, code, ins_info):
        ''' Push in a new instruction.

            When its code can be assembled, verify the result,
            otherwise add new information to current assembler.
            @return (flag, info):
                flag = True (Expected result)
                    "NewModi" / "NewVals" for new information
                    "Verified" for no new information, but the results is consistent
                flag = False (Unexpected result)
                    "NewConflict" for new conflict information
                    "KnownConflict" for known inconsistent assembling result
        '''

        if not all([m in self.m_InsModiSet for m in modi]):
            # If new instruction contains unknown modifier,
            # it's never possible to be assembled by current assembler.
            CuAsmLogger.logProcedure('Pushing new modi (%s, %-20s): %s' % (self.m_Arch.formatCode(code), self.m_InsKey, ins_info))
            updated = self.expandModiSet(modi)
            self.m_InsRepos.append((vals, modi, code))
            self.buildMatrix()
            self.m_InsRecords.append(ins_info)
            return True, 'NewModi'
        else:
            # If the vals of new instruction lies in the null space of
            # current ValMatrix, it does not contain new information.
            insvec = self.buildInsValVec(vals, modi)

            if self.m_ValNullMat is None:
                doVerify = True
            else:
                insrhs = self.m_ValNullMat * insvec
                doVerify = all([v==0 for v in insrhs])

            if doVerify:
                # return 'Verified'
                inscode = self.m_PSol.dot(insvec) / self.m_PSolFac

                if inscode != code:
                    if inscode.is_integer:
                        code_diff = inscode - code
                        if code_diff not in self.m_ErrRecords:
                            self.m_ErrRecords[code_diff] = ins_info
                            CuAsmLogger.logError("Error when verifying for %s" % self.m_InsKey)
                            CuAsmLogger.logError("    Asm : %s" % ins_info[-1])
                            CuAsmLogger.logError("    InputCode : %s" % self.m_Arch.formatCode(code))
                            CuAsmLogger.logError("    AsmCode   : %s" % self.m_Arch.formatCode(inscode))
                            return False, 'NewConflict'
                        else:
                            CuAsmLogger.logDebug("Known code conflict for %s!" % self.m_InsKey)
                            return False, 'KnownConflict'
                    else:
                        CuAsmLogger.logCritical("FATAL! Non-integral code assembled for %s" % self.m_InsKey)
                        CuAsmLogger.logCritical("    Asm : %s" % ins_info[-1])
                        CuAsmLogger.logCritical("    InputCode : %s" % self.m_Arch.formatCode(code))
                        CuAsmLogger.logCritical("    AsmCode   : (%s)!" % str(inscode))
                        
                        # It's very unlikely the diff is just the code it self. (usually opcode will match) 
                        code_diff = code
                        self.m_ErrRecords[code_diff] = ins_info

                        return False, 'NewConflict'

                    # print(self.__repr__())
                    # raise Exception("Inconsistent instruction code!")
                    # return False
                else:
                    # print("Verified: 0x%032x" % code)
                    return True, 'Verified'

            else:
                CuAsmLogger.logProcedure('Pushing new vals (%s, %-20s): %s' % (self.m_Arch.formatCode(code), self.m_InsKey, ins_info))
                self.m_InsRepos.append((vals, modi, code))
                self.m_InsRecords.append(ins_info)
                self.buildMatrix()
                return True, 'NewVals'

        # Never be here
        # return True

    def buildCode(self, vals, modi):
        '''Assemble with the input vals and modi.

        NOTE: This function didn't check the sufficiency of matrix.'''

        inscode = 0
        for v0, vs in zip(self.m_PSol[-len(vals):], vals):
            inscode += v0 * vs
            
        for m in modi:
            inscode += self.m_PSol[self.m_InsModiSet[m]]
        
        if self.m_PSolFac == 1:
            return int(inscode)
        else:
            return int(inscode//self.m_PSolFac)

    def buildMatrix(self, solve_method='LU'):
        if len(self.m_InsRepos) == 0:
            return None, None

        M = []
        b = []
        for vals, modis, code in self.m_InsRepos:
            l = self.buildInsValVec(vals, modis, outRawList=True)
            M.append(l)
            b.append(code)

        self.m_ValMatrix = sympy.Matrix(M)
        self.m_Rhs = sympy.Matrix(b)
        self.m_ValNullMat = self.getNullMatrix(self.m_ValMatrix)

        if self.m_ValNullMat is not None:
            M2 = self.m_ValMatrix.copy()
            b2 = self.m_Rhs.copy()
            for nn in range(self.m_ValNullMat.rows):
                M2 = M2.row_insert(0, self.m_ValNullMat.row(nn))
                b2 = b2.row_insert(0, sympy.Matrix([0]))
            self.m_PSol = M2.solve(b2, method=solve_method)
        else:
            self.m_PSol = self.m_ValMatrix.solve(self.m_Rhs, method=solve_method)

        self.m_PSol, self.m_PSolFac = self.getMatrixDenomLCM(self.m_PSol)
        return self.m_ValMatrix, self.m_Rhs

    def solve(self):
        ''' Try to solve every variable.

            This is possible only when ValNullMat is none.
        '''

        if self.m_ValNullMat is None:
            x = self.m_ValMatrix.solve(self.m_Rhs)
            print('Solution: ')
            for i,v in enumerate(x):
                print('%d : 0x%+033x' % (i,v))
            return x
        else:
            print('Not solvable!')
            return None
    
    def printSolution(self):
        print("InsKey = %s" % self.m_InsKey)
        nvals = len(self.m_InsRepos[0][0])
        nmodi = len(self.m_InsModiSet)

        names = ['V%d'%v for v in range(nvals)]
        names.extend([0] * nmodi)
        for m, midx in self.m_InsModiSet.items():
            names[midx+nvals] = m

        # the order of solutions are altered for better display.
        # vals are displayed before modis
        
        rev_sol = []
        rev_sol.extend(self.m_PSol[nmodi:])
        rev_sol.extend(self.m_PSol[:nmodi])
         
        for name, val in zip(names, rev_sol):
            if val % self.m_PSolFac == 0:
                print("  %24s :  %#32x" % (name, val // self.m_PSolFac))
            else:
                print("  %24s :  %#32x / %#x " % (name, val, self.m_PSolFac))

    def reprPSol(self):
        nvals = len(self.m_InsRepos[0][0])
        nmodi = len(self.m_InsModiSet)

        names = [0 for _ in range(nmodi)]
        for m, midx in self.m_InsModiSet.items():
            names[midx] = m
        
        names.append('Pred')
        names.extend(['V%d'%v for v in range(1, nvals)])
        
        slist = []
        vlist = []
        maxvlen = 0
        for ival in range(nvals+nmodi):
            sval = '%#x' % self.m_PSol[ival, 0]
            slist.append(sval)
            vlist.append(self.m_PSol[ival, 0])
            maxvlen = max(maxvlen, len(sval))
        
        maxnlen = 0
        for name in names:
            maxnlen = max(maxnlen, len(name))
        
        fac = int(self.m_PSolFac)

        sio = StringIO()
        sio.write('Matrix([\n')
        if self.m_PSolFac == 1:
            for vname, s in zip(names, slist):
                ss = ' '*(maxvlen-len(s)) + s
                sio.write(f'[ {ss}], # {vname}\n')
        else:
            for vname, s, v in zip(names, slist, vlist):
                ss = ' '*(maxvlen-len(s)) + s
                ns = ' '*(maxnlen-len(vname)) + vname
                vv = int(v)
                if vv % fac == 0:
                    vt = vv // fac
                    sio.write(f'[ {ss}], # {ns} : {vt:#32x}\n')
                else:
                    sio.write(f'[ {ss}], # {ns} : {vv:#32x} / {fac:#x}\n')
        sio.write('])')
        
        return sio.getvalue()
                
    def getNullMatrix(self, M):
        ''' Get the null space of current matrix M.

            And get the lcm for all fractional denominators.
            The null matrix is only for checking sufficiency of ValMatrix,
            thus it won't be affected by any non-zero common factor.
            Fractional seems much slower than integers.
        '''

        ns = M.nullspace(simplify=True)
        if len(ns)==0:
            return None
        else:
            nm = ns[0]
            for n in ns[1:]:
                nm = nm.row_join(n)

            # NullSpace won't be affected by a common factor.
            nmDenom, dm = self.getMatrixDenomLCM(nm.T)
            return nmDenom

    def getMatrixDenomLCM(self, M):
        ''' Get lcm of matrix denominator.

            In sympy, operation of fractionals seems much slower than integers.
            Thus we multiply a fraction matrix with the LCM of all denominators,
            then divide the result with the LCM.
        '''

        dm = 1
        for e in M:
            if isinstance(e, Rational):
                nom, denom = e.as_numer_denom()
                dm = sympy.lcm(denom, dm)
        return (M*dm, dm)

    def __repr__(self):
        ''' A string repr of current ins assembler.

            This will be used to dump it to text file and read back by setFromDict.
        '''
        sio = StringIO()

        sio.write('CuInsAssembler("", {"InsKey" : %s, \n' % repr(self.m_InsKey) )
        # sio.write('  "InsRepos" : %s, \n' % repr(self.m_InsRepos))
        sio.write('  "InsRepos" : ')
        reprList(sio, self.m_InsRepos)
        sio.write(', \n')

        sio.write('  "InsModiSet" : %s, \n' % repr(self.m_InsModiSet))
        sio.write('  "ValMatrix" : %s, \n' % repr(self.m_ValMatrix))
        sio.write('  "PSol" : %s, \n' % self.reprPSol())
        sio.write('  "PSolFac" : %s, \n' % repr(self.m_PSolFac))
        sio.write('  "ValNullMat" : %s, \n' % repr(self.m_ValNullMat))
        #sio.write('  "InsRecords" : %s, \n' % repr(self.m_InsRecords))

        sio.write('  "InsRecords" : [')
        #reprList(sio, self.m_InsRecords)
        for addr, code, s in self.m_InsRecords:
            sio.write('(%#08x, %s, "%s"),\n'%(addr, self.m_Arch.formatCode(code), s))
        sio.write('], \n')

        sio.write('  "ErrRecords" : {')
        for code_diff, (addr, code, s) in self.m_ErrRecords.items():
            sio.write('%#x : (%#08x, %s, "%s"),\n'%(code_diff, addr, self.m_Arch.formatCode(code), s))
        sio.write('}, ')

        sio.write('  "Rhs" : %s, \n' % reprHexMat(self.m_Rhs))
        sio.write('  "Arch" : %s })' % repr(self.m_Arch))

        return sio.getvalue()

    def __str__(self):

        return 'CuInsAssembler(%s)' % self.m_InsKey
