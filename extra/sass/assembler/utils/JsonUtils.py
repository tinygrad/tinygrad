# -*- coding: utf-8 -*-
import re

HexInt = re.compile('^[+-]?0x[0-9a-fA-F]+$')

def IntVal2Hex(d):
    ''' Translate all int values to hex strings. Used when dumping obj to json strings.'''
    if isinstance(d, dict):
        d2 = {}
        for k, v in d.items():
            d2[k] = IntVal2Hex(v)
        
        return d2
    elif isinstance(d, int) and not isinstance(d, bool): # True/False is also int
        return f'{d:#x}'
    elif isinstance(d, list):
        d2 = [IntVal2Hex(dv) for dv in d]
        return d2
    else:
        return d

def HexVal2Int(d):
    ''' Translate all hex strings to int values. Used when loading obj from json strings.'''
    if isinstance(d, dict):
        d2 = {}
        for k, v in d.items():
            d2[k] = HexVal2Int(v)
        return d2
    elif isinstance(d, str) and HexInt.match(d):
        return int(d, 16)
    elif isinstance(d, list):
        d2 = [HexVal2Int(dv) for dv in d]
        return d2
    else:
        return d
