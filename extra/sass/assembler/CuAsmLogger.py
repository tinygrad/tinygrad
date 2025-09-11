# -*- coding: utf-8 -*-

import logging
import logging.handlers
import sys
import time
import os
import tempfile
import random

class CuAsmLogger(object):
    ''' A logger private to current module.

        A customized logging style is used to show the progress better, 
        without affecting the logging of other modules.

    '''
    __LoggerRepos = {}
    __CurrLogger = None
    __LogFileRepos = {}
    __IndentLevel = 0
    __IndentString = ''

    # Predefined levels:
    # CRITICAL 50 
    # ERROR    40 
    # WARNING  30 
    # INFO     20 
    # DEBUG    10 
    # NOTSET    0 

    # Custom log levels

    ENTRY      = 35   # main entry of a module
    PROCEDURE  = 25   # procedures of some module
    SUBROUTINE = 15   # some internal subroutines

    @staticmethod
    def getDefaultLoggerFile(name):
        ''' Default log file in temp dir.
        
            NOTE: this is not safe, since several instances may run simultaneously.
        '''
        fpath = tempfile.gettempdir()
        return os.path.join(fpath, name + '.log')

    @staticmethod
    def getTemporaryLoggerFile(name):
        ''' Temporary logfile in temp dir.'''
        fpath = tempfile.gettempdir()
        while True:
            ttag = time.strftime('.%m%d-%H%M%S.', time.localtime())
            tmpname = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k = 8))
            fname = os.path.join(fpath, name + ttag + tmpname + '.log')
            if not os.path.exists(fname):
                break

        return fname

    @staticmethod
    def initLogger(log_file='', *, name='cuasm', file_level=logging.DEBUG, file_max_bytes=1<<30, file_backup_count=3, stdout_level=25):
        ''' Init a logger with given name and logfile.

            log_file: set to None for no file log;
                      set to '' for default temporary log file; (DEFAULT)
                      set to filename for user specified log file;

                      CuAsmLogger uses RotatingFileHandler for logging, thus if given log_file exists or file size exceeds the max_bytes,
                      it will roll over and rename previous files to logfile.log.1, logfile.log.2, etc...
                
                NOTE: Temporary logfiles will not be deleted automatically, since we usually need to check the log after running a program.

            name    : logger instance name, default to 'cuasm'
                      several loggers may exist simultaneously, use setActiveLogger(name) to switch between them.
            file_level : log level of file
            file_max_bytes: max size of logfile(in bytes), default to 1GB.
            file_backup_count: number of maximum rolling over files, default to 3.
            stdout_level: log level for standard output.
        '''
        # if name in CuAsmLogger.__LoggerRepos:
        #    CuAsmLogger.__CurrLogger = CuAsmLogger.__LoggerRepos[name]
        #    print('CuAsmLogger %s already exists! Skipping init...' % name)
        #    return
        
        logger = logging.getLogger(name)
        hs = [h for h in logger.handlers]
        for h in hs:
            logger.removeHandler(h)

        logger.setLevel(logging.DEBUG)

        fmt = logging.Formatter('%(asctime)s - %(message)s')
        if log_file is not None:
            if len(log_file) == 0:
                full_log_file = CuAsmLogger.getTemporaryLoggerFile(name)
            else:
                # fpath, fbase = os.path.split(log_file)

                # if fbase.lower().endswith('.log'):
                #     full_log_file = os.path.join(fpath, name + '.' + fbase)
                # else:
                #     full_log_file = os.path.join(fpath, name + '.' + fbase + '.log')
                if log_file.endswith('.log'):
                    full_log_file = log_file
                else:
                    full_log_file = log_file + '.log'
            
            # fh = logging.FileHandler(full_log_file, mode='a')
            print(f'InitLogger({name}) with logfile "{full_log_file}"...')
            
            # once RotatingFileHandler is created, the log file will be created at the same time
            # thus we need to detect whether the logfile needs to be rolled over before handler creation
            needsRollOver = os.path.exists(full_log_file)
            fh = logging.handlers.RotatingFileHandler(full_log_file, mode='a', maxBytes=file_max_bytes, backupCount=file_backup_count)

            # default mode is 'a', but we may want a new log for every run, but still keeping old logs as backup.
            if needsRollOver:
                print(f'Logfile {full_log_file} already exists! Rolling over...')
                fh.doRollover()

            fh.setFormatter(fmt)
            fh.setLevel(file_level)
            
            logger.addHandler(fh)
            CuAsmLogger.__LogFileRepos[name] = full_log_file
        else:
            CuAsmLogger.__LogFileRepos[name] = None

        if stdout_level is not None:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(fmt)
            sh.setLevel(stdout_level)
            logger.addHandler(sh)

        # 
        CuAsmLogger.__LoggerRepos[name] = logger
        CuAsmLogger.__CurrLogger = logger
    
    @staticmethod
    def setActiveLogger(name):
        if name in CuAsmLogger.__LoggerRepos:
            CuAsmLogger.__CurrLogger = CuAsmLogger.__LoggerRepos[name]
        else:
            print('CuAsmLogger %s does not exist! Keeping current logger...' % name)

    @staticmethod
    def getCurrentLogFile():
        return CuAsmLogger.__LogFileRepos[CuAsmLogger.__CurrLogger.name]

    @staticmethod
    def logDebug(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.debug('   DEBUG - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logInfo(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.info('    INFO - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logWarning(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.warning(' WARNING - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logError(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.error('   ERROR - ' + msg, *args, **kwargs)
        
    @staticmethod
    def logCritical(msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.critical('CRITICAL - ' + msg, *args, **kwargs)

    @staticmethod
    def logEntry(msg, *args, **kwargs):
        full_msg = '   ENTRY - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.ENTRY, full_msg, *args, **kwargs)
        

    @staticmethod
    def logProcedure(msg, *args, **kwargs):

        full_msg = '    PROC - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.PROCEDURE, full_msg, *args, **kwargs)
        
    
    @staticmethod
    def logSubroutine(msg, *args, **kwargs):
        full_msg = '     SUB - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.SUBROUTINE, full_msg, *args, **kwargs)
        

    @staticmethod
    def logLiteral(msg, *args, **kwargs):
        full_msg = '         - ' + CuAsmLogger.__IndentString + msg
        CuAsmLogger.__CurrLogger.log(CuAsmLogger.PROCEDURE, full_msg, *args, **kwargs)
        

    @staticmethod
    def log(level, msg, *args, **kwargs):
        CuAsmLogger.__CurrLogger.log(level, msg, *args, **kwargs)
        

    @staticmethod
    def logTimeIt(func):
        ''' Logging of a (usually) long running function.

        '''
        def wrapper(*args, **kwargs):
            CuAsmLogger.logLiteral('Running %s...'%func.__qualname__)
            CuAsmLogger.incIndent()
            
            t0 = time.time()
            ret = func(*args, **kwargs)
            t1 = time.time()

            CuAsmLogger.decIndent()
            CuAsmLogger.logLiteral('Func %s completed! Time=%8.4f secs.'%(func.__qualname__, t1-t0))
            
            return ret

        return wrapper
    
    @staticmethod
    def logIndentIt(func):
        '''
        '''
        def wrapper(*args, **kwargs):
            CuAsmLogger.incIndent()
            ret = func(*args, **kwargs)
            CuAsmLogger.decIndent()
            
            return ret

        return wrapper

    @staticmethod
    def logTraceIt(func):
        '''
        '''
        def wrapper(*args, **kwargs):
            CuAsmLogger.logLiteral('Running %s...'%func.__qualname__)
            CuAsmLogger.incIndent()
            
            ret = func(*args, **kwargs)
            CuAsmLogger.decIndent()

            return ret

        return wrapper

    @staticmethod
    def incIndent():
        CuAsmLogger.__IndentLevel += 1
        CuAsmLogger.__IndentString = '    ' * CuAsmLogger.__IndentLevel

    @staticmethod
    def decIndent():
        CuAsmLogger.__IndentLevel -= 1
        if CuAsmLogger.__IndentLevel < 0:
            CuAsmLogger.__IndentLevel = 0
        CuAsmLogger.__IndentString = '    ' * CuAsmLogger.__IndentLevel
    
    @staticmethod
    def resetIndent(val=0):
        if val<0:
            val = 0
        CuAsmLogger.__IndentLevel = val
        CuAsmLogger.__IndentString = '    ' * CuAsmLogger.__IndentLevel

    @staticmethod
    def setLevel(level):
        CuAsmLogger.__CurrLogger.setLevel(level)
    
    @staticmethod
    def disable():
        CuAsmLogger.__CurrLogger.setLevel(logging.ERROR)

# Init a default logger when the module is imported
CuAsmLogger.initLogger(log_file=None)
