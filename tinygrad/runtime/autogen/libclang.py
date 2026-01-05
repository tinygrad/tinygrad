# mypy: ignore-errors
import ctypes
from tinygrad.runtime.support.c import Array, DLL, Pointer, Struct, Union, field, CEnum, _IO, _IOW, _IOR, _IOWR
dll = DLL('libclang', ['clang-20', 'clang'])
CXIndex = ctypes.c_void_p
class struct_CXTargetInfoImpl(Struct): pass
CXTargetInfo = Pointer(struct_CXTargetInfoImpl)
class struct_CXTranslationUnitImpl(Struct): pass
CXTranslationUnit = Pointer(struct_CXTranslationUnitImpl)
CXClientData = ctypes.c_void_p
class struct_CXUnsavedFile(Struct): pass
struct_CXUnsavedFile.SIZE = 24
struct_CXUnsavedFile._fields_ = ['Filename', 'Contents', 'Length']
setattr(struct_CXUnsavedFile, 'Filename', field(0, Pointer(ctypes.c_char)))
setattr(struct_CXUnsavedFile, 'Contents', field(8, Pointer(ctypes.c_char)))
setattr(struct_CXUnsavedFile, 'Length', field(16, ctypes.c_uint64))
enum_CXAvailabilityKind = CEnum(ctypes.c_uint32)
CXAvailability_Available = enum_CXAvailabilityKind.define('CXAvailability_Available', 0)
CXAvailability_Deprecated = enum_CXAvailabilityKind.define('CXAvailability_Deprecated', 1)
CXAvailability_NotAvailable = enum_CXAvailabilityKind.define('CXAvailability_NotAvailable', 2)
CXAvailability_NotAccessible = enum_CXAvailabilityKind.define('CXAvailability_NotAccessible', 3)

class struct_CXVersion(Struct): pass
struct_CXVersion.SIZE = 12
struct_CXVersion._fields_ = ['Major', 'Minor', 'Subminor']
setattr(struct_CXVersion, 'Major', field(0, ctypes.c_int32))
setattr(struct_CXVersion, 'Minor', field(4, ctypes.c_int32))
setattr(struct_CXVersion, 'Subminor', field(8, ctypes.c_int32))
CXVersion = struct_CXVersion
enum_CXCursor_ExceptionSpecificationKind = CEnum(ctypes.c_uint32)
CXCursor_ExceptionSpecificationKind_None = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_None', 0)
CXCursor_ExceptionSpecificationKind_DynamicNone = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_DynamicNone', 1)
CXCursor_ExceptionSpecificationKind_Dynamic = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Dynamic', 2)
CXCursor_ExceptionSpecificationKind_MSAny = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_MSAny', 3)
CXCursor_ExceptionSpecificationKind_BasicNoexcept = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_BasicNoexcept', 4)
CXCursor_ExceptionSpecificationKind_ComputedNoexcept = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_ComputedNoexcept', 5)
CXCursor_ExceptionSpecificationKind_Unevaluated = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Unevaluated', 6)
CXCursor_ExceptionSpecificationKind_Uninstantiated = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Uninstantiated', 7)
CXCursor_ExceptionSpecificationKind_Unparsed = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Unparsed', 8)
CXCursor_ExceptionSpecificationKind_NoThrow = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_NoThrow', 9)

@dll.bind((ctypes.c_int32, ctypes.c_int32), CXIndex)
def clang_createIndex(excludeDeclarationsFromPCH, displayDiagnostics): ...
@dll.bind((CXIndex,), None)
def clang_disposeIndex(index): ...
CXChoice = CEnum(ctypes.c_uint32)
CXChoice_Default = CXChoice.define('CXChoice_Default', 0)
CXChoice_Enabled = CXChoice.define('CXChoice_Enabled', 1)
CXChoice_Disabled = CXChoice.define('CXChoice_Disabled', 2)

CXGlobalOptFlags = CEnum(ctypes.c_uint32)
CXGlobalOpt_None = CXGlobalOptFlags.define('CXGlobalOpt_None', 0)
CXGlobalOpt_ThreadBackgroundPriorityForIndexing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForIndexing', 1)
CXGlobalOpt_ThreadBackgroundPriorityForEditing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForEditing', 2)
CXGlobalOpt_ThreadBackgroundPriorityForAll = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForAll', 3)

class struct_CXIndexOptions(Struct): pass
struct_CXIndexOptions.SIZE = 24
struct_CXIndexOptions._fields_ = ['Size', 'ThreadBackgroundPriorityForIndexing', 'ThreadBackgroundPriorityForEditing', 'ExcludeDeclarationsFromPCH', 'DisplayDiagnostics', 'StorePreamblesInMemory', 'PreambleStoragePath', 'InvocationEmissionPath']
setattr(struct_CXIndexOptions, 'Size', field(0, ctypes.c_uint32))
setattr(struct_CXIndexOptions, 'ThreadBackgroundPriorityForIndexing', field(4, ctypes.c_ubyte))
setattr(struct_CXIndexOptions, 'ThreadBackgroundPriorityForEditing', field(5, ctypes.c_ubyte))
setattr(struct_CXIndexOptions, 'ExcludeDeclarationsFromPCH', field(6, ctypes.c_uint32, 1, 0))
setattr(struct_CXIndexOptions, 'DisplayDiagnostics', field(6, ctypes.c_uint32, 1, 1))
setattr(struct_CXIndexOptions, 'StorePreamblesInMemory', field(6, ctypes.c_uint32, 1, 2))
setattr(struct_CXIndexOptions, 'PreambleStoragePath', field(8, Pointer(ctypes.c_char)))
setattr(struct_CXIndexOptions, 'InvocationEmissionPath', field(16, Pointer(ctypes.c_char)))
CXIndexOptions = struct_CXIndexOptions
@dll.bind((Pointer(CXIndexOptions),), CXIndex)
def clang_createIndexWithOptions(options): ...
@dll.bind((CXIndex, ctypes.c_uint32), None)
def clang_CXIndex_setGlobalOptions(_0, options): ...
@dll.bind((CXIndex,), ctypes.c_uint32)
def clang_CXIndex_getGlobalOptions(_0): ...
@dll.bind((CXIndex, Pointer(ctypes.c_char)), None)
def clang_CXIndex_setInvocationEmissionPathOption(_0, Path): ...
CXFile = ctypes.c_void_p
@dll.bind((CXTranslationUnit, CXFile), ctypes.c_uint32)
def clang_isFileMultipleIncludeGuarded(tu, file): ...
@dll.bind((CXTranslationUnit, Pointer(ctypes.c_char)), CXFile)
def clang_getFile(tu, file_name): ...
size_t = ctypes.c_uint64
@dll.bind((CXTranslationUnit, CXFile, Pointer(size_t)), Pointer(ctypes.c_char))
def clang_getFileContents(tu, file, size): ...
class CXSourceLocation(Struct): pass
CXSourceLocation.SIZE = 24
CXSourceLocation._fields_ = ['ptr_data', 'int_data']
setattr(CXSourceLocation, 'ptr_data', field(0, Array(ctypes.c_void_p, 2)))
setattr(CXSourceLocation, 'int_data', field(16, ctypes.c_uint32))
@dll.bind((CXTranslationUnit, CXFile, ctypes.c_uint32, ctypes.c_uint32), CXSourceLocation)
def clang_getLocation(tu, file, line, column): ...
@dll.bind((CXTranslationUnit, CXFile, ctypes.c_uint32), CXSourceLocation)
def clang_getLocationForOffset(tu, file, offset): ...
class CXSourceRangeList(Struct): pass
class CXSourceRange(Struct): pass
CXSourceRange.SIZE = 24
CXSourceRange._fields_ = ['ptr_data', 'begin_int_data', 'end_int_data']
setattr(CXSourceRange, 'ptr_data', field(0, Array(ctypes.c_void_p, 2)))
setattr(CXSourceRange, 'begin_int_data', field(16, ctypes.c_uint32))
setattr(CXSourceRange, 'end_int_data', field(20, ctypes.c_uint32))
CXSourceRangeList.SIZE = 16
CXSourceRangeList._fields_ = ['count', 'ranges']
setattr(CXSourceRangeList, 'count', field(0, ctypes.c_uint32))
setattr(CXSourceRangeList, 'ranges', field(8, Pointer(CXSourceRange)))
@dll.bind((CXTranslationUnit, CXFile), Pointer(CXSourceRangeList))
def clang_getSkippedRanges(tu, file): ...
@dll.bind((CXTranslationUnit,), Pointer(CXSourceRangeList))
def clang_getAllSkippedRanges(tu): ...
@dll.bind((CXTranslationUnit,), ctypes.c_uint32)
def clang_getNumDiagnostics(Unit): ...
CXDiagnostic = ctypes.c_void_p
@dll.bind((CXTranslationUnit, ctypes.c_uint32), CXDiagnostic)
def clang_getDiagnostic(Unit, Index): ...
CXDiagnosticSet = ctypes.c_void_p
@dll.bind((CXTranslationUnit,), CXDiagnosticSet)
def clang_getDiagnosticSetFromTU(Unit): ...
class CXString(Struct): pass
CXString.SIZE = 16
CXString._fields_ = ['data', 'private_flags']
setattr(CXString, 'data', field(0, ctypes.c_void_p))
setattr(CXString, 'private_flags', field(8, ctypes.c_uint32))
@dll.bind((CXTranslationUnit,), CXString)
def clang_getTranslationUnitSpelling(CTUnit): ...
@dll.bind((CXIndex, Pointer(ctypes.c_char), ctypes.c_int32, Pointer(Pointer(ctypes.c_char)), ctypes.c_uint32, Pointer(struct_CXUnsavedFile)), CXTranslationUnit)
def clang_createTranslationUnitFromSourceFile(CIdx, source_filename, num_clang_command_line_args, clang_command_line_args, num_unsaved_files, unsaved_files): ...
@dll.bind((CXIndex, Pointer(ctypes.c_char)), CXTranslationUnit)
def clang_createTranslationUnit(CIdx, ast_filename): ...
enum_CXErrorCode = CEnum(ctypes.c_uint32)
CXError_Success = enum_CXErrorCode.define('CXError_Success', 0)
CXError_Failure = enum_CXErrorCode.define('CXError_Failure', 1)
CXError_Crashed = enum_CXErrorCode.define('CXError_Crashed', 2)
CXError_InvalidArguments = enum_CXErrorCode.define('CXError_InvalidArguments', 3)
CXError_ASTReadError = enum_CXErrorCode.define('CXError_ASTReadError', 4)

@dll.bind((CXIndex, Pointer(ctypes.c_char), Pointer(CXTranslationUnit)), enum_CXErrorCode)
def clang_createTranslationUnit2(CIdx, ast_filename, out_TU): ...
enum_CXTranslationUnit_Flags = CEnum(ctypes.c_uint32)
CXTranslationUnit_None = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_None', 0)
CXTranslationUnit_DetailedPreprocessingRecord = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_DetailedPreprocessingRecord', 1)
CXTranslationUnit_Incomplete = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_Incomplete', 2)
CXTranslationUnit_PrecompiledPreamble = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_PrecompiledPreamble', 4)
CXTranslationUnit_CacheCompletionResults = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CacheCompletionResults', 8)
CXTranslationUnit_ForSerialization = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_ForSerialization', 16)
CXTranslationUnit_CXXChainedPCH = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CXXChainedPCH', 32)
CXTranslationUnit_SkipFunctionBodies = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_SkipFunctionBodies', 64)
CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IncludeBriefCommentsInCodeCompletion', 128)
CXTranslationUnit_CreatePreambleOnFirstParse = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CreatePreambleOnFirstParse', 256)
CXTranslationUnit_KeepGoing = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_KeepGoing', 512)
CXTranslationUnit_SingleFileParse = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_SingleFileParse', 1024)
CXTranslationUnit_LimitSkipFunctionBodiesToPreamble = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_LimitSkipFunctionBodiesToPreamble', 2048)
CXTranslationUnit_IncludeAttributedTypes = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IncludeAttributedTypes', 4096)
CXTranslationUnit_VisitImplicitAttributes = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_VisitImplicitAttributes', 8192)
CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles', 16384)
CXTranslationUnit_RetainExcludedConditionalBlocks = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_RetainExcludedConditionalBlocks', 32768)

@dll.bind((), ctypes.c_uint32)
def clang_defaultEditingTranslationUnitOptions(): ...
@dll.bind((CXIndex, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char)), ctypes.c_int32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32), CXTranslationUnit)
def clang_parseTranslationUnit(CIdx, source_filename, command_line_args, num_command_line_args, unsaved_files, num_unsaved_files, options): ...
@dll.bind((CXIndex, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char)), ctypes.c_int32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32, Pointer(CXTranslationUnit)), enum_CXErrorCode)
def clang_parseTranslationUnit2(CIdx, source_filename, command_line_args, num_command_line_args, unsaved_files, num_unsaved_files, options, out_TU): ...
@dll.bind((CXIndex, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char)), ctypes.c_int32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32, Pointer(CXTranslationUnit)), enum_CXErrorCode)
def clang_parseTranslationUnit2FullArgv(CIdx, source_filename, command_line_args, num_command_line_args, unsaved_files, num_unsaved_files, options, out_TU): ...
enum_CXSaveTranslationUnit_Flags = CEnum(ctypes.c_uint32)
CXSaveTranslationUnit_None = enum_CXSaveTranslationUnit_Flags.define('CXSaveTranslationUnit_None', 0)

@dll.bind((CXTranslationUnit,), ctypes.c_uint32)
def clang_defaultSaveOptions(TU): ...
enum_CXSaveError = CEnum(ctypes.c_uint32)
CXSaveError_None = enum_CXSaveError.define('CXSaveError_None', 0)
CXSaveError_Unknown = enum_CXSaveError.define('CXSaveError_Unknown', 1)
CXSaveError_TranslationErrors = enum_CXSaveError.define('CXSaveError_TranslationErrors', 2)
CXSaveError_InvalidTU = enum_CXSaveError.define('CXSaveError_InvalidTU', 3)

@dll.bind((CXTranslationUnit, Pointer(ctypes.c_char), ctypes.c_uint32), ctypes.c_int32)
def clang_saveTranslationUnit(TU, FileName, options): ...
@dll.bind((CXTranslationUnit,), ctypes.c_uint32)
def clang_suspendTranslationUnit(_0): ...
@dll.bind((CXTranslationUnit,), None)
def clang_disposeTranslationUnit(_0): ...
enum_CXReparse_Flags = CEnum(ctypes.c_uint32)
CXReparse_None = enum_CXReparse_Flags.define('CXReparse_None', 0)

@dll.bind((CXTranslationUnit,), ctypes.c_uint32)
def clang_defaultReparseOptions(TU): ...
@dll.bind((CXTranslationUnit, ctypes.c_uint32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32), ctypes.c_int32)
def clang_reparseTranslationUnit(TU, num_unsaved_files, unsaved_files, options): ...
enum_CXTUResourceUsageKind = CEnum(ctypes.c_uint32)
CXTUResourceUsage_AST = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_AST', 1)
CXTUResourceUsage_Identifiers = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Identifiers', 2)
CXTUResourceUsage_Selectors = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Selectors', 3)
CXTUResourceUsage_GlobalCompletionResults = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_GlobalCompletionResults', 4)
CXTUResourceUsage_SourceManagerContentCache = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManagerContentCache', 5)
CXTUResourceUsage_AST_SideTables = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_AST_SideTables', 6)
CXTUResourceUsage_SourceManager_Membuffer_Malloc = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_Membuffer_Malloc', 7)
CXTUResourceUsage_SourceManager_Membuffer_MMap = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_Membuffer_MMap', 8)
CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc', 9)
CXTUResourceUsage_ExternalASTSource_Membuffer_MMap = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_ExternalASTSource_Membuffer_MMap', 10)
CXTUResourceUsage_Preprocessor = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Preprocessor', 11)
CXTUResourceUsage_PreprocessingRecord = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_PreprocessingRecord', 12)
CXTUResourceUsage_SourceManager_DataStructures = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_DataStructures', 13)
CXTUResourceUsage_Preprocessor_HeaderSearch = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Preprocessor_HeaderSearch', 14)
CXTUResourceUsage_MEMORY_IN_BYTES_BEGIN = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_MEMORY_IN_BYTES_BEGIN', 1)
CXTUResourceUsage_MEMORY_IN_BYTES_END = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_MEMORY_IN_BYTES_END', 14)
CXTUResourceUsage_First = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_First', 1)
CXTUResourceUsage_Last = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Last', 14)

@dll.bind((enum_CXTUResourceUsageKind,), Pointer(ctypes.c_char))
def clang_getTUResourceUsageName(kind): ...
class struct_CXTUResourceUsageEntry(Struct): pass
struct_CXTUResourceUsageEntry.SIZE = 16
struct_CXTUResourceUsageEntry._fields_ = ['kind', 'amount']
setattr(struct_CXTUResourceUsageEntry, 'kind', field(0, enum_CXTUResourceUsageKind))
setattr(struct_CXTUResourceUsageEntry, 'amount', field(8, ctypes.c_uint64))
CXTUResourceUsageEntry = struct_CXTUResourceUsageEntry
class struct_CXTUResourceUsage(Struct): pass
struct_CXTUResourceUsage.SIZE = 24
struct_CXTUResourceUsage._fields_ = ['data', 'numEntries', 'entries']
setattr(struct_CXTUResourceUsage, 'data', field(0, ctypes.c_void_p))
setattr(struct_CXTUResourceUsage, 'numEntries', field(8, ctypes.c_uint32))
setattr(struct_CXTUResourceUsage, 'entries', field(16, Pointer(CXTUResourceUsageEntry)))
CXTUResourceUsage = struct_CXTUResourceUsage
@dll.bind((CXTranslationUnit,), CXTUResourceUsage)
def clang_getCXTUResourceUsage(TU): ...
@dll.bind((CXTUResourceUsage,), None)
def clang_disposeCXTUResourceUsage(usage): ...
@dll.bind((CXTranslationUnit,), CXTargetInfo)
def clang_getTranslationUnitTargetInfo(CTUnit): ...
@dll.bind((CXTargetInfo,), None)
def clang_TargetInfo_dispose(Info): ...
@dll.bind((CXTargetInfo,), CXString)
def clang_TargetInfo_getTriple(Info): ...
@dll.bind((CXTargetInfo,), ctypes.c_int32)
def clang_TargetInfo_getPointerWidth(Info): ...
enum_CXCursorKind = CEnum(ctypes.c_uint32)
CXCursor_UnexposedDecl = enum_CXCursorKind.define('CXCursor_UnexposedDecl', 1)
CXCursor_StructDecl = enum_CXCursorKind.define('CXCursor_StructDecl', 2)
CXCursor_UnionDecl = enum_CXCursorKind.define('CXCursor_UnionDecl', 3)
CXCursor_ClassDecl = enum_CXCursorKind.define('CXCursor_ClassDecl', 4)
CXCursor_EnumDecl = enum_CXCursorKind.define('CXCursor_EnumDecl', 5)
CXCursor_FieldDecl = enum_CXCursorKind.define('CXCursor_FieldDecl', 6)
CXCursor_EnumConstantDecl = enum_CXCursorKind.define('CXCursor_EnumConstantDecl', 7)
CXCursor_FunctionDecl = enum_CXCursorKind.define('CXCursor_FunctionDecl', 8)
CXCursor_VarDecl = enum_CXCursorKind.define('CXCursor_VarDecl', 9)
CXCursor_ParmDecl = enum_CXCursorKind.define('CXCursor_ParmDecl', 10)
CXCursor_ObjCInterfaceDecl = enum_CXCursorKind.define('CXCursor_ObjCInterfaceDecl', 11)
CXCursor_ObjCCategoryDecl = enum_CXCursorKind.define('CXCursor_ObjCCategoryDecl', 12)
CXCursor_ObjCProtocolDecl = enum_CXCursorKind.define('CXCursor_ObjCProtocolDecl', 13)
CXCursor_ObjCPropertyDecl = enum_CXCursorKind.define('CXCursor_ObjCPropertyDecl', 14)
CXCursor_ObjCIvarDecl = enum_CXCursorKind.define('CXCursor_ObjCIvarDecl', 15)
CXCursor_ObjCInstanceMethodDecl = enum_CXCursorKind.define('CXCursor_ObjCInstanceMethodDecl', 16)
CXCursor_ObjCClassMethodDecl = enum_CXCursorKind.define('CXCursor_ObjCClassMethodDecl', 17)
CXCursor_ObjCImplementationDecl = enum_CXCursorKind.define('CXCursor_ObjCImplementationDecl', 18)
CXCursor_ObjCCategoryImplDecl = enum_CXCursorKind.define('CXCursor_ObjCCategoryImplDecl', 19)
CXCursor_TypedefDecl = enum_CXCursorKind.define('CXCursor_TypedefDecl', 20)
CXCursor_CXXMethod = enum_CXCursorKind.define('CXCursor_CXXMethod', 21)
CXCursor_Namespace = enum_CXCursorKind.define('CXCursor_Namespace', 22)
CXCursor_LinkageSpec = enum_CXCursorKind.define('CXCursor_LinkageSpec', 23)
CXCursor_Constructor = enum_CXCursorKind.define('CXCursor_Constructor', 24)
CXCursor_Destructor = enum_CXCursorKind.define('CXCursor_Destructor', 25)
CXCursor_ConversionFunction = enum_CXCursorKind.define('CXCursor_ConversionFunction', 26)
CXCursor_TemplateTypeParameter = enum_CXCursorKind.define('CXCursor_TemplateTypeParameter', 27)
CXCursor_NonTypeTemplateParameter = enum_CXCursorKind.define('CXCursor_NonTypeTemplateParameter', 28)
CXCursor_TemplateTemplateParameter = enum_CXCursorKind.define('CXCursor_TemplateTemplateParameter', 29)
CXCursor_FunctionTemplate = enum_CXCursorKind.define('CXCursor_FunctionTemplate', 30)
CXCursor_ClassTemplate = enum_CXCursorKind.define('CXCursor_ClassTemplate', 31)
CXCursor_ClassTemplatePartialSpecialization = enum_CXCursorKind.define('CXCursor_ClassTemplatePartialSpecialization', 32)
CXCursor_NamespaceAlias = enum_CXCursorKind.define('CXCursor_NamespaceAlias', 33)
CXCursor_UsingDirective = enum_CXCursorKind.define('CXCursor_UsingDirective', 34)
CXCursor_UsingDeclaration = enum_CXCursorKind.define('CXCursor_UsingDeclaration', 35)
CXCursor_TypeAliasDecl = enum_CXCursorKind.define('CXCursor_TypeAliasDecl', 36)
CXCursor_ObjCSynthesizeDecl = enum_CXCursorKind.define('CXCursor_ObjCSynthesizeDecl', 37)
CXCursor_ObjCDynamicDecl = enum_CXCursorKind.define('CXCursor_ObjCDynamicDecl', 38)
CXCursor_CXXAccessSpecifier = enum_CXCursorKind.define('CXCursor_CXXAccessSpecifier', 39)
CXCursor_FirstDecl = enum_CXCursorKind.define('CXCursor_FirstDecl', 1)
CXCursor_LastDecl = enum_CXCursorKind.define('CXCursor_LastDecl', 39)
CXCursor_FirstRef = enum_CXCursorKind.define('CXCursor_FirstRef', 40)
CXCursor_ObjCSuperClassRef = enum_CXCursorKind.define('CXCursor_ObjCSuperClassRef', 40)
CXCursor_ObjCProtocolRef = enum_CXCursorKind.define('CXCursor_ObjCProtocolRef', 41)
CXCursor_ObjCClassRef = enum_CXCursorKind.define('CXCursor_ObjCClassRef', 42)
CXCursor_TypeRef = enum_CXCursorKind.define('CXCursor_TypeRef', 43)
CXCursor_CXXBaseSpecifier = enum_CXCursorKind.define('CXCursor_CXXBaseSpecifier', 44)
CXCursor_TemplateRef = enum_CXCursorKind.define('CXCursor_TemplateRef', 45)
CXCursor_NamespaceRef = enum_CXCursorKind.define('CXCursor_NamespaceRef', 46)
CXCursor_MemberRef = enum_CXCursorKind.define('CXCursor_MemberRef', 47)
CXCursor_LabelRef = enum_CXCursorKind.define('CXCursor_LabelRef', 48)
CXCursor_OverloadedDeclRef = enum_CXCursorKind.define('CXCursor_OverloadedDeclRef', 49)
CXCursor_VariableRef = enum_CXCursorKind.define('CXCursor_VariableRef', 50)
CXCursor_LastRef = enum_CXCursorKind.define('CXCursor_LastRef', 50)
CXCursor_FirstInvalid = enum_CXCursorKind.define('CXCursor_FirstInvalid', 70)
CXCursor_InvalidFile = enum_CXCursorKind.define('CXCursor_InvalidFile', 70)
CXCursor_NoDeclFound = enum_CXCursorKind.define('CXCursor_NoDeclFound', 71)
CXCursor_NotImplemented = enum_CXCursorKind.define('CXCursor_NotImplemented', 72)
CXCursor_InvalidCode = enum_CXCursorKind.define('CXCursor_InvalidCode', 73)
CXCursor_LastInvalid = enum_CXCursorKind.define('CXCursor_LastInvalid', 73)
CXCursor_FirstExpr = enum_CXCursorKind.define('CXCursor_FirstExpr', 100)
CXCursor_UnexposedExpr = enum_CXCursorKind.define('CXCursor_UnexposedExpr', 100)
CXCursor_DeclRefExpr = enum_CXCursorKind.define('CXCursor_DeclRefExpr', 101)
CXCursor_MemberRefExpr = enum_CXCursorKind.define('CXCursor_MemberRefExpr', 102)
CXCursor_CallExpr = enum_CXCursorKind.define('CXCursor_CallExpr', 103)
CXCursor_ObjCMessageExpr = enum_CXCursorKind.define('CXCursor_ObjCMessageExpr', 104)
CXCursor_BlockExpr = enum_CXCursorKind.define('CXCursor_BlockExpr', 105)
CXCursor_IntegerLiteral = enum_CXCursorKind.define('CXCursor_IntegerLiteral', 106)
CXCursor_FloatingLiteral = enum_CXCursorKind.define('CXCursor_FloatingLiteral', 107)
CXCursor_ImaginaryLiteral = enum_CXCursorKind.define('CXCursor_ImaginaryLiteral', 108)
CXCursor_StringLiteral = enum_CXCursorKind.define('CXCursor_StringLiteral', 109)
CXCursor_CharacterLiteral = enum_CXCursorKind.define('CXCursor_CharacterLiteral', 110)
CXCursor_ParenExpr = enum_CXCursorKind.define('CXCursor_ParenExpr', 111)
CXCursor_UnaryOperator = enum_CXCursorKind.define('CXCursor_UnaryOperator', 112)
CXCursor_ArraySubscriptExpr = enum_CXCursorKind.define('CXCursor_ArraySubscriptExpr', 113)
CXCursor_BinaryOperator = enum_CXCursorKind.define('CXCursor_BinaryOperator', 114)
CXCursor_CompoundAssignOperator = enum_CXCursorKind.define('CXCursor_CompoundAssignOperator', 115)
CXCursor_ConditionalOperator = enum_CXCursorKind.define('CXCursor_ConditionalOperator', 116)
CXCursor_CStyleCastExpr = enum_CXCursorKind.define('CXCursor_CStyleCastExpr', 117)
CXCursor_CompoundLiteralExpr = enum_CXCursorKind.define('CXCursor_CompoundLiteralExpr', 118)
CXCursor_InitListExpr = enum_CXCursorKind.define('CXCursor_InitListExpr', 119)
CXCursor_AddrLabelExpr = enum_CXCursorKind.define('CXCursor_AddrLabelExpr', 120)
CXCursor_StmtExpr = enum_CXCursorKind.define('CXCursor_StmtExpr', 121)
CXCursor_GenericSelectionExpr = enum_CXCursorKind.define('CXCursor_GenericSelectionExpr', 122)
CXCursor_GNUNullExpr = enum_CXCursorKind.define('CXCursor_GNUNullExpr', 123)
CXCursor_CXXStaticCastExpr = enum_CXCursorKind.define('CXCursor_CXXStaticCastExpr', 124)
CXCursor_CXXDynamicCastExpr = enum_CXCursorKind.define('CXCursor_CXXDynamicCastExpr', 125)
CXCursor_CXXReinterpretCastExpr = enum_CXCursorKind.define('CXCursor_CXXReinterpretCastExpr', 126)
CXCursor_CXXConstCastExpr = enum_CXCursorKind.define('CXCursor_CXXConstCastExpr', 127)
CXCursor_CXXFunctionalCastExpr = enum_CXCursorKind.define('CXCursor_CXXFunctionalCastExpr', 128)
CXCursor_CXXTypeidExpr = enum_CXCursorKind.define('CXCursor_CXXTypeidExpr', 129)
CXCursor_CXXBoolLiteralExpr = enum_CXCursorKind.define('CXCursor_CXXBoolLiteralExpr', 130)
CXCursor_CXXNullPtrLiteralExpr = enum_CXCursorKind.define('CXCursor_CXXNullPtrLiteralExpr', 131)
CXCursor_CXXThisExpr = enum_CXCursorKind.define('CXCursor_CXXThisExpr', 132)
CXCursor_CXXThrowExpr = enum_CXCursorKind.define('CXCursor_CXXThrowExpr', 133)
CXCursor_CXXNewExpr = enum_CXCursorKind.define('CXCursor_CXXNewExpr', 134)
CXCursor_CXXDeleteExpr = enum_CXCursorKind.define('CXCursor_CXXDeleteExpr', 135)
CXCursor_UnaryExpr = enum_CXCursorKind.define('CXCursor_UnaryExpr', 136)
CXCursor_ObjCStringLiteral = enum_CXCursorKind.define('CXCursor_ObjCStringLiteral', 137)
CXCursor_ObjCEncodeExpr = enum_CXCursorKind.define('CXCursor_ObjCEncodeExpr', 138)
CXCursor_ObjCSelectorExpr = enum_CXCursorKind.define('CXCursor_ObjCSelectorExpr', 139)
CXCursor_ObjCProtocolExpr = enum_CXCursorKind.define('CXCursor_ObjCProtocolExpr', 140)
CXCursor_ObjCBridgedCastExpr = enum_CXCursorKind.define('CXCursor_ObjCBridgedCastExpr', 141)
CXCursor_PackExpansionExpr = enum_CXCursorKind.define('CXCursor_PackExpansionExpr', 142)
CXCursor_SizeOfPackExpr = enum_CXCursorKind.define('CXCursor_SizeOfPackExpr', 143)
CXCursor_LambdaExpr = enum_CXCursorKind.define('CXCursor_LambdaExpr', 144)
CXCursor_ObjCBoolLiteralExpr = enum_CXCursorKind.define('CXCursor_ObjCBoolLiteralExpr', 145)
CXCursor_ObjCSelfExpr = enum_CXCursorKind.define('CXCursor_ObjCSelfExpr', 146)
CXCursor_ArraySectionExpr = enum_CXCursorKind.define('CXCursor_ArraySectionExpr', 147)
CXCursor_ObjCAvailabilityCheckExpr = enum_CXCursorKind.define('CXCursor_ObjCAvailabilityCheckExpr', 148)
CXCursor_FixedPointLiteral = enum_CXCursorKind.define('CXCursor_FixedPointLiteral', 149)
CXCursor_OMPArrayShapingExpr = enum_CXCursorKind.define('CXCursor_OMPArrayShapingExpr', 150)
CXCursor_OMPIteratorExpr = enum_CXCursorKind.define('CXCursor_OMPIteratorExpr', 151)
CXCursor_CXXAddrspaceCastExpr = enum_CXCursorKind.define('CXCursor_CXXAddrspaceCastExpr', 152)
CXCursor_ConceptSpecializationExpr = enum_CXCursorKind.define('CXCursor_ConceptSpecializationExpr', 153)
CXCursor_RequiresExpr = enum_CXCursorKind.define('CXCursor_RequiresExpr', 154)
CXCursor_CXXParenListInitExpr = enum_CXCursorKind.define('CXCursor_CXXParenListInitExpr', 155)
CXCursor_PackIndexingExpr = enum_CXCursorKind.define('CXCursor_PackIndexingExpr', 156)
CXCursor_LastExpr = enum_CXCursorKind.define('CXCursor_LastExpr', 156)
CXCursor_FirstStmt = enum_CXCursorKind.define('CXCursor_FirstStmt', 200)
CXCursor_UnexposedStmt = enum_CXCursorKind.define('CXCursor_UnexposedStmt', 200)
CXCursor_LabelStmt = enum_CXCursorKind.define('CXCursor_LabelStmt', 201)
CXCursor_CompoundStmt = enum_CXCursorKind.define('CXCursor_CompoundStmt', 202)
CXCursor_CaseStmt = enum_CXCursorKind.define('CXCursor_CaseStmt', 203)
CXCursor_DefaultStmt = enum_CXCursorKind.define('CXCursor_DefaultStmt', 204)
CXCursor_IfStmt = enum_CXCursorKind.define('CXCursor_IfStmt', 205)
CXCursor_SwitchStmt = enum_CXCursorKind.define('CXCursor_SwitchStmt', 206)
CXCursor_WhileStmt = enum_CXCursorKind.define('CXCursor_WhileStmt', 207)
CXCursor_DoStmt = enum_CXCursorKind.define('CXCursor_DoStmt', 208)
CXCursor_ForStmt = enum_CXCursorKind.define('CXCursor_ForStmt', 209)
CXCursor_GotoStmt = enum_CXCursorKind.define('CXCursor_GotoStmt', 210)
CXCursor_IndirectGotoStmt = enum_CXCursorKind.define('CXCursor_IndirectGotoStmt', 211)
CXCursor_ContinueStmt = enum_CXCursorKind.define('CXCursor_ContinueStmt', 212)
CXCursor_BreakStmt = enum_CXCursorKind.define('CXCursor_BreakStmt', 213)
CXCursor_ReturnStmt = enum_CXCursorKind.define('CXCursor_ReturnStmt', 214)
CXCursor_GCCAsmStmt = enum_CXCursorKind.define('CXCursor_GCCAsmStmt', 215)
CXCursor_AsmStmt = enum_CXCursorKind.define('CXCursor_AsmStmt', 215)
CXCursor_ObjCAtTryStmt = enum_CXCursorKind.define('CXCursor_ObjCAtTryStmt', 216)
CXCursor_ObjCAtCatchStmt = enum_CXCursorKind.define('CXCursor_ObjCAtCatchStmt', 217)
CXCursor_ObjCAtFinallyStmt = enum_CXCursorKind.define('CXCursor_ObjCAtFinallyStmt', 218)
CXCursor_ObjCAtThrowStmt = enum_CXCursorKind.define('CXCursor_ObjCAtThrowStmt', 219)
CXCursor_ObjCAtSynchronizedStmt = enum_CXCursorKind.define('CXCursor_ObjCAtSynchronizedStmt', 220)
CXCursor_ObjCAutoreleasePoolStmt = enum_CXCursorKind.define('CXCursor_ObjCAutoreleasePoolStmt', 221)
CXCursor_ObjCForCollectionStmt = enum_CXCursorKind.define('CXCursor_ObjCForCollectionStmt', 222)
CXCursor_CXXCatchStmt = enum_CXCursorKind.define('CXCursor_CXXCatchStmt', 223)
CXCursor_CXXTryStmt = enum_CXCursorKind.define('CXCursor_CXXTryStmt', 224)
CXCursor_CXXForRangeStmt = enum_CXCursorKind.define('CXCursor_CXXForRangeStmt', 225)
CXCursor_SEHTryStmt = enum_CXCursorKind.define('CXCursor_SEHTryStmt', 226)
CXCursor_SEHExceptStmt = enum_CXCursorKind.define('CXCursor_SEHExceptStmt', 227)
CXCursor_SEHFinallyStmt = enum_CXCursorKind.define('CXCursor_SEHFinallyStmt', 228)
CXCursor_MSAsmStmt = enum_CXCursorKind.define('CXCursor_MSAsmStmt', 229)
CXCursor_NullStmt = enum_CXCursorKind.define('CXCursor_NullStmt', 230)
CXCursor_DeclStmt = enum_CXCursorKind.define('CXCursor_DeclStmt', 231)
CXCursor_OMPParallelDirective = enum_CXCursorKind.define('CXCursor_OMPParallelDirective', 232)
CXCursor_OMPSimdDirective = enum_CXCursorKind.define('CXCursor_OMPSimdDirective', 233)
CXCursor_OMPForDirective = enum_CXCursorKind.define('CXCursor_OMPForDirective', 234)
CXCursor_OMPSectionsDirective = enum_CXCursorKind.define('CXCursor_OMPSectionsDirective', 235)
CXCursor_OMPSectionDirective = enum_CXCursorKind.define('CXCursor_OMPSectionDirective', 236)
CXCursor_OMPSingleDirective = enum_CXCursorKind.define('CXCursor_OMPSingleDirective', 237)
CXCursor_OMPParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPParallelForDirective', 238)
CXCursor_OMPParallelSectionsDirective = enum_CXCursorKind.define('CXCursor_OMPParallelSectionsDirective', 239)
CXCursor_OMPTaskDirective = enum_CXCursorKind.define('CXCursor_OMPTaskDirective', 240)
CXCursor_OMPMasterDirective = enum_CXCursorKind.define('CXCursor_OMPMasterDirective', 241)
CXCursor_OMPCriticalDirective = enum_CXCursorKind.define('CXCursor_OMPCriticalDirective', 242)
CXCursor_OMPTaskyieldDirective = enum_CXCursorKind.define('CXCursor_OMPTaskyieldDirective', 243)
CXCursor_OMPBarrierDirective = enum_CXCursorKind.define('CXCursor_OMPBarrierDirective', 244)
CXCursor_OMPTaskwaitDirective = enum_CXCursorKind.define('CXCursor_OMPTaskwaitDirective', 245)
CXCursor_OMPFlushDirective = enum_CXCursorKind.define('CXCursor_OMPFlushDirective', 246)
CXCursor_SEHLeaveStmt = enum_CXCursorKind.define('CXCursor_SEHLeaveStmt', 247)
CXCursor_OMPOrderedDirective = enum_CXCursorKind.define('CXCursor_OMPOrderedDirective', 248)
CXCursor_OMPAtomicDirective = enum_CXCursorKind.define('CXCursor_OMPAtomicDirective', 249)
CXCursor_OMPForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPForSimdDirective', 250)
CXCursor_OMPParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelForSimdDirective', 251)
CXCursor_OMPTargetDirective = enum_CXCursorKind.define('CXCursor_OMPTargetDirective', 252)
CXCursor_OMPTeamsDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDirective', 253)
CXCursor_OMPTaskgroupDirective = enum_CXCursorKind.define('CXCursor_OMPTaskgroupDirective', 254)
CXCursor_OMPCancellationPointDirective = enum_CXCursorKind.define('CXCursor_OMPCancellationPointDirective', 255)
CXCursor_OMPCancelDirective = enum_CXCursorKind.define('CXCursor_OMPCancelDirective', 256)
CXCursor_OMPTargetDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetDataDirective', 257)
CXCursor_OMPTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTaskLoopDirective', 258)
CXCursor_OMPTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTaskLoopSimdDirective', 259)
CXCursor_OMPDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeDirective', 260)
CXCursor_OMPTargetEnterDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetEnterDataDirective', 261)
CXCursor_OMPTargetExitDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetExitDataDirective', 262)
CXCursor_OMPTargetParallelDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelDirective', 263)
CXCursor_OMPTargetParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelForDirective', 264)
CXCursor_OMPTargetUpdateDirective = enum_CXCursorKind.define('CXCursor_OMPTargetUpdateDirective', 265)
CXCursor_OMPDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeParallelForDirective', 266)
CXCursor_OMPDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeParallelForSimdDirective', 267)
CXCursor_OMPDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeSimdDirective', 268)
CXCursor_OMPTargetParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelForSimdDirective', 269)
CXCursor_OMPTargetSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetSimdDirective', 270)
CXCursor_OMPTeamsDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeDirective', 271)
CXCursor_OMPTeamsDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeSimdDirective', 272)
CXCursor_OMPTeamsDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeParallelForSimdDirective', 273)
CXCursor_OMPTeamsDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeParallelForDirective', 274)
CXCursor_OMPTargetTeamsDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDirective', 275)
CXCursor_OMPTargetTeamsDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeDirective', 276)
CXCursor_OMPTargetTeamsDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeParallelForDirective', 277)
CXCursor_OMPTargetTeamsDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeParallelForSimdDirective', 278)
CXCursor_OMPTargetTeamsDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeSimdDirective', 279)
CXCursor_BuiltinBitCastExpr = enum_CXCursorKind.define('CXCursor_BuiltinBitCastExpr', 280)
CXCursor_OMPMasterTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPMasterTaskLoopDirective', 281)
CXCursor_OMPParallelMasterTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterTaskLoopDirective', 282)
CXCursor_OMPMasterTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPMasterTaskLoopSimdDirective', 283)
CXCursor_OMPParallelMasterTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterTaskLoopSimdDirective', 284)
CXCursor_OMPParallelMasterDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterDirective', 285)
CXCursor_OMPDepobjDirective = enum_CXCursorKind.define('CXCursor_OMPDepobjDirective', 286)
CXCursor_OMPScanDirective = enum_CXCursorKind.define('CXCursor_OMPScanDirective', 287)
CXCursor_OMPTileDirective = enum_CXCursorKind.define('CXCursor_OMPTileDirective', 288)
CXCursor_OMPCanonicalLoop = enum_CXCursorKind.define('CXCursor_OMPCanonicalLoop', 289)
CXCursor_OMPInteropDirective = enum_CXCursorKind.define('CXCursor_OMPInteropDirective', 290)
CXCursor_OMPDispatchDirective = enum_CXCursorKind.define('CXCursor_OMPDispatchDirective', 291)
CXCursor_OMPMaskedDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedDirective', 292)
CXCursor_OMPUnrollDirective = enum_CXCursorKind.define('CXCursor_OMPUnrollDirective', 293)
CXCursor_OMPMetaDirective = enum_CXCursorKind.define('CXCursor_OMPMetaDirective', 294)
CXCursor_OMPGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPGenericLoopDirective', 295)
CXCursor_OMPTeamsGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsGenericLoopDirective', 296)
CXCursor_OMPTargetTeamsGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsGenericLoopDirective', 297)
CXCursor_OMPParallelGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelGenericLoopDirective', 298)
CXCursor_OMPTargetParallelGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelGenericLoopDirective', 299)
CXCursor_OMPParallelMaskedDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedDirective', 300)
CXCursor_OMPMaskedTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedTaskLoopDirective', 301)
CXCursor_OMPMaskedTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedTaskLoopSimdDirective', 302)
CXCursor_OMPParallelMaskedTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedTaskLoopDirective', 303)
CXCursor_OMPParallelMaskedTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedTaskLoopSimdDirective', 304)
CXCursor_OMPErrorDirective = enum_CXCursorKind.define('CXCursor_OMPErrorDirective', 305)
CXCursor_OMPScopeDirective = enum_CXCursorKind.define('CXCursor_OMPScopeDirective', 306)
CXCursor_OMPReverseDirective = enum_CXCursorKind.define('CXCursor_OMPReverseDirective', 307)
CXCursor_OMPInterchangeDirective = enum_CXCursorKind.define('CXCursor_OMPInterchangeDirective', 308)
CXCursor_OMPAssumeDirective = enum_CXCursorKind.define('CXCursor_OMPAssumeDirective', 309)
CXCursor_OpenACCComputeConstruct = enum_CXCursorKind.define('CXCursor_OpenACCComputeConstruct', 320)
CXCursor_OpenACCLoopConstruct = enum_CXCursorKind.define('CXCursor_OpenACCLoopConstruct', 321)
CXCursor_OpenACCCombinedConstruct = enum_CXCursorKind.define('CXCursor_OpenACCCombinedConstruct', 322)
CXCursor_OpenACCDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCDataConstruct', 323)
CXCursor_OpenACCEnterDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCEnterDataConstruct', 324)
CXCursor_OpenACCExitDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCExitDataConstruct', 325)
CXCursor_OpenACCHostDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCHostDataConstruct', 326)
CXCursor_OpenACCWaitConstruct = enum_CXCursorKind.define('CXCursor_OpenACCWaitConstruct', 327)
CXCursor_OpenACCInitConstruct = enum_CXCursorKind.define('CXCursor_OpenACCInitConstruct', 328)
CXCursor_OpenACCShutdownConstruct = enum_CXCursorKind.define('CXCursor_OpenACCShutdownConstruct', 329)
CXCursor_OpenACCSetConstruct = enum_CXCursorKind.define('CXCursor_OpenACCSetConstruct', 330)
CXCursor_OpenACCUpdateConstruct = enum_CXCursorKind.define('CXCursor_OpenACCUpdateConstruct', 331)
CXCursor_LastStmt = enum_CXCursorKind.define('CXCursor_LastStmt', 331)
CXCursor_TranslationUnit = enum_CXCursorKind.define('CXCursor_TranslationUnit', 350)
CXCursor_FirstAttr = enum_CXCursorKind.define('CXCursor_FirstAttr', 400)
CXCursor_UnexposedAttr = enum_CXCursorKind.define('CXCursor_UnexposedAttr', 400)
CXCursor_IBActionAttr = enum_CXCursorKind.define('CXCursor_IBActionAttr', 401)
CXCursor_IBOutletAttr = enum_CXCursorKind.define('CXCursor_IBOutletAttr', 402)
CXCursor_IBOutletCollectionAttr = enum_CXCursorKind.define('CXCursor_IBOutletCollectionAttr', 403)
CXCursor_CXXFinalAttr = enum_CXCursorKind.define('CXCursor_CXXFinalAttr', 404)
CXCursor_CXXOverrideAttr = enum_CXCursorKind.define('CXCursor_CXXOverrideAttr', 405)
CXCursor_AnnotateAttr = enum_CXCursorKind.define('CXCursor_AnnotateAttr', 406)
CXCursor_AsmLabelAttr = enum_CXCursorKind.define('CXCursor_AsmLabelAttr', 407)
CXCursor_PackedAttr = enum_CXCursorKind.define('CXCursor_PackedAttr', 408)
CXCursor_PureAttr = enum_CXCursorKind.define('CXCursor_PureAttr', 409)
CXCursor_ConstAttr = enum_CXCursorKind.define('CXCursor_ConstAttr', 410)
CXCursor_NoDuplicateAttr = enum_CXCursorKind.define('CXCursor_NoDuplicateAttr', 411)
CXCursor_CUDAConstantAttr = enum_CXCursorKind.define('CXCursor_CUDAConstantAttr', 412)
CXCursor_CUDADeviceAttr = enum_CXCursorKind.define('CXCursor_CUDADeviceAttr', 413)
CXCursor_CUDAGlobalAttr = enum_CXCursorKind.define('CXCursor_CUDAGlobalAttr', 414)
CXCursor_CUDAHostAttr = enum_CXCursorKind.define('CXCursor_CUDAHostAttr', 415)
CXCursor_CUDASharedAttr = enum_CXCursorKind.define('CXCursor_CUDASharedAttr', 416)
CXCursor_VisibilityAttr = enum_CXCursorKind.define('CXCursor_VisibilityAttr', 417)
CXCursor_DLLExport = enum_CXCursorKind.define('CXCursor_DLLExport', 418)
CXCursor_DLLImport = enum_CXCursorKind.define('CXCursor_DLLImport', 419)
CXCursor_NSReturnsRetained = enum_CXCursorKind.define('CXCursor_NSReturnsRetained', 420)
CXCursor_NSReturnsNotRetained = enum_CXCursorKind.define('CXCursor_NSReturnsNotRetained', 421)
CXCursor_NSReturnsAutoreleased = enum_CXCursorKind.define('CXCursor_NSReturnsAutoreleased', 422)
CXCursor_NSConsumesSelf = enum_CXCursorKind.define('CXCursor_NSConsumesSelf', 423)
CXCursor_NSConsumed = enum_CXCursorKind.define('CXCursor_NSConsumed', 424)
CXCursor_ObjCException = enum_CXCursorKind.define('CXCursor_ObjCException', 425)
CXCursor_ObjCNSObject = enum_CXCursorKind.define('CXCursor_ObjCNSObject', 426)
CXCursor_ObjCIndependentClass = enum_CXCursorKind.define('CXCursor_ObjCIndependentClass', 427)
CXCursor_ObjCPreciseLifetime = enum_CXCursorKind.define('CXCursor_ObjCPreciseLifetime', 428)
CXCursor_ObjCReturnsInnerPointer = enum_CXCursorKind.define('CXCursor_ObjCReturnsInnerPointer', 429)
CXCursor_ObjCRequiresSuper = enum_CXCursorKind.define('CXCursor_ObjCRequiresSuper', 430)
CXCursor_ObjCRootClass = enum_CXCursorKind.define('CXCursor_ObjCRootClass', 431)
CXCursor_ObjCSubclassingRestricted = enum_CXCursorKind.define('CXCursor_ObjCSubclassingRestricted', 432)
CXCursor_ObjCExplicitProtocolImpl = enum_CXCursorKind.define('CXCursor_ObjCExplicitProtocolImpl', 433)
CXCursor_ObjCDesignatedInitializer = enum_CXCursorKind.define('CXCursor_ObjCDesignatedInitializer', 434)
CXCursor_ObjCRuntimeVisible = enum_CXCursorKind.define('CXCursor_ObjCRuntimeVisible', 435)
CXCursor_ObjCBoxable = enum_CXCursorKind.define('CXCursor_ObjCBoxable', 436)
CXCursor_FlagEnum = enum_CXCursorKind.define('CXCursor_FlagEnum', 437)
CXCursor_ConvergentAttr = enum_CXCursorKind.define('CXCursor_ConvergentAttr', 438)
CXCursor_WarnUnusedAttr = enum_CXCursorKind.define('CXCursor_WarnUnusedAttr', 439)
CXCursor_WarnUnusedResultAttr = enum_CXCursorKind.define('CXCursor_WarnUnusedResultAttr', 440)
CXCursor_AlignedAttr = enum_CXCursorKind.define('CXCursor_AlignedAttr', 441)
CXCursor_LastAttr = enum_CXCursorKind.define('CXCursor_LastAttr', 441)
CXCursor_PreprocessingDirective = enum_CXCursorKind.define('CXCursor_PreprocessingDirective', 500)
CXCursor_MacroDefinition = enum_CXCursorKind.define('CXCursor_MacroDefinition', 501)
CXCursor_MacroExpansion = enum_CXCursorKind.define('CXCursor_MacroExpansion', 502)
CXCursor_MacroInstantiation = enum_CXCursorKind.define('CXCursor_MacroInstantiation', 502)
CXCursor_InclusionDirective = enum_CXCursorKind.define('CXCursor_InclusionDirective', 503)
CXCursor_FirstPreprocessing = enum_CXCursorKind.define('CXCursor_FirstPreprocessing', 500)
CXCursor_LastPreprocessing = enum_CXCursorKind.define('CXCursor_LastPreprocessing', 503)
CXCursor_ModuleImportDecl = enum_CXCursorKind.define('CXCursor_ModuleImportDecl', 600)
CXCursor_TypeAliasTemplateDecl = enum_CXCursorKind.define('CXCursor_TypeAliasTemplateDecl', 601)
CXCursor_StaticAssert = enum_CXCursorKind.define('CXCursor_StaticAssert', 602)
CXCursor_FriendDecl = enum_CXCursorKind.define('CXCursor_FriendDecl', 603)
CXCursor_ConceptDecl = enum_CXCursorKind.define('CXCursor_ConceptDecl', 604)
CXCursor_FirstExtraDecl = enum_CXCursorKind.define('CXCursor_FirstExtraDecl', 600)
CXCursor_LastExtraDecl = enum_CXCursorKind.define('CXCursor_LastExtraDecl', 604)
CXCursor_OverloadCandidate = enum_CXCursorKind.define('CXCursor_OverloadCandidate', 700)

class CXCursor(Struct): pass
CXCursor.SIZE = 32
CXCursor._fields_ = ['kind', 'xdata', 'data']
setattr(CXCursor, 'kind', field(0, enum_CXCursorKind))
setattr(CXCursor, 'xdata', field(4, ctypes.c_int32))
setattr(CXCursor, 'data', field(8, Array(ctypes.c_void_p, 3)))
@dll.bind((), CXCursor)
def clang_getNullCursor(): ...
@dll.bind((CXTranslationUnit,), CXCursor)
def clang_getTranslationUnitCursor(_0): ...
@dll.bind((CXCursor, CXCursor), ctypes.c_uint32)
def clang_equalCursors(_0, _1): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_isNull(cursor): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_hashCursor(_0): ...
@dll.bind((CXCursor,), enum_CXCursorKind)
def clang_getCursorKind(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isDeclaration(_0): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_isInvalidDeclaration(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isReference(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isExpression(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isStatement(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isAttribute(_0): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_hasAttrs(C): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isInvalid(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isTranslationUnit(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isPreprocessing(_0): ...
@dll.bind((enum_CXCursorKind,), ctypes.c_uint32)
def clang_isUnexposed(_0): ...
enum_CXLinkageKind = CEnum(ctypes.c_uint32)
CXLinkage_Invalid = enum_CXLinkageKind.define('CXLinkage_Invalid', 0)
CXLinkage_NoLinkage = enum_CXLinkageKind.define('CXLinkage_NoLinkage', 1)
CXLinkage_Internal = enum_CXLinkageKind.define('CXLinkage_Internal', 2)
CXLinkage_UniqueExternal = enum_CXLinkageKind.define('CXLinkage_UniqueExternal', 3)
CXLinkage_External = enum_CXLinkageKind.define('CXLinkage_External', 4)

@dll.bind((CXCursor,), enum_CXLinkageKind)
def clang_getCursorLinkage(cursor): ...
enum_CXVisibilityKind = CEnum(ctypes.c_uint32)
CXVisibility_Invalid = enum_CXVisibilityKind.define('CXVisibility_Invalid', 0)
CXVisibility_Hidden = enum_CXVisibilityKind.define('CXVisibility_Hidden', 1)
CXVisibility_Protected = enum_CXVisibilityKind.define('CXVisibility_Protected', 2)
CXVisibility_Default = enum_CXVisibilityKind.define('CXVisibility_Default', 3)

@dll.bind((CXCursor,), enum_CXVisibilityKind)
def clang_getCursorVisibility(cursor): ...
@dll.bind((CXCursor,), enum_CXAvailabilityKind)
def clang_getCursorAvailability(cursor): ...
class struct_CXPlatformAvailability(Struct): pass
struct_CXPlatformAvailability.SIZE = 72
struct_CXPlatformAvailability._fields_ = ['Platform', 'Introduced', 'Deprecated', 'Obsoleted', 'Unavailable', 'Message']
setattr(struct_CXPlatformAvailability, 'Platform', field(0, CXString))
setattr(struct_CXPlatformAvailability, 'Introduced', field(16, CXVersion))
setattr(struct_CXPlatformAvailability, 'Deprecated', field(28, CXVersion))
setattr(struct_CXPlatformAvailability, 'Obsoleted', field(40, CXVersion))
setattr(struct_CXPlatformAvailability, 'Unavailable', field(52, ctypes.c_int32))
setattr(struct_CXPlatformAvailability, 'Message', field(56, CXString))
CXPlatformAvailability = struct_CXPlatformAvailability
@dll.bind((CXCursor, Pointer(ctypes.c_int32), Pointer(CXString), Pointer(ctypes.c_int32), Pointer(CXString), Pointer(CXPlatformAvailability), ctypes.c_int32), ctypes.c_int32)
def clang_getCursorPlatformAvailability(cursor, always_deprecated, deprecated_message, always_unavailable, unavailable_message, availability, availability_size): ...
@dll.bind((Pointer(CXPlatformAvailability),), None)
def clang_disposeCXPlatformAvailability(availability): ...
@dll.bind((CXCursor,), CXCursor)
def clang_Cursor_getVarDeclInitializer(cursor): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_hasVarDeclGlobalStorage(cursor): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_hasVarDeclExternalStorage(cursor): ...
enum_CXLanguageKind = CEnum(ctypes.c_uint32)
CXLanguage_Invalid = enum_CXLanguageKind.define('CXLanguage_Invalid', 0)
CXLanguage_C = enum_CXLanguageKind.define('CXLanguage_C', 1)
CXLanguage_ObjC = enum_CXLanguageKind.define('CXLanguage_ObjC', 2)
CXLanguage_CPlusPlus = enum_CXLanguageKind.define('CXLanguage_CPlusPlus', 3)

@dll.bind((CXCursor,), enum_CXLanguageKind)
def clang_getCursorLanguage(cursor): ...
enum_CXTLSKind = CEnum(ctypes.c_uint32)
CXTLS_None = enum_CXTLSKind.define('CXTLS_None', 0)
CXTLS_Dynamic = enum_CXTLSKind.define('CXTLS_Dynamic', 1)
CXTLS_Static = enum_CXTLSKind.define('CXTLS_Static', 2)

@dll.bind((CXCursor,), enum_CXTLSKind)
def clang_getCursorTLSKind(cursor): ...
@dll.bind((CXCursor,), CXTranslationUnit)
def clang_Cursor_getTranslationUnit(_0): ...
class struct_CXCursorSetImpl(Struct): pass
CXCursorSet = Pointer(struct_CXCursorSetImpl)
@dll.bind((), CXCursorSet)
def clang_createCXCursorSet(): ...
@dll.bind((CXCursorSet,), None)
def clang_disposeCXCursorSet(cset): ...
@dll.bind((CXCursorSet, CXCursor), ctypes.c_uint32)
def clang_CXCursorSet_contains(cset, cursor): ...
@dll.bind((CXCursorSet, CXCursor), ctypes.c_uint32)
def clang_CXCursorSet_insert(cset, cursor): ...
@dll.bind((CXCursor,), CXCursor)
def clang_getCursorSemanticParent(cursor): ...
@dll.bind((CXCursor,), CXCursor)
def clang_getCursorLexicalParent(cursor): ...
@dll.bind((CXCursor, Pointer(Pointer(CXCursor)), Pointer(ctypes.c_uint32)), None)
def clang_getOverriddenCursors(cursor, overridden, num_overridden): ...
@dll.bind((Pointer(CXCursor),), None)
def clang_disposeOverriddenCursors(overridden): ...
@dll.bind((CXCursor,), CXFile)
def clang_getIncludedFile(cursor): ...
@dll.bind((CXTranslationUnit, CXSourceLocation), CXCursor)
def clang_getCursor(_0, _1): ...
@dll.bind((CXCursor,), CXSourceLocation)
def clang_getCursorLocation(_0): ...
@dll.bind((CXCursor,), CXSourceRange)
def clang_getCursorExtent(_0): ...
enum_CXTypeKind = CEnum(ctypes.c_uint32)
CXType_Invalid = enum_CXTypeKind.define('CXType_Invalid', 0)
CXType_Unexposed = enum_CXTypeKind.define('CXType_Unexposed', 1)
CXType_Void = enum_CXTypeKind.define('CXType_Void', 2)
CXType_Bool = enum_CXTypeKind.define('CXType_Bool', 3)
CXType_Char_U = enum_CXTypeKind.define('CXType_Char_U', 4)
CXType_UChar = enum_CXTypeKind.define('CXType_UChar', 5)
CXType_Char16 = enum_CXTypeKind.define('CXType_Char16', 6)
CXType_Char32 = enum_CXTypeKind.define('CXType_Char32', 7)
CXType_UShort = enum_CXTypeKind.define('CXType_UShort', 8)
CXType_UInt = enum_CXTypeKind.define('CXType_UInt', 9)
CXType_ULong = enum_CXTypeKind.define('CXType_ULong', 10)
CXType_ULongLong = enum_CXTypeKind.define('CXType_ULongLong', 11)
CXType_UInt128 = enum_CXTypeKind.define('CXType_UInt128', 12)
CXType_Char_S = enum_CXTypeKind.define('CXType_Char_S', 13)
CXType_SChar = enum_CXTypeKind.define('CXType_SChar', 14)
CXType_WChar = enum_CXTypeKind.define('CXType_WChar', 15)
CXType_Short = enum_CXTypeKind.define('CXType_Short', 16)
CXType_Int = enum_CXTypeKind.define('CXType_Int', 17)
CXType_Long = enum_CXTypeKind.define('CXType_Long', 18)
CXType_LongLong = enum_CXTypeKind.define('CXType_LongLong', 19)
CXType_Int128 = enum_CXTypeKind.define('CXType_Int128', 20)
CXType_Float = enum_CXTypeKind.define('CXType_Float', 21)
CXType_Double = enum_CXTypeKind.define('CXType_Double', 22)
CXType_LongDouble = enum_CXTypeKind.define('CXType_LongDouble', 23)
CXType_NullPtr = enum_CXTypeKind.define('CXType_NullPtr', 24)
CXType_Overload = enum_CXTypeKind.define('CXType_Overload', 25)
CXType_Dependent = enum_CXTypeKind.define('CXType_Dependent', 26)
CXType_ObjCId = enum_CXTypeKind.define('CXType_ObjCId', 27)
CXType_ObjCClass = enum_CXTypeKind.define('CXType_ObjCClass', 28)
CXType_ObjCSel = enum_CXTypeKind.define('CXType_ObjCSel', 29)
CXType_Float128 = enum_CXTypeKind.define('CXType_Float128', 30)
CXType_Half = enum_CXTypeKind.define('CXType_Half', 31)
CXType_Float16 = enum_CXTypeKind.define('CXType_Float16', 32)
CXType_ShortAccum = enum_CXTypeKind.define('CXType_ShortAccum', 33)
CXType_Accum = enum_CXTypeKind.define('CXType_Accum', 34)
CXType_LongAccum = enum_CXTypeKind.define('CXType_LongAccum', 35)
CXType_UShortAccum = enum_CXTypeKind.define('CXType_UShortAccum', 36)
CXType_UAccum = enum_CXTypeKind.define('CXType_UAccum', 37)
CXType_ULongAccum = enum_CXTypeKind.define('CXType_ULongAccum', 38)
CXType_BFloat16 = enum_CXTypeKind.define('CXType_BFloat16', 39)
CXType_Ibm128 = enum_CXTypeKind.define('CXType_Ibm128', 40)
CXType_FirstBuiltin = enum_CXTypeKind.define('CXType_FirstBuiltin', 2)
CXType_LastBuiltin = enum_CXTypeKind.define('CXType_LastBuiltin', 40)
CXType_Complex = enum_CXTypeKind.define('CXType_Complex', 100)
CXType_Pointer = enum_CXTypeKind.define('CXType_Pointer', 101)
CXType_BlockPointer = enum_CXTypeKind.define('CXType_BlockPointer', 102)
CXType_LValueReference = enum_CXTypeKind.define('CXType_LValueReference', 103)
CXType_RValueReference = enum_CXTypeKind.define('CXType_RValueReference', 104)
CXType_Record = enum_CXTypeKind.define('CXType_Record', 105)
CXType_Enum = enum_CXTypeKind.define('CXType_Enum', 106)
CXType_Typedef = enum_CXTypeKind.define('CXType_Typedef', 107)
CXType_ObjCInterface = enum_CXTypeKind.define('CXType_ObjCInterface', 108)
CXType_ObjCObjectPointer = enum_CXTypeKind.define('CXType_ObjCObjectPointer', 109)
CXType_FunctionNoProto = enum_CXTypeKind.define('CXType_FunctionNoProto', 110)
CXType_FunctionProto = enum_CXTypeKind.define('CXType_FunctionProto', 111)
CXType_ConstantArray = enum_CXTypeKind.define('CXType_ConstantArray', 112)
CXType_Vector = enum_CXTypeKind.define('CXType_Vector', 113)
CXType_IncompleteArray = enum_CXTypeKind.define('CXType_IncompleteArray', 114)
CXType_VariableArray = enum_CXTypeKind.define('CXType_VariableArray', 115)
CXType_DependentSizedArray = enum_CXTypeKind.define('CXType_DependentSizedArray', 116)
CXType_MemberPointer = enum_CXTypeKind.define('CXType_MemberPointer', 117)
CXType_Auto = enum_CXTypeKind.define('CXType_Auto', 118)
CXType_Elaborated = enum_CXTypeKind.define('CXType_Elaborated', 119)
CXType_Pipe = enum_CXTypeKind.define('CXType_Pipe', 120)
CXType_OCLImage1dRO = enum_CXTypeKind.define('CXType_OCLImage1dRO', 121)
CXType_OCLImage1dArrayRO = enum_CXTypeKind.define('CXType_OCLImage1dArrayRO', 122)
CXType_OCLImage1dBufferRO = enum_CXTypeKind.define('CXType_OCLImage1dBufferRO', 123)
CXType_OCLImage2dRO = enum_CXTypeKind.define('CXType_OCLImage2dRO', 124)
CXType_OCLImage2dArrayRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayRO', 125)
CXType_OCLImage2dDepthRO = enum_CXTypeKind.define('CXType_OCLImage2dDepthRO', 126)
CXType_OCLImage2dArrayDepthRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthRO', 127)
CXType_OCLImage2dMSAARO = enum_CXTypeKind.define('CXType_OCLImage2dMSAARO', 128)
CXType_OCLImage2dArrayMSAARO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAARO', 129)
CXType_OCLImage2dMSAADepthRO = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthRO', 130)
CXType_OCLImage2dArrayMSAADepthRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthRO', 131)
CXType_OCLImage3dRO = enum_CXTypeKind.define('CXType_OCLImage3dRO', 132)
CXType_OCLImage1dWO = enum_CXTypeKind.define('CXType_OCLImage1dWO', 133)
CXType_OCLImage1dArrayWO = enum_CXTypeKind.define('CXType_OCLImage1dArrayWO', 134)
CXType_OCLImage1dBufferWO = enum_CXTypeKind.define('CXType_OCLImage1dBufferWO', 135)
CXType_OCLImage2dWO = enum_CXTypeKind.define('CXType_OCLImage2dWO', 136)
CXType_OCLImage2dArrayWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayWO', 137)
CXType_OCLImage2dDepthWO = enum_CXTypeKind.define('CXType_OCLImage2dDepthWO', 138)
CXType_OCLImage2dArrayDepthWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthWO', 139)
CXType_OCLImage2dMSAAWO = enum_CXTypeKind.define('CXType_OCLImage2dMSAAWO', 140)
CXType_OCLImage2dArrayMSAAWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAAWO', 141)
CXType_OCLImage2dMSAADepthWO = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthWO', 142)
CXType_OCLImage2dArrayMSAADepthWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthWO', 143)
CXType_OCLImage3dWO = enum_CXTypeKind.define('CXType_OCLImage3dWO', 144)
CXType_OCLImage1dRW = enum_CXTypeKind.define('CXType_OCLImage1dRW', 145)
CXType_OCLImage1dArrayRW = enum_CXTypeKind.define('CXType_OCLImage1dArrayRW', 146)
CXType_OCLImage1dBufferRW = enum_CXTypeKind.define('CXType_OCLImage1dBufferRW', 147)
CXType_OCLImage2dRW = enum_CXTypeKind.define('CXType_OCLImage2dRW', 148)
CXType_OCLImage2dArrayRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayRW', 149)
CXType_OCLImage2dDepthRW = enum_CXTypeKind.define('CXType_OCLImage2dDepthRW', 150)
CXType_OCLImage2dArrayDepthRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthRW', 151)
CXType_OCLImage2dMSAARW = enum_CXTypeKind.define('CXType_OCLImage2dMSAARW', 152)
CXType_OCLImage2dArrayMSAARW = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAARW', 153)
CXType_OCLImage2dMSAADepthRW = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthRW', 154)
CXType_OCLImage2dArrayMSAADepthRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthRW', 155)
CXType_OCLImage3dRW = enum_CXTypeKind.define('CXType_OCLImage3dRW', 156)
CXType_OCLSampler = enum_CXTypeKind.define('CXType_OCLSampler', 157)
CXType_OCLEvent = enum_CXTypeKind.define('CXType_OCLEvent', 158)
CXType_OCLQueue = enum_CXTypeKind.define('CXType_OCLQueue', 159)
CXType_OCLReserveID = enum_CXTypeKind.define('CXType_OCLReserveID', 160)
CXType_ObjCObject = enum_CXTypeKind.define('CXType_ObjCObject', 161)
CXType_ObjCTypeParam = enum_CXTypeKind.define('CXType_ObjCTypeParam', 162)
CXType_Attributed = enum_CXTypeKind.define('CXType_Attributed', 163)
CXType_OCLIntelSubgroupAVCMcePayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCMcePayload', 164)
CXType_OCLIntelSubgroupAVCImePayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImePayload', 165)
CXType_OCLIntelSubgroupAVCRefPayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCRefPayload', 166)
CXType_OCLIntelSubgroupAVCSicPayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCSicPayload', 167)
CXType_OCLIntelSubgroupAVCMceResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCMceResult', 168)
CXType_OCLIntelSubgroupAVCImeResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResult', 169)
CXType_OCLIntelSubgroupAVCRefResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCRefResult', 170)
CXType_OCLIntelSubgroupAVCSicResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCSicResult', 171)
CXType_OCLIntelSubgroupAVCImeResultSingleReferenceStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultSingleReferenceStreamout', 172)
CXType_OCLIntelSubgroupAVCImeResultDualReferenceStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultDualReferenceStreamout', 173)
CXType_OCLIntelSubgroupAVCImeSingleReferenceStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeSingleReferenceStreamin', 174)
CXType_OCLIntelSubgroupAVCImeDualReferenceStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeDualReferenceStreamin', 175)
CXType_OCLIntelSubgroupAVCImeResultSingleRefStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultSingleRefStreamout', 172)
CXType_OCLIntelSubgroupAVCImeResultDualRefStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultDualRefStreamout', 173)
CXType_OCLIntelSubgroupAVCImeSingleRefStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeSingleRefStreamin', 174)
CXType_OCLIntelSubgroupAVCImeDualRefStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeDualRefStreamin', 175)
CXType_ExtVector = enum_CXTypeKind.define('CXType_ExtVector', 176)
CXType_Atomic = enum_CXTypeKind.define('CXType_Atomic', 177)
CXType_BTFTagAttributed = enum_CXTypeKind.define('CXType_BTFTagAttributed', 178)
CXType_HLSLResource = enum_CXTypeKind.define('CXType_HLSLResource', 179)
CXType_HLSLAttributedResource = enum_CXTypeKind.define('CXType_HLSLAttributedResource', 180)

enum_CXCallingConv = CEnum(ctypes.c_uint32)
CXCallingConv_Default = enum_CXCallingConv.define('CXCallingConv_Default', 0)
CXCallingConv_C = enum_CXCallingConv.define('CXCallingConv_C', 1)
CXCallingConv_X86StdCall = enum_CXCallingConv.define('CXCallingConv_X86StdCall', 2)
CXCallingConv_X86FastCall = enum_CXCallingConv.define('CXCallingConv_X86FastCall', 3)
CXCallingConv_X86ThisCall = enum_CXCallingConv.define('CXCallingConv_X86ThisCall', 4)
CXCallingConv_X86Pascal = enum_CXCallingConv.define('CXCallingConv_X86Pascal', 5)
CXCallingConv_AAPCS = enum_CXCallingConv.define('CXCallingConv_AAPCS', 6)
CXCallingConv_AAPCS_VFP = enum_CXCallingConv.define('CXCallingConv_AAPCS_VFP', 7)
CXCallingConv_X86RegCall = enum_CXCallingConv.define('CXCallingConv_X86RegCall', 8)
CXCallingConv_IntelOclBicc = enum_CXCallingConv.define('CXCallingConv_IntelOclBicc', 9)
CXCallingConv_Win64 = enum_CXCallingConv.define('CXCallingConv_Win64', 10)
CXCallingConv_X86_64Win64 = enum_CXCallingConv.define('CXCallingConv_X86_64Win64', 10)
CXCallingConv_X86_64SysV = enum_CXCallingConv.define('CXCallingConv_X86_64SysV', 11)
CXCallingConv_X86VectorCall = enum_CXCallingConv.define('CXCallingConv_X86VectorCall', 12)
CXCallingConv_Swift = enum_CXCallingConv.define('CXCallingConv_Swift', 13)
CXCallingConv_PreserveMost = enum_CXCallingConv.define('CXCallingConv_PreserveMost', 14)
CXCallingConv_PreserveAll = enum_CXCallingConv.define('CXCallingConv_PreserveAll', 15)
CXCallingConv_AArch64VectorCall = enum_CXCallingConv.define('CXCallingConv_AArch64VectorCall', 16)
CXCallingConv_SwiftAsync = enum_CXCallingConv.define('CXCallingConv_SwiftAsync', 17)
CXCallingConv_AArch64SVEPCS = enum_CXCallingConv.define('CXCallingConv_AArch64SVEPCS', 18)
CXCallingConv_M68kRTD = enum_CXCallingConv.define('CXCallingConv_M68kRTD', 19)
CXCallingConv_PreserveNone = enum_CXCallingConv.define('CXCallingConv_PreserveNone', 20)
CXCallingConv_RISCVVectorCall = enum_CXCallingConv.define('CXCallingConv_RISCVVectorCall', 21)
CXCallingConv_Invalid = enum_CXCallingConv.define('CXCallingConv_Invalid', 100)
CXCallingConv_Unexposed = enum_CXCallingConv.define('CXCallingConv_Unexposed', 200)

class CXType(Struct): pass
CXType.SIZE = 24
CXType._fields_ = ['kind', 'data']
setattr(CXType, 'kind', field(0, enum_CXTypeKind))
setattr(CXType, 'data', field(8, Array(ctypes.c_void_p, 2)))
@dll.bind((CXCursor,), CXType)
def clang_getCursorType(C): ...
@dll.bind((CXType,), CXString)
def clang_getTypeSpelling(CT): ...
@dll.bind((CXCursor,), CXType)
def clang_getTypedefDeclUnderlyingType(C): ...
@dll.bind((CXCursor,), CXType)
def clang_getEnumDeclIntegerType(C): ...
@dll.bind((CXCursor,), ctypes.c_int64)
def clang_getEnumConstantDeclValue(C): ...
@dll.bind((CXCursor,), ctypes.c_uint64)
def clang_getEnumConstantDeclUnsignedValue(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isBitField(C): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_getFieldDeclBitWidth(C): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_getNumArguments(C): ...
@dll.bind((CXCursor, ctypes.c_uint32), CXCursor)
def clang_Cursor_getArgument(C, i): ...
enum_CXTemplateArgumentKind = CEnum(ctypes.c_uint32)
CXTemplateArgumentKind_Null = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Null', 0)
CXTemplateArgumentKind_Type = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Type', 1)
CXTemplateArgumentKind_Declaration = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Declaration', 2)
CXTemplateArgumentKind_NullPtr = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_NullPtr', 3)
CXTemplateArgumentKind_Integral = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Integral', 4)
CXTemplateArgumentKind_Template = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Template', 5)
CXTemplateArgumentKind_TemplateExpansion = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_TemplateExpansion', 6)
CXTemplateArgumentKind_Expression = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Expression', 7)
CXTemplateArgumentKind_Pack = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Pack', 8)
CXTemplateArgumentKind_Invalid = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Invalid', 9)

@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_getNumTemplateArguments(C): ...
@dll.bind((CXCursor, ctypes.c_uint32), enum_CXTemplateArgumentKind)
def clang_Cursor_getTemplateArgumentKind(C, I): ...
@dll.bind((CXCursor, ctypes.c_uint32), CXType)
def clang_Cursor_getTemplateArgumentType(C, I): ...
@dll.bind((CXCursor, ctypes.c_uint32), ctypes.c_int64)
def clang_Cursor_getTemplateArgumentValue(C, I): ...
@dll.bind((CXCursor, ctypes.c_uint32), ctypes.c_uint64)
def clang_Cursor_getTemplateArgumentUnsignedValue(C, I): ...
@dll.bind((CXType, CXType), ctypes.c_uint32)
def clang_equalTypes(A, B): ...
@dll.bind((CXType,), CXType)
def clang_getCanonicalType(T): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_isConstQualifiedType(T): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isMacroFunctionLike(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isMacroBuiltin(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isFunctionInlined(C): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_isVolatileQualifiedType(T): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_isRestrictQualifiedType(T): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_getAddressSpace(T): ...
@dll.bind((CXType,), CXString)
def clang_getTypedefName(CT): ...
@dll.bind((CXType,), CXType)
def clang_getPointeeType(T): ...
@dll.bind((CXType,), CXType)
def clang_getUnqualifiedType(CT): ...
@dll.bind((CXType,), CXType)
def clang_getNonReferenceType(CT): ...
@dll.bind((CXType,), CXCursor)
def clang_getTypeDeclaration(T): ...
@dll.bind((CXCursor,), CXString)
def clang_getDeclObjCTypeEncoding(C): ...
@dll.bind((CXType,), CXString)
def clang_Type_getObjCEncoding(type): ...
@dll.bind((enum_CXTypeKind,), CXString)
def clang_getTypeKindSpelling(K): ...
@dll.bind((CXType,), enum_CXCallingConv)
def clang_getFunctionTypeCallingConv(T): ...
@dll.bind((CXType,), CXType)
def clang_getResultType(T): ...
@dll.bind((CXType,), ctypes.c_int32)
def clang_getExceptionSpecificationType(T): ...
@dll.bind((CXType,), ctypes.c_int32)
def clang_getNumArgTypes(T): ...
@dll.bind((CXType, ctypes.c_uint32), CXType)
def clang_getArgType(T, i): ...
@dll.bind((CXType,), CXType)
def clang_Type_getObjCObjectBaseType(T): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_Type_getNumObjCProtocolRefs(T): ...
@dll.bind((CXType, ctypes.c_uint32), CXCursor)
def clang_Type_getObjCProtocolDecl(T, i): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_Type_getNumObjCTypeArgs(T): ...
@dll.bind((CXType, ctypes.c_uint32), CXType)
def clang_Type_getObjCTypeArg(T, i): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_isFunctionTypeVariadic(T): ...
@dll.bind((CXCursor,), CXType)
def clang_getCursorResultType(C): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_getCursorExceptionSpecificationType(C): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_isPODType(T): ...
@dll.bind((CXType,), CXType)
def clang_getElementType(T): ...
@dll.bind((CXType,), ctypes.c_int64)
def clang_getNumElements(T): ...
@dll.bind((CXType,), CXType)
def clang_getArrayElementType(T): ...
@dll.bind((CXType,), ctypes.c_int64)
def clang_getArraySize(T): ...
@dll.bind((CXType,), CXType)
def clang_Type_getNamedType(T): ...
@dll.bind((CXType,), ctypes.c_uint32)
def clang_Type_isTransparentTagTypedef(T): ...
enum_CXTypeNullabilityKind = CEnum(ctypes.c_uint32)
CXTypeNullability_NonNull = enum_CXTypeNullabilityKind.define('CXTypeNullability_NonNull', 0)
CXTypeNullability_Nullable = enum_CXTypeNullabilityKind.define('CXTypeNullability_Nullable', 1)
CXTypeNullability_Unspecified = enum_CXTypeNullabilityKind.define('CXTypeNullability_Unspecified', 2)
CXTypeNullability_Invalid = enum_CXTypeNullabilityKind.define('CXTypeNullability_Invalid', 3)
CXTypeNullability_NullableResult = enum_CXTypeNullabilityKind.define('CXTypeNullability_NullableResult', 4)

@dll.bind((CXType,), enum_CXTypeNullabilityKind)
def clang_Type_getNullability(T): ...
enum_CXTypeLayoutError = CEnum(ctypes.c_int32)
CXTypeLayoutError_Invalid = enum_CXTypeLayoutError.define('CXTypeLayoutError_Invalid', -1)
CXTypeLayoutError_Incomplete = enum_CXTypeLayoutError.define('CXTypeLayoutError_Incomplete', -2)
CXTypeLayoutError_Dependent = enum_CXTypeLayoutError.define('CXTypeLayoutError_Dependent', -3)
CXTypeLayoutError_NotConstantSize = enum_CXTypeLayoutError.define('CXTypeLayoutError_NotConstantSize', -4)
CXTypeLayoutError_InvalidFieldName = enum_CXTypeLayoutError.define('CXTypeLayoutError_InvalidFieldName', -5)
CXTypeLayoutError_Undeduced = enum_CXTypeLayoutError.define('CXTypeLayoutError_Undeduced', -6)

@dll.bind((CXType,), ctypes.c_int64)
def clang_Type_getAlignOf(T): ...
@dll.bind((CXType,), CXType)
def clang_Type_getClassType(T): ...
@dll.bind((CXType,), ctypes.c_int64)
def clang_Type_getSizeOf(T): ...
@dll.bind((CXType, Pointer(ctypes.c_char)), ctypes.c_int64)
def clang_Type_getOffsetOf(T, S): ...
@dll.bind((CXType,), CXType)
def clang_Type_getModifiedType(T): ...
@dll.bind((CXType,), CXType)
def clang_Type_getValueType(CT): ...
@dll.bind((CXCursor,), ctypes.c_int64)
def clang_Cursor_getOffsetOfField(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isAnonymous(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isAnonymousRecordDecl(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isInlineNamespace(C): ...
enum_CXRefQualifierKind = CEnum(ctypes.c_uint32)
CXRefQualifier_None = enum_CXRefQualifierKind.define('CXRefQualifier_None', 0)
CXRefQualifier_LValue = enum_CXRefQualifierKind.define('CXRefQualifier_LValue', 1)
CXRefQualifier_RValue = enum_CXRefQualifierKind.define('CXRefQualifier_RValue', 2)

@dll.bind((CXType,), ctypes.c_int32)
def clang_Type_getNumTemplateArguments(T): ...
@dll.bind((CXType, ctypes.c_uint32), CXType)
def clang_Type_getTemplateArgumentAsType(T, i): ...
@dll.bind((CXType,), enum_CXRefQualifierKind)
def clang_Type_getCXXRefQualifier(T): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_isVirtualBase(_0): ...
@dll.bind((CXCursor, CXCursor), ctypes.c_int64)
def clang_getOffsetOfBase(Parent, Base): ...
enum_CX_CXXAccessSpecifier = CEnum(ctypes.c_uint32)
CX_CXXInvalidAccessSpecifier = enum_CX_CXXAccessSpecifier.define('CX_CXXInvalidAccessSpecifier', 0)
CX_CXXPublic = enum_CX_CXXAccessSpecifier.define('CX_CXXPublic', 1)
CX_CXXProtected = enum_CX_CXXAccessSpecifier.define('CX_CXXProtected', 2)
CX_CXXPrivate = enum_CX_CXXAccessSpecifier.define('CX_CXXPrivate', 3)

@dll.bind((CXCursor,), enum_CX_CXXAccessSpecifier)
def clang_getCXXAccessSpecifier(_0): ...
enum_CX_StorageClass = CEnum(ctypes.c_uint32)
CX_SC_Invalid = enum_CX_StorageClass.define('CX_SC_Invalid', 0)
CX_SC_None = enum_CX_StorageClass.define('CX_SC_None', 1)
CX_SC_Extern = enum_CX_StorageClass.define('CX_SC_Extern', 2)
CX_SC_Static = enum_CX_StorageClass.define('CX_SC_Static', 3)
CX_SC_PrivateExtern = enum_CX_StorageClass.define('CX_SC_PrivateExtern', 4)
CX_SC_OpenCLWorkGroupLocal = enum_CX_StorageClass.define('CX_SC_OpenCLWorkGroupLocal', 5)
CX_SC_Auto = enum_CX_StorageClass.define('CX_SC_Auto', 6)
CX_SC_Register = enum_CX_StorageClass.define('CX_SC_Register', 7)

enum_CX_BinaryOperatorKind = CEnum(ctypes.c_uint32)
CX_BO_Invalid = enum_CX_BinaryOperatorKind.define('CX_BO_Invalid', 0)
CX_BO_PtrMemD = enum_CX_BinaryOperatorKind.define('CX_BO_PtrMemD', 1)
CX_BO_PtrMemI = enum_CX_BinaryOperatorKind.define('CX_BO_PtrMemI', 2)
CX_BO_Mul = enum_CX_BinaryOperatorKind.define('CX_BO_Mul', 3)
CX_BO_Div = enum_CX_BinaryOperatorKind.define('CX_BO_Div', 4)
CX_BO_Rem = enum_CX_BinaryOperatorKind.define('CX_BO_Rem', 5)
CX_BO_Add = enum_CX_BinaryOperatorKind.define('CX_BO_Add', 6)
CX_BO_Sub = enum_CX_BinaryOperatorKind.define('CX_BO_Sub', 7)
CX_BO_Shl = enum_CX_BinaryOperatorKind.define('CX_BO_Shl', 8)
CX_BO_Shr = enum_CX_BinaryOperatorKind.define('CX_BO_Shr', 9)
CX_BO_Cmp = enum_CX_BinaryOperatorKind.define('CX_BO_Cmp', 10)
CX_BO_LT = enum_CX_BinaryOperatorKind.define('CX_BO_LT', 11)
CX_BO_GT = enum_CX_BinaryOperatorKind.define('CX_BO_GT', 12)
CX_BO_LE = enum_CX_BinaryOperatorKind.define('CX_BO_LE', 13)
CX_BO_GE = enum_CX_BinaryOperatorKind.define('CX_BO_GE', 14)
CX_BO_EQ = enum_CX_BinaryOperatorKind.define('CX_BO_EQ', 15)
CX_BO_NE = enum_CX_BinaryOperatorKind.define('CX_BO_NE', 16)
CX_BO_And = enum_CX_BinaryOperatorKind.define('CX_BO_And', 17)
CX_BO_Xor = enum_CX_BinaryOperatorKind.define('CX_BO_Xor', 18)
CX_BO_Or = enum_CX_BinaryOperatorKind.define('CX_BO_Or', 19)
CX_BO_LAnd = enum_CX_BinaryOperatorKind.define('CX_BO_LAnd', 20)
CX_BO_LOr = enum_CX_BinaryOperatorKind.define('CX_BO_LOr', 21)
CX_BO_Assign = enum_CX_BinaryOperatorKind.define('CX_BO_Assign', 22)
CX_BO_MulAssign = enum_CX_BinaryOperatorKind.define('CX_BO_MulAssign', 23)
CX_BO_DivAssign = enum_CX_BinaryOperatorKind.define('CX_BO_DivAssign', 24)
CX_BO_RemAssign = enum_CX_BinaryOperatorKind.define('CX_BO_RemAssign', 25)
CX_BO_AddAssign = enum_CX_BinaryOperatorKind.define('CX_BO_AddAssign', 26)
CX_BO_SubAssign = enum_CX_BinaryOperatorKind.define('CX_BO_SubAssign', 27)
CX_BO_ShlAssign = enum_CX_BinaryOperatorKind.define('CX_BO_ShlAssign', 28)
CX_BO_ShrAssign = enum_CX_BinaryOperatorKind.define('CX_BO_ShrAssign', 29)
CX_BO_AndAssign = enum_CX_BinaryOperatorKind.define('CX_BO_AndAssign', 30)
CX_BO_XorAssign = enum_CX_BinaryOperatorKind.define('CX_BO_XorAssign', 31)
CX_BO_OrAssign = enum_CX_BinaryOperatorKind.define('CX_BO_OrAssign', 32)
CX_BO_Comma = enum_CX_BinaryOperatorKind.define('CX_BO_Comma', 33)
CX_BO_LAST = enum_CX_BinaryOperatorKind.define('CX_BO_LAST', 33)

@dll.bind((CXCursor,), enum_CX_BinaryOperatorKind)
def clang_Cursor_getBinaryOpcode(C): ...
@dll.bind((enum_CX_BinaryOperatorKind,), CXString)
def clang_Cursor_getBinaryOpcodeStr(Op): ...
@dll.bind((CXCursor,), enum_CX_StorageClass)
def clang_Cursor_getStorageClass(_0): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_getNumOverloadedDecls(cursor): ...
@dll.bind((CXCursor, ctypes.c_uint32), CXCursor)
def clang_getOverloadedDecl(cursor, index): ...
@dll.bind((CXCursor,), CXType)
def clang_getIBOutletCollectionType(_0): ...
enum_CXChildVisitResult = CEnum(ctypes.c_uint32)
CXChildVisit_Break = enum_CXChildVisitResult.define('CXChildVisit_Break', 0)
CXChildVisit_Continue = enum_CXChildVisitResult.define('CXChildVisit_Continue', 1)
CXChildVisit_Recurse = enum_CXChildVisitResult.define('CXChildVisit_Recurse', 2)

CXCursorVisitor = ctypes.CFUNCTYPE(enum_CXChildVisitResult, CXCursor, CXCursor, ctypes.c_void_p)
@dll.bind((CXCursor, CXCursorVisitor, CXClientData), ctypes.c_uint32)
def clang_visitChildren(parent, visitor, client_data): ...
class struct__CXChildVisitResult(Struct): pass
CXCursorVisitorBlock = Pointer(struct__CXChildVisitResult)
@dll.bind((CXCursor, CXCursorVisitorBlock), ctypes.c_uint32)
def clang_visitChildrenWithBlock(parent, block): ...
@dll.bind((CXCursor,), CXString)
def clang_getCursorUSR(_0): ...
@dll.bind((Pointer(ctypes.c_char),), CXString)
def clang_constructUSR_ObjCClass(class_name): ...
@dll.bind((Pointer(ctypes.c_char), Pointer(ctypes.c_char)), CXString)
def clang_constructUSR_ObjCCategory(class_name, category_name): ...
@dll.bind((Pointer(ctypes.c_char),), CXString)
def clang_constructUSR_ObjCProtocol(protocol_name): ...
@dll.bind((Pointer(ctypes.c_char), CXString), CXString)
def clang_constructUSR_ObjCIvar(name, classUSR): ...
@dll.bind((Pointer(ctypes.c_char), ctypes.c_uint32, CXString), CXString)
def clang_constructUSR_ObjCMethod(name, isInstanceMethod, classUSR): ...
@dll.bind((Pointer(ctypes.c_char), CXString), CXString)
def clang_constructUSR_ObjCProperty(property, classUSR): ...
@dll.bind((CXCursor,), CXString)
def clang_getCursorSpelling(_0): ...
@dll.bind((CXCursor, ctypes.c_uint32, ctypes.c_uint32), CXSourceRange)
def clang_Cursor_getSpellingNameRange(_0, pieceIndex, options): ...
CXPrintingPolicy = ctypes.c_void_p
enum_CXPrintingPolicyProperty = CEnum(ctypes.c_uint32)
CXPrintingPolicy_Indentation = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Indentation', 0)
CXPrintingPolicy_SuppressSpecifiers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressSpecifiers', 1)
CXPrintingPolicy_SuppressTagKeyword = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressTagKeyword', 2)
CXPrintingPolicy_IncludeTagDefinition = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_IncludeTagDefinition', 3)
CXPrintingPolicy_SuppressScope = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressScope', 4)
CXPrintingPolicy_SuppressUnwrittenScope = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressUnwrittenScope', 5)
CXPrintingPolicy_SuppressInitializers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressInitializers', 6)
CXPrintingPolicy_ConstantArraySizeAsWritten = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_ConstantArraySizeAsWritten', 7)
CXPrintingPolicy_AnonymousTagLocations = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_AnonymousTagLocations', 8)
CXPrintingPolicy_SuppressStrongLifetime = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressStrongLifetime', 9)
CXPrintingPolicy_SuppressLifetimeQualifiers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressLifetimeQualifiers', 10)
CXPrintingPolicy_SuppressTemplateArgsInCXXConstructors = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressTemplateArgsInCXXConstructors', 11)
CXPrintingPolicy_Bool = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Bool', 12)
CXPrintingPolicy_Restrict = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Restrict', 13)
CXPrintingPolicy_Alignof = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Alignof', 14)
CXPrintingPolicy_UnderscoreAlignof = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_UnderscoreAlignof', 15)
CXPrintingPolicy_UseVoidForZeroParams = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_UseVoidForZeroParams', 16)
CXPrintingPolicy_TerseOutput = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_TerseOutput', 17)
CXPrintingPolicy_PolishForDeclaration = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_PolishForDeclaration', 18)
CXPrintingPolicy_Half = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Half', 19)
CXPrintingPolicy_MSWChar = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_MSWChar', 20)
CXPrintingPolicy_IncludeNewlines = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_IncludeNewlines', 21)
CXPrintingPolicy_MSVCFormatting = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_MSVCFormatting', 22)
CXPrintingPolicy_ConstantsAsWritten = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_ConstantsAsWritten', 23)
CXPrintingPolicy_SuppressImplicitBase = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressImplicitBase', 24)
CXPrintingPolicy_FullyQualifiedName = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_FullyQualifiedName', 25)
CXPrintingPolicy_LastProperty = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_LastProperty', 25)

@dll.bind((CXPrintingPolicy, enum_CXPrintingPolicyProperty), ctypes.c_uint32)
def clang_PrintingPolicy_getProperty(Policy, Property): ...
@dll.bind((CXPrintingPolicy, enum_CXPrintingPolicyProperty, ctypes.c_uint32), None)
def clang_PrintingPolicy_setProperty(Policy, Property, Value): ...
@dll.bind((CXCursor,), CXPrintingPolicy)
def clang_getCursorPrintingPolicy(_0): ...
@dll.bind((CXPrintingPolicy,), None)
def clang_PrintingPolicy_dispose(Policy): ...
@dll.bind((CXCursor, CXPrintingPolicy), CXString)
def clang_getCursorPrettyPrinted(Cursor, Policy): ...
@dll.bind((CXType, CXPrintingPolicy), CXString)
def clang_getTypePrettyPrinted(CT, cxPolicy): ...
@dll.bind((CXCursor,), CXString)
def clang_getCursorDisplayName(_0): ...
@dll.bind((CXCursor,), CXCursor)
def clang_getCursorReferenced(_0): ...
@dll.bind((CXCursor,), CXCursor)
def clang_getCursorDefinition(_0): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_isCursorDefinition(_0): ...
@dll.bind((CXCursor,), CXCursor)
def clang_getCanonicalCursor(_0): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_getObjCSelectorIndex(_0): ...
@dll.bind((CXCursor,), ctypes.c_int32)
def clang_Cursor_isDynamicCall(C): ...
@dll.bind((CXCursor,), CXType)
def clang_Cursor_getReceiverType(C): ...
CXObjCPropertyAttrKind = CEnum(ctypes.c_uint32)
CXObjCPropertyAttr_noattr = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_noattr', 0)
CXObjCPropertyAttr_readonly = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_readonly', 1)
CXObjCPropertyAttr_getter = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_getter', 2)
CXObjCPropertyAttr_assign = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_assign', 4)
CXObjCPropertyAttr_readwrite = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_readwrite', 8)
CXObjCPropertyAttr_retain = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_retain', 16)
CXObjCPropertyAttr_copy = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_copy', 32)
CXObjCPropertyAttr_nonatomic = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_nonatomic', 64)
CXObjCPropertyAttr_setter = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_setter', 128)
CXObjCPropertyAttr_atomic = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_atomic', 256)
CXObjCPropertyAttr_weak = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_weak', 512)
CXObjCPropertyAttr_strong = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_strong', 1024)
CXObjCPropertyAttr_unsafe_unretained = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_unsafe_unretained', 2048)
CXObjCPropertyAttr_class = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_class', 4096)

@dll.bind((CXCursor, ctypes.c_uint32), ctypes.c_uint32)
def clang_Cursor_getObjCPropertyAttributes(C, reserved): ...
@dll.bind((CXCursor,), CXString)
def clang_Cursor_getObjCPropertyGetterName(C): ...
@dll.bind((CXCursor,), CXString)
def clang_Cursor_getObjCPropertySetterName(C): ...
CXObjCDeclQualifierKind = CEnum(ctypes.c_uint32)
CXObjCDeclQualifier_None = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_None', 0)
CXObjCDeclQualifier_In = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_In', 1)
CXObjCDeclQualifier_Inout = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Inout', 2)
CXObjCDeclQualifier_Out = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Out', 4)
CXObjCDeclQualifier_Bycopy = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Bycopy', 8)
CXObjCDeclQualifier_Byref = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Byref', 16)
CXObjCDeclQualifier_Oneway = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Oneway', 32)

@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_getObjCDeclQualifiers(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isObjCOptional(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_Cursor_isVariadic(C): ...
@dll.bind((CXCursor, Pointer(CXString), Pointer(CXString), Pointer(ctypes.c_uint32)), ctypes.c_uint32)
def clang_Cursor_isExternalSymbol(C, language, definedIn, isGenerated): ...
@dll.bind((CXCursor,), CXSourceRange)
def clang_Cursor_getCommentRange(C): ...
@dll.bind((CXCursor,), CXString)
def clang_Cursor_getRawCommentText(C): ...
@dll.bind((CXCursor,), CXString)
def clang_Cursor_getBriefCommentText(C): ...
@dll.bind((CXCursor,), CXString)
def clang_Cursor_getMangling(_0): ...
class CXStringSet(Struct): pass
CXStringSet.SIZE = 16
CXStringSet._fields_ = ['Strings', 'Count']
setattr(CXStringSet, 'Strings', field(0, Pointer(CXString)))
setattr(CXStringSet, 'Count', field(8, ctypes.c_uint32))
@dll.bind((CXCursor,), Pointer(CXStringSet))
def clang_Cursor_getCXXManglings(_0): ...
@dll.bind((CXCursor,), Pointer(CXStringSet))
def clang_Cursor_getObjCManglings(_0): ...
CXModule = ctypes.c_void_p
@dll.bind((CXCursor,), CXModule)
def clang_Cursor_getModule(C): ...
@dll.bind((CXTranslationUnit, CXFile), CXModule)
def clang_getModuleForFile(_0, _1): ...
@dll.bind((CXModule,), CXFile)
def clang_Module_getASTFile(Module): ...
@dll.bind((CXModule,), CXModule)
def clang_Module_getParent(Module): ...
@dll.bind((CXModule,), CXString)
def clang_Module_getName(Module): ...
@dll.bind((CXModule,), CXString)
def clang_Module_getFullName(Module): ...
@dll.bind((CXModule,), ctypes.c_int32)
def clang_Module_isSystem(Module): ...
@dll.bind((CXTranslationUnit, CXModule), ctypes.c_uint32)
def clang_Module_getNumTopLevelHeaders(_0, Module): ...
@dll.bind((CXTranslationUnit, CXModule, ctypes.c_uint32), CXFile)
def clang_Module_getTopLevelHeader(_0, Module, Index): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXConstructor_isConvertingConstructor(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXConstructor_isCopyConstructor(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXConstructor_isDefaultConstructor(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXConstructor_isMoveConstructor(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXField_isMutable(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isDefaulted(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isDeleted(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isPureVirtual(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isStatic(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isVirtual(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isCopyAssignmentOperator(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isMoveAssignmentOperator(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isExplicit(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXRecord_isAbstract(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_EnumDecl_isScoped(C): ...
@dll.bind((CXCursor,), ctypes.c_uint32)
def clang_CXXMethod_isConst(C): ...
@dll.bind((CXCursor,), enum_CXCursorKind)
def clang_getTemplateCursorKind(C): ...
@dll.bind((CXCursor,), CXCursor)
def clang_getSpecializedCursorTemplate(C): ...
@dll.bind((CXCursor, ctypes.c_uint32, ctypes.c_uint32), CXSourceRange)
def clang_getCursorReferenceNameRange(C, NameFlags, PieceIndex): ...
enum_CXNameRefFlags = CEnum(ctypes.c_uint32)
CXNameRange_WantQualifier = enum_CXNameRefFlags.define('CXNameRange_WantQualifier', 1)
CXNameRange_WantTemplateArgs = enum_CXNameRefFlags.define('CXNameRange_WantTemplateArgs', 2)
CXNameRange_WantSinglePiece = enum_CXNameRefFlags.define('CXNameRange_WantSinglePiece', 4)

enum_CXTokenKind = CEnum(ctypes.c_uint32)
CXToken_Punctuation = enum_CXTokenKind.define('CXToken_Punctuation', 0)
CXToken_Keyword = enum_CXTokenKind.define('CXToken_Keyword', 1)
CXToken_Identifier = enum_CXTokenKind.define('CXToken_Identifier', 2)
CXToken_Literal = enum_CXTokenKind.define('CXToken_Literal', 3)
CXToken_Comment = enum_CXTokenKind.define('CXToken_Comment', 4)

CXTokenKind = enum_CXTokenKind
class CXToken(Struct): pass
CXToken.SIZE = 24
CXToken._fields_ = ['int_data', 'ptr_data']
setattr(CXToken, 'int_data', field(0, Array(ctypes.c_uint32, 4)))
setattr(CXToken, 'ptr_data', field(16, ctypes.c_void_p))
@dll.bind((CXTranslationUnit, CXSourceLocation), Pointer(CXToken))
def clang_getToken(TU, Location): ...
@dll.bind((CXToken,), CXTokenKind)
def clang_getTokenKind(_0): ...
@dll.bind((CXTranslationUnit, CXToken), CXString)
def clang_getTokenSpelling(_0, _1): ...
@dll.bind((CXTranslationUnit, CXToken), CXSourceLocation)
def clang_getTokenLocation(_0, _1): ...
@dll.bind((CXTranslationUnit, CXToken), CXSourceRange)
def clang_getTokenExtent(_0, _1): ...
@dll.bind((CXTranslationUnit, CXSourceRange, Pointer(Pointer(CXToken)), Pointer(ctypes.c_uint32)), None)
def clang_tokenize(TU, Range, Tokens, NumTokens): ...
@dll.bind((CXTranslationUnit, Pointer(CXToken), ctypes.c_uint32, Pointer(CXCursor)), None)
def clang_annotateTokens(TU, Tokens, NumTokens, Cursors): ...
@dll.bind((CXTranslationUnit, Pointer(CXToken), ctypes.c_uint32), None)
def clang_disposeTokens(TU, Tokens, NumTokens): ...
@dll.bind((enum_CXCursorKind,), CXString)
def clang_getCursorKindSpelling(Kind): ...
@dll.bind((CXCursor, Pointer(Pointer(ctypes.c_char)), Pointer(Pointer(ctypes.c_char)), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_getDefinitionSpellingAndExtent(_0, startBuf, endBuf, startLine, startColumn, endLine, endColumn): ...
@dll.bind((), None)
def clang_enableStackTraces(): ...
@dll.bind((ctypes.CFUNCTYPE(None, ctypes.c_void_p), ctypes.c_void_p, ctypes.c_uint32), None)
def clang_executeOnThread(fn, user_data, stack_size): ...
CXCompletionString = ctypes.c_void_p
class CXCompletionResult(Struct): pass
CXCompletionResult.SIZE = 16
CXCompletionResult._fields_ = ['CursorKind', 'CompletionString']
setattr(CXCompletionResult, 'CursorKind', field(0, enum_CXCursorKind))
setattr(CXCompletionResult, 'CompletionString', field(8, CXCompletionString))
enum_CXCompletionChunkKind = CEnum(ctypes.c_uint32)
CXCompletionChunk_Optional = enum_CXCompletionChunkKind.define('CXCompletionChunk_Optional', 0)
CXCompletionChunk_TypedText = enum_CXCompletionChunkKind.define('CXCompletionChunk_TypedText', 1)
CXCompletionChunk_Text = enum_CXCompletionChunkKind.define('CXCompletionChunk_Text', 2)
CXCompletionChunk_Placeholder = enum_CXCompletionChunkKind.define('CXCompletionChunk_Placeholder', 3)
CXCompletionChunk_Informative = enum_CXCompletionChunkKind.define('CXCompletionChunk_Informative', 4)
CXCompletionChunk_CurrentParameter = enum_CXCompletionChunkKind.define('CXCompletionChunk_CurrentParameter', 5)
CXCompletionChunk_LeftParen = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftParen', 6)
CXCompletionChunk_RightParen = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightParen', 7)
CXCompletionChunk_LeftBracket = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftBracket', 8)
CXCompletionChunk_RightBracket = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightBracket', 9)
CXCompletionChunk_LeftBrace = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftBrace', 10)
CXCompletionChunk_RightBrace = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightBrace', 11)
CXCompletionChunk_LeftAngle = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftAngle', 12)
CXCompletionChunk_RightAngle = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightAngle', 13)
CXCompletionChunk_Comma = enum_CXCompletionChunkKind.define('CXCompletionChunk_Comma', 14)
CXCompletionChunk_ResultType = enum_CXCompletionChunkKind.define('CXCompletionChunk_ResultType', 15)
CXCompletionChunk_Colon = enum_CXCompletionChunkKind.define('CXCompletionChunk_Colon', 16)
CXCompletionChunk_SemiColon = enum_CXCompletionChunkKind.define('CXCompletionChunk_SemiColon', 17)
CXCompletionChunk_Equal = enum_CXCompletionChunkKind.define('CXCompletionChunk_Equal', 18)
CXCompletionChunk_HorizontalSpace = enum_CXCompletionChunkKind.define('CXCompletionChunk_HorizontalSpace', 19)
CXCompletionChunk_VerticalSpace = enum_CXCompletionChunkKind.define('CXCompletionChunk_VerticalSpace', 20)

@dll.bind((CXCompletionString, ctypes.c_uint32), enum_CXCompletionChunkKind)
def clang_getCompletionChunkKind(completion_string, chunk_number): ...
@dll.bind((CXCompletionString, ctypes.c_uint32), CXString)
def clang_getCompletionChunkText(completion_string, chunk_number): ...
@dll.bind((CXCompletionString, ctypes.c_uint32), CXCompletionString)
def clang_getCompletionChunkCompletionString(completion_string, chunk_number): ...
@dll.bind((CXCompletionString,), ctypes.c_uint32)
def clang_getNumCompletionChunks(completion_string): ...
@dll.bind((CXCompletionString,), ctypes.c_uint32)
def clang_getCompletionPriority(completion_string): ...
@dll.bind((CXCompletionString,), enum_CXAvailabilityKind)
def clang_getCompletionAvailability(completion_string): ...
@dll.bind((CXCompletionString,), ctypes.c_uint32)
def clang_getCompletionNumAnnotations(completion_string): ...
@dll.bind((CXCompletionString, ctypes.c_uint32), CXString)
def clang_getCompletionAnnotation(completion_string, annotation_number): ...
@dll.bind((CXCompletionString, Pointer(enum_CXCursorKind)), CXString)
def clang_getCompletionParent(completion_string, kind): ...
@dll.bind((CXCompletionString,), CXString)
def clang_getCompletionBriefComment(completion_string): ...
@dll.bind((CXCursor,), CXCompletionString)
def clang_getCursorCompletionString(cursor): ...
class CXCodeCompleteResults(Struct): pass
CXCodeCompleteResults.SIZE = 16
CXCodeCompleteResults._fields_ = ['Results', 'NumResults']
setattr(CXCodeCompleteResults, 'Results', field(0, Pointer(CXCompletionResult)))
setattr(CXCodeCompleteResults, 'NumResults', field(8, ctypes.c_uint32))
@dll.bind((Pointer(CXCodeCompleteResults), ctypes.c_uint32), ctypes.c_uint32)
def clang_getCompletionNumFixIts(results, completion_index): ...
@dll.bind((Pointer(CXCodeCompleteResults), ctypes.c_uint32, ctypes.c_uint32, Pointer(CXSourceRange)), CXString)
def clang_getCompletionFixIt(results, completion_index, fixit_index, replacement_range): ...
enum_CXCodeComplete_Flags = CEnum(ctypes.c_uint32)
CXCodeComplete_IncludeMacros = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeMacros', 1)
CXCodeComplete_IncludeCodePatterns = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCodePatterns', 2)
CXCodeComplete_IncludeBriefComments = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeBriefComments', 4)
CXCodeComplete_SkipPreamble = enum_CXCodeComplete_Flags.define('CXCodeComplete_SkipPreamble', 8)
CXCodeComplete_IncludeCompletionsWithFixIts = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCompletionsWithFixIts', 16)

enum_CXCompletionContext = CEnum(ctypes.c_uint32)
CXCompletionContext_Unexposed = enum_CXCompletionContext.define('CXCompletionContext_Unexposed', 0)
CXCompletionContext_AnyType = enum_CXCompletionContext.define('CXCompletionContext_AnyType', 1)
CXCompletionContext_AnyValue = enum_CXCompletionContext.define('CXCompletionContext_AnyValue', 2)
CXCompletionContext_ObjCObjectValue = enum_CXCompletionContext.define('CXCompletionContext_ObjCObjectValue', 4)
CXCompletionContext_ObjCSelectorValue = enum_CXCompletionContext.define('CXCompletionContext_ObjCSelectorValue', 8)
CXCompletionContext_CXXClassTypeValue = enum_CXCompletionContext.define('CXCompletionContext_CXXClassTypeValue', 16)
CXCompletionContext_DotMemberAccess = enum_CXCompletionContext.define('CXCompletionContext_DotMemberAccess', 32)
CXCompletionContext_ArrowMemberAccess = enum_CXCompletionContext.define('CXCompletionContext_ArrowMemberAccess', 64)
CXCompletionContext_ObjCPropertyAccess = enum_CXCompletionContext.define('CXCompletionContext_ObjCPropertyAccess', 128)
CXCompletionContext_EnumTag = enum_CXCompletionContext.define('CXCompletionContext_EnumTag', 256)
CXCompletionContext_UnionTag = enum_CXCompletionContext.define('CXCompletionContext_UnionTag', 512)
CXCompletionContext_StructTag = enum_CXCompletionContext.define('CXCompletionContext_StructTag', 1024)
CXCompletionContext_ClassTag = enum_CXCompletionContext.define('CXCompletionContext_ClassTag', 2048)
CXCompletionContext_Namespace = enum_CXCompletionContext.define('CXCompletionContext_Namespace', 4096)
CXCompletionContext_NestedNameSpecifier = enum_CXCompletionContext.define('CXCompletionContext_NestedNameSpecifier', 8192)
CXCompletionContext_ObjCInterface = enum_CXCompletionContext.define('CXCompletionContext_ObjCInterface', 16384)
CXCompletionContext_ObjCProtocol = enum_CXCompletionContext.define('CXCompletionContext_ObjCProtocol', 32768)
CXCompletionContext_ObjCCategory = enum_CXCompletionContext.define('CXCompletionContext_ObjCCategory', 65536)
CXCompletionContext_ObjCInstanceMessage = enum_CXCompletionContext.define('CXCompletionContext_ObjCInstanceMessage', 131072)
CXCompletionContext_ObjCClassMessage = enum_CXCompletionContext.define('CXCompletionContext_ObjCClassMessage', 262144)
CXCompletionContext_ObjCSelectorName = enum_CXCompletionContext.define('CXCompletionContext_ObjCSelectorName', 524288)
CXCompletionContext_MacroName = enum_CXCompletionContext.define('CXCompletionContext_MacroName', 1048576)
CXCompletionContext_NaturalLanguage = enum_CXCompletionContext.define('CXCompletionContext_NaturalLanguage', 2097152)
CXCompletionContext_IncludedFile = enum_CXCompletionContext.define('CXCompletionContext_IncludedFile', 4194304)
CXCompletionContext_Unknown = enum_CXCompletionContext.define('CXCompletionContext_Unknown', 8388607)

@dll.bind((), ctypes.c_uint32)
def clang_defaultCodeCompleteOptions(): ...
@dll.bind((CXTranslationUnit, Pointer(ctypes.c_char), ctypes.c_uint32, ctypes.c_uint32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32, ctypes.c_uint32), Pointer(CXCodeCompleteResults))
def clang_codeCompleteAt(TU, complete_filename, complete_line, complete_column, unsaved_files, num_unsaved_files, options): ...
@dll.bind((Pointer(CXCompletionResult), ctypes.c_uint32), None)
def clang_sortCodeCompletionResults(Results, NumResults): ...
@dll.bind((Pointer(CXCodeCompleteResults),), None)
def clang_disposeCodeCompleteResults(Results): ...
@dll.bind((Pointer(CXCodeCompleteResults),), ctypes.c_uint32)
def clang_codeCompleteGetNumDiagnostics(Results): ...
@dll.bind((Pointer(CXCodeCompleteResults), ctypes.c_uint32), CXDiagnostic)
def clang_codeCompleteGetDiagnostic(Results, Index): ...
@dll.bind((Pointer(CXCodeCompleteResults),), ctypes.c_uint64)
def clang_codeCompleteGetContexts(Results): ...
@dll.bind((Pointer(CXCodeCompleteResults), Pointer(ctypes.c_uint32)), enum_CXCursorKind)
def clang_codeCompleteGetContainerKind(Results, IsIncomplete): ...
@dll.bind((Pointer(CXCodeCompleteResults),), CXString)
def clang_codeCompleteGetContainerUSR(Results): ...
@dll.bind((Pointer(CXCodeCompleteResults),), CXString)
def clang_codeCompleteGetObjCSelector(Results): ...
@dll.bind((), CXString)
def clang_getClangVersion(): ...
@dll.bind((ctypes.c_uint32,), None)
def clang_toggleCrashRecovery(isEnabled): ...
CXInclusionVisitor = ctypes.CFUNCTYPE(None, ctypes.c_void_p, Pointer(CXSourceLocation), ctypes.c_uint32, ctypes.c_void_p)
@dll.bind((CXTranslationUnit, CXInclusionVisitor, CXClientData), None)
def clang_getInclusions(tu, visitor, client_data): ...
CXEvalResultKind = CEnum(ctypes.c_uint32)
CXEval_Int = CXEvalResultKind.define('CXEval_Int', 1)
CXEval_Float = CXEvalResultKind.define('CXEval_Float', 2)
CXEval_ObjCStrLiteral = CXEvalResultKind.define('CXEval_ObjCStrLiteral', 3)
CXEval_StrLiteral = CXEvalResultKind.define('CXEval_StrLiteral', 4)
CXEval_CFStr = CXEvalResultKind.define('CXEval_CFStr', 5)
CXEval_Other = CXEvalResultKind.define('CXEval_Other', 6)
CXEval_UnExposed = CXEvalResultKind.define('CXEval_UnExposed', 0)

CXEvalResult = ctypes.c_void_p
@dll.bind((CXCursor,), CXEvalResult)
def clang_Cursor_Evaluate(C): ...
@dll.bind((CXEvalResult,), CXEvalResultKind)
def clang_EvalResult_getKind(E): ...
@dll.bind((CXEvalResult,), ctypes.c_int32)
def clang_EvalResult_getAsInt(E): ...
@dll.bind((CXEvalResult,), ctypes.c_int64)
def clang_EvalResult_getAsLongLong(E): ...
@dll.bind((CXEvalResult,), ctypes.c_uint32)
def clang_EvalResult_isUnsignedInt(E): ...
@dll.bind((CXEvalResult,), ctypes.c_uint64)
def clang_EvalResult_getAsUnsigned(E): ...
@dll.bind((CXEvalResult,), ctypes.c_double)
def clang_EvalResult_getAsDouble(E): ...
@dll.bind((CXEvalResult,), Pointer(ctypes.c_char))
def clang_EvalResult_getAsStr(E): ...
@dll.bind((CXEvalResult,), None)
def clang_EvalResult_dispose(E): ...
CXRemapping = ctypes.c_void_p
@dll.bind((Pointer(ctypes.c_char),), CXRemapping)
def clang_getRemappings(path): ...
@dll.bind((Pointer(Pointer(ctypes.c_char)), ctypes.c_uint32), CXRemapping)
def clang_getRemappingsFromFileList(filePaths, numFiles): ...
@dll.bind((CXRemapping,), ctypes.c_uint32)
def clang_remap_getNumFiles(_0): ...
@dll.bind((CXRemapping, ctypes.c_uint32, Pointer(CXString), Pointer(CXString)), None)
def clang_remap_getFilenames(_0, index, original, transformed): ...
@dll.bind((CXRemapping,), None)
def clang_remap_dispose(_0): ...
enum_CXVisitorResult = CEnum(ctypes.c_uint32)
CXVisit_Break = enum_CXVisitorResult.define('CXVisit_Break', 0)
CXVisit_Continue = enum_CXVisitorResult.define('CXVisit_Continue', 1)

class struct_CXCursorAndRangeVisitor(Struct): pass
struct_CXCursorAndRangeVisitor.SIZE = 16
struct_CXCursorAndRangeVisitor._fields_ = ['context', 'visit']
setattr(struct_CXCursorAndRangeVisitor, 'context', field(0, ctypes.c_void_p))
setattr(struct_CXCursorAndRangeVisitor, 'visit', field(8, ctypes.CFUNCTYPE(enum_CXVisitorResult, ctypes.c_void_p, CXCursor, CXSourceRange)))
CXCursorAndRangeVisitor = struct_CXCursorAndRangeVisitor
CXResult = CEnum(ctypes.c_uint32)
CXResult_Success = CXResult.define('CXResult_Success', 0)
CXResult_Invalid = CXResult.define('CXResult_Invalid', 1)
CXResult_VisitBreak = CXResult.define('CXResult_VisitBreak', 2)

@dll.bind((CXCursor, CXFile, CXCursorAndRangeVisitor), CXResult)
def clang_findReferencesInFile(cursor, file, visitor): ...
@dll.bind((CXTranslationUnit, CXFile, CXCursorAndRangeVisitor), CXResult)
def clang_findIncludesInFile(TU, file, visitor): ...
class struct__CXCursorAndRangeVisitorBlock(Struct): pass
CXCursorAndRangeVisitorBlock = Pointer(struct__CXCursorAndRangeVisitorBlock)
@dll.bind((CXCursor, CXFile, CXCursorAndRangeVisitorBlock), CXResult)
def clang_findReferencesInFileWithBlock(_0, _1, _2): ...
@dll.bind((CXTranslationUnit, CXFile, CXCursorAndRangeVisitorBlock), CXResult)
def clang_findIncludesInFileWithBlock(_0, _1, _2): ...
CXIdxClientFile = ctypes.c_void_p
CXIdxClientEntity = ctypes.c_void_p
CXIdxClientContainer = ctypes.c_void_p
CXIdxClientASTFile = ctypes.c_void_p
class CXIdxLoc(Struct): pass
CXIdxLoc.SIZE = 24
CXIdxLoc._fields_ = ['ptr_data', 'int_data']
setattr(CXIdxLoc, 'ptr_data', field(0, Array(ctypes.c_void_p, 2)))
setattr(CXIdxLoc, 'int_data', field(16, ctypes.c_uint32))
class CXIdxIncludedFileInfo(Struct): pass
CXIdxIncludedFileInfo.SIZE = 56
CXIdxIncludedFileInfo._fields_ = ['hashLoc', 'filename', 'file', 'isImport', 'isAngled', 'isModuleImport']
setattr(CXIdxIncludedFileInfo, 'hashLoc', field(0, CXIdxLoc))
setattr(CXIdxIncludedFileInfo, 'filename', field(24, Pointer(ctypes.c_char)))
setattr(CXIdxIncludedFileInfo, 'file', field(32, CXFile))
setattr(CXIdxIncludedFileInfo, 'isImport', field(40, ctypes.c_int32))
setattr(CXIdxIncludedFileInfo, 'isAngled', field(44, ctypes.c_int32))
setattr(CXIdxIncludedFileInfo, 'isModuleImport', field(48, ctypes.c_int32))
class CXIdxImportedASTFileInfo(Struct): pass
CXIdxImportedASTFileInfo.SIZE = 48
CXIdxImportedASTFileInfo._fields_ = ['file', 'module', 'loc', 'isImplicit']
setattr(CXIdxImportedASTFileInfo, 'file', field(0, CXFile))
setattr(CXIdxImportedASTFileInfo, 'module', field(8, CXModule))
setattr(CXIdxImportedASTFileInfo, 'loc', field(16, CXIdxLoc))
setattr(CXIdxImportedASTFileInfo, 'isImplicit', field(40, ctypes.c_int32))
CXIdxEntityKind = CEnum(ctypes.c_uint32)
CXIdxEntity_Unexposed = CXIdxEntityKind.define('CXIdxEntity_Unexposed', 0)
CXIdxEntity_Typedef = CXIdxEntityKind.define('CXIdxEntity_Typedef', 1)
CXIdxEntity_Function = CXIdxEntityKind.define('CXIdxEntity_Function', 2)
CXIdxEntity_Variable = CXIdxEntityKind.define('CXIdxEntity_Variable', 3)
CXIdxEntity_Field = CXIdxEntityKind.define('CXIdxEntity_Field', 4)
CXIdxEntity_EnumConstant = CXIdxEntityKind.define('CXIdxEntity_EnumConstant', 5)
CXIdxEntity_ObjCClass = CXIdxEntityKind.define('CXIdxEntity_ObjCClass', 6)
CXIdxEntity_ObjCProtocol = CXIdxEntityKind.define('CXIdxEntity_ObjCProtocol', 7)
CXIdxEntity_ObjCCategory = CXIdxEntityKind.define('CXIdxEntity_ObjCCategory', 8)
CXIdxEntity_ObjCInstanceMethod = CXIdxEntityKind.define('CXIdxEntity_ObjCInstanceMethod', 9)
CXIdxEntity_ObjCClassMethod = CXIdxEntityKind.define('CXIdxEntity_ObjCClassMethod', 10)
CXIdxEntity_ObjCProperty = CXIdxEntityKind.define('CXIdxEntity_ObjCProperty', 11)
CXIdxEntity_ObjCIvar = CXIdxEntityKind.define('CXIdxEntity_ObjCIvar', 12)
CXIdxEntity_Enum = CXIdxEntityKind.define('CXIdxEntity_Enum', 13)
CXIdxEntity_Struct = CXIdxEntityKind.define('CXIdxEntity_Struct', 14)
CXIdxEntity_Union = CXIdxEntityKind.define('CXIdxEntity_Union', 15)
CXIdxEntity_CXXClass = CXIdxEntityKind.define('CXIdxEntity_CXXClass', 16)
CXIdxEntity_CXXNamespace = CXIdxEntityKind.define('CXIdxEntity_CXXNamespace', 17)
CXIdxEntity_CXXNamespaceAlias = CXIdxEntityKind.define('CXIdxEntity_CXXNamespaceAlias', 18)
CXIdxEntity_CXXStaticVariable = CXIdxEntityKind.define('CXIdxEntity_CXXStaticVariable', 19)
CXIdxEntity_CXXStaticMethod = CXIdxEntityKind.define('CXIdxEntity_CXXStaticMethod', 20)
CXIdxEntity_CXXInstanceMethod = CXIdxEntityKind.define('CXIdxEntity_CXXInstanceMethod', 21)
CXIdxEntity_CXXConstructor = CXIdxEntityKind.define('CXIdxEntity_CXXConstructor', 22)
CXIdxEntity_CXXDestructor = CXIdxEntityKind.define('CXIdxEntity_CXXDestructor', 23)
CXIdxEntity_CXXConversionFunction = CXIdxEntityKind.define('CXIdxEntity_CXXConversionFunction', 24)
CXIdxEntity_CXXTypeAlias = CXIdxEntityKind.define('CXIdxEntity_CXXTypeAlias', 25)
CXIdxEntity_CXXInterface = CXIdxEntityKind.define('CXIdxEntity_CXXInterface', 26)
CXIdxEntity_CXXConcept = CXIdxEntityKind.define('CXIdxEntity_CXXConcept', 27)

CXIdxEntityLanguage = CEnum(ctypes.c_uint32)
CXIdxEntityLang_None = CXIdxEntityLanguage.define('CXIdxEntityLang_None', 0)
CXIdxEntityLang_C = CXIdxEntityLanguage.define('CXIdxEntityLang_C', 1)
CXIdxEntityLang_ObjC = CXIdxEntityLanguage.define('CXIdxEntityLang_ObjC', 2)
CXIdxEntityLang_CXX = CXIdxEntityLanguage.define('CXIdxEntityLang_CXX', 3)
CXIdxEntityLang_Swift = CXIdxEntityLanguage.define('CXIdxEntityLang_Swift', 4)

CXIdxEntityCXXTemplateKind = CEnum(ctypes.c_uint32)
CXIdxEntity_NonTemplate = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_NonTemplate', 0)
CXIdxEntity_Template = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_Template', 1)
CXIdxEntity_TemplatePartialSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplatePartialSpecialization', 2)
CXIdxEntity_TemplateSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplateSpecialization', 3)

CXIdxAttrKind = CEnum(ctypes.c_uint32)
CXIdxAttr_Unexposed = CXIdxAttrKind.define('CXIdxAttr_Unexposed', 0)
CXIdxAttr_IBAction = CXIdxAttrKind.define('CXIdxAttr_IBAction', 1)
CXIdxAttr_IBOutlet = CXIdxAttrKind.define('CXIdxAttr_IBOutlet', 2)
CXIdxAttr_IBOutletCollection = CXIdxAttrKind.define('CXIdxAttr_IBOutletCollection', 3)

class CXIdxAttrInfo(Struct): pass
CXIdxAttrInfo.SIZE = 64
CXIdxAttrInfo._fields_ = ['kind', 'cursor', 'loc']
setattr(CXIdxAttrInfo, 'kind', field(0, CXIdxAttrKind))
setattr(CXIdxAttrInfo, 'cursor', field(8, CXCursor))
setattr(CXIdxAttrInfo, 'loc', field(40, CXIdxLoc))
class CXIdxEntityInfo(Struct): pass
CXIdxEntityInfo.SIZE = 80
CXIdxEntityInfo._fields_ = ['kind', 'templateKind', 'lang', 'name', 'USR', 'cursor', 'attributes', 'numAttributes']
setattr(CXIdxEntityInfo, 'kind', field(0, CXIdxEntityKind))
setattr(CXIdxEntityInfo, 'templateKind', field(4, CXIdxEntityCXXTemplateKind))
setattr(CXIdxEntityInfo, 'lang', field(8, CXIdxEntityLanguage))
setattr(CXIdxEntityInfo, 'name', field(16, Pointer(ctypes.c_char)))
setattr(CXIdxEntityInfo, 'USR', field(24, Pointer(ctypes.c_char)))
setattr(CXIdxEntityInfo, 'cursor', field(32, CXCursor))
setattr(CXIdxEntityInfo, 'attributes', field(64, Pointer(Pointer(CXIdxAttrInfo))))
setattr(CXIdxEntityInfo, 'numAttributes', field(72, ctypes.c_uint32))
class CXIdxContainerInfo(Struct): pass
CXIdxContainerInfo.SIZE = 32
CXIdxContainerInfo._fields_ = ['cursor']
setattr(CXIdxContainerInfo, 'cursor', field(0, CXCursor))
class CXIdxIBOutletCollectionAttrInfo(Struct): pass
CXIdxIBOutletCollectionAttrInfo.SIZE = 72
CXIdxIBOutletCollectionAttrInfo._fields_ = ['attrInfo', 'objcClass', 'classCursor', 'classLoc']
setattr(CXIdxIBOutletCollectionAttrInfo, 'attrInfo', field(0, Pointer(CXIdxAttrInfo)))
setattr(CXIdxIBOutletCollectionAttrInfo, 'objcClass', field(8, Pointer(CXIdxEntityInfo)))
setattr(CXIdxIBOutletCollectionAttrInfo, 'classCursor', field(16, CXCursor))
setattr(CXIdxIBOutletCollectionAttrInfo, 'classLoc', field(48, CXIdxLoc))
CXIdxDeclInfoFlags = CEnum(ctypes.c_uint32)
CXIdxDeclFlag_Skipped = CXIdxDeclInfoFlags.define('CXIdxDeclFlag_Skipped', 1)

class CXIdxDeclInfo(Struct): pass
CXIdxDeclInfo.SIZE = 128
CXIdxDeclInfo._fields_ = ['entityInfo', 'cursor', 'loc', 'semanticContainer', 'lexicalContainer', 'isRedeclaration', 'isDefinition', 'isContainer', 'declAsContainer', 'isImplicit', 'attributes', 'numAttributes', 'flags']
setattr(CXIdxDeclInfo, 'entityInfo', field(0, Pointer(CXIdxEntityInfo)))
setattr(CXIdxDeclInfo, 'cursor', field(8, CXCursor))
setattr(CXIdxDeclInfo, 'loc', field(40, CXIdxLoc))
setattr(CXIdxDeclInfo, 'semanticContainer', field(64, Pointer(CXIdxContainerInfo)))
setattr(CXIdxDeclInfo, 'lexicalContainer', field(72, Pointer(CXIdxContainerInfo)))
setattr(CXIdxDeclInfo, 'isRedeclaration', field(80, ctypes.c_int32))
setattr(CXIdxDeclInfo, 'isDefinition', field(84, ctypes.c_int32))
setattr(CXIdxDeclInfo, 'isContainer', field(88, ctypes.c_int32))
setattr(CXIdxDeclInfo, 'declAsContainer', field(96, Pointer(CXIdxContainerInfo)))
setattr(CXIdxDeclInfo, 'isImplicit', field(104, ctypes.c_int32))
setattr(CXIdxDeclInfo, 'attributes', field(112, Pointer(Pointer(CXIdxAttrInfo))))
setattr(CXIdxDeclInfo, 'numAttributes', field(120, ctypes.c_uint32))
setattr(CXIdxDeclInfo, 'flags', field(124, ctypes.c_uint32))
CXIdxObjCContainerKind = CEnum(ctypes.c_uint32)
CXIdxObjCContainer_ForwardRef = CXIdxObjCContainerKind.define('CXIdxObjCContainer_ForwardRef', 0)
CXIdxObjCContainer_Interface = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Interface', 1)
CXIdxObjCContainer_Implementation = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Implementation', 2)

class CXIdxObjCContainerDeclInfo(Struct): pass
CXIdxObjCContainerDeclInfo.SIZE = 16
CXIdxObjCContainerDeclInfo._fields_ = ['declInfo', 'kind']
setattr(CXIdxObjCContainerDeclInfo, 'declInfo', field(0, Pointer(CXIdxDeclInfo)))
setattr(CXIdxObjCContainerDeclInfo, 'kind', field(8, CXIdxObjCContainerKind))
class CXIdxBaseClassInfo(Struct): pass
CXIdxBaseClassInfo.SIZE = 64
CXIdxBaseClassInfo._fields_ = ['base', 'cursor', 'loc']
setattr(CXIdxBaseClassInfo, 'base', field(0, Pointer(CXIdxEntityInfo)))
setattr(CXIdxBaseClassInfo, 'cursor', field(8, CXCursor))
setattr(CXIdxBaseClassInfo, 'loc', field(40, CXIdxLoc))
class CXIdxObjCProtocolRefInfo(Struct): pass
CXIdxObjCProtocolRefInfo.SIZE = 64
CXIdxObjCProtocolRefInfo._fields_ = ['protocol', 'cursor', 'loc']
setattr(CXIdxObjCProtocolRefInfo, 'protocol', field(0, Pointer(CXIdxEntityInfo)))
setattr(CXIdxObjCProtocolRefInfo, 'cursor', field(8, CXCursor))
setattr(CXIdxObjCProtocolRefInfo, 'loc', field(40, CXIdxLoc))
class CXIdxObjCProtocolRefListInfo(Struct): pass
CXIdxObjCProtocolRefListInfo.SIZE = 16
CXIdxObjCProtocolRefListInfo._fields_ = ['protocols', 'numProtocols']
setattr(CXIdxObjCProtocolRefListInfo, 'protocols', field(0, Pointer(Pointer(CXIdxObjCProtocolRefInfo))))
setattr(CXIdxObjCProtocolRefListInfo, 'numProtocols', field(8, ctypes.c_uint32))
class CXIdxObjCInterfaceDeclInfo(Struct): pass
CXIdxObjCInterfaceDeclInfo.SIZE = 24
CXIdxObjCInterfaceDeclInfo._fields_ = ['containerInfo', 'superInfo', 'protocols']
setattr(CXIdxObjCInterfaceDeclInfo, 'containerInfo', field(0, Pointer(CXIdxObjCContainerDeclInfo)))
setattr(CXIdxObjCInterfaceDeclInfo, 'superInfo', field(8, Pointer(CXIdxBaseClassInfo)))
setattr(CXIdxObjCInterfaceDeclInfo, 'protocols', field(16, Pointer(CXIdxObjCProtocolRefListInfo)))
class CXIdxObjCCategoryDeclInfo(Struct): pass
CXIdxObjCCategoryDeclInfo.SIZE = 80
CXIdxObjCCategoryDeclInfo._fields_ = ['containerInfo', 'objcClass', 'classCursor', 'classLoc', 'protocols']
setattr(CXIdxObjCCategoryDeclInfo, 'containerInfo', field(0, Pointer(CXIdxObjCContainerDeclInfo)))
setattr(CXIdxObjCCategoryDeclInfo, 'objcClass', field(8, Pointer(CXIdxEntityInfo)))
setattr(CXIdxObjCCategoryDeclInfo, 'classCursor', field(16, CXCursor))
setattr(CXIdxObjCCategoryDeclInfo, 'classLoc', field(48, CXIdxLoc))
setattr(CXIdxObjCCategoryDeclInfo, 'protocols', field(72, Pointer(CXIdxObjCProtocolRefListInfo)))
class CXIdxObjCPropertyDeclInfo(Struct): pass
CXIdxObjCPropertyDeclInfo.SIZE = 24
CXIdxObjCPropertyDeclInfo._fields_ = ['declInfo', 'getter', 'setter']
setattr(CXIdxObjCPropertyDeclInfo, 'declInfo', field(0, Pointer(CXIdxDeclInfo)))
setattr(CXIdxObjCPropertyDeclInfo, 'getter', field(8, Pointer(CXIdxEntityInfo)))
setattr(CXIdxObjCPropertyDeclInfo, 'setter', field(16, Pointer(CXIdxEntityInfo)))
class CXIdxCXXClassDeclInfo(Struct): pass
CXIdxCXXClassDeclInfo.SIZE = 24
CXIdxCXXClassDeclInfo._fields_ = ['declInfo', 'bases', 'numBases']
setattr(CXIdxCXXClassDeclInfo, 'declInfo', field(0, Pointer(CXIdxDeclInfo)))
setattr(CXIdxCXXClassDeclInfo, 'bases', field(8, Pointer(Pointer(CXIdxBaseClassInfo))))
setattr(CXIdxCXXClassDeclInfo, 'numBases', field(16, ctypes.c_uint32))
CXIdxEntityRefKind = CEnum(ctypes.c_uint32)
CXIdxEntityRef_Direct = CXIdxEntityRefKind.define('CXIdxEntityRef_Direct', 1)
CXIdxEntityRef_Implicit = CXIdxEntityRefKind.define('CXIdxEntityRef_Implicit', 2)

CXSymbolRole = CEnum(ctypes.c_uint32)
CXSymbolRole_None = CXSymbolRole.define('CXSymbolRole_None', 0)
CXSymbolRole_Declaration = CXSymbolRole.define('CXSymbolRole_Declaration', 1)
CXSymbolRole_Definition = CXSymbolRole.define('CXSymbolRole_Definition', 2)
CXSymbolRole_Reference = CXSymbolRole.define('CXSymbolRole_Reference', 4)
CXSymbolRole_Read = CXSymbolRole.define('CXSymbolRole_Read', 8)
CXSymbolRole_Write = CXSymbolRole.define('CXSymbolRole_Write', 16)
CXSymbolRole_Call = CXSymbolRole.define('CXSymbolRole_Call', 32)
CXSymbolRole_Dynamic = CXSymbolRole.define('CXSymbolRole_Dynamic', 64)
CXSymbolRole_AddressOf = CXSymbolRole.define('CXSymbolRole_AddressOf', 128)
CXSymbolRole_Implicit = CXSymbolRole.define('CXSymbolRole_Implicit', 256)

class CXIdxEntityRefInfo(Struct): pass
CXIdxEntityRefInfo.SIZE = 96
CXIdxEntityRefInfo._fields_ = ['kind', 'cursor', 'loc', 'referencedEntity', 'parentEntity', 'container', 'role']
setattr(CXIdxEntityRefInfo, 'kind', field(0, CXIdxEntityRefKind))
setattr(CXIdxEntityRefInfo, 'cursor', field(8, CXCursor))
setattr(CXIdxEntityRefInfo, 'loc', field(40, CXIdxLoc))
setattr(CXIdxEntityRefInfo, 'referencedEntity', field(64, Pointer(CXIdxEntityInfo)))
setattr(CXIdxEntityRefInfo, 'parentEntity', field(72, Pointer(CXIdxEntityInfo)))
setattr(CXIdxEntityRefInfo, 'container', field(80, Pointer(CXIdxContainerInfo)))
setattr(CXIdxEntityRefInfo, 'role', field(88, CXSymbolRole))
class IndexerCallbacks(Struct): pass
IndexerCallbacks.SIZE = 64
IndexerCallbacks._fields_ = ['abortQuery', 'diagnostic', 'enteredMainFile', 'ppIncludedFile', 'importedASTFile', 'startedTranslationUnit', 'indexDeclaration', 'indexEntityReference']
setattr(IndexerCallbacks, 'abortQuery', field(0, ctypes.CFUNCTYPE(ctypes.c_int32, CXClientData, ctypes.c_void_p)))
setattr(IndexerCallbacks, 'diagnostic', field(8, ctypes.CFUNCTYPE(None, CXClientData, CXDiagnosticSet, ctypes.c_void_p)))
setattr(IndexerCallbacks, 'enteredMainFile', field(16, ctypes.CFUNCTYPE(CXIdxClientFile, CXClientData, CXFile, ctypes.c_void_p)))
setattr(IndexerCallbacks, 'ppIncludedFile', field(24, ctypes.CFUNCTYPE(CXIdxClientFile, CXClientData, Pointer(CXIdxIncludedFileInfo))))
setattr(IndexerCallbacks, 'importedASTFile', field(32, ctypes.CFUNCTYPE(CXIdxClientASTFile, CXClientData, Pointer(CXIdxImportedASTFileInfo))))
setattr(IndexerCallbacks, 'startedTranslationUnit', field(40, ctypes.CFUNCTYPE(CXIdxClientContainer, CXClientData, ctypes.c_void_p)))
setattr(IndexerCallbacks, 'indexDeclaration', field(48, ctypes.CFUNCTYPE(None, CXClientData, Pointer(CXIdxDeclInfo))))
setattr(IndexerCallbacks, 'indexEntityReference', field(56, ctypes.CFUNCTYPE(None, CXClientData, Pointer(CXIdxEntityRefInfo))))
@dll.bind((CXIdxEntityKind,), ctypes.c_int32)
def clang_index_isEntityObjCContainerKind(_0): ...
@dll.bind((Pointer(CXIdxDeclInfo),), Pointer(CXIdxObjCContainerDeclInfo))
def clang_index_getObjCContainerDeclInfo(_0): ...
@dll.bind((Pointer(CXIdxDeclInfo),), Pointer(CXIdxObjCInterfaceDeclInfo))
def clang_index_getObjCInterfaceDeclInfo(_0): ...
@dll.bind((Pointer(CXIdxDeclInfo),), Pointer(CXIdxObjCCategoryDeclInfo))
def clang_index_getObjCCategoryDeclInfo(_0): ...
@dll.bind((Pointer(CXIdxDeclInfo),), Pointer(CXIdxObjCProtocolRefListInfo))
def clang_index_getObjCProtocolRefListInfo(_0): ...
@dll.bind((Pointer(CXIdxDeclInfo),), Pointer(CXIdxObjCPropertyDeclInfo))
def clang_index_getObjCPropertyDeclInfo(_0): ...
@dll.bind((Pointer(CXIdxAttrInfo),), Pointer(CXIdxIBOutletCollectionAttrInfo))
def clang_index_getIBOutletCollectionAttrInfo(_0): ...
@dll.bind((Pointer(CXIdxDeclInfo),), Pointer(CXIdxCXXClassDeclInfo))
def clang_index_getCXXClassDeclInfo(_0): ...
@dll.bind((Pointer(CXIdxContainerInfo),), CXIdxClientContainer)
def clang_index_getClientContainer(_0): ...
@dll.bind((Pointer(CXIdxContainerInfo), CXIdxClientContainer), None)
def clang_index_setClientContainer(_0, _1): ...
@dll.bind((Pointer(CXIdxEntityInfo),), CXIdxClientEntity)
def clang_index_getClientEntity(_0): ...
@dll.bind((Pointer(CXIdxEntityInfo), CXIdxClientEntity), None)
def clang_index_setClientEntity(_0, _1): ...
CXIndexAction = ctypes.c_void_p
@dll.bind((CXIndex,), CXIndexAction)
def clang_IndexAction_create(CIdx): ...
@dll.bind((CXIndexAction,), None)
def clang_IndexAction_dispose(_0): ...
CXIndexOptFlags = CEnum(ctypes.c_uint32)
CXIndexOpt_None = CXIndexOptFlags.define('CXIndexOpt_None', 0)
CXIndexOpt_SuppressRedundantRefs = CXIndexOptFlags.define('CXIndexOpt_SuppressRedundantRefs', 1)
CXIndexOpt_IndexFunctionLocalSymbols = CXIndexOptFlags.define('CXIndexOpt_IndexFunctionLocalSymbols', 2)
CXIndexOpt_IndexImplicitTemplateInstantiations = CXIndexOptFlags.define('CXIndexOpt_IndexImplicitTemplateInstantiations', 4)
CXIndexOpt_SuppressWarnings = CXIndexOptFlags.define('CXIndexOpt_SuppressWarnings', 8)
CXIndexOpt_SkipParsedBodiesInSession = CXIndexOptFlags.define('CXIndexOpt_SkipParsedBodiesInSession', 16)

@dll.bind((CXIndexAction, CXClientData, Pointer(IndexerCallbacks), ctypes.c_uint32, ctypes.c_uint32, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char)), ctypes.c_int32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32, Pointer(CXTranslationUnit), ctypes.c_uint32), ctypes.c_int32)
def clang_indexSourceFile(_0, client_data, index_callbacks, index_callbacks_size, index_options, source_filename, command_line_args, num_command_line_args, unsaved_files, num_unsaved_files, out_TU, TU_options): ...
@dll.bind((CXIndexAction, CXClientData, Pointer(IndexerCallbacks), ctypes.c_uint32, ctypes.c_uint32, Pointer(ctypes.c_char), Pointer(Pointer(ctypes.c_char)), ctypes.c_int32, Pointer(struct_CXUnsavedFile), ctypes.c_uint32, Pointer(CXTranslationUnit), ctypes.c_uint32), ctypes.c_int32)
def clang_indexSourceFileFullArgv(_0, client_data, index_callbacks, index_callbacks_size, index_options, source_filename, command_line_args, num_command_line_args, unsaved_files, num_unsaved_files, out_TU, TU_options): ...
@dll.bind((CXIndexAction, CXClientData, Pointer(IndexerCallbacks), ctypes.c_uint32, ctypes.c_uint32, CXTranslationUnit), ctypes.c_int32)
def clang_indexTranslationUnit(_0, client_data, index_callbacks, index_callbacks_size, index_options, _5): ...
@dll.bind((CXIdxLoc, Pointer(CXIdxClientFile), Pointer(CXFile), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_indexLoc_getFileLocation(loc, indexFile, file, line, column, offset): ...
@dll.bind((CXIdxLoc,), CXSourceLocation)
def clang_indexLoc_getCXSourceLocation(loc): ...
CXFieldVisitor = ctypes.CFUNCTYPE(enum_CXVisitorResult, CXCursor, ctypes.c_void_p)
@dll.bind((CXType, CXFieldVisitor, CXClientData), ctypes.c_uint32)
def clang_Type_visitFields(T, visitor, client_data): ...
@dll.bind((CXType, CXFieldVisitor, CXClientData), ctypes.c_uint32)
def clang_visitCXXBaseClasses(T, visitor, client_data): ...
enum_CXBinaryOperatorKind = CEnum(ctypes.c_uint32)
CXBinaryOperator_Invalid = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Invalid', 0)
CXBinaryOperator_PtrMemD = enum_CXBinaryOperatorKind.define('CXBinaryOperator_PtrMemD', 1)
CXBinaryOperator_PtrMemI = enum_CXBinaryOperatorKind.define('CXBinaryOperator_PtrMemI', 2)
CXBinaryOperator_Mul = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Mul', 3)
CXBinaryOperator_Div = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Div', 4)
CXBinaryOperator_Rem = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Rem', 5)
CXBinaryOperator_Add = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Add', 6)
CXBinaryOperator_Sub = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Sub', 7)
CXBinaryOperator_Shl = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Shl', 8)
CXBinaryOperator_Shr = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Shr', 9)
CXBinaryOperator_Cmp = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Cmp', 10)
CXBinaryOperator_LT = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LT', 11)
CXBinaryOperator_GT = enum_CXBinaryOperatorKind.define('CXBinaryOperator_GT', 12)
CXBinaryOperator_LE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LE', 13)
CXBinaryOperator_GE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_GE', 14)
CXBinaryOperator_EQ = enum_CXBinaryOperatorKind.define('CXBinaryOperator_EQ', 15)
CXBinaryOperator_NE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_NE', 16)
CXBinaryOperator_And = enum_CXBinaryOperatorKind.define('CXBinaryOperator_And', 17)
CXBinaryOperator_Xor = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Xor', 18)
CXBinaryOperator_Or = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Or', 19)
CXBinaryOperator_LAnd = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LAnd', 20)
CXBinaryOperator_LOr = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LOr', 21)
CXBinaryOperator_Assign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Assign', 22)
CXBinaryOperator_MulAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_MulAssign', 23)
CXBinaryOperator_DivAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_DivAssign', 24)
CXBinaryOperator_RemAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_RemAssign', 25)
CXBinaryOperator_AddAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_AddAssign', 26)
CXBinaryOperator_SubAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_SubAssign', 27)
CXBinaryOperator_ShlAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_ShlAssign', 28)
CXBinaryOperator_ShrAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_ShrAssign', 29)
CXBinaryOperator_AndAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_AndAssign', 30)
CXBinaryOperator_XorAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_XorAssign', 31)
CXBinaryOperator_OrAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_OrAssign', 32)
CXBinaryOperator_Comma = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Comma', 33)

@dll.bind((enum_CXBinaryOperatorKind,), CXString)
def clang_getBinaryOperatorKindSpelling(kind): ...
@dll.bind((CXCursor,), enum_CXBinaryOperatorKind)
def clang_getCursorBinaryOperatorKind(cursor): ...
enum_CXUnaryOperatorKind = CEnum(ctypes.c_uint32)
CXUnaryOperator_Invalid = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Invalid', 0)
CXUnaryOperator_PostInc = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PostInc', 1)
CXUnaryOperator_PostDec = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PostDec', 2)
CXUnaryOperator_PreInc = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PreInc', 3)
CXUnaryOperator_PreDec = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PreDec', 4)
CXUnaryOperator_AddrOf = enum_CXUnaryOperatorKind.define('CXUnaryOperator_AddrOf', 5)
CXUnaryOperator_Deref = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Deref', 6)
CXUnaryOperator_Plus = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Plus', 7)
CXUnaryOperator_Minus = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Minus', 8)
CXUnaryOperator_Not = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Not', 9)
CXUnaryOperator_LNot = enum_CXUnaryOperatorKind.define('CXUnaryOperator_LNot', 10)
CXUnaryOperator_Real = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Real', 11)
CXUnaryOperator_Imag = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Imag', 12)
CXUnaryOperator_Extension = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Extension', 13)
CXUnaryOperator_Coawait = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Coawait', 14)

@dll.bind((enum_CXUnaryOperatorKind,), CXString)
def clang_getUnaryOperatorKindSpelling(kind): ...
@dll.bind((CXCursor,), enum_CXUnaryOperatorKind)
def clang_getCursorUnaryOperatorKind(cursor): ...
@dll.bind((CXString,), Pointer(ctypes.c_char))
def clang_getCString(string): ...
@dll.bind((CXString,), None)
def clang_disposeString(string): ...
@dll.bind((Pointer(CXStringSet),), None)
def clang_disposeStringSet(set): ...
@dll.bind((), CXSourceLocation)
def clang_getNullLocation(): ...
@dll.bind((CXSourceLocation, CXSourceLocation), ctypes.c_uint32)
def clang_equalLocations(loc1, loc2): ...
@dll.bind((CXSourceLocation, CXSourceLocation), ctypes.c_uint32)
def clang_isBeforeInTranslationUnit(loc1, loc2): ...
@dll.bind((CXSourceLocation,), ctypes.c_int32)
def clang_Location_isInSystemHeader(location): ...
@dll.bind((CXSourceLocation,), ctypes.c_int32)
def clang_Location_isFromMainFile(location): ...
@dll.bind((), CXSourceRange)
def clang_getNullRange(): ...
@dll.bind((CXSourceLocation, CXSourceLocation), CXSourceRange)
def clang_getRange(begin, end): ...
@dll.bind((CXSourceRange, CXSourceRange), ctypes.c_uint32)
def clang_equalRanges(range1, range2): ...
@dll.bind((CXSourceRange,), ctypes.c_int32)
def clang_Range_isNull(range): ...
@dll.bind((CXSourceLocation, Pointer(CXFile), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_getExpansionLocation(location, file, line, column, offset): ...
@dll.bind((CXSourceLocation, Pointer(CXString), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_getPresumedLocation(location, filename, line, column): ...
@dll.bind((CXSourceLocation, Pointer(CXFile), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_getInstantiationLocation(location, file, line, column, offset): ...
@dll.bind((CXSourceLocation, Pointer(CXFile), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_getSpellingLocation(location, file, line, column, offset): ...
@dll.bind((CXSourceLocation, Pointer(CXFile), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32), Pointer(ctypes.c_uint32)), None)
def clang_getFileLocation(location, file, line, column, offset): ...
@dll.bind((CXSourceRange,), CXSourceLocation)
def clang_getRangeStart(range): ...
@dll.bind((CXSourceRange,), CXSourceLocation)
def clang_getRangeEnd(range): ...
@dll.bind((Pointer(CXSourceRangeList),), None)
def clang_disposeSourceRangeList(ranges): ...
@dll.bind((CXFile,), CXString)
def clang_getFileName(SFile): ...
time_t = ctypes.c_int64
@dll.bind((CXFile,), time_t)
def clang_getFileTime(SFile): ...
class CXFileUniqueID(Struct): pass
CXFileUniqueID.SIZE = 24
CXFileUniqueID._fields_ = ['data']
setattr(CXFileUniqueID, 'data', field(0, Array(ctypes.c_uint64, 3)))
@dll.bind((CXFile, Pointer(CXFileUniqueID)), ctypes.c_int32)
def clang_getFileUniqueID(file, outID): ...
@dll.bind((CXFile, CXFile), ctypes.c_int32)
def clang_File_isEqual(file1, file2): ...
@dll.bind((CXFile,), CXString)
def clang_File_tryGetRealPathName(file): ...
CINDEX_VERSION_MAJOR = 0
CINDEX_VERSION_MINOR = 64
CINDEX_VERSION_ENCODE = lambda major,minor: (((major)*10000) + ((minor)*1))
CINDEX_VERSION = CINDEX_VERSION_ENCODE(CINDEX_VERSION_MAJOR, CINDEX_VERSION_MINOR)
CINDEX_VERSION_STRINGIZE = lambda major,minor: CINDEX_VERSION_STRINGIZE_(major, minor)