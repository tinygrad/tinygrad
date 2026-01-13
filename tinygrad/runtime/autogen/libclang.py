# mypy: disable-error-code="empty-body"
from __future__ import annotations
import ctypes
from typing import Annotated, Literal, TypeAlias
from tinygrad.runtime.support.c import CEnum, _IO, _IOW, _IOR, _IOWR
from tinygrad.runtime.support import c
dll = c.DLL('libclang', ['clang-20', 'clang'])
CXIndex = c.POINTER[None]
class struct_CXTargetInfoImpl(ctypes.Structure): pass
CXTargetInfo = c.POINTER[struct_CXTargetInfoImpl]
class struct_CXTranslationUnitImpl(ctypes.Structure): pass
CXTranslationUnit = c.POINTER[struct_CXTranslationUnitImpl]
CXClientData = c.POINTER[None]
@c.record
class struct_CXUnsavedFile(c.Struct):
  SIZE = 24
  Filename: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 0]
  Contents: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 8]
  Length: Annotated[Annotated[int, ctypes.c_uint64], 16]
enum_CXAvailabilityKind = CEnum(Annotated[int, ctypes.c_uint32])
CXAvailability_Available = enum_CXAvailabilityKind.define('CXAvailability_Available', 0) # type: ignore
CXAvailability_Deprecated = enum_CXAvailabilityKind.define('CXAvailability_Deprecated', 1) # type: ignore
CXAvailability_NotAvailable = enum_CXAvailabilityKind.define('CXAvailability_NotAvailable', 2) # type: ignore
CXAvailability_NotAccessible = enum_CXAvailabilityKind.define('CXAvailability_NotAccessible', 3) # type: ignore

@c.record
class struct_CXVersion(c.Struct):
  SIZE = 12
  Major: Annotated[Annotated[int, ctypes.c_int32], 0]
  Minor: Annotated[Annotated[int, ctypes.c_int32], 4]
  Subminor: Annotated[Annotated[int, ctypes.c_int32], 8]
CXVersion = struct_CXVersion
enum_CXCursor_ExceptionSpecificationKind = CEnum(Annotated[int, ctypes.c_uint32])
CXCursor_ExceptionSpecificationKind_None = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_None', 0) # type: ignore
CXCursor_ExceptionSpecificationKind_DynamicNone = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_DynamicNone', 1) # type: ignore
CXCursor_ExceptionSpecificationKind_Dynamic = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Dynamic', 2) # type: ignore
CXCursor_ExceptionSpecificationKind_MSAny = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_MSAny', 3) # type: ignore
CXCursor_ExceptionSpecificationKind_BasicNoexcept = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_BasicNoexcept', 4) # type: ignore
CXCursor_ExceptionSpecificationKind_ComputedNoexcept = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_ComputedNoexcept', 5) # type: ignore
CXCursor_ExceptionSpecificationKind_Unevaluated = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Unevaluated', 6) # type: ignore
CXCursor_ExceptionSpecificationKind_Uninstantiated = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Uninstantiated', 7) # type: ignore
CXCursor_ExceptionSpecificationKind_Unparsed = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_Unparsed', 8) # type: ignore
CXCursor_ExceptionSpecificationKind_NoThrow = enum_CXCursor_ExceptionSpecificationKind.define('CXCursor_ExceptionSpecificationKind_NoThrow', 9) # type: ignore

@dll.bind
def clang_createIndex(excludeDeclarationsFromPCH:Annotated[int, ctypes.c_int32], displayDiagnostics:Annotated[int, ctypes.c_int32]) -> CXIndex: ...
@dll.bind
def clang_disposeIndex(index:CXIndex) -> None: ...
CXChoice = CEnum(Annotated[int, ctypes.c_uint32])
CXChoice_Default = CXChoice.define('CXChoice_Default', 0) # type: ignore
CXChoice_Enabled = CXChoice.define('CXChoice_Enabled', 1) # type: ignore
CXChoice_Disabled = CXChoice.define('CXChoice_Disabled', 2) # type: ignore

CXGlobalOptFlags = CEnum(Annotated[int, ctypes.c_uint32])
CXGlobalOpt_None = CXGlobalOptFlags.define('CXGlobalOpt_None', 0) # type: ignore
CXGlobalOpt_ThreadBackgroundPriorityForIndexing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForIndexing', 1) # type: ignore
CXGlobalOpt_ThreadBackgroundPriorityForEditing = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForEditing', 2) # type: ignore
CXGlobalOpt_ThreadBackgroundPriorityForAll = CXGlobalOptFlags.define('CXGlobalOpt_ThreadBackgroundPriorityForAll', 3) # type: ignore

@c.record
class struct_CXIndexOptions(c.Struct):
  SIZE = 24
  Size: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ThreadBackgroundPriorityForIndexing: Annotated[Annotated[int, ctypes.c_ubyte], 4]
  ThreadBackgroundPriorityForEditing: Annotated[Annotated[int, ctypes.c_ubyte], 5]
  ExcludeDeclarationsFromPCH: Annotated[Annotated[int, ctypes.c_uint32], 6, 1, 0]
  DisplayDiagnostics: Annotated[Annotated[int, ctypes.c_uint32], 6, 1, 1]
  StorePreamblesInMemory: Annotated[Annotated[int, ctypes.c_uint32], 6, 1, 2]
  PreambleStoragePath: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 8]
  InvocationEmissionPath: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 16]
CXIndexOptions = struct_CXIndexOptions
@dll.bind
def clang_createIndexWithOptions(options:c.POINTER[CXIndexOptions]) -> CXIndex: ...
@dll.bind
def clang_CXIndex_setGlobalOptions(_0:CXIndex, options:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_CXIndex_getGlobalOptions(_0:CXIndex) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXIndex_setInvocationEmissionPathOption(_0:CXIndex, Path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> None: ...
CXFile = c.POINTER[None]
@dll.bind
def clang_isFileMultipleIncludeGuarded(tu:CXTranslationUnit, file:CXFile) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getFile(tu:CXTranslationUnit, file_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXFile: ...
size_t = Annotated[int, ctypes.c_uint64]
@dll.bind
def clang_getFileContents(tu:CXTranslationUnit, file:CXFile, size:c.POINTER[size_t]) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@c.record
class CXSourceLocation(c.Struct):
  SIZE = 24
  ptr_data: Annotated[c.Array[c.POINTER[None], Literal[2]], 0]
  int_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
@dll.bind
def clang_getLocation(tu:CXTranslationUnit, file:CXFile, line:Annotated[int, ctypes.c_uint32], column:Annotated[int, ctypes.c_uint32]) -> CXSourceLocation: ...
@dll.bind
def clang_getLocationForOffset(tu:CXTranslationUnit, file:CXFile, offset:Annotated[int, ctypes.c_uint32]) -> CXSourceLocation: ...
@c.record
class CXSourceRangeList(c.Struct):
  SIZE = 16
  count: Annotated[Annotated[int, ctypes.c_uint32], 0]
  ranges: Annotated[c.POINTER[CXSourceRange], 8]
@c.record
class CXSourceRange(c.Struct):
  SIZE = 24
  ptr_data: Annotated[c.Array[c.POINTER[None], Literal[2]], 0]
  begin_int_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
  end_int_data: Annotated[Annotated[int, ctypes.c_uint32], 20]
@dll.bind
def clang_getSkippedRanges(tu:CXTranslationUnit, file:CXFile) -> c.POINTER[CXSourceRangeList]: ...
@dll.bind
def clang_getAllSkippedRanges(tu:CXTranslationUnit) -> c.POINTER[CXSourceRangeList]: ...
@dll.bind
def clang_getNumDiagnostics(Unit:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
CXDiagnostic = c.POINTER[None]
@dll.bind
def clang_getDiagnostic(Unit:CXTranslationUnit, Index:Annotated[int, ctypes.c_uint32]) -> CXDiagnostic: ...
CXDiagnosticSet = c.POINTER[None]
@dll.bind
def clang_getDiagnosticSetFromTU(Unit:CXTranslationUnit) -> CXDiagnosticSet: ...
@c.record
class CXString(c.Struct):
  SIZE = 16
  data: Annotated[c.POINTER[None], 0]
  private_flags: Annotated[Annotated[int, ctypes.c_uint32], 8]
@dll.bind
def clang_getTranslationUnitSpelling(CTUnit:CXTranslationUnit) -> CXString: ...
@dll.bind
def clang_createTranslationUnitFromSourceFile(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], num_clang_command_line_args:Annotated[int, ctypes.c_int32], clang_command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_unsaved_files:Annotated[int, ctypes.c_uint32], unsaved_files:c.POINTER[struct_CXUnsavedFile]) -> CXTranslationUnit: ...
@dll.bind
def clang_createTranslationUnit(CIdx:CXIndex, ast_filename:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXTranslationUnit: ...
enum_CXErrorCode = CEnum(Annotated[int, ctypes.c_uint32])
CXError_Success = enum_CXErrorCode.define('CXError_Success', 0) # type: ignore
CXError_Failure = enum_CXErrorCode.define('CXError_Failure', 1) # type: ignore
CXError_Crashed = enum_CXErrorCode.define('CXError_Crashed', 2) # type: ignore
CXError_InvalidArguments = enum_CXErrorCode.define('CXError_InvalidArguments', 3) # type: ignore
CXError_ASTReadError = enum_CXErrorCode.define('CXError_ASTReadError', 4) # type: ignore

@dll.bind
def clang_createTranslationUnit2(CIdx:CXIndex, ast_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], out_TU:c.POINTER[CXTranslationUnit]) -> enum_CXErrorCode: ...
enum_CXTranslationUnit_Flags = CEnum(Annotated[int, ctypes.c_uint32])
CXTranslationUnit_None = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_None', 0) # type: ignore
CXTranslationUnit_DetailedPreprocessingRecord = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_DetailedPreprocessingRecord', 1) # type: ignore
CXTranslationUnit_Incomplete = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_Incomplete', 2) # type: ignore
CXTranslationUnit_PrecompiledPreamble = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_PrecompiledPreamble', 4) # type: ignore
CXTranslationUnit_CacheCompletionResults = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CacheCompletionResults', 8) # type: ignore
CXTranslationUnit_ForSerialization = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_ForSerialization', 16) # type: ignore
CXTranslationUnit_CXXChainedPCH = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CXXChainedPCH', 32) # type: ignore
CXTranslationUnit_SkipFunctionBodies = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_SkipFunctionBodies', 64) # type: ignore
CXTranslationUnit_IncludeBriefCommentsInCodeCompletion = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IncludeBriefCommentsInCodeCompletion', 128) # type: ignore
CXTranslationUnit_CreatePreambleOnFirstParse = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_CreatePreambleOnFirstParse', 256) # type: ignore
CXTranslationUnit_KeepGoing = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_KeepGoing', 512) # type: ignore
CXTranslationUnit_SingleFileParse = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_SingleFileParse', 1024) # type: ignore
CXTranslationUnit_LimitSkipFunctionBodiesToPreamble = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_LimitSkipFunctionBodiesToPreamble', 2048) # type: ignore
CXTranslationUnit_IncludeAttributedTypes = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IncludeAttributedTypes', 4096) # type: ignore
CXTranslationUnit_VisitImplicitAttributes = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_VisitImplicitAttributes', 8192) # type: ignore
CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_IgnoreNonErrorsFromIncludedFiles', 16384) # type: ignore
CXTranslationUnit_RetainExcludedConditionalBlocks = enum_CXTranslationUnit_Flags.define('CXTranslationUnit_RetainExcludedConditionalBlocks', 32768) # type: ignore

@dll.bind
def clang_defaultEditingTranslationUnitOptions() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_parseTranslationUnit(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32]) -> CXTranslationUnit: ...
@dll.bind
def clang_parseTranslationUnit2(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit]) -> enum_CXErrorCode: ...
@dll.bind
def clang_parseTranslationUnit2FullArgv(CIdx:CXIndex, source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit]) -> enum_CXErrorCode: ...
enum_CXSaveTranslationUnit_Flags = CEnum(Annotated[int, ctypes.c_uint32])
CXSaveTranslationUnit_None = enum_CXSaveTranslationUnit_Flags.define('CXSaveTranslationUnit_None', 0) # type: ignore

@dll.bind
def clang_defaultSaveOptions(TU:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
enum_CXSaveError = CEnum(Annotated[int, ctypes.c_uint32])
CXSaveError_None = enum_CXSaveError.define('CXSaveError_None', 0) # type: ignore
CXSaveError_Unknown = enum_CXSaveError.define('CXSaveError_Unknown', 1) # type: ignore
CXSaveError_TranslationErrors = enum_CXSaveError.define('CXSaveError_TranslationErrors', 2) # type: ignore
CXSaveError_InvalidTU = enum_CXSaveError.define('CXSaveError_InvalidTU', 3) # type: ignore

@dll.bind
def clang_saveTranslationUnit(TU:CXTranslationUnit, FileName:c.POINTER[Annotated[bytes, ctypes.c_char]], options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_suspendTranslationUnit(_0:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_disposeTranslationUnit(_0:CXTranslationUnit) -> None: ...
enum_CXReparse_Flags = CEnum(Annotated[int, ctypes.c_uint32])
CXReparse_None = enum_CXReparse_Flags.define('CXReparse_None', 0) # type: ignore

@dll.bind
def clang_defaultReparseOptions(TU:CXTranslationUnit) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_reparseTranslationUnit(TU:CXTranslationUnit, num_unsaved_files:Annotated[int, ctypes.c_uint32], unsaved_files:c.POINTER[struct_CXUnsavedFile], options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
enum_CXTUResourceUsageKind = CEnum(Annotated[int, ctypes.c_uint32])
CXTUResourceUsage_AST = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_AST', 1) # type: ignore
CXTUResourceUsage_Identifiers = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Identifiers', 2) # type: ignore
CXTUResourceUsage_Selectors = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Selectors', 3) # type: ignore
CXTUResourceUsage_GlobalCompletionResults = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_GlobalCompletionResults', 4) # type: ignore
CXTUResourceUsage_SourceManagerContentCache = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManagerContentCache', 5) # type: ignore
CXTUResourceUsage_AST_SideTables = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_AST_SideTables', 6) # type: ignore
CXTUResourceUsage_SourceManager_Membuffer_Malloc = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_Membuffer_Malloc', 7) # type: ignore
CXTUResourceUsage_SourceManager_Membuffer_MMap = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_Membuffer_MMap', 8) # type: ignore
CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_ExternalASTSource_Membuffer_Malloc', 9) # type: ignore
CXTUResourceUsage_ExternalASTSource_Membuffer_MMap = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_ExternalASTSource_Membuffer_MMap', 10) # type: ignore
CXTUResourceUsage_Preprocessor = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Preprocessor', 11) # type: ignore
CXTUResourceUsage_PreprocessingRecord = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_PreprocessingRecord', 12) # type: ignore
CXTUResourceUsage_SourceManager_DataStructures = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_SourceManager_DataStructures', 13) # type: ignore
CXTUResourceUsage_Preprocessor_HeaderSearch = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Preprocessor_HeaderSearch', 14) # type: ignore
CXTUResourceUsage_MEMORY_IN_BYTES_BEGIN = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_MEMORY_IN_BYTES_BEGIN', 1) # type: ignore
CXTUResourceUsage_MEMORY_IN_BYTES_END = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_MEMORY_IN_BYTES_END', 14) # type: ignore
CXTUResourceUsage_First = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_First', 1) # type: ignore
CXTUResourceUsage_Last = enum_CXTUResourceUsageKind.define('CXTUResourceUsage_Last', 14) # type: ignore

@dll.bind
def clang_getTUResourceUsageName(kind:enum_CXTUResourceUsageKind) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@c.record
class struct_CXTUResourceUsageEntry(c.Struct):
  SIZE = 16
  kind: Annotated[enum_CXTUResourceUsageKind, 0]
  amount: Annotated[Annotated[int, ctypes.c_uint64], 8]
CXTUResourceUsageEntry = struct_CXTUResourceUsageEntry
@c.record
class struct_CXTUResourceUsage(c.Struct):
  SIZE = 24
  data: Annotated[c.POINTER[None], 0]
  numEntries: Annotated[Annotated[int, ctypes.c_uint32], 8]
  entries: Annotated[c.POINTER[CXTUResourceUsageEntry], 16]
CXTUResourceUsage = struct_CXTUResourceUsage
@dll.bind
def clang_getCXTUResourceUsage(TU:CXTranslationUnit) -> CXTUResourceUsage: ...
@dll.bind
def clang_disposeCXTUResourceUsage(usage:CXTUResourceUsage) -> None: ...
@dll.bind
def clang_getTranslationUnitTargetInfo(CTUnit:CXTranslationUnit) -> CXTargetInfo: ...
@dll.bind
def clang_TargetInfo_dispose(Info:CXTargetInfo) -> None: ...
@dll.bind
def clang_TargetInfo_getTriple(Info:CXTargetInfo) -> CXString: ...
@dll.bind
def clang_TargetInfo_getPointerWidth(Info:CXTargetInfo) -> Annotated[int, ctypes.c_int32]: ...
enum_CXCursorKind = CEnum(Annotated[int, ctypes.c_uint32])
CXCursor_UnexposedDecl = enum_CXCursorKind.define('CXCursor_UnexposedDecl', 1) # type: ignore
CXCursor_StructDecl = enum_CXCursorKind.define('CXCursor_StructDecl', 2) # type: ignore
CXCursor_UnionDecl = enum_CXCursorKind.define('CXCursor_UnionDecl', 3) # type: ignore
CXCursor_ClassDecl = enum_CXCursorKind.define('CXCursor_ClassDecl', 4) # type: ignore
CXCursor_EnumDecl = enum_CXCursorKind.define('CXCursor_EnumDecl', 5) # type: ignore
CXCursor_FieldDecl = enum_CXCursorKind.define('CXCursor_FieldDecl', 6) # type: ignore
CXCursor_EnumConstantDecl = enum_CXCursorKind.define('CXCursor_EnumConstantDecl', 7) # type: ignore
CXCursor_FunctionDecl = enum_CXCursorKind.define('CXCursor_FunctionDecl', 8) # type: ignore
CXCursor_VarDecl = enum_CXCursorKind.define('CXCursor_VarDecl', 9) # type: ignore
CXCursor_ParmDecl = enum_CXCursorKind.define('CXCursor_ParmDecl', 10) # type: ignore
CXCursor_ObjCInterfaceDecl = enum_CXCursorKind.define('CXCursor_ObjCInterfaceDecl', 11) # type: ignore
CXCursor_ObjCCategoryDecl = enum_CXCursorKind.define('CXCursor_ObjCCategoryDecl', 12) # type: ignore
CXCursor_ObjCProtocolDecl = enum_CXCursorKind.define('CXCursor_ObjCProtocolDecl', 13) # type: ignore
CXCursor_ObjCPropertyDecl = enum_CXCursorKind.define('CXCursor_ObjCPropertyDecl', 14) # type: ignore
CXCursor_ObjCIvarDecl = enum_CXCursorKind.define('CXCursor_ObjCIvarDecl', 15) # type: ignore
CXCursor_ObjCInstanceMethodDecl = enum_CXCursorKind.define('CXCursor_ObjCInstanceMethodDecl', 16) # type: ignore
CXCursor_ObjCClassMethodDecl = enum_CXCursorKind.define('CXCursor_ObjCClassMethodDecl', 17) # type: ignore
CXCursor_ObjCImplementationDecl = enum_CXCursorKind.define('CXCursor_ObjCImplementationDecl', 18) # type: ignore
CXCursor_ObjCCategoryImplDecl = enum_CXCursorKind.define('CXCursor_ObjCCategoryImplDecl', 19) # type: ignore
CXCursor_TypedefDecl = enum_CXCursorKind.define('CXCursor_TypedefDecl', 20) # type: ignore
CXCursor_CXXMethod = enum_CXCursorKind.define('CXCursor_CXXMethod', 21) # type: ignore
CXCursor_Namespace = enum_CXCursorKind.define('CXCursor_Namespace', 22) # type: ignore
CXCursor_LinkageSpec = enum_CXCursorKind.define('CXCursor_LinkageSpec', 23) # type: ignore
CXCursor_Constructor = enum_CXCursorKind.define('CXCursor_Constructor', 24) # type: ignore
CXCursor_Destructor = enum_CXCursorKind.define('CXCursor_Destructor', 25) # type: ignore
CXCursor_ConversionFunction = enum_CXCursorKind.define('CXCursor_ConversionFunction', 26) # type: ignore
CXCursor_TemplateTypeParameter = enum_CXCursorKind.define('CXCursor_TemplateTypeParameter', 27) # type: ignore
CXCursor_NonTypeTemplateParameter = enum_CXCursorKind.define('CXCursor_NonTypeTemplateParameter', 28) # type: ignore
CXCursor_TemplateTemplateParameter = enum_CXCursorKind.define('CXCursor_TemplateTemplateParameter', 29) # type: ignore
CXCursor_FunctionTemplate = enum_CXCursorKind.define('CXCursor_FunctionTemplate', 30) # type: ignore
CXCursor_ClassTemplate = enum_CXCursorKind.define('CXCursor_ClassTemplate', 31) # type: ignore
CXCursor_ClassTemplatePartialSpecialization = enum_CXCursorKind.define('CXCursor_ClassTemplatePartialSpecialization', 32) # type: ignore
CXCursor_NamespaceAlias = enum_CXCursorKind.define('CXCursor_NamespaceAlias', 33) # type: ignore
CXCursor_UsingDirective = enum_CXCursorKind.define('CXCursor_UsingDirective', 34) # type: ignore
CXCursor_UsingDeclaration = enum_CXCursorKind.define('CXCursor_UsingDeclaration', 35) # type: ignore
CXCursor_TypeAliasDecl = enum_CXCursorKind.define('CXCursor_TypeAliasDecl', 36) # type: ignore
CXCursor_ObjCSynthesizeDecl = enum_CXCursorKind.define('CXCursor_ObjCSynthesizeDecl', 37) # type: ignore
CXCursor_ObjCDynamicDecl = enum_CXCursorKind.define('CXCursor_ObjCDynamicDecl', 38) # type: ignore
CXCursor_CXXAccessSpecifier = enum_CXCursorKind.define('CXCursor_CXXAccessSpecifier', 39) # type: ignore
CXCursor_FirstDecl = enum_CXCursorKind.define('CXCursor_FirstDecl', 1) # type: ignore
CXCursor_LastDecl = enum_CXCursorKind.define('CXCursor_LastDecl', 39) # type: ignore
CXCursor_FirstRef = enum_CXCursorKind.define('CXCursor_FirstRef', 40) # type: ignore
CXCursor_ObjCSuperClassRef = enum_CXCursorKind.define('CXCursor_ObjCSuperClassRef', 40) # type: ignore
CXCursor_ObjCProtocolRef = enum_CXCursorKind.define('CXCursor_ObjCProtocolRef', 41) # type: ignore
CXCursor_ObjCClassRef = enum_CXCursorKind.define('CXCursor_ObjCClassRef', 42) # type: ignore
CXCursor_TypeRef = enum_CXCursorKind.define('CXCursor_TypeRef', 43) # type: ignore
CXCursor_CXXBaseSpecifier = enum_CXCursorKind.define('CXCursor_CXXBaseSpecifier', 44) # type: ignore
CXCursor_TemplateRef = enum_CXCursorKind.define('CXCursor_TemplateRef', 45) # type: ignore
CXCursor_NamespaceRef = enum_CXCursorKind.define('CXCursor_NamespaceRef', 46) # type: ignore
CXCursor_MemberRef = enum_CXCursorKind.define('CXCursor_MemberRef', 47) # type: ignore
CXCursor_LabelRef = enum_CXCursorKind.define('CXCursor_LabelRef', 48) # type: ignore
CXCursor_OverloadedDeclRef = enum_CXCursorKind.define('CXCursor_OverloadedDeclRef', 49) # type: ignore
CXCursor_VariableRef = enum_CXCursorKind.define('CXCursor_VariableRef', 50) # type: ignore
CXCursor_LastRef = enum_CXCursorKind.define('CXCursor_LastRef', 50) # type: ignore
CXCursor_FirstInvalid = enum_CXCursorKind.define('CXCursor_FirstInvalid', 70) # type: ignore
CXCursor_InvalidFile = enum_CXCursorKind.define('CXCursor_InvalidFile', 70) # type: ignore
CXCursor_NoDeclFound = enum_CXCursorKind.define('CXCursor_NoDeclFound', 71) # type: ignore
CXCursor_NotImplemented = enum_CXCursorKind.define('CXCursor_NotImplemented', 72) # type: ignore
CXCursor_InvalidCode = enum_CXCursorKind.define('CXCursor_InvalidCode', 73) # type: ignore
CXCursor_LastInvalid = enum_CXCursorKind.define('CXCursor_LastInvalid', 73) # type: ignore
CXCursor_FirstExpr = enum_CXCursorKind.define('CXCursor_FirstExpr', 100) # type: ignore
CXCursor_UnexposedExpr = enum_CXCursorKind.define('CXCursor_UnexposedExpr', 100) # type: ignore
CXCursor_DeclRefExpr = enum_CXCursorKind.define('CXCursor_DeclRefExpr', 101) # type: ignore
CXCursor_MemberRefExpr = enum_CXCursorKind.define('CXCursor_MemberRefExpr', 102) # type: ignore
CXCursor_CallExpr = enum_CXCursorKind.define('CXCursor_CallExpr', 103) # type: ignore
CXCursor_ObjCMessageExpr = enum_CXCursorKind.define('CXCursor_ObjCMessageExpr', 104) # type: ignore
CXCursor_BlockExpr = enum_CXCursorKind.define('CXCursor_BlockExpr', 105) # type: ignore
CXCursor_IntegerLiteral = enum_CXCursorKind.define('CXCursor_IntegerLiteral', 106) # type: ignore
CXCursor_FloatingLiteral = enum_CXCursorKind.define('CXCursor_FloatingLiteral', 107) # type: ignore
CXCursor_ImaginaryLiteral = enum_CXCursorKind.define('CXCursor_ImaginaryLiteral', 108) # type: ignore
CXCursor_StringLiteral = enum_CXCursorKind.define('CXCursor_StringLiteral', 109) # type: ignore
CXCursor_CharacterLiteral = enum_CXCursorKind.define('CXCursor_CharacterLiteral', 110) # type: ignore
CXCursor_ParenExpr = enum_CXCursorKind.define('CXCursor_ParenExpr', 111) # type: ignore
CXCursor_UnaryOperator = enum_CXCursorKind.define('CXCursor_UnaryOperator', 112) # type: ignore
CXCursor_ArraySubscriptExpr = enum_CXCursorKind.define('CXCursor_ArraySubscriptExpr', 113) # type: ignore
CXCursor_BinaryOperator = enum_CXCursorKind.define('CXCursor_BinaryOperator', 114) # type: ignore
CXCursor_CompoundAssignOperator = enum_CXCursorKind.define('CXCursor_CompoundAssignOperator', 115) # type: ignore
CXCursor_ConditionalOperator = enum_CXCursorKind.define('CXCursor_ConditionalOperator', 116) # type: ignore
CXCursor_CStyleCastExpr = enum_CXCursorKind.define('CXCursor_CStyleCastExpr', 117) # type: ignore
CXCursor_CompoundLiteralExpr = enum_CXCursorKind.define('CXCursor_CompoundLiteralExpr', 118) # type: ignore
CXCursor_InitListExpr = enum_CXCursorKind.define('CXCursor_InitListExpr', 119) # type: ignore
CXCursor_AddrLabelExpr = enum_CXCursorKind.define('CXCursor_AddrLabelExpr', 120) # type: ignore
CXCursor_StmtExpr = enum_CXCursorKind.define('CXCursor_StmtExpr', 121) # type: ignore
CXCursor_GenericSelectionExpr = enum_CXCursorKind.define('CXCursor_GenericSelectionExpr', 122) # type: ignore
CXCursor_GNUNullExpr = enum_CXCursorKind.define('CXCursor_GNUNullExpr', 123) # type: ignore
CXCursor_CXXStaticCastExpr = enum_CXCursorKind.define('CXCursor_CXXStaticCastExpr', 124) # type: ignore
CXCursor_CXXDynamicCastExpr = enum_CXCursorKind.define('CXCursor_CXXDynamicCastExpr', 125) # type: ignore
CXCursor_CXXReinterpretCastExpr = enum_CXCursorKind.define('CXCursor_CXXReinterpretCastExpr', 126) # type: ignore
CXCursor_CXXConstCastExpr = enum_CXCursorKind.define('CXCursor_CXXConstCastExpr', 127) # type: ignore
CXCursor_CXXFunctionalCastExpr = enum_CXCursorKind.define('CXCursor_CXXFunctionalCastExpr', 128) # type: ignore
CXCursor_CXXTypeidExpr = enum_CXCursorKind.define('CXCursor_CXXTypeidExpr', 129) # type: ignore
CXCursor_CXXBoolLiteralExpr = enum_CXCursorKind.define('CXCursor_CXXBoolLiteralExpr', 130) # type: ignore
CXCursor_CXXNullPtrLiteralExpr = enum_CXCursorKind.define('CXCursor_CXXNullPtrLiteralExpr', 131) # type: ignore
CXCursor_CXXThisExpr = enum_CXCursorKind.define('CXCursor_CXXThisExpr', 132) # type: ignore
CXCursor_CXXThrowExpr = enum_CXCursorKind.define('CXCursor_CXXThrowExpr', 133) # type: ignore
CXCursor_CXXNewExpr = enum_CXCursorKind.define('CXCursor_CXXNewExpr', 134) # type: ignore
CXCursor_CXXDeleteExpr = enum_CXCursorKind.define('CXCursor_CXXDeleteExpr', 135) # type: ignore
CXCursor_UnaryExpr = enum_CXCursorKind.define('CXCursor_UnaryExpr', 136) # type: ignore
CXCursor_ObjCStringLiteral = enum_CXCursorKind.define('CXCursor_ObjCStringLiteral', 137) # type: ignore
CXCursor_ObjCEncodeExpr = enum_CXCursorKind.define('CXCursor_ObjCEncodeExpr', 138) # type: ignore
CXCursor_ObjCSelectorExpr = enum_CXCursorKind.define('CXCursor_ObjCSelectorExpr', 139) # type: ignore
CXCursor_ObjCProtocolExpr = enum_CXCursorKind.define('CXCursor_ObjCProtocolExpr', 140) # type: ignore
CXCursor_ObjCBridgedCastExpr = enum_CXCursorKind.define('CXCursor_ObjCBridgedCastExpr', 141) # type: ignore
CXCursor_PackExpansionExpr = enum_CXCursorKind.define('CXCursor_PackExpansionExpr', 142) # type: ignore
CXCursor_SizeOfPackExpr = enum_CXCursorKind.define('CXCursor_SizeOfPackExpr', 143) # type: ignore
CXCursor_LambdaExpr = enum_CXCursorKind.define('CXCursor_LambdaExpr', 144) # type: ignore
CXCursor_ObjCBoolLiteralExpr = enum_CXCursorKind.define('CXCursor_ObjCBoolLiteralExpr', 145) # type: ignore
CXCursor_ObjCSelfExpr = enum_CXCursorKind.define('CXCursor_ObjCSelfExpr', 146) # type: ignore
CXCursor_ArraySectionExpr = enum_CXCursorKind.define('CXCursor_ArraySectionExpr', 147) # type: ignore
CXCursor_ObjCAvailabilityCheckExpr = enum_CXCursorKind.define('CXCursor_ObjCAvailabilityCheckExpr', 148) # type: ignore
CXCursor_FixedPointLiteral = enum_CXCursorKind.define('CXCursor_FixedPointLiteral', 149) # type: ignore
CXCursor_OMPArrayShapingExpr = enum_CXCursorKind.define('CXCursor_OMPArrayShapingExpr', 150) # type: ignore
CXCursor_OMPIteratorExpr = enum_CXCursorKind.define('CXCursor_OMPIteratorExpr', 151) # type: ignore
CXCursor_CXXAddrspaceCastExpr = enum_CXCursorKind.define('CXCursor_CXXAddrspaceCastExpr', 152) # type: ignore
CXCursor_ConceptSpecializationExpr = enum_CXCursorKind.define('CXCursor_ConceptSpecializationExpr', 153) # type: ignore
CXCursor_RequiresExpr = enum_CXCursorKind.define('CXCursor_RequiresExpr', 154) # type: ignore
CXCursor_CXXParenListInitExpr = enum_CXCursorKind.define('CXCursor_CXXParenListInitExpr', 155) # type: ignore
CXCursor_PackIndexingExpr = enum_CXCursorKind.define('CXCursor_PackIndexingExpr', 156) # type: ignore
CXCursor_LastExpr = enum_CXCursorKind.define('CXCursor_LastExpr', 156) # type: ignore
CXCursor_FirstStmt = enum_CXCursorKind.define('CXCursor_FirstStmt', 200) # type: ignore
CXCursor_UnexposedStmt = enum_CXCursorKind.define('CXCursor_UnexposedStmt', 200) # type: ignore
CXCursor_LabelStmt = enum_CXCursorKind.define('CXCursor_LabelStmt', 201) # type: ignore
CXCursor_CompoundStmt = enum_CXCursorKind.define('CXCursor_CompoundStmt', 202) # type: ignore
CXCursor_CaseStmt = enum_CXCursorKind.define('CXCursor_CaseStmt', 203) # type: ignore
CXCursor_DefaultStmt = enum_CXCursorKind.define('CXCursor_DefaultStmt', 204) # type: ignore
CXCursor_IfStmt = enum_CXCursorKind.define('CXCursor_IfStmt', 205) # type: ignore
CXCursor_SwitchStmt = enum_CXCursorKind.define('CXCursor_SwitchStmt', 206) # type: ignore
CXCursor_WhileStmt = enum_CXCursorKind.define('CXCursor_WhileStmt', 207) # type: ignore
CXCursor_DoStmt = enum_CXCursorKind.define('CXCursor_DoStmt', 208) # type: ignore
CXCursor_ForStmt = enum_CXCursorKind.define('CXCursor_ForStmt', 209) # type: ignore
CXCursor_GotoStmt = enum_CXCursorKind.define('CXCursor_GotoStmt', 210) # type: ignore
CXCursor_IndirectGotoStmt = enum_CXCursorKind.define('CXCursor_IndirectGotoStmt', 211) # type: ignore
CXCursor_ContinueStmt = enum_CXCursorKind.define('CXCursor_ContinueStmt', 212) # type: ignore
CXCursor_BreakStmt = enum_CXCursorKind.define('CXCursor_BreakStmt', 213) # type: ignore
CXCursor_ReturnStmt = enum_CXCursorKind.define('CXCursor_ReturnStmt', 214) # type: ignore
CXCursor_GCCAsmStmt = enum_CXCursorKind.define('CXCursor_GCCAsmStmt', 215) # type: ignore
CXCursor_AsmStmt = enum_CXCursorKind.define('CXCursor_AsmStmt', 215) # type: ignore
CXCursor_ObjCAtTryStmt = enum_CXCursorKind.define('CXCursor_ObjCAtTryStmt', 216) # type: ignore
CXCursor_ObjCAtCatchStmt = enum_CXCursorKind.define('CXCursor_ObjCAtCatchStmt', 217) # type: ignore
CXCursor_ObjCAtFinallyStmt = enum_CXCursorKind.define('CXCursor_ObjCAtFinallyStmt', 218) # type: ignore
CXCursor_ObjCAtThrowStmt = enum_CXCursorKind.define('CXCursor_ObjCAtThrowStmt', 219) # type: ignore
CXCursor_ObjCAtSynchronizedStmt = enum_CXCursorKind.define('CXCursor_ObjCAtSynchronizedStmt', 220) # type: ignore
CXCursor_ObjCAutoreleasePoolStmt = enum_CXCursorKind.define('CXCursor_ObjCAutoreleasePoolStmt', 221) # type: ignore
CXCursor_ObjCForCollectionStmt = enum_CXCursorKind.define('CXCursor_ObjCForCollectionStmt', 222) # type: ignore
CXCursor_CXXCatchStmt = enum_CXCursorKind.define('CXCursor_CXXCatchStmt', 223) # type: ignore
CXCursor_CXXTryStmt = enum_CXCursorKind.define('CXCursor_CXXTryStmt', 224) # type: ignore
CXCursor_CXXForRangeStmt = enum_CXCursorKind.define('CXCursor_CXXForRangeStmt', 225) # type: ignore
CXCursor_SEHTryStmt = enum_CXCursorKind.define('CXCursor_SEHTryStmt', 226) # type: ignore
CXCursor_SEHExceptStmt = enum_CXCursorKind.define('CXCursor_SEHExceptStmt', 227) # type: ignore
CXCursor_SEHFinallyStmt = enum_CXCursorKind.define('CXCursor_SEHFinallyStmt', 228) # type: ignore
CXCursor_MSAsmStmt = enum_CXCursorKind.define('CXCursor_MSAsmStmt', 229) # type: ignore
CXCursor_NullStmt = enum_CXCursorKind.define('CXCursor_NullStmt', 230) # type: ignore
CXCursor_DeclStmt = enum_CXCursorKind.define('CXCursor_DeclStmt', 231) # type: ignore
CXCursor_OMPParallelDirective = enum_CXCursorKind.define('CXCursor_OMPParallelDirective', 232) # type: ignore
CXCursor_OMPSimdDirective = enum_CXCursorKind.define('CXCursor_OMPSimdDirective', 233) # type: ignore
CXCursor_OMPForDirective = enum_CXCursorKind.define('CXCursor_OMPForDirective', 234) # type: ignore
CXCursor_OMPSectionsDirective = enum_CXCursorKind.define('CXCursor_OMPSectionsDirective', 235) # type: ignore
CXCursor_OMPSectionDirective = enum_CXCursorKind.define('CXCursor_OMPSectionDirective', 236) # type: ignore
CXCursor_OMPSingleDirective = enum_CXCursorKind.define('CXCursor_OMPSingleDirective', 237) # type: ignore
CXCursor_OMPParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPParallelForDirective', 238) # type: ignore
CXCursor_OMPParallelSectionsDirective = enum_CXCursorKind.define('CXCursor_OMPParallelSectionsDirective', 239) # type: ignore
CXCursor_OMPTaskDirective = enum_CXCursorKind.define('CXCursor_OMPTaskDirective', 240) # type: ignore
CXCursor_OMPMasterDirective = enum_CXCursorKind.define('CXCursor_OMPMasterDirective', 241) # type: ignore
CXCursor_OMPCriticalDirective = enum_CXCursorKind.define('CXCursor_OMPCriticalDirective', 242) # type: ignore
CXCursor_OMPTaskyieldDirective = enum_CXCursorKind.define('CXCursor_OMPTaskyieldDirective', 243) # type: ignore
CXCursor_OMPBarrierDirective = enum_CXCursorKind.define('CXCursor_OMPBarrierDirective', 244) # type: ignore
CXCursor_OMPTaskwaitDirective = enum_CXCursorKind.define('CXCursor_OMPTaskwaitDirective', 245) # type: ignore
CXCursor_OMPFlushDirective = enum_CXCursorKind.define('CXCursor_OMPFlushDirective', 246) # type: ignore
CXCursor_SEHLeaveStmt = enum_CXCursorKind.define('CXCursor_SEHLeaveStmt', 247) # type: ignore
CXCursor_OMPOrderedDirective = enum_CXCursorKind.define('CXCursor_OMPOrderedDirective', 248) # type: ignore
CXCursor_OMPAtomicDirective = enum_CXCursorKind.define('CXCursor_OMPAtomicDirective', 249) # type: ignore
CXCursor_OMPForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPForSimdDirective', 250) # type: ignore
CXCursor_OMPParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelForSimdDirective', 251) # type: ignore
CXCursor_OMPTargetDirective = enum_CXCursorKind.define('CXCursor_OMPTargetDirective', 252) # type: ignore
CXCursor_OMPTeamsDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDirective', 253) # type: ignore
CXCursor_OMPTaskgroupDirective = enum_CXCursorKind.define('CXCursor_OMPTaskgroupDirective', 254) # type: ignore
CXCursor_OMPCancellationPointDirective = enum_CXCursorKind.define('CXCursor_OMPCancellationPointDirective', 255) # type: ignore
CXCursor_OMPCancelDirective = enum_CXCursorKind.define('CXCursor_OMPCancelDirective', 256) # type: ignore
CXCursor_OMPTargetDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetDataDirective', 257) # type: ignore
CXCursor_OMPTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTaskLoopDirective', 258) # type: ignore
CXCursor_OMPTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTaskLoopSimdDirective', 259) # type: ignore
CXCursor_OMPDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeDirective', 260) # type: ignore
CXCursor_OMPTargetEnterDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetEnterDataDirective', 261) # type: ignore
CXCursor_OMPTargetExitDataDirective = enum_CXCursorKind.define('CXCursor_OMPTargetExitDataDirective', 262) # type: ignore
CXCursor_OMPTargetParallelDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelDirective', 263) # type: ignore
CXCursor_OMPTargetParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelForDirective', 264) # type: ignore
CXCursor_OMPTargetUpdateDirective = enum_CXCursorKind.define('CXCursor_OMPTargetUpdateDirective', 265) # type: ignore
CXCursor_OMPDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeParallelForDirective', 266) # type: ignore
CXCursor_OMPDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeParallelForSimdDirective', 267) # type: ignore
CXCursor_OMPDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPDistributeSimdDirective', 268) # type: ignore
CXCursor_OMPTargetParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelForSimdDirective', 269) # type: ignore
CXCursor_OMPTargetSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetSimdDirective', 270) # type: ignore
CXCursor_OMPTeamsDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeDirective', 271) # type: ignore
CXCursor_OMPTeamsDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeSimdDirective', 272) # type: ignore
CXCursor_OMPTeamsDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeParallelForSimdDirective', 273) # type: ignore
CXCursor_OMPTeamsDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsDistributeParallelForDirective', 274) # type: ignore
CXCursor_OMPTargetTeamsDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDirective', 275) # type: ignore
CXCursor_OMPTargetTeamsDistributeDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeDirective', 276) # type: ignore
CXCursor_OMPTargetTeamsDistributeParallelForDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeParallelForDirective', 277) # type: ignore
CXCursor_OMPTargetTeamsDistributeParallelForSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeParallelForSimdDirective', 278) # type: ignore
CXCursor_OMPTargetTeamsDistributeSimdDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsDistributeSimdDirective', 279) # type: ignore
CXCursor_BuiltinBitCastExpr = enum_CXCursorKind.define('CXCursor_BuiltinBitCastExpr', 280) # type: ignore
CXCursor_OMPMasterTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPMasterTaskLoopDirective', 281) # type: ignore
CXCursor_OMPParallelMasterTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterTaskLoopDirective', 282) # type: ignore
CXCursor_OMPMasterTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPMasterTaskLoopSimdDirective', 283) # type: ignore
CXCursor_OMPParallelMasterTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterTaskLoopSimdDirective', 284) # type: ignore
CXCursor_OMPParallelMasterDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMasterDirective', 285) # type: ignore
CXCursor_OMPDepobjDirective = enum_CXCursorKind.define('CXCursor_OMPDepobjDirective', 286) # type: ignore
CXCursor_OMPScanDirective = enum_CXCursorKind.define('CXCursor_OMPScanDirective', 287) # type: ignore
CXCursor_OMPTileDirective = enum_CXCursorKind.define('CXCursor_OMPTileDirective', 288) # type: ignore
CXCursor_OMPCanonicalLoop = enum_CXCursorKind.define('CXCursor_OMPCanonicalLoop', 289) # type: ignore
CXCursor_OMPInteropDirective = enum_CXCursorKind.define('CXCursor_OMPInteropDirective', 290) # type: ignore
CXCursor_OMPDispatchDirective = enum_CXCursorKind.define('CXCursor_OMPDispatchDirective', 291) # type: ignore
CXCursor_OMPMaskedDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedDirective', 292) # type: ignore
CXCursor_OMPUnrollDirective = enum_CXCursorKind.define('CXCursor_OMPUnrollDirective', 293) # type: ignore
CXCursor_OMPMetaDirective = enum_CXCursorKind.define('CXCursor_OMPMetaDirective', 294) # type: ignore
CXCursor_OMPGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPGenericLoopDirective', 295) # type: ignore
CXCursor_OMPTeamsGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTeamsGenericLoopDirective', 296) # type: ignore
CXCursor_OMPTargetTeamsGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTargetTeamsGenericLoopDirective', 297) # type: ignore
CXCursor_OMPParallelGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelGenericLoopDirective', 298) # type: ignore
CXCursor_OMPTargetParallelGenericLoopDirective = enum_CXCursorKind.define('CXCursor_OMPTargetParallelGenericLoopDirective', 299) # type: ignore
CXCursor_OMPParallelMaskedDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedDirective', 300) # type: ignore
CXCursor_OMPMaskedTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedTaskLoopDirective', 301) # type: ignore
CXCursor_OMPMaskedTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPMaskedTaskLoopSimdDirective', 302) # type: ignore
CXCursor_OMPParallelMaskedTaskLoopDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedTaskLoopDirective', 303) # type: ignore
CXCursor_OMPParallelMaskedTaskLoopSimdDirective = enum_CXCursorKind.define('CXCursor_OMPParallelMaskedTaskLoopSimdDirective', 304) # type: ignore
CXCursor_OMPErrorDirective = enum_CXCursorKind.define('CXCursor_OMPErrorDirective', 305) # type: ignore
CXCursor_OMPScopeDirective = enum_CXCursorKind.define('CXCursor_OMPScopeDirective', 306) # type: ignore
CXCursor_OMPReverseDirective = enum_CXCursorKind.define('CXCursor_OMPReverseDirective', 307) # type: ignore
CXCursor_OMPInterchangeDirective = enum_CXCursorKind.define('CXCursor_OMPInterchangeDirective', 308) # type: ignore
CXCursor_OMPAssumeDirective = enum_CXCursorKind.define('CXCursor_OMPAssumeDirective', 309) # type: ignore
CXCursor_OpenACCComputeConstruct = enum_CXCursorKind.define('CXCursor_OpenACCComputeConstruct', 320) # type: ignore
CXCursor_OpenACCLoopConstruct = enum_CXCursorKind.define('CXCursor_OpenACCLoopConstruct', 321) # type: ignore
CXCursor_OpenACCCombinedConstruct = enum_CXCursorKind.define('CXCursor_OpenACCCombinedConstruct', 322) # type: ignore
CXCursor_OpenACCDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCDataConstruct', 323) # type: ignore
CXCursor_OpenACCEnterDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCEnterDataConstruct', 324) # type: ignore
CXCursor_OpenACCExitDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCExitDataConstruct', 325) # type: ignore
CXCursor_OpenACCHostDataConstruct = enum_CXCursorKind.define('CXCursor_OpenACCHostDataConstruct', 326) # type: ignore
CXCursor_OpenACCWaitConstruct = enum_CXCursorKind.define('CXCursor_OpenACCWaitConstruct', 327) # type: ignore
CXCursor_OpenACCInitConstruct = enum_CXCursorKind.define('CXCursor_OpenACCInitConstruct', 328) # type: ignore
CXCursor_OpenACCShutdownConstruct = enum_CXCursorKind.define('CXCursor_OpenACCShutdownConstruct', 329) # type: ignore
CXCursor_OpenACCSetConstruct = enum_CXCursorKind.define('CXCursor_OpenACCSetConstruct', 330) # type: ignore
CXCursor_OpenACCUpdateConstruct = enum_CXCursorKind.define('CXCursor_OpenACCUpdateConstruct', 331) # type: ignore
CXCursor_LastStmt = enum_CXCursorKind.define('CXCursor_LastStmt', 331) # type: ignore
CXCursor_TranslationUnit = enum_CXCursorKind.define('CXCursor_TranslationUnit', 350) # type: ignore
CXCursor_FirstAttr = enum_CXCursorKind.define('CXCursor_FirstAttr', 400) # type: ignore
CXCursor_UnexposedAttr = enum_CXCursorKind.define('CXCursor_UnexposedAttr', 400) # type: ignore
CXCursor_IBActionAttr = enum_CXCursorKind.define('CXCursor_IBActionAttr', 401) # type: ignore
CXCursor_IBOutletAttr = enum_CXCursorKind.define('CXCursor_IBOutletAttr', 402) # type: ignore
CXCursor_IBOutletCollectionAttr = enum_CXCursorKind.define('CXCursor_IBOutletCollectionAttr', 403) # type: ignore
CXCursor_CXXFinalAttr = enum_CXCursorKind.define('CXCursor_CXXFinalAttr', 404) # type: ignore
CXCursor_CXXOverrideAttr = enum_CXCursorKind.define('CXCursor_CXXOverrideAttr', 405) # type: ignore
CXCursor_AnnotateAttr = enum_CXCursorKind.define('CXCursor_AnnotateAttr', 406) # type: ignore
CXCursor_AsmLabelAttr = enum_CXCursorKind.define('CXCursor_AsmLabelAttr', 407) # type: ignore
CXCursor_PackedAttr = enum_CXCursorKind.define('CXCursor_PackedAttr', 408) # type: ignore
CXCursor_PureAttr = enum_CXCursorKind.define('CXCursor_PureAttr', 409) # type: ignore
CXCursor_ConstAttr = enum_CXCursorKind.define('CXCursor_ConstAttr', 410) # type: ignore
CXCursor_NoDuplicateAttr = enum_CXCursorKind.define('CXCursor_NoDuplicateAttr', 411) # type: ignore
CXCursor_CUDAConstantAttr = enum_CXCursorKind.define('CXCursor_CUDAConstantAttr', 412) # type: ignore
CXCursor_CUDADeviceAttr = enum_CXCursorKind.define('CXCursor_CUDADeviceAttr', 413) # type: ignore
CXCursor_CUDAGlobalAttr = enum_CXCursorKind.define('CXCursor_CUDAGlobalAttr', 414) # type: ignore
CXCursor_CUDAHostAttr = enum_CXCursorKind.define('CXCursor_CUDAHostAttr', 415) # type: ignore
CXCursor_CUDASharedAttr = enum_CXCursorKind.define('CXCursor_CUDASharedAttr', 416) # type: ignore
CXCursor_VisibilityAttr = enum_CXCursorKind.define('CXCursor_VisibilityAttr', 417) # type: ignore
CXCursor_DLLExport = enum_CXCursorKind.define('CXCursor_DLLExport', 418) # type: ignore
CXCursor_DLLImport = enum_CXCursorKind.define('CXCursor_DLLImport', 419) # type: ignore
CXCursor_NSReturnsRetained = enum_CXCursorKind.define('CXCursor_NSReturnsRetained', 420) # type: ignore
CXCursor_NSReturnsNotRetained = enum_CXCursorKind.define('CXCursor_NSReturnsNotRetained', 421) # type: ignore
CXCursor_NSReturnsAutoreleased = enum_CXCursorKind.define('CXCursor_NSReturnsAutoreleased', 422) # type: ignore
CXCursor_NSConsumesSelf = enum_CXCursorKind.define('CXCursor_NSConsumesSelf', 423) # type: ignore
CXCursor_NSConsumed = enum_CXCursorKind.define('CXCursor_NSConsumed', 424) # type: ignore
CXCursor_ObjCException = enum_CXCursorKind.define('CXCursor_ObjCException', 425) # type: ignore
CXCursor_ObjCNSObject = enum_CXCursorKind.define('CXCursor_ObjCNSObject', 426) # type: ignore
CXCursor_ObjCIndependentClass = enum_CXCursorKind.define('CXCursor_ObjCIndependentClass', 427) # type: ignore
CXCursor_ObjCPreciseLifetime = enum_CXCursorKind.define('CXCursor_ObjCPreciseLifetime', 428) # type: ignore
CXCursor_ObjCReturnsInnerPointer = enum_CXCursorKind.define('CXCursor_ObjCReturnsInnerPointer', 429) # type: ignore
CXCursor_ObjCRequiresSuper = enum_CXCursorKind.define('CXCursor_ObjCRequiresSuper', 430) # type: ignore
CXCursor_ObjCRootClass = enum_CXCursorKind.define('CXCursor_ObjCRootClass', 431) # type: ignore
CXCursor_ObjCSubclassingRestricted = enum_CXCursorKind.define('CXCursor_ObjCSubclassingRestricted', 432) # type: ignore
CXCursor_ObjCExplicitProtocolImpl = enum_CXCursorKind.define('CXCursor_ObjCExplicitProtocolImpl', 433) # type: ignore
CXCursor_ObjCDesignatedInitializer = enum_CXCursorKind.define('CXCursor_ObjCDesignatedInitializer', 434) # type: ignore
CXCursor_ObjCRuntimeVisible = enum_CXCursorKind.define('CXCursor_ObjCRuntimeVisible', 435) # type: ignore
CXCursor_ObjCBoxable = enum_CXCursorKind.define('CXCursor_ObjCBoxable', 436) # type: ignore
CXCursor_FlagEnum = enum_CXCursorKind.define('CXCursor_FlagEnum', 437) # type: ignore
CXCursor_ConvergentAttr = enum_CXCursorKind.define('CXCursor_ConvergentAttr', 438) # type: ignore
CXCursor_WarnUnusedAttr = enum_CXCursorKind.define('CXCursor_WarnUnusedAttr', 439) # type: ignore
CXCursor_WarnUnusedResultAttr = enum_CXCursorKind.define('CXCursor_WarnUnusedResultAttr', 440) # type: ignore
CXCursor_AlignedAttr = enum_CXCursorKind.define('CXCursor_AlignedAttr', 441) # type: ignore
CXCursor_LastAttr = enum_CXCursorKind.define('CXCursor_LastAttr', 441) # type: ignore
CXCursor_PreprocessingDirective = enum_CXCursorKind.define('CXCursor_PreprocessingDirective', 500) # type: ignore
CXCursor_MacroDefinition = enum_CXCursorKind.define('CXCursor_MacroDefinition', 501) # type: ignore
CXCursor_MacroExpansion = enum_CXCursorKind.define('CXCursor_MacroExpansion', 502) # type: ignore
CXCursor_MacroInstantiation = enum_CXCursorKind.define('CXCursor_MacroInstantiation', 502) # type: ignore
CXCursor_InclusionDirective = enum_CXCursorKind.define('CXCursor_InclusionDirective', 503) # type: ignore
CXCursor_FirstPreprocessing = enum_CXCursorKind.define('CXCursor_FirstPreprocessing', 500) # type: ignore
CXCursor_LastPreprocessing = enum_CXCursorKind.define('CXCursor_LastPreprocessing', 503) # type: ignore
CXCursor_ModuleImportDecl = enum_CXCursorKind.define('CXCursor_ModuleImportDecl', 600) # type: ignore
CXCursor_TypeAliasTemplateDecl = enum_CXCursorKind.define('CXCursor_TypeAliasTemplateDecl', 601) # type: ignore
CXCursor_StaticAssert = enum_CXCursorKind.define('CXCursor_StaticAssert', 602) # type: ignore
CXCursor_FriendDecl = enum_CXCursorKind.define('CXCursor_FriendDecl', 603) # type: ignore
CXCursor_ConceptDecl = enum_CXCursorKind.define('CXCursor_ConceptDecl', 604) # type: ignore
CXCursor_FirstExtraDecl = enum_CXCursorKind.define('CXCursor_FirstExtraDecl', 600) # type: ignore
CXCursor_LastExtraDecl = enum_CXCursorKind.define('CXCursor_LastExtraDecl', 604) # type: ignore
CXCursor_OverloadCandidate = enum_CXCursorKind.define('CXCursor_OverloadCandidate', 700) # type: ignore

@c.record
class CXCursor(c.Struct):
  SIZE = 32
  kind: Annotated[enum_CXCursorKind, 0]
  xdata: Annotated[Annotated[int, ctypes.c_int32], 4]
  data: Annotated[c.Array[c.POINTER[None], Literal[3]], 8]
@dll.bind
def clang_getNullCursor() -> CXCursor: ...
@dll.bind
def clang_getTranslationUnitCursor(_0:CXTranslationUnit) -> CXCursor: ...
@dll.bind
def clang_equalCursors(_0:CXCursor, _1:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isNull(cursor:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_hashCursor(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorKind(_0:CXCursor) -> enum_CXCursorKind: ...
@dll.bind
def clang_isDeclaration(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isInvalidDeclaration(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isReference(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isExpression(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isStatement(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isAttribute(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_hasAttrs(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isInvalid(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isTranslationUnit(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isPreprocessing(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isUnexposed(_0:enum_CXCursorKind) -> Annotated[int, ctypes.c_uint32]: ...
enum_CXLinkageKind = CEnum(Annotated[int, ctypes.c_uint32])
CXLinkage_Invalid = enum_CXLinkageKind.define('CXLinkage_Invalid', 0) # type: ignore
CXLinkage_NoLinkage = enum_CXLinkageKind.define('CXLinkage_NoLinkage', 1) # type: ignore
CXLinkage_Internal = enum_CXLinkageKind.define('CXLinkage_Internal', 2) # type: ignore
CXLinkage_UniqueExternal = enum_CXLinkageKind.define('CXLinkage_UniqueExternal', 3) # type: ignore
CXLinkage_External = enum_CXLinkageKind.define('CXLinkage_External', 4) # type: ignore

@dll.bind
def clang_getCursorLinkage(cursor:CXCursor) -> enum_CXLinkageKind: ...
enum_CXVisibilityKind = CEnum(Annotated[int, ctypes.c_uint32])
CXVisibility_Invalid = enum_CXVisibilityKind.define('CXVisibility_Invalid', 0) # type: ignore
CXVisibility_Hidden = enum_CXVisibilityKind.define('CXVisibility_Hidden', 1) # type: ignore
CXVisibility_Protected = enum_CXVisibilityKind.define('CXVisibility_Protected', 2) # type: ignore
CXVisibility_Default = enum_CXVisibilityKind.define('CXVisibility_Default', 3) # type: ignore

@dll.bind
def clang_getCursorVisibility(cursor:CXCursor) -> enum_CXVisibilityKind: ...
@dll.bind
def clang_getCursorAvailability(cursor:CXCursor) -> enum_CXAvailabilityKind: ...
@c.record
class struct_CXPlatformAvailability(c.Struct):
  SIZE = 72
  Platform: Annotated[CXString, 0]
  Introduced: Annotated[CXVersion, 16]
  Deprecated: Annotated[CXVersion, 28]
  Obsoleted: Annotated[CXVersion, 40]
  Unavailable: Annotated[Annotated[int, ctypes.c_int32], 52]
  Message: Annotated[CXString, 56]
CXPlatformAvailability = struct_CXPlatformAvailability
@dll.bind
def clang_getCursorPlatformAvailability(cursor:CXCursor, always_deprecated:c.POINTER[Annotated[int, ctypes.c_int32]], deprecated_message:c.POINTER[CXString], always_unavailable:c.POINTER[Annotated[int, ctypes.c_int32]], unavailable_message:c.POINTER[CXString], availability:c.POINTER[CXPlatformAvailability], availability_size:Annotated[int, ctypes.c_int32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_disposeCXPlatformAvailability(availability:c.POINTER[CXPlatformAvailability]) -> None: ...
@dll.bind
def clang_Cursor_getVarDeclInitializer(cursor:CXCursor) -> CXCursor: ...
@dll.bind
def clang_Cursor_hasVarDeclGlobalStorage(cursor:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_hasVarDeclExternalStorage(cursor:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
enum_CXLanguageKind = CEnum(Annotated[int, ctypes.c_uint32])
CXLanguage_Invalid = enum_CXLanguageKind.define('CXLanguage_Invalid', 0) # type: ignore
CXLanguage_C = enum_CXLanguageKind.define('CXLanguage_C', 1) # type: ignore
CXLanguage_ObjC = enum_CXLanguageKind.define('CXLanguage_ObjC', 2) # type: ignore
CXLanguage_CPlusPlus = enum_CXLanguageKind.define('CXLanguage_CPlusPlus', 3) # type: ignore

@dll.bind
def clang_getCursorLanguage(cursor:CXCursor) -> enum_CXLanguageKind: ...
enum_CXTLSKind = CEnum(Annotated[int, ctypes.c_uint32])
CXTLS_None = enum_CXTLSKind.define('CXTLS_None', 0) # type: ignore
CXTLS_Dynamic = enum_CXTLSKind.define('CXTLS_Dynamic', 1) # type: ignore
CXTLS_Static = enum_CXTLSKind.define('CXTLS_Static', 2) # type: ignore

@dll.bind
def clang_getCursorTLSKind(cursor:CXCursor) -> enum_CXTLSKind: ...
@dll.bind
def clang_Cursor_getTranslationUnit(_0:CXCursor) -> CXTranslationUnit: ...
class struct_CXCursorSetImpl(ctypes.Structure): pass
CXCursorSet = c.POINTER[struct_CXCursorSetImpl]
@dll.bind
def clang_createCXCursorSet() -> CXCursorSet: ...
@dll.bind
def clang_disposeCXCursorSet(cset:CXCursorSet) -> None: ...
@dll.bind
def clang_CXCursorSet_contains(cset:CXCursorSet, cursor:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXCursorSet_insert(cset:CXCursorSet, cursor:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorSemanticParent(cursor:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getCursorLexicalParent(cursor:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getOverriddenCursors(cursor:CXCursor, overridden:c.POINTER[c.POINTER[CXCursor]], num_overridden:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_disposeOverriddenCursors(overridden:c.POINTER[CXCursor]) -> None: ...
@dll.bind
def clang_getIncludedFile(cursor:CXCursor) -> CXFile: ...
@dll.bind
def clang_getCursor(_0:CXTranslationUnit, _1:CXSourceLocation) -> CXCursor: ...
@dll.bind
def clang_getCursorLocation(_0:CXCursor) -> CXSourceLocation: ...
@dll.bind
def clang_getCursorExtent(_0:CXCursor) -> CXSourceRange: ...
enum_CXTypeKind = CEnum(Annotated[int, ctypes.c_uint32])
CXType_Invalid = enum_CXTypeKind.define('CXType_Invalid', 0) # type: ignore
CXType_Unexposed = enum_CXTypeKind.define('CXType_Unexposed', 1) # type: ignore
CXType_Void = enum_CXTypeKind.define('CXType_Void', 2) # type: ignore
CXType_Bool = enum_CXTypeKind.define('CXType_Bool', 3) # type: ignore
CXType_Char_U = enum_CXTypeKind.define('CXType_Char_U', 4) # type: ignore
CXType_UChar = enum_CXTypeKind.define('CXType_UChar', 5) # type: ignore
CXType_Char16 = enum_CXTypeKind.define('CXType_Char16', 6) # type: ignore
CXType_Char32 = enum_CXTypeKind.define('CXType_Char32', 7) # type: ignore
CXType_UShort = enum_CXTypeKind.define('CXType_UShort', 8) # type: ignore
CXType_UInt = enum_CXTypeKind.define('CXType_UInt', 9) # type: ignore
CXType_ULong = enum_CXTypeKind.define('CXType_ULong', 10) # type: ignore
CXType_ULongLong = enum_CXTypeKind.define('CXType_ULongLong', 11) # type: ignore
CXType_UInt128 = enum_CXTypeKind.define('CXType_UInt128', 12) # type: ignore
CXType_Char_S = enum_CXTypeKind.define('CXType_Char_S', 13) # type: ignore
CXType_SChar = enum_CXTypeKind.define('CXType_SChar', 14) # type: ignore
CXType_WChar = enum_CXTypeKind.define('CXType_WChar', 15) # type: ignore
CXType_Short = enum_CXTypeKind.define('CXType_Short', 16) # type: ignore
CXType_Int = enum_CXTypeKind.define('CXType_Int', 17) # type: ignore
CXType_Long = enum_CXTypeKind.define('CXType_Long', 18) # type: ignore
CXType_LongLong = enum_CXTypeKind.define('CXType_LongLong', 19) # type: ignore
CXType_Int128 = enum_CXTypeKind.define('CXType_Int128', 20) # type: ignore
CXType_Float = enum_CXTypeKind.define('CXType_Float', 21) # type: ignore
CXType_Double = enum_CXTypeKind.define('CXType_Double', 22) # type: ignore
CXType_LongDouble = enum_CXTypeKind.define('CXType_LongDouble', 23) # type: ignore
CXType_NullPtr = enum_CXTypeKind.define('CXType_NullPtr', 24) # type: ignore
CXType_Overload = enum_CXTypeKind.define('CXType_Overload', 25) # type: ignore
CXType_Dependent = enum_CXTypeKind.define('CXType_Dependent', 26) # type: ignore
CXType_ObjCId = enum_CXTypeKind.define('CXType_ObjCId', 27) # type: ignore
CXType_ObjCClass = enum_CXTypeKind.define('CXType_ObjCClass', 28) # type: ignore
CXType_ObjCSel = enum_CXTypeKind.define('CXType_ObjCSel', 29) # type: ignore
CXType_Float128 = enum_CXTypeKind.define('CXType_Float128', 30) # type: ignore
CXType_Half = enum_CXTypeKind.define('CXType_Half', 31) # type: ignore
CXType_Float16 = enum_CXTypeKind.define('CXType_Float16', 32) # type: ignore
CXType_ShortAccum = enum_CXTypeKind.define('CXType_ShortAccum', 33) # type: ignore
CXType_Accum = enum_CXTypeKind.define('CXType_Accum', 34) # type: ignore
CXType_LongAccum = enum_CXTypeKind.define('CXType_LongAccum', 35) # type: ignore
CXType_UShortAccum = enum_CXTypeKind.define('CXType_UShortAccum', 36) # type: ignore
CXType_UAccum = enum_CXTypeKind.define('CXType_UAccum', 37) # type: ignore
CXType_ULongAccum = enum_CXTypeKind.define('CXType_ULongAccum', 38) # type: ignore
CXType_BFloat16 = enum_CXTypeKind.define('CXType_BFloat16', 39) # type: ignore
CXType_Ibm128 = enum_CXTypeKind.define('CXType_Ibm128', 40) # type: ignore
CXType_FirstBuiltin = enum_CXTypeKind.define('CXType_FirstBuiltin', 2) # type: ignore
CXType_LastBuiltin = enum_CXTypeKind.define('CXType_LastBuiltin', 40) # type: ignore
CXType_Complex = enum_CXTypeKind.define('CXType_Complex', 100) # type: ignore
CXType_Pointer = enum_CXTypeKind.define('CXType_Pointer', 101) # type: ignore
CXType_BlockPointer = enum_CXTypeKind.define('CXType_BlockPointer', 102) # type: ignore
CXType_LValueReference = enum_CXTypeKind.define('CXType_LValueReference', 103) # type: ignore
CXType_RValueReference = enum_CXTypeKind.define('CXType_RValueReference', 104) # type: ignore
CXType_Record = enum_CXTypeKind.define('CXType_Record', 105) # type: ignore
CXType_Enum = enum_CXTypeKind.define('CXType_Enum', 106) # type: ignore
CXType_Typedef = enum_CXTypeKind.define('CXType_Typedef', 107) # type: ignore
CXType_ObjCInterface = enum_CXTypeKind.define('CXType_ObjCInterface', 108) # type: ignore
CXType_ObjCObjectPointer = enum_CXTypeKind.define('CXType_ObjCObjectPointer', 109) # type: ignore
CXType_FunctionNoProto = enum_CXTypeKind.define('CXType_FunctionNoProto', 110) # type: ignore
CXType_FunctionProto = enum_CXTypeKind.define('CXType_FunctionProto', 111) # type: ignore
CXType_ConstantArray = enum_CXTypeKind.define('CXType_ConstantArray', 112) # type: ignore
CXType_Vector = enum_CXTypeKind.define('CXType_Vector', 113) # type: ignore
CXType_IncompleteArray = enum_CXTypeKind.define('CXType_IncompleteArray', 114) # type: ignore
CXType_VariableArray = enum_CXTypeKind.define('CXType_VariableArray', 115) # type: ignore
CXType_DependentSizedArray = enum_CXTypeKind.define('CXType_DependentSizedArray', 116) # type: ignore
CXType_MemberPointer = enum_CXTypeKind.define('CXType_MemberPointer', 117) # type: ignore
CXType_Auto = enum_CXTypeKind.define('CXType_Auto', 118) # type: ignore
CXType_Elaborated = enum_CXTypeKind.define('CXType_Elaborated', 119) # type: ignore
CXType_Pipe = enum_CXTypeKind.define('CXType_Pipe', 120) # type: ignore
CXType_OCLImage1dRO = enum_CXTypeKind.define('CXType_OCLImage1dRO', 121) # type: ignore
CXType_OCLImage1dArrayRO = enum_CXTypeKind.define('CXType_OCLImage1dArrayRO', 122) # type: ignore
CXType_OCLImage1dBufferRO = enum_CXTypeKind.define('CXType_OCLImage1dBufferRO', 123) # type: ignore
CXType_OCLImage2dRO = enum_CXTypeKind.define('CXType_OCLImage2dRO', 124) # type: ignore
CXType_OCLImage2dArrayRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayRO', 125) # type: ignore
CXType_OCLImage2dDepthRO = enum_CXTypeKind.define('CXType_OCLImage2dDepthRO', 126) # type: ignore
CXType_OCLImage2dArrayDepthRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthRO', 127) # type: ignore
CXType_OCLImage2dMSAARO = enum_CXTypeKind.define('CXType_OCLImage2dMSAARO', 128) # type: ignore
CXType_OCLImage2dArrayMSAARO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAARO', 129) # type: ignore
CXType_OCLImage2dMSAADepthRO = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthRO', 130) # type: ignore
CXType_OCLImage2dArrayMSAADepthRO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthRO', 131) # type: ignore
CXType_OCLImage3dRO = enum_CXTypeKind.define('CXType_OCLImage3dRO', 132) # type: ignore
CXType_OCLImage1dWO = enum_CXTypeKind.define('CXType_OCLImage1dWO', 133) # type: ignore
CXType_OCLImage1dArrayWO = enum_CXTypeKind.define('CXType_OCLImage1dArrayWO', 134) # type: ignore
CXType_OCLImage1dBufferWO = enum_CXTypeKind.define('CXType_OCLImage1dBufferWO', 135) # type: ignore
CXType_OCLImage2dWO = enum_CXTypeKind.define('CXType_OCLImage2dWO', 136) # type: ignore
CXType_OCLImage2dArrayWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayWO', 137) # type: ignore
CXType_OCLImage2dDepthWO = enum_CXTypeKind.define('CXType_OCLImage2dDepthWO', 138) # type: ignore
CXType_OCLImage2dArrayDepthWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthWO', 139) # type: ignore
CXType_OCLImage2dMSAAWO = enum_CXTypeKind.define('CXType_OCLImage2dMSAAWO', 140) # type: ignore
CXType_OCLImage2dArrayMSAAWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAAWO', 141) # type: ignore
CXType_OCLImage2dMSAADepthWO = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthWO', 142) # type: ignore
CXType_OCLImage2dArrayMSAADepthWO = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthWO', 143) # type: ignore
CXType_OCLImage3dWO = enum_CXTypeKind.define('CXType_OCLImage3dWO', 144) # type: ignore
CXType_OCLImage1dRW = enum_CXTypeKind.define('CXType_OCLImage1dRW', 145) # type: ignore
CXType_OCLImage1dArrayRW = enum_CXTypeKind.define('CXType_OCLImage1dArrayRW', 146) # type: ignore
CXType_OCLImage1dBufferRW = enum_CXTypeKind.define('CXType_OCLImage1dBufferRW', 147) # type: ignore
CXType_OCLImage2dRW = enum_CXTypeKind.define('CXType_OCLImage2dRW', 148) # type: ignore
CXType_OCLImage2dArrayRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayRW', 149) # type: ignore
CXType_OCLImage2dDepthRW = enum_CXTypeKind.define('CXType_OCLImage2dDepthRW', 150) # type: ignore
CXType_OCLImage2dArrayDepthRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayDepthRW', 151) # type: ignore
CXType_OCLImage2dMSAARW = enum_CXTypeKind.define('CXType_OCLImage2dMSAARW', 152) # type: ignore
CXType_OCLImage2dArrayMSAARW = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAARW', 153) # type: ignore
CXType_OCLImage2dMSAADepthRW = enum_CXTypeKind.define('CXType_OCLImage2dMSAADepthRW', 154) # type: ignore
CXType_OCLImage2dArrayMSAADepthRW = enum_CXTypeKind.define('CXType_OCLImage2dArrayMSAADepthRW', 155) # type: ignore
CXType_OCLImage3dRW = enum_CXTypeKind.define('CXType_OCLImage3dRW', 156) # type: ignore
CXType_OCLSampler = enum_CXTypeKind.define('CXType_OCLSampler', 157) # type: ignore
CXType_OCLEvent = enum_CXTypeKind.define('CXType_OCLEvent', 158) # type: ignore
CXType_OCLQueue = enum_CXTypeKind.define('CXType_OCLQueue', 159) # type: ignore
CXType_OCLReserveID = enum_CXTypeKind.define('CXType_OCLReserveID', 160) # type: ignore
CXType_ObjCObject = enum_CXTypeKind.define('CXType_ObjCObject', 161) # type: ignore
CXType_ObjCTypeParam = enum_CXTypeKind.define('CXType_ObjCTypeParam', 162) # type: ignore
CXType_Attributed = enum_CXTypeKind.define('CXType_Attributed', 163) # type: ignore
CXType_OCLIntelSubgroupAVCMcePayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCMcePayload', 164) # type: ignore
CXType_OCLIntelSubgroupAVCImePayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImePayload', 165) # type: ignore
CXType_OCLIntelSubgroupAVCRefPayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCRefPayload', 166) # type: ignore
CXType_OCLIntelSubgroupAVCSicPayload = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCSicPayload', 167) # type: ignore
CXType_OCLIntelSubgroupAVCMceResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCMceResult', 168) # type: ignore
CXType_OCLIntelSubgroupAVCImeResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResult', 169) # type: ignore
CXType_OCLIntelSubgroupAVCRefResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCRefResult', 170) # type: ignore
CXType_OCLIntelSubgroupAVCSicResult = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCSicResult', 171) # type: ignore
CXType_OCLIntelSubgroupAVCImeResultSingleReferenceStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultSingleReferenceStreamout', 172) # type: ignore
CXType_OCLIntelSubgroupAVCImeResultDualReferenceStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultDualReferenceStreamout', 173) # type: ignore
CXType_OCLIntelSubgroupAVCImeSingleReferenceStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeSingleReferenceStreamin', 174) # type: ignore
CXType_OCLIntelSubgroupAVCImeDualReferenceStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeDualReferenceStreamin', 175) # type: ignore
CXType_OCLIntelSubgroupAVCImeResultSingleRefStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultSingleRefStreamout', 172) # type: ignore
CXType_OCLIntelSubgroupAVCImeResultDualRefStreamout = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeResultDualRefStreamout', 173) # type: ignore
CXType_OCLIntelSubgroupAVCImeSingleRefStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeSingleRefStreamin', 174) # type: ignore
CXType_OCLIntelSubgroupAVCImeDualRefStreamin = enum_CXTypeKind.define('CXType_OCLIntelSubgroupAVCImeDualRefStreamin', 175) # type: ignore
CXType_ExtVector = enum_CXTypeKind.define('CXType_ExtVector', 176) # type: ignore
CXType_Atomic = enum_CXTypeKind.define('CXType_Atomic', 177) # type: ignore
CXType_BTFTagAttributed = enum_CXTypeKind.define('CXType_BTFTagAttributed', 178) # type: ignore
CXType_HLSLResource = enum_CXTypeKind.define('CXType_HLSLResource', 179) # type: ignore
CXType_HLSLAttributedResource = enum_CXTypeKind.define('CXType_HLSLAttributedResource', 180) # type: ignore

enum_CXCallingConv = CEnum(Annotated[int, ctypes.c_uint32])
CXCallingConv_Default = enum_CXCallingConv.define('CXCallingConv_Default', 0) # type: ignore
CXCallingConv_C = enum_CXCallingConv.define('CXCallingConv_C', 1) # type: ignore
CXCallingConv_X86StdCall = enum_CXCallingConv.define('CXCallingConv_X86StdCall', 2) # type: ignore
CXCallingConv_X86FastCall = enum_CXCallingConv.define('CXCallingConv_X86FastCall', 3) # type: ignore
CXCallingConv_X86ThisCall = enum_CXCallingConv.define('CXCallingConv_X86ThisCall', 4) # type: ignore
CXCallingConv_X86Pascal = enum_CXCallingConv.define('CXCallingConv_X86Pascal', 5) # type: ignore
CXCallingConv_AAPCS = enum_CXCallingConv.define('CXCallingConv_AAPCS', 6) # type: ignore
CXCallingConv_AAPCS_VFP = enum_CXCallingConv.define('CXCallingConv_AAPCS_VFP', 7) # type: ignore
CXCallingConv_X86RegCall = enum_CXCallingConv.define('CXCallingConv_X86RegCall', 8) # type: ignore
CXCallingConv_IntelOclBicc = enum_CXCallingConv.define('CXCallingConv_IntelOclBicc', 9) # type: ignore
CXCallingConv_Win64 = enum_CXCallingConv.define('CXCallingConv_Win64', 10) # type: ignore
CXCallingConv_X86_64Win64 = enum_CXCallingConv.define('CXCallingConv_X86_64Win64', 10) # type: ignore
CXCallingConv_X86_64SysV = enum_CXCallingConv.define('CXCallingConv_X86_64SysV', 11) # type: ignore
CXCallingConv_X86VectorCall = enum_CXCallingConv.define('CXCallingConv_X86VectorCall', 12) # type: ignore
CXCallingConv_Swift = enum_CXCallingConv.define('CXCallingConv_Swift', 13) # type: ignore
CXCallingConv_PreserveMost = enum_CXCallingConv.define('CXCallingConv_PreserveMost', 14) # type: ignore
CXCallingConv_PreserveAll = enum_CXCallingConv.define('CXCallingConv_PreserveAll', 15) # type: ignore
CXCallingConv_AArch64VectorCall = enum_CXCallingConv.define('CXCallingConv_AArch64VectorCall', 16) # type: ignore
CXCallingConv_SwiftAsync = enum_CXCallingConv.define('CXCallingConv_SwiftAsync', 17) # type: ignore
CXCallingConv_AArch64SVEPCS = enum_CXCallingConv.define('CXCallingConv_AArch64SVEPCS', 18) # type: ignore
CXCallingConv_M68kRTD = enum_CXCallingConv.define('CXCallingConv_M68kRTD', 19) # type: ignore
CXCallingConv_PreserveNone = enum_CXCallingConv.define('CXCallingConv_PreserveNone', 20) # type: ignore
CXCallingConv_RISCVVectorCall = enum_CXCallingConv.define('CXCallingConv_RISCVVectorCall', 21) # type: ignore
CXCallingConv_Invalid = enum_CXCallingConv.define('CXCallingConv_Invalid', 100) # type: ignore
CXCallingConv_Unexposed = enum_CXCallingConv.define('CXCallingConv_Unexposed', 200) # type: ignore

@c.record
class CXType(c.Struct):
  SIZE = 24
  kind: Annotated[enum_CXTypeKind, 0]
  data: Annotated[c.Array[c.POINTER[None], Literal[2]], 8]
@dll.bind
def clang_getCursorType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getTypeSpelling(CT:CXType) -> CXString: ...
@dll.bind
def clang_getTypedefDeclUnderlyingType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getEnumDeclIntegerType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getEnumConstantDeclValue(C:CXCursor) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_getEnumConstantDeclUnsignedValue(C:CXCursor) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_Cursor_isBitField(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getFieldDeclBitWidth(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getNumArguments(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getArgument(C:CXCursor, i:Annotated[int, ctypes.c_uint32]) -> CXCursor: ...
enum_CXTemplateArgumentKind = CEnum(Annotated[int, ctypes.c_uint32])
CXTemplateArgumentKind_Null = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Null', 0) # type: ignore
CXTemplateArgumentKind_Type = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Type', 1) # type: ignore
CXTemplateArgumentKind_Declaration = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Declaration', 2) # type: ignore
CXTemplateArgumentKind_NullPtr = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_NullPtr', 3) # type: ignore
CXTemplateArgumentKind_Integral = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Integral', 4) # type: ignore
CXTemplateArgumentKind_Template = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Template', 5) # type: ignore
CXTemplateArgumentKind_TemplateExpansion = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_TemplateExpansion', 6) # type: ignore
CXTemplateArgumentKind_Expression = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Expression', 7) # type: ignore
CXTemplateArgumentKind_Pack = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Pack', 8) # type: ignore
CXTemplateArgumentKind_Invalid = enum_CXTemplateArgumentKind.define('CXTemplateArgumentKind_Invalid', 9) # type: ignore

@dll.bind
def clang_Cursor_getNumTemplateArguments(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getTemplateArgumentKind(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> enum_CXTemplateArgumentKind: ...
@dll.bind
def clang_Cursor_getTemplateArgumentType(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_Cursor_getTemplateArgumentValue(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Cursor_getTemplateArgumentUnsignedValue(C:CXCursor, I:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_equalTypes(A:CXType, B:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCanonicalType(T:CXType) -> CXType: ...
@dll.bind
def clang_isConstQualifiedType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isMacroFunctionLike(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isMacroBuiltin(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isFunctionInlined(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isVolatileQualifiedType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isRestrictQualifiedType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getAddressSpace(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getTypedefName(CT:CXType) -> CXString: ...
@dll.bind
def clang_getPointeeType(T:CXType) -> CXType: ...
@dll.bind
def clang_getUnqualifiedType(CT:CXType) -> CXType: ...
@dll.bind
def clang_getNonReferenceType(CT:CXType) -> CXType: ...
@dll.bind
def clang_getTypeDeclaration(T:CXType) -> CXCursor: ...
@dll.bind
def clang_getDeclObjCTypeEncoding(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Type_getObjCEncoding(type:CXType) -> CXString: ...
@dll.bind
def clang_getTypeKindSpelling(K:enum_CXTypeKind) -> CXString: ...
@dll.bind
def clang_getFunctionTypeCallingConv(T:CXType) -> enum_CXCallingConv: ...
@dll.bind
def clang_getResultType(T:CXType) -> CXType: ...
@dll.bind
def clang_getExceptionSpecificationType(T:CXType) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getNumArgTypes(T:CXType) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getArgType(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_Type_getObjCObjectBaseType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_getNumObjCProtocolRefs(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Type_getObjCProtocolDecl(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXCursor: ...
@dll.bind
def clang_Type_getNumObjCTypeArgs(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Type_getObjCTypeArg(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_isFunctionTypeVariadic(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorResultType(C:CXCursor) -> CXType: ...
@dll.bind
def clang_getCursorExceptionSpecificationType(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_isPODType(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getElementType(T:CXType) -> CXType: ...
@dll.bind
def clang_getNumElements(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_getArrayElementType(T:CXType) -> CXType: ...
@dll.bind
def clang_getArraySize(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getNamedType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_isTransparentTagTypedef(T:CXType) -> Annotated[int, ctypes.c_uint32]: ...
enum_CXTypeNullabilityKind = CEnum(Annotated[int, ctypes.c_uint32])
CXTypeNullability_NonNull = enum_CXTypeNullabilityKind.define('CXTypeNullability_NonNull', 0) # type: ignore
CXTypeNullability_Nullable = enum_CXTypeNullabilityKind.define('CXTypeNullability_Nullable', 1) # type: ignore
CXTypeNullability_Unspecified = enum_CXTypeNullabilityKind.define('CXTypeNullability_Unspecified', 2) # type: ignore
CXTypeNullability_Invalid = enum_CXTypeNullabilityKind.define('CXTypeNullability_Invalid', 3) # type: ignore
CXTypeNullability_NullableResult = enum_CXTypeNullabilityKind.define('CXTypeNullability_NullableResult', 4) # type: ignore

@dll.bind
def clang_Type_getNullability(T:CXType) -> enum_CXTypeNullabilityKind: ...
enum_CXTypeLayoutError = CEnum(Annotated[int, ctypes.c_int32])
CXTypeLayoutError_Invalid = enum_CXTypeLayoutError.define('CXTypeLayoutError_Invalid', -1) # type: ignore
CXTypeLayoutError_Incomplete = enum_CXTypeLayoutError.define('CXTypeLayoutError_Incomplete', -2) # type: ignore
CXTypeLayoutError_Dependent = enum_CXTypeLayoutError.define('CXTypeLayoutError_Dependent', -3) # type: ignore
CXTypeLayoutError_NotConstantSize = enum_CXTypeLayoutError.define('CXTypeLayoutError_NotConstantSize', -4) # type: ignore
CXTypeLayoutError_InvalidFieldName = enum_CXTypeLayoutError.define('CXTypeLayoutError_InvalidFieldName', -5) # type: ignore
CXTypeLayoutError_Undeduced = enum_CXTypeLayoutError.define('CXTypeLayoutError_Undeduced', -6) # type: ignore

@dll.bind
def clang_Type_getAlignOf(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getClassType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_getSizeOf(T:CXType) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getOffsetOf(T:CXType, S:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Type_getModifiedType(T:CXType) -> CXType: ...
@dll.bind
def clang_Type_getValueType(CT:CXType) -> CXType: ...
@dll.bind
def clang_Cursor_getOffsetOfField(C:CXCursor) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_Cursor_isAnonymous(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isAnonymousRecordDecl(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isInlineNamespace(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
enum_CXRefQualifierKind = CEnum(Annotated[int, ctypes.c_uint32])
CXRefQualifier_None = enum_CXRefQualifierKind.define('CXRefQualifier_None', 0) # type: ignore
CXRefQualifier_LValue = enum_CXRefQualifierKind.define('CXRefQualifier_LValue', 1) # type: ignore
CXRefQualifier_RValue = enum_CXRefQualifierKind.define('CXRefQualifier_RValue', 2) # type: ignore

@dll.bind
def clang_Type_getNumTemplateArguments(T:CXType) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Type_getTemplateArgumentAsType(T:CXType, i:Annotated[int, ctypes.c_uint32]) -> CXType: ...
@dll.bind
def clang_Type_getCXXRefQualifier(T:CXType) -> enum_CXRefQualifierKind: ...
@dll.bind
def clang_isVirtualBase(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getOffsetOfBase(Parent:CXCursor, Base:CXCursor) -> Annotated[int, ctypes.c_int64]: ...
enum_CX_CXXAccessSpecifier = CEnum(Annotated[int, ctypes.c_uint32])
CX_CXXInvalidAccessSpecifier = enum_CX_CXXAccessSpecifier.define('CX_CXXInvalidAccessSpecifier', 0) # type: ignore
CX_CXXPublic = enum_CX_CXXAccessSpecifier.define('CX_CXXPublic', 1) # type: ignore
CX_CXXProtected = enum_CX_CXXAccessSpecifier.define('CX_CXXProtected', 2) # type: ignore
CX_CXXPrivate = enum_CX_CXXAccessSpecifier.define('CX_CXXPrivate', 3) # type: ignore

@dll.bind
def clang_getCXXAccessSpecifier(_0:CXCursor) -> enum_CX_CXXAccessSpecifier: ...
enum_CX_StorageClass = CEnum(Annotated[int, ctypes.c_uint32])
CX_SC_Invalid = enum_CX_StorageClass.define('CX_SC_Invalid', 0) # type: ignore
CX_SC_None = enum_CX_StorageClass.define('CX_SC_None', 1) # type: ignore
CX_SC_Extern = enum_CX_StorageClass.define('CX_SC_Extern', 2) # type: ignore
CX_SC_Static = enum_CX_StorageClass.define('CX_SC_Static', 3) # type: ignore
CX_SC_PrivateExtern = enum_CX_StorageClass.define('CX_SC_PrivateExtern', 4) # type: ignore
CX_SC_OpenCLWorkGroupLocal = enum_CX_StorageClass.define('CX_SC_OpenCLWorkGroupLocal', 5) # type: ignore
CX_SC_Auto = enum_CX_StorageClass.define('CX_SC_Auto', 6) # type: ignore
CX_SC_Register = enum_CX_StorageClass.define('CX_SC_Register', 7) # type: ignore

enum_CX_BinaryOperatorKind = CEnum(Annotated[int, ctypes.c_uint32])
CX_BO_Invalid = enum_CX_BinaryOperatorKind.define('CX_BO_Invalid', 0) # type: ignore
CX_BO_PtrMemD = enum_CX_BinaryOperatorKind.define('CX_BO_PtrMemD', 1) # type: ignore
CX_BO_PtrMemI = enum_CX_BinaryOperatorKind.define('CX_BO_PtrMemI', 2) # type: ignore
CX_BO_Mul = enum_CX_BinaryOperatorKind.define('CX_BO_Mul', 3) # type: ignore
CX_BO_Div = enum_CX_BinaryOperatorKind.define('CX_BO_Div', 4) # type: ignore
CX_BO_Rem = enum_CX_BinaryOperatorKind.define('CX_BO_Rem', 5) # type: ignore
CX_BO_Add = enum_CX_BinaryOperatorKind.define('CX_BO_Add', 6) # type: ignore
CX_BO_Sub = enum_CX_BinaryOperatorKind.define('CX_BO_Sub', 7) # type: ignore
CX_BO_Shl = enum_CX_BinaryOperatorKind.define('CX_BO_Shl', 8) # type: ignore
CX_BO_Shr = enum_CX_BinaryOperatorKind.define('CX_BO_Shr', 9) # type: ignore
CX_BO_Cmp = enum_CX_BinaryOperatorKind.define('CX_BO_Cmp', 10) # type: ignore
CX_BO_LT = enum_CX_BinaryOperatorKind.define('CX_BO_LT', 11) # type: ignore
CX_BO_GT = enum_CX_BinaryOperatorKind.define('CX_BO_GT', 12) # type: ignore
CX_BO_LE = enum_CX_BinaryOperatorKind.define('CX_BO_LE', 13) # type: ignore
CX_BO_GE = enum_CX_BinaryOperatorKind.define('CX_BO_GE', 14) # type: ignore
CX_BO_EQ = enum_CX_BinaryOperatorKind.define('CX_BO_EQ', 15) # type: ignore
CX_BO_NE = enum_CX_BinaryOperatorKind.define('CX_BO_NE', 16) # type: ignore
CX_BO_And = enum_CX_BinaryOperatorKind.define('CX_BO_And', 17) # type: ignore
CX_BO_Xor = enum_CX_BinaryOperatorKind.define('CX_BO_Xor', 18) # type: ignore
CX_BO_Or = enum_CX_BinaryOperatorKind.define('CX_BO_Or', 19) # type: ignore
CX_BO_LAnd = enum_CX_BinaryOperatorKind.define('CX_BO_LAnd', 20) # type: ignore
CX_BO_LOr = enum_CX_BinaryOperatorKind.define('CX_BO_LOr', 21) # type: ignore
CX_BO_Assign = enum_CX_BinaryOperatorKind.define('CX_BO_Assign', 22) # type: ignore
CX_BO_MulAssign = enum_CX_BinaryOperatorKind.define('CX_BO_MulAssign', 23) # type: ignore
CX_BO_DivAssign = enum_CX_BinaryOperatorKind.define('CX_BO_DivAssign', 24) # type: ignore
CX_BO_RemAssign = enum_CX_BinaryOperatorKind.define('CX_BO_RemAssign', 25) # type: ignore
CX_BO_AddAssign = enum_CX_BinaryOperatorKind.define('CX_BO_AddAssign', 26) # type: ignore
CX_BO_SubAssign = enum_CX_BinaryOperatorKind.define('CX_BO_SubAssign', 27) # type: ignore
CX_BO_ShlAssign = enum_CX_BinaryOperatorKind.define('CX_BO_ShlAssign', 28) # type: ignore
CX_BO_ShrAssign = enum_CX_BinaryOperatorKind.define('CX_BO_ShrAssign', 29) # type: ignore
CX_BO_AndAssign = enum_CX_BinaryOperatorKind.define('CX_BO_AndAssign', 30) # type: ignore
CX_BO_XorAssign = enum_CX_BinaryOperatorKind.define('CX_BO_XorAssign', 31) # type: ignore
CX_BO_OrAssign = enum_CX_BinaryOperatorKind.define('CX_BO_OrAssign', 32) # type: ignore
CX_BO_Comma = enum_CX_BinaryOperatorKind.define('CX_BO_Comma', 33) # type: ignore
CX_BO_LAST = enum_CX_BinaryOperatorKind.define('CX_BO_LAST', 33) # type: ignore

@dll.bind
def clang_Cursor_getBinaryOpcode(C:CXCursor) -> enum_CX_BinaryOperatorKind: ...
@dll.bind
def clang_Cursor_getBinaryOpcodeStr(Op:enum_CX_BinaryOperatorKind) -> CXString: ...
@dll.bind
def clang_Cursor_getStorageClass(_0:CXCursor) -> enum_CX_StorageClass: ...
@dll.bind
def clang_getNumOverloadedDecls(cursor:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getOverloadedDecl(cursor:CXCursor, index:Annotated[int, ctypes.c_uint32]) -> CXCursor: ...
@dll.bind
def clang_getIBOutletCollectionType(_0:CXCursor) -> CXType: ...
enum_CXChildVisitResult = CEnum(Annotated[int, ctypes.c_uint32])
CXChildVisit_Break = enum_CXChildVisitResult.define('CXChildVisit_Break', 0) # type: ignore
CXChildVisit_Continue = enum_CXChildVisitResult.define('CXChildVisit_Continue', 1) # type: ignore
CXChildVisit_Recurse = enum_CXChildVisitResult.define('CXChildVisit_Recurse', 2) # type: ignore

CXCursorVisitor: TypeAlias = c.CFUNCTYPE(enum_CXChildVisitResult, CXCursor, CXCursor, c.POINTER[None])
@dll.bind
def clang_visitChildren(parent:CXCursor, visitor:CXCursorVisitor, client_data:CXClientData) -> Annotated[int, ctypes.c_uint32]: ...
class struct__CXChildVisitResult(ctypes.Structure): pass
CXCursorVisitorBlock = c.POINTER[struct__CXChildVisitResult]
@dll.bind
def clang_visitChildrenWithBlock(parent:CXCursor, block:CXCursorVisitorBlock) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCursorUSR(_0:CXCursor) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCClass(class_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCCategory(class_name:c.POINTER[Annotated[bytes, ctypes.c_char]], category_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCProtocol(protocol_name:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCIvar(name:c.POINTER[Annotated[bytes, ctypes.c_char]], classUSR:CXString) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCMethod(name:c.POINTER[Annotated[bytes, ctypes.c_char]], isInstanceMethod:Annotated[int, ctypes.c_uint32], classUSR:CXString) -> CXString: ...
@dll.bind
def clang_constructUSR_ObjCProperty(property:c.POINTER[Annotated[bytes, ctypes.c_char]], classUSR:CXString) -> CXString: ...
@dll.bind
def clang_getCursorSpelling(_0:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getSpellingNameRange(_0:CXCursor, pieceIndex:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32]) -> CXSourceRange: ...
CXPrintingPolicy = c.POINTER[None]
enum_CXPrintingPolicyProperty = CEnum(Annotated[int, ctypes.c_uint32])
CXPrintingPolicy_Indentation = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Indentation', 0) # type: ignore
CXPrintingPolicy_SuppressSpecifiers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressSpecifiers', 1) # type: ignore
CXPrintingPolicy_SuppressTagKeyword = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressTagKeyword', 2) # type: ignore
CXPrintingPolicy_IncludeTagDefinition = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_IncludeTagDefinition', 3) # type: ignore
CXPrintingPolicy_SuppressScope = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressScope', 4) # type: ignore
CXPrintingPolicy_SuppressUnwrittenScope = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressUnwrittenScope', 5) # type: ignore
CXPrintingPolicy_SuppressInitializers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressInitializers', 6) # type: ignore
CXPrintingPolicy_ConstantArraySizeAsWritten = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_ConstantArraySizeAsWritten', 7) # type: ignore
CXPrintingPolicy_AnonymousTagLocations = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_AnonymousTagLocations', 8) # type: ignore
CXPrintingPolicy_SuppressStrongLifetime = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressStrongLifetime', 9) # type: ignore
CXPrintingPolicy_SuppressLifetimeQualifiers = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressLifetimeQualifiers', 10) # type: ignore
CXPrintingPolicy_SuppressTemplateArgsInCXXConstructors = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressTemplateArgsInCXXConstructors', 11) # type: ignore
CXPrintingPolicy_Bool = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Bool', 12) # type: ignore
CXPrintingPolicy_Restrict = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Restrict', 13) # type: ignore
CXPrintingPolicy_Alignof = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Alignof', 14) # type: ignore
CXPrintingPolicy_UnderscoreAlignof = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_UnderscoreAlignof', 15) # type: ignore
CXPrintingPolicy_UseVoidForZeroParams = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_UseVoidForZeroParams', 16) # type: ignore
CXPrintingPolicy_TerseOutput = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_TerseOutput', 17) # type: ignore
CXPrintingPolicy_PolishForDeclaration = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_PolishForDeclaration', 18) # type: ignore
CXPrintingPolicy_Half = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_Half', 19) # type: ignore
CXPrintingPolicy_MSWChar = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_MSWChar', 20) # type: ignore
CXPrintingPolicy_IncludeNewlines = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_IncludeNewlines', 21) # type: ignore
CXPrintingPolicy_MSVCFormatting = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_MSVCFormatting', 22) # type: ignore
CXPrintingPolicy_ConstantsAsWritten = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_ConstantsAsWritten', 23) # type: ignore
CXPrintingPolicy_SuppressImplicitBase = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_SuppressImplicitBase', 24) # type: ignore
CXPrintingPolicy_FullyQualifiedName = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_FullyQualifiedName', 25) # type: ignore
CXPrintingPolicy_LastProperty = enum_CXPrintingPolicyProperty.define('CXPrintingPolicy_LastProperty', 25) # type: ignore

@dll.bind
def clang_PrintingPolicy_getProperty(Policy:CXPrintingPolicy, Property:enum_CXPrintingPolicyProperty) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_PrintingPolicy_setProperty(Policy:CXPrintingPolicy, Property:enum_CXPrintingPolicyProperty, Value:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_getCursorPrintingPolicy(_0:CXCursor) -> CXPrintingPolicy: ...
@dll.bind
def clang_PrintingPolicy_dispose(Policy:CXPrintingPolicy) -> None: ...
@dll.bind
def clang_getCursorPrettyPrinted(Cursor:CXCursor, Policy:CXPrintingPolicy) -> CXString: ...
@dll.bind
def clang_getTypePrettyPrinted(CT:CXType, cxPolicy:CXPrintingPolicy) -> CXString: ...
@dll.bind
def clang_getCursorDisplayName(_0:CXCursor) -> CXString: ...
@dll.bind
def clang_getCursorReferenced(_0:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getCursorDefinition(_0:CXCursor) -> CXCursor: ...
@dll.bind
def clang_isCursorDefinition(_0:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCanonicalCursor(_0:CXCursor) -> CXCursor: ...
@dll.bind
def clang_Cursor_getObjCSelectorIndex(_0:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_isDynamicCall(C:CXCursor) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Cursor_getReceiverType(C:CXCursor) -> CXType: ...
CXObjCPropertyAttrKind = CEnum(Annotated[int, ctypes.c_uint32])
CXObjCPropertyAttr_noattr = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_noattr', 0) # type: ignore
CXObjCPropertyAttr_readonly = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_readonly', 1) # type: ignore
CXObjCPropertyAttr_getter = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_getter', 2) # type: ignore
CXObjCPropertyAttr_assign = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_assign', 4) # type: ignore
CXObjCPropertyAttr_readwrite = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_readwrite', 8) # type: ignore
CXObjCPropertyAttr_retain = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_retain', 16) # type: ignore
CXObjCPropertyAttr_copy = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_copy', 32) # type: ignore
CXObjCPropertyAttr_nonatomic = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_nonatomic', 64) # type: ignore
CXObjCPropertyAttr_setter = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_setter', 128) # type: ignore
CXObjCPropertyAttr_atomic = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_atomic', 256) # type: ignore
CXObjCPropertyAttr_weak = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_weak', 512) # type: ignore
CXObjCPropertyAttr_strong = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_strong', 1024) # type: ignore
CXObjCPropertyAttr_unsafe_unretained = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_unsafe_unretained', 2048) # type: ignore
CXObjCPropertyAttr_class = CXObjCPropertyAttrKind.define('CXObjCPropertyAttr_class', 4096) # type: ignore

@dll.bind
def clang_Cursor_getObjCPropertyAttributes(C:CXCursor, reserved:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_getObjCPropertyGetterName(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getObjCPropertySetterName(C:CXCursor) -> CXString: ...
CXObjCDeclQualifierKind = CEnum(Annotated[int, ctypes.c_uint32])
CXObjCDeclQualifier_None = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_None', 0) # type: ignore
CXObjCDeclQualifier_In = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_In', 1) # type: ignore
CXObjCDeclQualifier_Inout = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Inout', 2) # type: ignore
CXObjCDeclQualifier_Out = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Out', 4) # type: ignore
CXObjCDeclQualifier_Bycopy = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Bycopy', 8) # type: ignore
CXObjCDeclQualifier_Byref = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Byref', 16) # type: ignore
CXObjCDeclQualifier_Oneway = CXObjCDeclQualifierKind.define('CXObjCDeclQualifier_Oneway', 32) # type: ignore

@dll.bind
def clang_Cursor_getObjCDeclQualifiers(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isObjCOptional(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isVariadic(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_isExternalSymbol(C:CXCursor, language:c.POINTER[CXString], definedIn:c.POINTER[CXString], isGenerated:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Cursor_getCommentRange(C:CXCursor) -> CXSourceRange: ...
@dll.bind
def clang_Cursor_getRawCommentText(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getBriefCommentText(C:CXCursor) -> CXString: ...
@dll.bind
def clang_Cursor_getMangling(_0:CXCursor) -> CXString: ...
@c.record
class CXStringSet(c.Struct):
  SIZE = 16
  Strings: Annotated[c.POINTER[CXString], 0]
  Count: Annotated[Annotated[int, ctypes.c_uint32], 8]
@dll.bind
def clang_Cursor_getCXXManglings(_0:CXCursor) -> c.POINTER[CXStringSet]: ...
@dll.bind
def clang_Cursor_getObjCManglings(_0:CXCursor) -> c.POINTER[CXStringSet]: ...
CXModule = c.POINTER[None]
@dll.bind
def clang_Cursor_getModule(C:CXCursor) -> CXModule: ...
@dll.bind
def clang_getModuleForFile(_0:CXTranslationUnit, _1:CXFile) -> CXModule: ...
@dll.bind
def clang_Module_getASTFile(Module:CXModule) -> CXFile: ...
@dll.bind
def clang_Module_getParent(Module:CXModule) -> CXModule: ...
@dll.bind
def clang_Module_getName(Module:CXModule) -> CXString: ...
@dll.bind
def clang_Module_getFullName(Module:CXModule) -> CXString: ...
@dll.bind
def clang_Module_isSystem(Module:CXModule) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Module_getNumTopLevelHeaders(_0:CXTranslationUnit, Module:CXModule) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Module_getTopLevelHeader(_0:CXTranslationUnit, Module:CXModule, Index:Annotated[int, ctypes.c_uint32]) -> CXFile: ...
@dll.bind
def clang_CXXConstructor_isConvertingConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXConstructor_isCopyConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXConstructor_isDefaultConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXConstructor_isMoveConstructor(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXField_isMutable(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isDefaulted(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isDeleted(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isPureVirtual(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isStatic(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isVirtual(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isCopyAssignmentOperator(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isMoveAssignmentOperator(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isExplicit(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXRecord_isAbstract(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_EnumDecl_isScoped(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_CXXMethod_isConst(C:CXCursor) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getTemplateCursorKind(C:CXCursor) -> enum_CXCursorKind: ...
@dll.bind
def clang_getSpecializedCursorTemplate(C:CXCursor) -> CXCursor: ...
@dll.bind
def clang_getCursorReferenceNameRange(C:CXCursor, NameFlags:Annotated[int, ctypes.c_uint32], PieceIndex:Annotated[int, ctypes.c_uint32]) -> CXSourceRange: ...
enum_CXNameRefFlags = CEnum(Annotated[int, ctypes.c_uint32])
CXNameRange_WantQualifier = enum_CXNameRefFlags.define('CXNameRange_WantQualifier', 1) # type: ignore
CXNameRange_WantTemplateArgs = enum_CXNameRefFlags.define('CXNameRange_WantTemplateArgs', 2) # type: ignore
CXNameRange_WantSinglePiece = enum_CXNameRefFlags.define('CXNameRange_WantSinglePiece', 4) # type: ignore

enum_CXTokenKind = CEnum(Annotated[int, ctypes.c_uint32])
CXToken_Punctuation = enum_CXTokenKind.define('CXToken_Punctuation', 0) # type: ignore
CXToken_Keyword = enum_CXTokenKind.define('CXToken_Keyword', 1) # type: ignore
CXToken_Identifier = enum_CXTokenKind.define('CXToken_Identifier', 2) # type: ignore
CXToken_Literal = enum_CXTokenKind.define('CXToken_Literal', 3) # type: ignore
CXToken_Comment = enum_CXTokenKind.define('CXToken_Comment', 4) # type: ignore

CXTokenKind: TypeAlias = enum_CXTokenKind
@c.record
class CXToken(c.Struct):
  SIZE = 24
  int_data: Annotated[c.Array[Annotated[int, ctypes.c_uint32], Literal[4]], 0]
  ptr_data: Annotated[c.POINTER[None], 16]
@dll.bind
def clang_getToken(TU:CXTranslationUnit, Location:CXSourceLocation) -> c.POINTER[CXToken]: ...
@dll.bind
def clang_getTokenKind(_0:CXToken) -> CXTokenKind: ...
@dll.bind
def clang_getTokenSpelling(_0:CXTranslationUnit, _1:CXToken) -> CXString: ...
@dll.bind
def clang_getTokenLocation(_0:CXTranslationUnit, _1:CXToken) -> CXSourceLocation: ...
@dll.bind
def clang_getTokenExtent(_0:CXTranslationUnit, _1:CXToken) -> CXSourceRange: ...
@dll.bind
def clang_tokenize(TU:CXTranslationUnit, Range:CXSourceRange, Tokens:c.POINTER[c.POINTER[CXToken]], NumTokens:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_annotateTokens(TU:CXTranslationUnit, Tokens:c.POINTER[CXToken], NumTokens:Annotated[int, ctypes.c_uint32], Cursors:c.POINTER[CXCursor]) -> None: ...
@dll.bind
def clang_disposeTokens(TU:CXTranslationUnit, Tokens:c.POINTER[CXToken], NumTokens:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_getCursorKindSpelling(Kind:enum_CXCursorKind) -> CXString: ...
@dll.bind
def clang_getDefinitionSpellingAndExtent(_0:CXCursor, startBuf:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], endBuf:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], startLine:c.POINTER[Annotated[int, ctypes.c_uint32]], startColumn:c.POINTER[Annotated[int, ctypes.c_uint32]], endLine:c.POINTER[Annotated[int, ctypes.c_uint32]], endColumn:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_enableStackTraces() -> None: ...
@dll.bind
def clang_executeOnThread(fn:c.CFUNCTYPE(None, c.POINTER[None]), user_data:c.POINTER[None], stack_size:Annotated[int, ctypes.c_uint32]) -> None: ...
CXCompletionString = c.POINTER[None]
@c.record
class CXCompletionResult(c.Struct):
  SIZE = 16
  CursorKind: Annotated[enum_CXCursorKind, 0]
  CompletionString: Annotated[CXCompletionString, 8]
enum_CXCompletionChunkKind = CEnum(Annotated[int, ctypes.c_uint32])
CXCompletionChunk_Optional = enum_CXCompletionChunkKind.define('CXCompletionChunk_Optional', 0) # type: ignore
CXCompletionChunk_TypedText = enum_CXCompletionChunkKind.define('CXCompletionChunk_TypedText', 1) # type: ignore
CXCompletionChunk_Text = enum_CXCompletionChunkKind.define('CXCompletionChunk_Text', 2) # type: ignore
CXCompletionChunk_Placeholder = enum_CXCompletionChunkKind.define('CXCompletionChunk_Placeholder', 3) # type: ignore
CXCompletionChunk_Informative = enum_CXCompletionChunkKind.define('CXCompletionChunk_Informative', 4) # type: ignore
CXCompletionChunk_CurrentParameter = enum_CXCompletionChunkKind.define('CXCompletionChunk_CurrentParameter', 5) # type: ignore
CXCompletionChunk_LeftParen = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftParen', 6) # type: ignore
CXCompletionChunk_RightParen = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightParen', 7) # type: ignore
CXCompletionChunk_LeftBracket = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftBracket', 8) # type: ignore
CXCompletionChunk_RightBracket = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightBracket', 9) # type: ignore
CXCompletionChunk_LeftBrace = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftBrace', 10) # type: ignore
CXCompletionChunk_RightBrace = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightBrace', 11) # type: ignore
CXCompletionChunk_LeftAngle = enum_CXCompletionChunkKind.define('CXCompletionChunk_LeftAngle', 12) # type: ignore
CXCompletionChunk_RightAngle = enum_CXCompletionChunkKind.define('CXCompletionChunk_RightAngle', 13) # type: ignore
CXCompletionChunk_Comma = enum_CXCompletionChunkKind.define('CXCompletionChunk_Comma', 14) # type: ignore
CXCompletionChunk_ResultType = enum_CXCompletionChunkKind.define('CXCompletionChunk_ResultType', 15) # type: ignore
CXCompletionChunk_Colon = enum_CXCompletionChunkKind.define('CXCompletionChunk_Colon', 16) # type: ignore
CXCompletionChunk_SemiColon = enum_CXCompletionChunkKind.define('CXCompletionChunk_SemiColon', 17) # type: ignore
CXCompletionChunk_Equal = enum_CXCompletionChunkKind.define('CXCompletionChunk_Equal', 18) # type: ignore
CXCompletionChunk_HorizontalSpace = enum_CXCompletionChunkKind.define('CXCompletionChunk_HorizontalSpace', 19) # type: ignore
CXCompletionChunk_VerticalSpace = enum_CXCompletionChunkKind.define('CXCompletionChunk_VerticalSpace', 20) # type: ignore

@dll.bind
def clang_getCompletionChunkKind(completion_string:CXCompletionString, chunk_number:Annotated[int, ctypes.c_uint32]) -> enum_CXCompletionChunkKind: ...
@dll.bind
def clang_getCompletionChunkText(completion_string:CXCompletionString, chunk_number:Annotated[int, ctypes.c_uint32]) -> CXString: ...
@dll.bind
def clang_getCompletionChunkCompletionString(completion_string:CXCompletionString, chunk_number:Annotated[int, ctypes.c_uint32]) -> CXCompletionString: ...
@dll.bind
def clang_getNumCompletionChunks(completion_string:CXCompletionString) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionPriority(completion_string:CXCompletionString) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionAvailability(completion_string:CXCompletionString) -> enum_CXAvailabilityKind: ...
@dll.bind
def clang_getCompletionNumAnnotations(completion_string:CXCompletionString) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionAnnotation(completion_string:CXCompletionString, annotation_number:Annotated[int, ctypes.c_uint32]) -> CXString: ...
@dll.bind
def clang_getCompletionParent(completion_string:CXCompletionString, kind:c.POINTER[enum_CXCursorKind]) -> CXString: ...
@dll.bind
def clang_getCompletionBriefComment(completion_string:CXCompletionString) -> CXString: ...
@dll.bind
def clang_getCursorCompletionString(cursor:CXCursor) -> CXCompletionString: ...
@c.record
class CXCodeCompleteResults(c.Struct):
  SIZE = 16
  Results: Annotated[c.POINTER[CXCompletionResult], 0]
  NumResults: Annotated[Annotated[int, ctypes.c_uint32], 8]
@dll.bind
def clang_getCompletionNumFixIts(results:c.POINTER[CXCodeCompleteResults], completion_index:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_getCompletionFixIt(results:c.POINTER[CXCodeCompleteResults], completion_index:Annotated[int, ctypes.c_uint32], fixit_index:Annotated[int, ctypes.c_uint32], replacement_range:c.POINTER[CXSourceRange]) -> CXString: ...
enum_CXCodeComplete_Flags = CEnum(Annotated[int, ctypes.c_uint32])
CXCodeComplete_IncludeMacros = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeMacros', 1) # type: ignore
CXCodeComplete_IncludeCodePatterns = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCodePatterns', 2) # type: ignore
CXCodeComplete_IncludeBriefComments = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeBriefComments', 4) # type: ignore
CXCodeComplete_SkipPreamble = enum_CXCodeComplete_Flags.define('CXCodeComplete_SkipPreamble', 8) # type: ignore
CXCodeComplete_IncludeCompletionsWithFixIts = enum_CXCodeComplete_Flags.define('CXCodeComplete_IncludeCompletionsWithFixIts', 16) # type: ignore

enum_CXCompletionContext = CEnum(Annotated[int, ctypes.c_uint32])
CXCompletionContext_Unexposed = enum_CXCompletionContext.define('CXCompletionContext_Unexposed', 0) # type: ignore
CXCompletionContext_AnyType = enum_CXCompletionContext.define('CXCompletionContext_AnyType', 1) # type: ignore
CXCompletionContext_AnyValue = enum_CXCompletionContext.define('CXCompletionContext_AnyValue', 2) # type: ignore
CXCompletionContext_ObjCObjectValue = enum_CXCompletionContext.define('CXCompletionContext_ObjCObjectValue', 4) # type: ignore
CXCompletionContext_ObjCSelectorValue = enum_CXCompletionContext.define('CXCompletionContext_ObjCSelectorValue', 8) # type: ignore
CXCompletionContext_CXXClassTypeValue = enum_CXCompletionContext.define('CXCompletionContext_CXXClassTypeValue', 16) # type: ignore
CXCompletionContext_DotMemberAccess = enum_CXCompletionContext.define('CXCompletionContext_DotMemberAccess', 32) # type: ignore
CXCompletionContext_ArrowMemberAccess = enum_CXCompletionContext.define('CXCompletionContext_ArrowMemberAccess', 64) # type: ignore
CXCompletionContext_ObjCPropertyAccess = enum_CXCompletionContext.define('CXCompletionContext_ObjCPropertyAccess', 128) # type: ignore
CXCompletionContext_EnumTag = enum_CXCompletionContext.define('CXCompletionContext_EnumTag', 256) # type: ignore
CXCompletionContext_UnionTag = enum_CXCompletionContext.define('CXCompletionContext_UnionTag', 512) # type: ignore
CXCompletionContext_StructTag = enum_CXCompletionContext.define('CXCompletionContext_StructTag', 1024) # type: ignore
CXCompletionContext_ClassTag = enum_CXCompletionContext.define('CXCompletionContext_ClassTag', 2048) # type: ignore
CXCompletionContext_Namespace = enum_CXCompletionContext.define('CXCompletionContext_Namespace', 4096) # type: ignore
CXCompletionContext_NestedNameSpecifier = enum_CXCompletionContext.define('CXCompletionContext_NestedNameSpecifier', 8192) # type: ignore
CXCompletionContext_ObjCInterface = enum_CXCompletionContext.define('CXCompletionContext_ObjCInterface', 16384) # type: ignore
CXCompletionContext_ObjCProtocol = enum_CXCompletionContext.define('CXCompletionContext_ObjCProtocol', 32768) # type: ignore
CXCompletionContext_ObjCCategory = enum_CXCompletionContext.define('CXCompletionContext_ObjCCategory', 65536) # type: ignore
CXCompletionContext_ObjCInstanceMessage = enum_CXCompletionContext.define('CXCompletionContext_ObjCInstanceMessage', 131072) # type: ignore
CXCompletionContext_ObjCClassMessage = enum_CXCompletionContext.define('CXCompletionContext_ObjCClassMessage', 262144) # type: ignore
CXCompletionContext_ObjCSelectorName = enum_CXCompletionContext.define('CXCompletionContext_ObjCSelectorName', 524288) # type: ignore
CXCompletionContext_MacroName = enum_CXCompletionContext.define('CXCompletionContext_MacroName', 1048576) # type: ignore
CXCompletionContext_NaturalLanguage = enum_CXCompletionContext.define('CXCompletionContext_NaturalLanguage', 2097152) # type: ignore
CXCompletionContext_IncludedFile = enum_CXCompletionContext.define('CXCompletionContext_IncludedFile', 4194304) # type: ignore
CXCompletionContext_Unknown = enum_CXCompletionContext.define('CXCompletionContext_Unknown', 8388607) # type: ignore

@dll.bind
def clang_defaultCodeCompleteOptions() -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_codeCompleteAt(TU:CXTranslationUnit, complete_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], complete_line:Annotated[int, ctypes.c_uint32], complete_column:Annotated[int, ctypes.c_uint32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], options:Annotated[int, ctypes.c_uint32]) -> c.POINTER[CXCodeCompleteResults]: ...
@dll.bind
def clang_sortCodeCompletionResults(Results:c.POINTER[CXCompletionResult], NumResults:Annotated[int, ctypes.c_uint32]) -> None: ...
@dll.bind
def clang_disposeCodeCompleteResults(Results:c.POINTER[CXCodeCompleteResults]) -> None: ...
@dll.bind
def clang_codeCompleteGetNumDiagnostics(Results:c.POINTER[CXCodeCompleteResults]) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_codeCompleteGetDiagnostic(Results:c.POINTER[CXCodeCompleteResults], Index:Annotated[int, ctypes.c_uint32]) -> CXDiagnostic: ...
@dll.bind
def clang_codeCompleteGetContexts(Results:c.POINTER[CXCodeCompleteResults]) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_codeCompleteGetContainerKind(Results:c.POINTER[CXCodeCompleteResults], IsIncomplete:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> enum_CXCursorKind: ...
@dll.bind
def clang_codeCompleteGetContainerUSR(Results:c.POINTER[CXCodeCompleteResults]) -> CXString: ...
@dll.bind
def clang_codeCompleteGetObjCSelector(Results:c.POINTER[CXCodeCompleteResults]) -> CXString: ...
@dll.bind
def clang_getClangVersion() -> CXString: ...
@dll.bind
def clang_toggleCrashRecovery(isEnabled:Annotated[int, ctypes.c_uint32]) -> None: ...
CXInclusionVisitor = c.CFUNCTYPE(None, c.POINTER[None], c.POINTER[CXSourceLocation], Annotated[int, ctypes.c_uint32], c.POINTER[None])
@dll.bind
def clang_getInclusions(tu:CXTranslationUnit, visitor:CXInclusionVisitor, client_data:CXClientData) -> None: ...
CXEvalResultKind = CEnum(Annotated[int, ctypes.c_uint32])
CXEval_Int = CXEvalResultKind.define('CXEval_Int', 1) # type: ignore
CXEval_Float = CXEvalResultKind.define('CXEval_Float', 2) # type: ignore
CXEval_ObjCStrLiteral = CXEvalResultKind.define('CXEval_ObjCStrLiteral', 3) # type: ignore
CXEval_StrLiteral = CXEvalResultKind.define('CXEval_StrLiteral', 4) # type: ignore
CXEval_CFStr = CXEvalResultKind.define('CXEval_CFStr', 5) # type: ignore
CXEval_Other = CXEvalResultKind.define('CXEval_Other', 6) # type: ignore
CXEval_UnExposed = CXEvalResultKind.define('CXEval_UnExposed', 0) # type: ignore

CXEvalResult = c.POINTER[None]
@dll.bind
def clang_Cursor_Evaluate(C:CXCursor) -> CXEvalResult: ...
@dll.bind
def clang_EvalResult_getKind(E:CXEvalResult) -> CXEvalResultKind: ...
@dll.bind
def clang_EvalResult_getAsInt(E:CXEvalResult) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_EvalResult_getAsLongLong(E:CXEvalResult) -> Annotated[int, ctypes.c_int64]: ...
@dll.bind
def clang_EvalResult_isUnsignedInt(E:CXEvalResult) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_EvalResult_getAsUnsigned(E:CXEvalResult) -> Annotated[int, ctypes.c_uint64]: ...
@dll.bind
def clang_EvalResult_getAsDouble(E:CXEvalResult) -> Annotated[float, ctypes.c_double]: ...
@dll.bind
def clang_EvalResult_getAsStr(E:CXEvalResult) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def clang_EvalResult_dispose(E:CXEvalResult) -> None: ...
CXRemapping = c.POINTER[None]
@dll.bind
def clang_getRemappings(path:c.POINTER[Annotated[bytes, ctypes.c_char]]) -> CXRemapping: ...
@dll.bind
def clang_getRemappingsFromFileList(filePaths:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], numFiles:Annotated[int, ctypes.c_uint32]) -> CXRemapping: ...
@dll.bind
def clang_remap_getNumFiles(_0:CXRemapping) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_remap_getFilenames(_0:CXRemapping, index:Annotated[int, ctypes.c_uint32], original:c.POINTER[CXString], transformed:c.POINTER[CXString]) -> None: ...
@dll.bind
def clang_remap_dispose(_0:CXRemapping) -> None: ...
enum_CXVisitorResult = CEnum(Annotated[int, ctypes.c_uint32])
CXVisit_Break = enum_CXVisitorResult.define('CXVisit_Break', 0) # type: ignore
CXVisit_Continue = enum_CXVisitorResult.define('CXVisit_Continue', 1) # type: ignore

@c.record
class struct_CXCursorAndRangeVisitor(c.Struct):
  SIZE = 16
  context: Annotated[c.POINTER[None], 0]
  visit: Annotated[c.CFUNCTYPE(enum_CXVisitorResult, c.POINTER[None], CXCursor, CXSourceRange), 8]
CXCursorAndRangeVisitor = struct_CXCursorAndRangeVisitor
CXResult = CEnum(Annotated[int, ctypes.c_uint32])
CXResult_Success = CXResult.define('CXResult_Success', 0) # type: ignore
CXResult_Invalid = CXResult.define('CXResult_Invalid', 1) # type: ignore
CXResult_VisitBreak = CXResult.define('CXResult_VisitBreak', 2) # type: ignore

@dll.bind
def clang_findReferencesInFile(cursor:CXCursor, file:CXFile, visitor:CXCursorAndRangeVisitor) -> CXResult: ...
@dll.bind
def clang_findIncludesInFile(TU:CXTranslationUnit, file:CXFile, visitor:CXCursorAndRangeVisitor) -> CXResult: ...
class struct__CXCursorAndRangeVisitorBlock(ctypes.Structure): pass
CXCursorAndRangeVisitorBlock = c.POINTER[struct__CXCursorAndRangeVisitorBlock]
@dll.bind
def clang_findReferencesInFileWithBlock(_0:CXCursor, _1:CXFile, _2:CXCursorAndRangeVisitorBlock) -> CXResult: ...
@dll.bind
def clang_findIncludesInFileWithBlock(_0:CXTranslationUnit, _1:CXFile, _2:CXCursorAndRangeVisitorBlock) -> CXResult: ...
CXIdxClientFile = c.POINTER[None]
CXIdxClientEntity = c.POINTER[None]
CXIdxClientContainer = c.POINTER[None]
CXIdxClientASTFile = c.POINTER[None]
@c.record
class CXIdxLoc(c.Struct):
  SIZE = 24
  ptr_data: Annotated[c.Array[c.POINTER[None], Literal[2]], 0]
  int_data: Annotated[Annotated[int, ctypes.c_uint32], 16]
@c.record
class CXIdxIncludedFileInfo(c.Struct):
  SIZE = 56
  hashLoc: Annotated[CXIdxLoc, 0]
  filename: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 24]
  file: Annotated[CXFile, 32]
  isImport: Annotated[Annotated[int, ctypes.c_int32], 40]
  isAngled: Annotated[Annotated[int, ctypes.c_int32], 44]
  isModuleImport: Annotated[Annotated[int, ctypes.c_int32], 48]
@c.record
class CXIdxImportedASTFileInfo(c.Struct):
  SIZE = 48
  file: Annotated[CXFile, 0]
  module: Annotated[CXModule, 8]
  loc: Annotated[CXIdxLoc, 16]
  isImplicit: Annotated[Annotated[int, ctypes.c_int32], 40]
CXIdxEntityKind = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxEntity_Unexposed = CXIdxEntityKind.define('CXIdxEntity_Unexposed', 0) # type: ignore
CXIdxEntity_Typedef = CXIdxEntityKind.define('CXIdxEntity_Typedef', 1) # type: ignore
CXIdxEntity_Function = CXIdxEntityKind.define('CXIdxEntity_Function', 2) # type: ignore
CXIdxEntity_Variable = CXIdxEntityKind.define('CXIdxEntity_Variable', 3) # type: ignore
CXIdxEntity_Field = CXIdxEntityKind.define('CXIdxEntity_Field', 4) # type: ignore
CXIdxEntity_EnumConstant = CXIdxEntityKind.define('CXIdxEntity_EnumConstant', 5) # type: ignore
CXIdxEntity_ObjCClass = CXIdxEntityKind.define('CXIdxEntity_ObjCClass', 6) # type: ignore
CXIdxEntity_ObjCProtocol = CXIdxEntityKind.define('CXIdxEntity_ObjCProtocol', 7) # type: ignore
CXIdxEntity_ObjCCategory = CXIdxEntityKind.define('CXIdxEntity_ObjCCategory', 8) # type: ignore
CXIdxEntity_ObjCInstanceMethod = CXIdxEntityKind.define('CXIdxEntity_ObjCInstanceMethod', 9) # type: ignore
CXIdxEntity_ObjCClassMethod = CXIdxEntityKind.define('CXIdxEntity_ObjCClassMethod', 10) # type: ignore
CXIdxEntity_ObjCProperty = CXIdxEntityKind.define('CXIdxEntity_ObjCProperty', 11) # type: ignore
CXIdxEntity_ObjCIvar = CXIdxEntityKind.define('CXIdxEntity_ObjCIvar', 12) # type: ignore
CXIdxEntity_Enum = CXIdxEntityKind.define('CXIdxEntity_Enum', 13) # type: ignore
CXIdxEntity_Struct = CXIdxEntityKind.define('CXIdxEntity_Struct', 14) # type: ignore
CXIdxEntity_Union = CXIdxEntityKind.define('CXIdxEntity_Union', 15) # type: ignore
CXIdxEntity_CXXClass = CXIdxEntityKind.define('CXIdxEntity_CXXClass', 16) # type: ignore
CXIdxEntity_CXXNamespace = CXIdxEntityKind.define('CXIdxEntity_CXXNamespace', 17) # type: ignore
CXIdxEntity_CXXNamespaceAlias = CXIdxEntityKind.define('CXIdxEntity_CXXNamespaceAlias', 18) # type: ignore
CXIdxEntity_CXXStaticVariable = CXIdxEntityKind.define('CXIdxEntity_CXXStaticVariable', 19) # type: ignore
CXIdxEntity_CXXStaticMethod = CXIdxEntityKind.define('CXIdxEntity_CXXStaticMethod', 20) # type: ignore
CXIdxEntity_CXXInstanceMethod = CXIdxEntityKind.define('CXIdxEntity_CXXInstanceMethod', 21) # type: ignore
CXIdxEntity_CXXConstructor = CXIdxEntityKind.define('CXIdxEntity_CXXConstructor', 22) # type: ignore
CXIdxEntity_CXXDestructor = CXIdxEntityKind.define('CXIdxEntity_CXXDestructor', 23) # type: ignore
CXIdxEntity_CXXConversionFunction = CXIdxEntityKind.define('CXIdxEntity_CXXConversionFunction', 24) # type: ignore
CXIdxEntity_CXXTypeAlias = CXIdxEntityKind.define('CXIdxEntity_CXXTypeAlias', 25) # type: ignore
CXIdxEntity_CXXInterface = CXIdxEntityKind.define('CXIdxEntity_CXXInterface', 26) # type: ignore
CXIdxEntity_CXXConcept = CXIdxEntityKind.define('CXIdxEntity_CXXConcept', 27) # type: ignore

CXIdxEntityLanguage = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxEntityLang_None = CXIdxEntityLanguage.define('CXIdxEntityLang_None', 0) # type: ignore
CXIdxEntityLang_C = CXIdxEntityLanguage.define('CXIdxEntityLang_C', 1) # type: ignore
CXIdxEntityLang_ObjC = CXIdxEntityLanguage.define('CXIdxEntityLang_ObjC', 2) # type: ignore
CXIdxEntityLang_CXX = CXIdxEntityLanguage.define('CXIdxEntityLang_CXX', 3) # type: ignore
CXIdxEntityLang_Swift = CXIdxEntityLanguage.define('CXIdxEntityLang_Swift', 4) # type: ignore

CXIdxEntityCXXTemplateKind = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxEntity_NonTemplate = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_NonTemplate', 0) # type: ignore
CXIdxEntity_Template = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_Template', 1) # type: ignore
CXIdxEntity_TemplatePartialSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplatePartialSpecialization', 2) # type: ignore
CXIdxEntity_TemplateSpecialization = CXIdxEntityCXXTemplateKind.define('CXIdxEntity_TemplateSpecialization', 3) # type: ignore

CXIdxAttrKind = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxAttr_Unexposed = CXIdxAttrKind.define('CXIdxAttr_Unexposed', 0) # type: ignore
CXIdxAttr_IBAction = CXIdxAttrKind.define('CXIdxAttr_IBAction', 1) # type: ignore
CXIdxAttr_IBOutlet = CXIdxAttrKind.define('CXIdxAttr_IBOutlet', 2) # type: ignore
CXIdxAttr_IBOutletCollection = CXIdxAttrKind.define('CXIdxAttr_IBOutletCollection', 3) # type: ignore

@c.record
class CXIdxAttrInfo(c.Struct):
  SIZE = 64
  kind: Annotated[CXIdxAttrKind, 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
@c.record
class CXIdxEntityInfo(c.Struct):
  SIZE = 80
  kind: Annotated[CXIdxEntityKind, 0]
  templateKind: Annotated[CXIdxEntityCXXTemplateKind, 4]
  lang: Annotated[CXIdxEntityLanguage, 8]
  name: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 16]
  USR: Annotated[c.POINTER[Annotated[bytes, ctypes.c_char]], 24]
  cursor: Annotated[CXCursor, 32]
  attributes: Annotated[c.POINTER[c.POINTER[CXIdxAttrInfo]], 64]
  numAttributes: Annotated[Annotated[int, ctypes.c_uint32], 72]
@c.record
class CXIdxContainerInfo(c.Struct):
  SIZE = 32
  cursor: Annotated[CXCursor, 0]
@c.record
class CXIdxIBOutletCollectionAttrInfo(c.Struct):
  SIZE = 72
  attrInfo: Annotated[c.POINTER[CXIdxAttrInfo], 0]
  objcClass: Annotated[c.POINTER[CXIdxEntityInfo], 8]
  classCursor: Annotated[CXCursor, 16]
  classLoc: Annotated[CXIdxLoc, 48]
CXIdxDeclInfoFlags = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxDeclFlag_Skipped = CXIdxDeclInfoFlags.define('CXIdxDeclFlag_Skipped', 1) # type: ignore

@c.record
class CXIdxDeclInfo(c.Struct):
  SIZE = 128
  entityInfo: Annotated[c.POINTER[CXIdxEntityInfo], 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
  semanticContainer: Annotated[c.POINTER[CXIdxContainerInfo], 64]
  lexicalContainer: Annotated[c.POINTER[CXIdxContainerInfo], 72]
  isRedeclaration: Annotated[Annotated[int, ctypes.c_int32], 80]
  isDefinition: Annotated[Annotated[int, ctypes.c_int32], 84]
  isContainer: Annotated[Annotated[int, ctypes.c_int32], 88]
  declAsContainer: Annotated[c.POINTER[CXIdxContainerInfo], 96]
  isImplicit: Annotated[Annotated[int, ctypes.c_int32], 104]
  attributes: Annotated[c.POINTER[c.POINTER[CXIdxAttrInfo]], 112]
  numAttributes: Annotated[Annotated[int, ctypes.c_uint32], 120]
  flags: Annotated[Annotated[int, ctypes.c_uint32], 124]
CXIdxObjCContainerKind = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxObjCContainer_ForwardRef = CXIdxObjCContainerKind.define('CXIdxObjCContainer_ForwardRef', 0) # type: ignore
CXIdxObjCContainer_Interface = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Interface', 1) # type: ignore
CXIdxObjCContainer_Implementation = CXIdxObjCContainerKind.define('CXIdxObjCContainer_Implementation', 2) # type: ignore

@c.record
class CXIdxObjCContainerDeclInfo(c.Struct):
  SIZE = 16
  declInfo: Annotated[c.POINTER[CXIdxDeclInfo], 0]
  kind: Annotated[CXIdxObjCContainerKind, 8]
@c.record
class CXIdxBaseClassInfo(c.Struct):
  SIZE = 64
  base: Annotated[c.POINTER[CXIdxEntityInfo], 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
@c.record
class CXIdxObjCProtocolRefInfo(c.Struct):
  SIZE = 64
  protocol: Annotated[c.POINTER[CXIdxEntityInfo], 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
@c.record
class CXIdxObjCProtocolRefListInfo(c.Struct):
  SIZE = 16
  protocols: Annotated[c.POINTER[c.POINTER[CXIdxObjCProtocolRefInfo]], 0]
  numProtocols: Annotated[Annotated[int, ctypes.c_uint32], 8]
@c.record
class CXIdxObjCInterfaceDeclInfo(c.Struct):
  SIZE = 24
  containerInfo: Annotated[c.POINTER[CXIdxObjCContainerDeclInfo], 0]
  superInfo: Annotated[c.POINTER[CXIdxBaseClassInfo], 8]
  protocols: Annotated[c.POINTER[CXIdxObjCProtocolRefListInfo], 16]
@c.record
class CXIdxObjCCategoryDeclInfo(c.Struct):
  SIZE = 80
  containerInfo: Annotated[c.POINTER[CXIdxObjCContainerDeclInfo], 0]
  objcClass: Annotated[c.POINTER[CXIdxEntityInfo], 8]
  classCursor: Annotated[CXCursor, 16]
  classLoc: Annotated[CXIdxLoc, 48]
  protocols: Annotated[c.POINTER[CXIdxObjCProtocolRefListInfo], 72]
@c.record
class CXIdxObjCPropertyDeclInfo(c.Struct):
  SIZE = 24
  declInfo: Annotated[c.POINTER[CXIdxDeclInfo], 0]
  getter: Annotated[c.POINTER[CXIdxEntityInfo], 8]
  setter: Annotated[c.POINTER[CXIdxEntityInfo], 16]
@c.record
class CXIdxCXXClassDeclInfo(c.Struct):
  SIZE = 24
  declInfo: Annotated[c.POINTER[CXIdxDeclInfo], 0]
  bases: Annotated[c.POINTER[c.POINTER[CXIdxBaseClassInfo]], 8]
  numBases: Annotated[Annotated[int, ctypes.c_uint32], 16]
CXIdxEntityRefKind = CEnum(Annotated[int, ctypes.c_uint32])
CXIdxEntityRef_Direct = CXIdxEntityRefKind.define('CXIdxEntityRef_Direct', 1) # type: ignore
CXIdxEntityRef_Implicit = CXIdxEntityRefKind.define('CXIdxEntityRef_Implicit', 2) # type: ignore

CXSymbolRole = CEnum(Annotated[int, ctypes.c_uint32])
CXSymbolRole_None = CXSymbolRole.define('CXSymbolRole_None', 0) # type: ignore
CXSymbolRole_Declaration = CXSymbolRole.define('CXSymbolRole_Declaration', 1) # type: ignore
CXSymbolRole_Definition = CXSymbolRole.define('CXSymbolRole_Definition', 2) # type: ignore
CXSymbolRole_Reference = CXSymbolRole.define('CXSymbolRole_Reference', 4) # type: ignore
CXSymbolRole_Read = CXSymbolRole.define('CXSymbolRole_Read', 8) # type: ignore
CXSymbolRole_Write = CXSymbolRole.define('CXSymbolRole_Write', 16) # type: ignore
CXSymbolRole_Call = CXSymbolRole.define('CXSymbolRole_Call', 32) # type: ignore
CXSymbolRole_Dynamic = CXSymbolRole.define('CXSymbolRole_Dynamic', 64) # type: ignore
CXSymbolRole_AddressOf = CXSymbolRole.define('CXSymbolRole_AddressOf', 128) # type: ignore
CXSymbolRole_Implicit = CXSymbolRole.define('CXSymbolRole_Implicit', 256) # type: ignore

@c.record
class CXIdxEntityRefInfo(c.Struct):
  SIZE = 96
  kind: Annotated[CXIdxEntityRefKind, 0]
  cursor: Annotated[CXCursor, 8]
  loc: Annotated[CXIdxLoc, 40]
  referencedEntity: Annotated[c.POINTER[CXIdxEntityInfo], 64]
  parentEntity: Annotated[c.POINTER[CXIdxEntityInfo], 72]
  container: Annotated[c.POINTER[CXIdxContainerInfo], 80]
  role: Annotated[CXSymbolRole, 88]
@c.record
class IndexerCallbacks(c.Struct):
  SIZE = 64
  abortQuery: Annotated[c.CFUNCTYPE(Annotated[int, ctypes.c_int32], CXClientData, c.POINTER[None]), 0]
  diagnostic: Annotated[c.CFUNCTYPE(None, CXClientData, CXDiagnosticSet, c.POINTER[None]), 8]
  enteredMainFile: Annotated[c.CFUNCTYPE(CXIdxClientFile, CXClientData, CXFile, c.POINTER[None]), 16]
  ppIncludedFile: Annotated[c.CFUNCTYPE(CXIdxClientFile, CXClientData, c.POINTER[CXIdxIncludedFileInfo]), 24]
  importedASTFile: Annotated[c.CFUNCTYPE(CXIdxClientASTFile, CXClientData, c.POINTER[CXIdxImportedASTFileInfo]), 32]
  startedTranslationUnit: Annotated[c.CFUNCTYPE(CXIdxClientContainer, CXClientData, c.POINTER[None]), 40]
  indexDeclaration: Annotated[c.CFUNCTYPE(None, CXClientData, c.POINTER[CXIdxDeclInfo]), 48]
  indexEntityReference: Annotated[c.CFUNCTYPE(None, CXClientData, c.POINTER[CXIdxEntityRefInfo]), 56]
@dll.bind
def clang_index_isEntityObjCContainerKind(_0:CXIdxEntityKind) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_index_getObjCContainerDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCContainerDeclInfo]: ...
@dll.bind
def clang_index_getObjCInterfaceDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCInterfaceDeclInfo]: ...
@dll.bind
def clang_index_getObjCCategoryDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCCategoryDeclInfo]: ...
@dll.bind
def clang_index_getObjCProtocolRefListInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCProtocolRefListInfo]: ...
@dll.bind
def clang_index_getObjCPropertyDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxObjCPropertyDeclInfo]: ...
@dll.bind
def clang_index_getIBOutletCollectionAttrInfo(_0:c.POINTER[CXIdxAttrInfo]) -> c.POINTER[CXIdxIBOutletCollectionAttrInfo]: ...
@dll.bind
def clang_index_getCXXClassDeclInfo(_0:c.POINTER[CXIdxDeclInfo]) -> c.POINTER[CXIdxCXXClassDeclInfo]: ...
@dll.bind
def clang_index_getClientContainer(_0:c.POINTER[CXIdxContainerInfo]) -> CXIdxClientContainer: ...
@dll.bind
def clang_index_setClientContainer(_0:c.POINTER[CXIdxContainerInfo], _1:CXIdxClientContainer) -> None: ...
@dll.bind
def clang_index_getClientEntity(_0:c.POINTER[CXIdxEntityInfo]) -> CXIdxClientEntity: ...
@dll.bind
def clang_index_setClientEntity(_0:c.POINTER[CXIdxEntityInfo], _1:CXIdxClientEntity) -> None: ...
CXIndexAction = c.POINTER[None]
@dll.bind
def clang_IndexAction_create(CIdx:CXIndex) -> CXIndexAction: ...
@dll.bind
def clang_IndexAction_dispose(_0:CXIndexAction) -> None: ...
CXIndexOptFlags = CEnum(Annotated[int, ctypes.c_uint32])
CXIndexOpt_None = CXIndexOptFlags.define('CXIndexOpt_None', 0) # type: ignore
CXIndexOpt_SuppressRedundantRefs = CXIndexOptFlags.define('CXIndexOpt_SuppressRedundantRefs', 1) # type: ignore
CXIndexOpt_IndexFunctionLocalSymbols = CXIndexOptFlags.define('CXIndexOpt_IndexFunctionLocalSymbols', 2) # type: ignore
CXIndexOpt_IndexImplicitTemplateInstantiations = CXIndexOptFlags.define('CXIndexOpt_IndexImplicitTemplateInstantiations', 4) # type: ignore
CXIndexOpt_SuppressWarnings = CXIndexOptFlags.define('CXIndexOpt_SuppressWarnings', 8) # type: ignore
CXIndexOpt_SkipParsedBodiesInSession = CXIndexOptFlags.define('CXIndexOpt_SkipParsedBodiesInSession', 16) # type: ignore

@dll.bind
def clang_indexSourceFile(_0:CXIndexAction, client_data:CXClientData, index_callbacks:c.POINTER[IndexerCallbacks], index_callbacks_size:Annotated[int, ctypes.c_uint32], index_options:Annotated[int, ctypes.c_uint32], source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit], TU_options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_indexSourceFileFullArgv(_0:CXIndexAction, client_data:CXClientData, index_callbacks:c.POINTER[IndexerCallbacks], index_callbacks_size:Annotated[int, ctypes.c_uint32], index_options:Annotated[int, ctypes.c_uint32], source_filename:c.POINTER[Annotated[bytes, ctypes.c_char]], command_line_args:c.POINTER[c.POINTER[Annotated[bytes, ctypes.c_char]]], num_command_line_args:Annotated[int, ctypes.c_int32], unsaved_files:c.POINTER[struct_CXUnsavedFile], num_unsaved_files:Annotated[int, ctypes.c_uint32], out_TU:c.POINTER[CXTranslationUnit], TU_options:Annotated[int, ctypes.c_uint32]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_indexTranslationUnit(_0:CXIndexAction, client_data:CXClientData, index_callbacks:c.POINTER[IndexerCallbacks], index_callbacks_size:Annotated[int, ctypes.c_uint32], index_options:Annotated[int, ctypes.c_uint32], _5:CXTranslationUnit) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_indexLoc_getFileLocation(loc:CXIdxLoc, indexFile:c.POINTER[CXIdxClientFile], file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_indexLoc_getCXSourceLocation(loc:CXIdxLoc) -> CXSourceLocation: ...
CXFieldVisitor: TypeAlias = c.CFUNCTYPE(enum_CXVisitorResult, CXCursor, c.POINTER[None])
@dll.bind
def clang_Type_visitFields(T:CXType, visitor:CXFieldVisitor, client_data:CXClientData) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_visitCXXBaseClasses(T:CXType, visitor:CXFieldVisitor, client_data:CXClientData) -> Annotated[int, ctypes.c_uint32]: ...
enum_CXBinaryOperatorKind = CEnum(Annotated[int, ctypes.c_uint32])
CXBinaryOperator_Invalid = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Invalid', 0) # type: ignore
CXBinaryOperator_PtrMemD = enum_CXBinaryOperatorKind.define('CXBinaryOperator_PtrMemD', 1) # type: ignore
CXBinaryOperator_PtrMemI = enum_CXBinaryOperatorKind.define('CXBinaryOperator_PtrMemI', 2) # type: ignore
CXBinaryOperator_Mul = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Mul', 3) # type: ignore
CXBinaryOperator_Div = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Div', 4) # type: ignore
CXBinaryOperator_Rem = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Rem', 5) # type: ignore
CXBinaryOperator_Add = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Add', 6) # type: ignore
CXBinaryOperator_Sub = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Sub', 7) # type: ignore
CXBinaryOperator_Shl = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Shl', 8) # type: ignore
CXBinaryOperator_Shr = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Shr', 9) # type: ignore
CXBinaryOperator_Cmp = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Cmp', 10) # type: ignore
CXBinaryOperator_LT = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LT', 11) # type: ignore
CXBinaryOperator_GT = enum_CXBinaryOperatorKind.define('CXBinaryOperator_GT', 12) # type: ignore
CXBinaryOperator_LE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LE', 13) # type: ignore
CXBinaryOperator_GE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_GE', 14) # type: ignore
CXBinaryOperator_EQ = enum_CXBinaryOperatorKind.define('CXBinaryOperator_EQ', 15) # type: ignore
CXBinaryOperator_NE = enum_CXBinaryOperatorKind.define('CXBinaryOperator_NE', 16) # type: ignore
CXBinaryOperator_And = enum_CXBinaryOperatorKind.define('CXBinaryOperator_And', 17) # type: ignore
CXBinaryOperator_Xor = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Xor', 18) # type: ignore
CXBinaryOperator_Or = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Or', 19) # type: ignore
CXBinaryOperator_LAnd = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LAnd', 20) # type: ignore
CXBinaryOperator_LOr = enum_CXBinaryOperatorKind.define('CXBinaryOperator_LOr', 21) # type: ignore
CXBinaryOperator_Assign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Assign', 22) # type: ignore
CXBinaryOperator_MulAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_MulAssign', 23) # type: ignore
CXBinaryOperator_DivAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_DivAssign', 24) # type: ignore
CXBinaryOperator_RemAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_RemAssign', 25) # type: ignore
CXBinaryOperator_AddAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_AddAssign', 26) # type: ignore
CXBinaryOperator_SubAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_SubAssign', 27) # type: ignore
CXBinaryOperator_ShlAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_ShlAssign', 28) # type: ignore
CXBinaryOperator_ShrAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_ShrAssign', 29) # type: ignore
CXBinaryOperator_AndAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_AndAssign', 30) # type: ignore
CXBinaryOperator_XorAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_XorAssign', 31) # type: ignore
CXBinaryOperator_OrAssign = enum_CXBinaryOperatorKind.define('CXBinaryOperator_OrAssign', 32) # type: ignore
CXBinaryOperator_Comma = enum_CXBinaryOperatorKind.define('CXBinaryOperator_Comma', 33) # type: ignore

@dll.bind
def clang_getBinaryOperatorKindSpelling(kind:enum_CXBinaryOperatorKind) -> CXString: ...
@dll.bind
def clang_getCursorBinaryOperatorKind(cursor:CXCursor) -> enum_CXBinaryOperatorKind: ...
enum_CXUnaryOperatorKind = CEnum(Annotated[int, ctypes.c_uint32])
CXUnaryOperator_Invalid = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Invalid', 0) # type: ignore
CXUnaryOperator_PostInc = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PostInc', 1) # type: ignore
CXUnaryOperator_PostDec = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PostDec', 2) # type: ignore
CXUnaryOperator_PreInc = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PreInc', 3) # type: ignore
CXUnaryOperator_PreDec = enum_CXUnaryOperatorKind.define('CXUnaryOperator_PreDec', 4) # type: ignore
CXUnaryOperator_AddrOf = enum_CXUnaryOperatorKind.define('CXUnaryOperator_AddrOf', 5) # type: ignore
CXUnaryOperator_Deref = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Deref', 6) # type: ignore
CXUnaryOperator_Plus = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Plus', 7) # type: ignore
CXUnaryOperator_Minus = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Minus', 8) # type: ignore
CXUnaryOperator_Not = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Not', 9) # type: ignore
CXUnaryOperator_LNot = enum_CXUnaryOperatorKind.define('CXUnaryOperator_LNot', 10) # type: ignore
CXUnaryOperator_Real = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Real', 11) # type: ignore
CXUnaryOperator_Imag = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Imag', 12) # type: ignore
CXUnaryOperator_Extension = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Extension', 13) # type: ignore
CXUnaryOperator_Coawait = enum_CXUnaryOperatorKind.define('CXUnaryOperator_Coawait', 14) # type: ignore

@dll.bind
def clang_getUnaryOperatorKindSpelling(kind:enum_CXUnaryOperatorKind) -> CXString: ...
@dll.bind
def clang_getCursorUnaryOperatorKind(cursor:CXCursor) -> enum_CXUnaryOperatorKind: ...
@dll.bind
def clang_getCString(string:CXString) -> c.POINTER[Annotated[bytes, ctypes.c_char]]: ...
@dll.bind
def clang_disposeString(string:CXString) -> None: ...
@dll.bind
def clang_disposeStringSet(set:c.POINTER[CXStringSet]) -> None: ...
@dll.bind
def clang_getNullLocation() -> CXSourceLocation: ...
@dll.bind
def clang_equalLocations(loc1:CXSourceLocation, loc2:CXSourceLocation) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_isBeforeInTranslationUnit(loc1:CXSourceLocation, loc2:CXSourceLocation) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Location_isInSystemHeader(location:CXSourceLocation) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_Location_isFromMainFile(location:CXSourceLocation) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getNullRange() -> CXSourceRange: ...
@dll.bind
def clang_getRange(begin:CXSourceLocation, end:CXSourceLocation) -> CXSourceRange: ...
@dll.bind
def clang_equalRanges(range1:CXSourceRange, range2:CXSourceRange) -> Annotated[int, ctypes.c_uint32]: ...
@dll.bind
def clang_Range_isNull(range:CXSourceRange) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_getExpansionLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getPresumedLocation(location:CXSourceLocation, filename:c.POINTER[CXString], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getInstantiationLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getSpellingLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getFileLocation(location:CXSourceLocation, file:c.POINTER[CXFile], line:c.POINTER[Annotated[int, ctypes.c_uint32]], column:c.POINTER[Annotated[int, ctypes.c_uint32]], offset:c.POINTER[Annotated[int, ctypes.c_uint32]]) -> None: ...
@dll.bind
def clang_getRangeStart(range:CXSourceRange) -> CXSourceLocation: ...
@dll.bind
def clang_getRangeEnd(range:CXSourceRange) -> CXSourceLocation: ...
@dll.bind
def clang_disposeSourceRangeList(ranges:c.POINTER[CXSourceRangeList]) -> None: ...
@dll.bind
def clang_getFileName(SFile:CXFile) -> CXString: ...
time_t = Annotated[int, ctypes.c_int64]
@dll.bind
def clang_getFileTime(SFile:CXFile) -> time_t: ...
@c.record
class CXFileUniqueID(c.Struct):
  SIZE = 24
  data: Annotated[c.Array[Annotated[int, ctypes.c_uint64], Literal[3]], 0]
@dll.bind
def clang_getFileUniqueID(file:CXFile, outID:c.POINTER[CXFileUniqueID]) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_File_isEqual(file1:CXFile, file2:CXFile) -> Annotated[int, ctypes.c_int32]: ...
@dll.bind
def clang_File_tryGetRealPathName(file:CXFile) -> CXString: ...
c.init_records()
CINDEX_VERSION_MAJOR = 0 # type: ignore
CINDEX_VERSION_MINOR = 64 # type: ignore
CINDEX_VERSION_ENCODE = lambda major,minor: (((major)*10000) + ((minor)*1)) # type: ignore
CINDEX_VERSION = CINDEX_VERSION_ENCODE(CINDEX_VERSION_MAJOR, CINDEX_VERSION_MINOR) # type: ignore
CINDEX_VERSION_STRINGIZE = lambda major,minor: CINDEX_VERSION_STRINGIZE_(major, minor) # type: ignore