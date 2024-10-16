class GLSLLanguage(CStyleLanguage):
  type_map = {dtypes.float: "float", dtypes.half: "float", dtypes.int32: "int", dtypes.uint32: "uint", dtypes.bool: "bool"}
  sampler_prefix = {dtypes.float64: "d", dtypes.float: "", dtypes.half: "", dtypes.int32: "i", dtypes.uint32: "u", dtypes.bool: "i"}
  fragment_center_offset = 0.5
  code_for_workitem = {"i": lambda x, offset=fragment_center_offset:f"int(gl_FragCoord.y-{offset}) * width + int(gl_FragCoord.x-{offset})"}
  code_for_op = {**CStyleLanguage().code_for_op, **{op: lambda a,b,dtype,charforop=charforop: f"bool(int({a}){charforop}int({b}))" \
    if dtype == dtypes.bool else f"({a}{charforop}{b})" for op,charforop in [(BinaryOps.MUL,"*"),(BinaryOps.ADD,"+"),(BinaryOps.DIV,"/")]},
    BinaryOps.CMPLT: lambda a,b,dtype: f"(float({a})<float({b}))" if dtype == dtypes.bool else f"({a}<{b})",
    BinaryOps.MOD: lambda a,b,dtype: f"(int({a})%int({b}))", TernaryOps.WHERE: lambda a,b,c,dtype: f"(float({a})!=0.0?{b}:{c})"}

  def render_const(self, x:Union[float,int], var_dtype) -> str:
    if math.isnan(x): return "(0.0 / 0.0)"
    elif math.isinf(x): return ("-" if x < 0 else "") + "(1./0.)"
    return self.render_cast(["({:.1f})".format(x) if x == int(x) and dtypes.is_float(var_dtype) else f"({x})"]*var_dtype.sz, var_dtype)

  def render_kernel(self, function_name:str, kernel:List[str], bufs:List[Tuple[str,DType]], local_size:List[int], prekernel:List[str]) -> str:
    prg = "#version 330\nprecision highp float;\nprecision highp int;\nin vec2 uv;\nuniform int width;\n"
    prg += "\n".join([f"uniform {self.sampler_prefix[dtype]}sampler2D {name};" for name,dtype in bufs if name != "data0"])
    prg += f"\nout {'int' if bufs[0][1] == dtypes.bool else self.type_map[bufs[0][1]]} out_data;\n"
    return prg + "\nvoid main() {\n" + "\n".join(kernel) + "\n}"

  def render_cast(self, x:List[str], var_dtype:DType, bitcast=False) -> str:
    if self.type_map[var_dtype]: return f"{self.type_map[var_dtype]}({x[0]})"
    raise NotImplementedError(f"no cast for {var_dtype}")

  def render_load(self, output_dtype, buf_name, buf_dtype, idx, local=False) -> str:
    x_calc = f"float(int({idx})%textureSize({buf_name}, 0).x)"
    y_calc = f"float(int({idx})/textureSize({buf_name}, 0).x)"
    out_val = f"texture({buf_name}, vec2(float({x_calc} + {self.fragment_center_offset}f)/float(textureSize({buf_name}, 0).x),\
    float({y_calc} + {self.fragment_center_offset}f)/float(textureSize({buf_name}, 0).y))).r"
    return f"{self.render_cast([out_val], output_dtype)}"

  def render_store(self, buf_name:str, buf_dtype:DType, var_name:str, var_dtype:DType, idx, local=False) -> str:
    return f"out_data = {'int' if buf_dtype == dtypes.bool else self.type_map[buf_dtype]}({var_name});"
