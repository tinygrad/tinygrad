from tinygrad import Tensor,dtypes,Device,TinyJit
from tinygrad.device import Buffer, BufferOptions
from tinygrad.helpers import Context, dedup, DEBUG
from tinygrad.lazy import LazyBuffer
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.view import View, strides_for_shape
from tinygrad.ops import ReduceOps
import ctypes
import math
import dataclasses
import copy
import functools
import tinygrad.runtime.autogen.cuda as cuda
import tinygrad.runtime.ops_cuda
Device.DEFAULT="CUDA"

class RandomUniform:
  def __init__(self,gpus,seeds=None):
    import ctypes, ctypes.util
    CURAND_RNG_PSEUDO_XORWOW = 101
    lib = ctypes.CDLL(ctypes.util.find_library('curand'))
    self.f = lib["curandGenerateUniform"]
    lib["curandCreateGenerator"].argtypes = [ctypes.POINTER(ctypes.c_void_p),ctypes.c_uint]
    lib["curandSetPseudoRandomGeneratorSeed"].argtypes = [ctypes.c_void_p,ctypes.c_ulonglong]
    self.generators = [ctypes.c_void_p() for gpu in gpus]
    if seeds is None:
      seeds = range(len(gpus))
    for generator,gpu,seed, in zip(self.generators, gpus,seeds):
      tinygrad.runtime.ops_cuda.check(cuda.cuCtxSetCurrent(Device[gpu].context))
      lib["curandCreateGenerator"](ctypes.byref(generator),CURAND_RNG_PSEUDO_XORWOW)
      lib["curandSetPseudoRandomGeneratorSeed"](generator, ctypes.c_ulonglong(seed))

  def __call__(self, bufs):
    for generator, buf in zip(self.generators, bufs):
      tinygrad.runtime.ops_cuda.check(cuda.cuCtxSetCurrent(Device[buf.lazydata.realized.device].context))
      self.f(generator,buf.lazydata.realized._buf, buf.lazydata.realized.size)

  def forward(self, buf, generator):
    tinygrad.runtime.ops_cuda.check(cuda.cuCtxSetCurrent(Device[buf.lazydata.realized.device].context))
    self.f(generator,buf.lazydata.realized._buf, buf.lazydata.realized.size)

def tonumpy(x,axis=0):
  import numpy as np
  return np.concatenate([el.numpy() for el in x],axis=axis)

def fastrun(f,*args):
    if not isinstance(f,TinyJit) or f.cnt < 2:
      f(*args)
    else:
      input_rawbuffers = [v.lazydata.base.realized for v in args if v.__class__ is Tensor]
      # input_rawbuffers = [lb.base.realized for v in args for lb in v.lazydata.lbs if v.__class__ is Tensor]
      for (j,i),input_idx in f.input_replace.items():
        f.jit_cache[j].rawbufs[i] = input_rawbuffers[input_idx]
      for ji in f.jit_cache:
        ji.prg(ji.rawbufs, {}, wait=DEBUG>=2)
      for (j,i),input_idx in f.input_replace.items():
        f.jit_cache[j].rawbufs[i] = None

class runparallel:
  def __init__(self,f=None,gpus=None,n=1):
    self.args = []
    self.jitted = {}
    self.cnt = 0
    self.argcount = None
    if isinstance(f,self.__class__):
      self.gpus = f.gpus
      self.f = f.f
      self.n = f.n
    elif hasattr(f,"__self__"):
      self.gpus = f.__self__.gpus
      self.f = f
      self.n = n
    else:
      self.gpus = gpus
      self.f = f
      self.n = n

  def __get__(self,obj,objtype):
    newself = self.__class__(functools.partial(self.f,obj),obj.gpus,self.n)
    newself.name = self.name
    setattr(obj,self.name,newself)
    return newself

  def __set_name__(self,owner,name):
    self.name = name

  def __call__(self,*args):
    if self.f is None:
      self.f = args[0]
      return self
    if self.argcount == None: self.argcount = len(args)
    assert self.argcount == len(args)
    lists = [el for el in args if isinstance(el,list)]
    assert all(len(el) == len(lists[0]) for el in lists)
    args = [el if isinstance(el,list) else [el]*len(lists[0]) for el in args]
    gpuargs = [[arg for arg in gpuargs] for gpuargs in zip(*args)]
    self.args.extend(gpuargs)
    self.cnt += 1
    if self.cnt == self.n:
      self.flush()

  def addtojitted(self):
    flatargs = [arg for argset in self.args for arg in argset]
    if self.cnt not in self.jitted:
      flatargs_rawbuffers = [v.lazydata.base.realized._buf.value if isinstance(v,Tensor) else None for v in flatargs]
      input_rawbuffers = [v.lazydata.base.realized._buf.value for v in flatargs if isinstance(v,Tensor)]
      other_inds = [n for n,el in enumerate(flatargs) if not isinstance(el,Tensor)]
      nodupflatargs = dedup(input_rawbuffers)
      nodupflatinds = [flatargs_rawbuffers.index(el) for el in nodupflatargs]
      arginds = [nodupflatargs.index(el.lazydata.base.realized._buf.value) if n not in other_inds else len(nodupflatargs)+other_inds.index(n) for n,el in enumerate(flatargs)]
      self.jitted[self.cnt] = {"inds": arginds, "flatten_inds": nodupflatinds+other_inds}
      def f(*args):
        for n2 in range(self.cnt*len(self.gpus)):
          self.f(*[args[k] for k in self.jitted[self.cnt]["inds"][n2*self.argcount:(n2+1)*self.argcount]])
      self.jitted[self.cnt]["f"] = TinyJit(f)

  def flush(self):
    if self.cnt==0:
      return
    if self.cnt not in self.jitted:
      self.addtojitted()
    flatargs = [arg for argset in self.args for arg in argset]
    nodupargs = [flatargs[i] for i in self.jitted[self.cnt]["flatten_inds"]]
    fastrun(self.jitted[self.cnt]["f"],*nodupargs)
    self.cnt = 0
    self.args = []

class Buffer_nodel(Buffer):
  def __del__(self):
    pass

def realize_pointer(tens):
  assert hasattr(tens.lazydata.base, "realized")
  assert tens.lazydata.base.st.contiguous
  assert len(tens.lazydata.st.views) == 1
  base = tens.lazydata.base
  st = tens.lazydata.st
  offset = st.views[-1].offset
  new_view = dataclasses.replace(st.views[-1], offset=0, contiguous=st.views[-1].strides==strides_for_shape(st.views[-1].shape))
  new_st = ShapeTracker((new_view,))
  options = BufferOptions(uncached=True,nolru=True) if not tens.lazydata.base.realized.options else dataclasses.replace(tens.lazydata.base.realized.options, uncached=True,nolru=True)
  newbuf = Buffer_nodel(tens.lazydata.device,tens.lazydata.size,tens.lazydata.dtype,type(tens.lazydata.base.realized._buf)(tens.lazydata.base.realized._buf.value + offset*tens.dtype.itemsize),options)

  new_base = LazyBuffer.__new__(LazyBuffer)
  new_base = copy.copy(base)

  new_base.buffer = newbuf

  newlb = new_base._view(new_st)
  newt = copy.copy(tens)
  newt.lazydata = newlb
  return newt

class Transducer:
  import numpy as np
  def __init__(self,B,maxT,maxU,K,gpus,loss_factor,dtype=dtypes.float32,beam=1,debug=0):
    self.beam = beam
    self.debug=debug
    self.B, self.maxT, self.maxU, self.K = B,maxT,maxU,K
    self.nullind = K-1
    self.gpus = gpus
    self.factor = [Tensor([loss_factor],dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.loss = [Tensor.zeros(B,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.loss2 = [Tensor.zeros(B,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.a = [Tensor.zeros(B,maxT,maxU,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.b = [Tensor.zeros(B,maxT,maxU,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]

    self.helperk = [Tensor.arange(0,K,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.helpergt = [Tensor.ones(1,dtype=dtype,device=gpu).pad(((K-1,0),)).contiguous().realize() for gpu in gpus]
    self.helperu = [Tensor.arange(0,self.maxU,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.helpert = [Tensor.arange(0,self.maxT,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.helperab = [Tensor.arange(0,self.maxU+self.maxT-1,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.masku = [Tensor.zeros(B,maxU-1,K,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.maskt = [Tensor.ones(1,dtype=dtype,device=gpu).pad(((K-1,0),)).contiguous().realize() for gpu in gpus]
    self.prtxt_pad = [Tensor.zeros(B,maxT,maxU+2*(maxT-1),dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.prnull_pad = [Tensor.zeros(B,maxT,maxU+2*(maxT-1),dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]

    self.prtxt = [realize_pointer(el[:,:,maxT-1:maxU+(maxT-1)]) for el in self.prtxt_pad]
    self.prnull = [realize_pointer(el[:,:,maxT-1:maxU+(maxT-1)]) for el in self.prnull_pad]
    self.a_dall = [Tensor.zeros(maxU-1+maxT-1+1,B,maxT,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.b_dall = [Tensor.zeros(maxU-1+maxT-1+1,B,maxT,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]
    self.a_view = [self.makeab(el) for el in self.a_dall]
    self.b_view = [self.makeab(el) for el in self.b_dall]
    self.a_d = [[realize_pointer(self.a_dall[ngpu][n]) for ngpu in range(len(self.gpus))] for n in range(self.a_dall[0].shape[0])]
    self.b_d = [[realize_pointer(self.b_dall[ngpu][n]) for ngpu in range(len(self.gpus))] for n in range(self.b_dall[0].shape[0])]
    self.prtxt_diag = [list(x) for x in zip(*[self.diagonals(el) for el in self.prtxt_pad])]
    self.prnull_diag = [list(x) for x in zip(*[self.diagonals(el) for el in self.prnull_pad])]

  def makeab(self, dall):
    newshape = (self.B,self.maxT,self.maxU)
    shape = dall.lazydata.shape
    strides = dall.lazydata.st.views[-1].strides
    newstrides = (strides[1],strides[0]+1,strides[0])
    newshape = (shape[1],shape[2],shape[0]-(shape[2]-1))
    res = Tensor(dall.lazydata._view(ShapeTracker((View.create(shape=newshape,strides=newstrides,offset=0),))),device=dall.device)
    return res

  def logsumexp(self,x1,x2):
    return (x1==float("-inf")).where(x2,(x2==float("-inf")).where(x1,(x1>x2).where(x1+(1+(x2-x1).exp()).log(), x2+(1+(x1-x2).exp()).log())))

  def diagonals(self,x):
    B,T,dim3 = x.shape
    U = dim3-2*(T-1)
    diags = []
    for d in range((U-1)+(T-1)+1):
      diag = x[:,:,T-1+d]
      views = diag.lazydata.st.views
      views = (dataclasses.replace(views[-1], strides=views[-1].strides[:-1]+(views[-1].strides[-1]-1,)),)
      diag.lazydata.st = ShapeTracker(views)
      diags.append(realize_pointer(diag))
    return diags

  @runparallel
  def forward_probs(self,logits,txt,helperk,prtxt,prnull,masku,helpert,helperu,lent,lenu,ainit,binit,helperab):
    masku.assign((txt.unsqueeze(-1)==helperk).cast(masku.dtype)).realize()
    with Context(BEAM=min(1,self.beam or 0),DEBUG=self.debug):
      tempu = (masku.unsqueeze(1)*logits[:,:,:-1,:]).sum(-1).realize()
    masklen1 = (lent.unsqueeze(1)<=helpert).unsqueeze(2) + (lenu.unsqueeze(1)<=helperu).unsqueeze(1)
    tempu = (masklen1).where(float("-inf"),tempu.pad((None,None,(0,1)),value=float("-inf")))
    prtxt.assign( tempu.pad((None,None,(self.maxT-1,self.maxT-1)),value=float("-inf")) ).realize()

    tempt = logits.shrink((None,None,None,(self.nullind,self.nullind+1))).reshape(logits.shape[:3])
    masklen2 = (lent.unsqueeze(1)-1<=helpert).unsqueeze(2) + (lenu.unsqueeze(1)<helperu).unsqueeze(1)
    tempt = (masklen2).where(float("-inf"),tempt)
    prnull.assign( tempt.pad((None,None,(self.maxT-1,self.maxT-1)),value=float("-inf")) ).realize()

    masklen3 = (lent[:,None]<=helpert)[None] + ((lenu)[None,:,None]+helpert<helperab[:,None,None])+ (helpert>helperab[:,None,None])
    ainit.assign(masklen3.where(float("-inf"),((helpert[None,None,:]==0) * (helperab[:,None,None]==0)).where(0,float("-inf"))).cast(ainit.dtype)).realize()
    batch_final_prnull = (logits[:,:,:,self.nullind]*(lenu[:,None,None]==helperu[None,None,:])*((lent-1)[:,None,None]==helpert[None,:,None])).sum((1,2))[None,:,None]
    mask_final_prnull = ((lent-1)[None,:,None]==helpert[None,None,:]) * (lenu[None,:,None]+helpert == helperab[:,None,None])
    binit.assign(masklen3.where(float("-inf"),mask_final_prnull.where(batch_final_prnull,float("-inf")))).realize()

  @runparallel(n=5)
  def akern(self, ad, adp, ptxt, pnull):
    term1 = adp+ptxt
    term2 = (adp+pnull).pad((None,(1,0)),value=float("-inf"))[:,:-1]
    ad.assign(self.logsumexp(term1,term2))
    ad.realize()

  @runparallel(n=5)
  def bkern(self, bd, bdp, ptxt, pnull):
    term1 = bdp+ptxt
    term2 = pnull + bdp.pad((None,(0,1)),value=float("-inf"))[:,1:]
    res = self.logsumexp(term1,term2)
    bd.assign(((bd==float("-inf"))).where(res,bd))
    bd.realize()

  @runparallel
  def init_ab(self,a,b,prnull):
    a.assign(Tensor.zeros_like(a.shrink((None,(0,1)))).pad((None,(0,a.shape[1]-1)),value=float("-inf"))).realize()
    b.assign(prnull[:,-1:,-1].pad((None,(b.shape[1]-1,0)),value=float("-inf"))).realize()

  @runparallel
  def getloss(self,b,loss,loss2,factor):
    loss.assign(-b[0,:,0]).realize()
    loss2.assign(loss*factor).realize()

  @runparallel
  def getab(self,a,a_view):
    a.assign(a_view).realize()

  def forward(self,logits,txt,audio_len,txt_len,nullind):
    self.forward_probs(logits,txt,self.helperk,self.prtxt_pad,self.prnull_pad,self.masku,self.helpert,self.helperu,audio_len,txt_len,self.a_dall,self.b_dall,self.helperab)
    adp = self.a_d[0]
    bdp = self.b_d[-1]
    for n in range(1,(self.maxT-1)+(self.maxU-1)+1):
      ad = self.a_d[n]
      self.akern(ad,adp,self.prtxt_diag[n-1],self.prnull_diag[n-1])
      adp = ad
      bd = self.b_d[-n-1]
      self.bkern(bd,bdp,self.prtxt_diag[-n-1],self.prnull_diag[-n-1])
      bdp = bd
    self.akern.flush()
    self.bkern.flush()
    self.getloss(self.b_dall,self.loss,self.loss2,self.factor)
    self.getab(self.a,self.a_view)
    self.getab(self.b,self.b_view)
    return self.loss2

  @runparallel
  def backward_kernel(self,logits,txt,logits_grad,lent,lenu,prtxt,prnull,a,b,loss,helpergt,helperk,helperu,helpert,factor):
    with Context(DEBUG=self.debug,BEAM=0):
      import numpy as np
      bnt = b.shrink((None,(1,b.shape[1]),None)).pad((None,(0,1),None),value=float("-inf"))
      bnt = (((lent-1)[:,None]==helpert)[:,:,None]*(lenu[:,None]==helperu)[:,None,:]).where(0,bnt)
      bnu = b.shrink((None,None,(1,b.shape[2]))).pad((None,None,(0,1)),value=float("-inf"))
      facnt = loss[:,None,None] + a + bnt + logits[:,:,:,self.nullind]
      facnu = loss[:,None,None] + a + bnu + prtxt
      masku = (txt.unsqueeze(-1)==helperk).cast(logits_grad.dtype).pad((None,(0,1),None))[:,None,:,:]
      grad = -facnt.exp()[:,:,:,None]*(helpergt==1).where(1-logits.exp(),-logits.exp()) - facnu.exp()[:,:,:,None]*(masku).where(1-logits.exp(),-logits.exp())
      logits_grad.assign(grad*factor).realize()

  def backward(self,logits,logits_grad,txt,audio_len,txt_len,nullind):
    self.backward_kernel(logits,txt,logits_grad,audio_len,txt_len,self.prtxt,self.prnull,self.a_view,self.b_view,self.loss,self.helpergt,self.helperk,self.helperu,self.helpert,self.factor)

def mmulsplitk(m1,m2,parts):
  M,K = m1.shape[:2]
  N = m2.shape[1]
  data = (m1.reshape((M,1,K))*m2.reshape((1,K,N)).transpose(1,2)).lazydata.reshape((M,N,parts,K//parts))._reduce_op(ReduceOps.SUM, [3]).reshape((M,N,parts))._reduce_op(ReduceOps.SUM, [2]).reshape((M,N))
  return Tensor(data,device=m1.device)

class LSTM:
  def togpus(self,tens):
    return [tens.realize()] + [tens.to(gpu).realize() for gpu in self.gpus[1:]]

  def zeros(self,*shape):
    return self.togpus(Tensor.zeros(*shape,device=self.gpus[0],dtype=self.dtype).contiguous().realize())

  def __init__(self, batch_size, maxT, input_size, hid_size, layers, forget_gate_bias=1.0, weights_init_scale=0.45, beam=8, gpus=[Device.DEFAULT], dtype=dtypes.float32, debug=0, dropout=0, eval=False):
    self.batch_size = batch_size
    self.maxT = maxT
    self.input_size = input_size
    self.hid_size = hid_size
    self.layers = layers
    self.beam=beam
    self.gpus=gpus
    self.debug = debug
    self.dropout = dropout
    self.eval = eval
    self.dtype=dtype

    self.forget_gate_bias = forget_gate_bias
    self.weights_init_scale = weights_init_scale

    if not eval:
      self.uniform_generate = RandomUniform(self.gpus)

    def rand_same_all_gpus(M,N1=None,N2=None,bound=1.0):
      res = [Tensor.uniform([M] if N1 is None else [M,N1 if layer==0 or N2 is None else N2],low=-bound,high=bound,dtype=dtype,device=gpus[0]).realize() for layer in range(layers)]
      return [[layer]+[layer.to(gpu).realize() for gpu in gpus[1:]] for layer in res]

    def zeros_weights(M,N1=None,N2=None):
      return [[Tensor.zeros([M] if N1 is None else [M,N1 if layer==0 or N2 is None else N2],dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus] for layer in range(layers)]

    def rand_bias_set_forget(H,forget,bound):
      res = []
      for n in range(self.layers):
        f = Tensor.full((H,),fill_value=forget,dtype=dtype,device=gpus[0])
        i,g,o = [Tensor.uniform((H,),low=-bound,high=bound,dtype=dtype,device=gpus[0]).realize() for _ in range(3)]
        stacked = i.cat(f,g,o).realize()
        res.append([stacked] + [stacked.to(gpu).realize() for gpu in gpus[1:]])
      return res

    bound = self.weights_init_scale/(self.hid_size**0.5)
    self.w1 = rand_same_all_gpus(4*hid_size,input_size,hid_size,bound=bound)
    self.w2 = rand_same_all_gpus(4*hid_size,hid_size,bound=bound)
    self.b1 = rand_bias_set_forget(hid_size,self.forget_gate_bias*self.weights_init_scale,bound)
    self.b2 = rand_bias_set_forget(hid_size,0,bound)
    if not eval:
      self.w1_grad = zeros_weights(4*hid_size,input_size,hid_size)
      self.w2_grad = zeros_weights(4*hid_size,hid_size)
      self.b1_grad = zeros_weights(4*hid_size)
      self.b2_grad = zeros_weights(4*hid_size)

    def zeros(*args):
      return [[Tensor.zeros(*args,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus] for layer in range(layers)]

    self.adotp = zeros(maxT,batch_size,4*hid_size)
    self.ahph = zeros(maxT+1,batch_size,hid_size)
    self.ahp = [[gpuels.shrink(((0,maxT),None,None)) for gpuels in lay] for lay in self.ahph]
    self.ah = [[realize_pointer(gpuels[1:]) for gpuels in lay] for lay in self.ahph]
    self.ac = zeros(maxT,batch_size,hid_size)
    if not eval:
      self.adotp_grad = zeros(maxT,batch_size,4*hid_size)
      self.ah_grad = zeros(maxT,batch_size,hid_size)
      self.ac_grad = zeros(maxT,batch_size,hid_size)
      self.ax_grad = [Tensor.zeros(maxT,batch_size,input_size,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus]

      self.dropouts = [[Tensor.zeros(maxT,batch_size,hid_size,dtype=dtype,device=gpu).contiguous().realize() for gpu in gpus] for layer in range(layers-1)]

    self.zc = zeros(batch_size,hid_size)
    self.zhp = zeros(batch_size,hid_size)
    self.zcp = zeros(batch_size,hid_size)
    if not eval:
      self.zh_grad = zeros(batch_size,hid_size)
      self.zc_grad = zeros(batch_size,hid_size)

    self.dotp = [[list(el) for el in zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in lay])] for lay in self.adotp]
    self.h = [[list(el) for el in zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in lay])] for lay in self.ah]
    self.hp = [[self.zhp[lay]]+self.h[lay][:-1] for lay in range(layers)]
    self.c = [[list(el) for el in zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in lay])] for lay in self.ac]
    self.cp = [[self.zcp[lay]]+self.c[lay][:-1] for lay in range(layers)]
    if not eval:
      self.dotp_grad = [[list(el) for el in zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in lay])] for lay in self.adotp_grad]
      self.h_grad = [[list(el) for el in zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in lay])] for lay in self.ah_grad]
      self.c_grad = [[list(el) for el in zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in lay])] for lay in self.ac_grad]
      self.x_grad = list(zip(*[[realize_pointer(gpuels[n]) for n in range(gpuels.shape[0])] for gpuels in self.ax_grad]))

    self.forwardcell2 = runparallel(self.forwardcell) if self.input_size!=self.hid_size else self.forwardcell
    self.forwardcell_a2 = runparallel(self.forwardcell_a) if self.input_size!=self.hid_size else self.forwardcell_a

    self.backwardcell2 = runparallel(self.backwardcell) if self.input_size!=self.hid_size else self.backwardcell
    self.backwardcell_a2 = runparallel(self.backwardcell_a) if self.input_size!=self.hid_size else self.backwardcell_a

    self.parameters = [el for lay in zip(self.w1,self.w2,self.b1,self.b2) for el in lay]
    if not self.eval:
      self.grads = [el for lay in zip(self.w1_grad,self.w2_grad,self.b1_grad,self.b2_grad) for el in lay]
      self.other_grads = [el for lay in zip(self.adotp_grad,self.ah_grad,self.ac_grad,self.ax_grad,self.zh_grad,self.zc_grad) for el in lay]

  @runparallel
  def forwardcell_a(self,dotp,x,w1,b1,b2):
    with Context(DEBUG=self.debug, BEAM=self.beam):
      dotp.assign(x@w1.T + b1 + b2).realize()

  @runparallel
  def evaluate_forwardcell(self,w1,w2,b1,b2,x,cp,hp,c,h):
    with Context(DEBUG=self.debug, BEAM=self.beam):
      with Context(BEAM=self.beam):
        temp = x@w1.T + hp@w2.T + b1 + b2
        temp.realize()
      with Context(BEAM=min(self.beam or 0, 1)):
        i0,f0,g0,o0 = temp.chunk(4,1)
        c.assign(cp*f0.sigmoid() + i0.sigmoid()*g0.tanh()).realize()
        h.assign(c.tanh()*o0.sigmoid()).realize()

  @runparallel(n=10)
  def forwardcell(self,dotp,w2,hp,cp,h,c):
    with Context(DEBUG=self.debug, BEAM=self.beam):
      dotp += hp@w2.T
      dotp.realize()
    with Context(DEBUG=self.debug, BEAM=min(self.beam or 0, 1)):
      i0,f0,g0,o0 = dotp.chunk(4,1)
      c.assign(cp*f0.sigmoid() + i0.sigmoid()*g0.tanh()).realize()
      h.assign(c.tanh()*o0.sigmoid()).realize()

  @runparallel
  def dropoutcell(self,x,dom):
    dom.assign((dom>=self.dropout).cast(dom.dtype)*1.0/(1.0-self.dropout)).realize()
    x.assign(dom*x).realize()

  @runparallel
  def dropoutcell_backward(self,x_grad,dom):
    x_grad.assign(x_grad*dom*1.0/(1.0-self.dropout)).realize()

  def forward(self,x,T):
    self.ax = x
    self.T = T
    if not self.eval:
      for dom in self.dropouts:
        self.uniform_generate(dom)
    for layer in range(self.layers):
      if layer != 0 and not self.eval:
        self.dropoutcell(x,self.dropouts[layer-1])
      forward_a = self.forwardcell_a if layer == 0 else self.forwardcell_a2
      forward_a(self.adotp[layer],x,self.w1[layer],self.b1[layer],self.b2[layer])
      forward_f = self.forwardcell if layer == 0 else self.forwardcell2
      cp = self.zcp[layer]
      hp = self.zhp[layer]
      w2 = self.w2[layer]
      dotp = self.dotp[layer]
      self.synchronize()
      for n in range(0,T):
        h = self.h[layer][n]
        c = self.c[layer][n]
        hp = self.hp[layer][n]
        cp = self.cp[layer][n]
        forward_f(dotp[n],w2,hp,cp,h,c)
      forward_f.flush()
      self.synchronize()
      x = self.ah[layer]
    return self.ah[-1]

  @runparallel(n=3)
  def backwardcell(self,dotp,dotp_grad,h_grad,c_grad,grad_hp,grad_cp,c,cp,w2):
    with Context(DEBUG=self.debug, BEAM=min(self.beam or 0, 0)):
      i0,f0,g0,o0 = dotp.chunk(4,1)
      c_grad += h_grad*o0.sigmoid()*(1-c.tanh()**2)
      c_grad.realize()
      gi0 = c_grad*(1-(i0/2).tanh()**2)/4*g0.tanh()
      gf0 = c_grad*cp*(1-(f0/2).tanh()**2)/4
      gg0 = c_grad*i0.sigmoid()*(1-g0.tanh()**2)
      go0 = h_grad*c.tanh()*(1-(o0/2).tanh()**2)/4
      dotp_grad.assign(gi0.cat(gf0,gg0,go0,dim=1))
      dotp_grad.realize()
      grad_cp.assign(c_grad*f0.sigmoid())
      grad_cp.realize()
    with Context(DEBUG=self.debug, BEAM=self.beam):
      grad_hp += dotp_grad@w2
      grad_hp.realize()

  @runparallel
  def backwardcell_a(self,dotp_grad,x,x_grad,ahp,w1,grad_w1,grad_w2,grad_b1,grad_b2):
    with Context(DEBUG=self.debug, BEAM=0):
      grad_b = dotp_grad.sum(axis=(0,1)).realize()
      grad_b1 += grad_b
      grad_b2 += grad_b
      grad_b1.realize()
      grad_b2.realize()

    with Context(DEBUG=self.debug, BEAM=self.beam):
      def reshape(mat):
        return mat.reshape((-1,mat.shape[-1]))
      m1 = reshape(dotp_grad)
      m2 = reshape(x)
      m3 = reshape(ahp)
      parts = min(math.gcd(4,m1.shape[0]),m1.shape[0])
      grad_w1 += mmulsplitk(m1.T,m2, parts=parts)
      grad_w2 += mmulsplitk(m1.T, m3, parts=parts)

      x_grad.assign(dotp_grad@w1)
      grad_w1.realize()
      grad_w2.realize()
      x_grad.realize()

  @runparallel
  def add_ah_grad_in(self, ah_grad_in, ah_grad):
    with Context(DEBUG=self.debug):
      ah_grad.assign(ah_grad_in)
      ah_grad.realize()

  @runparallel
  def zero_grads(self,*args):
    for arg in args:
      arg.assign(Tensor.zeros_like(arg)).realize()

  def backward(self, ah_grad_in):
    self.add_ah_grad_in(ah_grad_in,self.ah_grad[-1])
    self.zero_grads(*self.zh_grad,*self.zc_grad,*self.ac_grad)
    for layer in range(self.layers-1,-1,-1):
      backward_f = self.backwardcell if layer == 0 else self.backwardcell2
      backward_a = self.backwardcell_a if layer == 0 else self.backwardcell_a2
      w1, w2= self.w1[layer], self.w2[layer]
      w1_grad, w2_grad, b1_grad, b2_grad = self.w1_grad[layer], self.w2_grad[layer], self.b1_grad[layer], self.b2_grad[layer]
      dotp = self.dotp[layer]
      dotp_grad = self.dotp_grad[layer]
      h,c = self.h[layer], self.c[layer]
      h_grad = self.h_grad[layer]
      c_grad = self.c_grad[layer]
      ax = self.ah[layer-1] if layer>0 else self.ax
      ax_grad = self.ah_grad[layer-1] if layer>0 else self.ax_grad

      self.synchronize()
      for t in range(self.T-1,-1,-1):
        cp = c[t-1]if t>0 else self.zc[layer]
        grad_hp = h_grad[t-1] if t>0 else self.zh_grad[layer]
        grad_cp = c_grad[t-1] if t>0 else self.zc_grad[layer]
        backward_f(dotp[t],dotp_grad[t],h_grad[t],c_grad[t],grad_hp,grad_cp,c[t],cp,w2)
      backward_f.flush()
      self.synchronize()
      backward_a(self.adotp_grad[layer],ax,ax_grad,self.ahp[layer],w1,w1_grad,w2_grad,b1_grad,b2_grad)
      if layer != 0 and not self.eval:
        self.dropoutcell_backward(ax_grad,self.dropouts[layer-1])
    return self.ax_grad

  def synchronize(self):
    for gpu in self.gpus:
      Device[gpu].synchronize()

class RNNT:
  def togpus(self,tens):
    return [tens.realize()] + [tens.to(gpu).realize() for gpu in self.gpus[1:]]

  def zeros(self,*shape,dtype=None):
    return self.togpus(Tensor.zeros(*shape,device=self.gpus[0],dtype=self.dtype if dtype is None else dtype).contiguous().realize())

  def __init__(self, gpus,

    nclasses = 1024,
    enc_input_size = 256,
    enc_hid_size = 1024,
    enc1_layers = 2,
    enc2_layers = 3,
    pred_layers = 2,
    batch_size = 32,
    pred_input_size = 512,
    pred_hid_size = 512,
    lin_outputsize = 512,
    enc_dropout = 0.1,
    pred_dropout = 0.3,
    joint_dropout = 0.3,
    forget_gate_bias = 1.0,
    weights_init_scale = 0.45,

    maxT=642,
    maxU=126,

    opt_eps=1e-9,
    wd=1e-3,
    ema=0.994,
    opt_b1 = 0.9,
    opt_b2 = 0.9985,
    lr=0.0062,
    min_lr=1e-5,
    lr_exp_gamma=0.915,
    grad_accumulation_factor=1.0,
    max_global_norm = 1.0,
    warmup_epochs = 1,
    hold_epochs = 11,

    beam=8, debug=3,
    dtype=dtypes.float32,
    eval=False):

    self.eval = eval

    self.enc1_layers,self.enc2_layers,self.pred_layers,self.batch_size,self.enc_input_size,self.enc_hid_size,self.pred_input_size,self.pred_hid_size,self.lin_outputsize,self.nclasses = enc1_layers,enc2_layers,pred_layers,batch_size,enc_input_size,enc_hid_size,pred_input_size,pred_hid_size,lin_outputsize,nclasses
    assert maxT%2 == 0

    self.enc_dropout = enc_dropout
    self.pred_dropout = pred_dropout
    self.joint_dropout = joint_dropout

    self.nullind = self.nclasses-1

    self.beam = beam
    self.gpus = gpus
    self.dtype = dtype

    self.maxT = maxT
    self.maxU = maxU

    self.debug = debug
    togpus=self.togpus
    zeros=self.zeros
    self.grad_accumulation_factor = grad_accumulation_factor

    def lin_init(*shape):
      initval = 1/math.sqrt(shape[-1])
      return togpus(Tensor.uniform(shape, low=-initval, high=initval,device=gpus[0],dtype=dtype).realize())

    if not eval:
      self.transducer = Transducer(self.batch_size, self.maxT//2, self.maxU, self.nclasses,gpus=self.gpus,loss_factor=self.grad_accumulation_factor/(self.batch_size*len(gpus)),dtype=dtype,beam=self.beam,debug=self.debug)

    if self.beam and not eval:
      self.sum = zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
      self.f = zeros(self.maxT//2,self.batch_size,self.lin_outputsize)
      self.g = zeros(self.maxU,self.batch_size,self.lin_outputsize)
      self.joint_lin = zeros(self.nclasses,self.lin_outputsize)
      self.joint_lin_bias = zeros(self.nclasses)
      self.logits = zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)

      self.joint(self.sum,self.f,self.g,self.joint_lin,self.joint_lin_bias,self.logits)
      del self.sum,self.f,self.g,self.joint_lin,self.joint_lin_bias,self.logits
      for gpu in gpus:
        Device[gpu].allocator.free_cache()

      self.joint_lin_grad = zeros(self.nclasses,self.lin_outputsize)
      self.joint_lin_bias_grad = zeros(self.nclasses)
      self.joint_lin = zeros(self.nclasses,self.lin_outputsize)
      self.sum = zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
      self.sum2_grad = zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)
      self.sum_grad = zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
      self.f_grad = zeros(self.maxT//2,self.batch_size,self.lin_outputsize)
      self.g_grad = zeros(self.maxU,self.batch_size,self.lin_outputsize)
      self.joint_grad(self.joint_lin_grad,self.joint_lin_bias_grad,self.joint_lin,self.sum,self.sum2_grad,self.sum_grad,self.f_grad,self.g_grad)
      del self.joint_lin_grad,self.joint_lin_bias_grad,self.joint_lin,self.sum,self.sum2_grad,self.sum_grad,self.f_grad,self.g_grad
      for gpu in gpus:
        Device[gpu].allocator.free_cache()

      logits = zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)
      txt = zeros(self.batch_size,self.maxU-1, dtype=dtypes.int32)
      audio_len = zeros(self.batch_size, dtype=dtypes.int32)
      txt_len = zeros(self.batch_size, dtype=dtypes.int32)
      self.transducer.forward(logits,txt,audio_len,txt_len,self.nullind)
      del logits,txt,audio_len,txt_len
      for gpu in gpus:
        Device[gpu].allocator.free_cache()

    self.enc_lstm1 = LSTM(self.batch_size,self.maxT,self.enc_input_size,self.enc_hid_size,self.enc1_layers,forget_gate_bias,weights_init_scale,gpus=gpus,beam=self.beam,debug=debug,dropout=self.enc_dropout,dtype=dtype,eval=eval)
    self.enc_lstm2 = LSTM(self.batch_size,self.maxT//2,self.enc_hid_size*2,self.enc_hid_size,self.enc2_layers,forget_gate_bias,weights_init_scale,gpus=gpus,beam=self.beam,debug=debug,dropout=self.enc_dropout,dtype=dtype,eval=eval)

    self.enc_lin = lin_init(self.lin_outputsize,self.enc_hid_size)
    self.enc_lin_bias = lin_init(self.lin_outputsize)
    if not eval:
      self.enc_lin_bias_grad = zeros(self.lin_outputsize)
      self.enc_lin_grad = zeros(self.lin_outputsize,self.enc_hid_size)

    self.embed_weight = togpus(Tensor.normal(self.nclasses-1,self.pred_input_size,mean=0,std=1,dtype=dtype,device=gpus[0]).realize())
    self.embed_txt = zeros(self.maxU-1,self.batch_size,self.nclasses-1)
    if not eval:
      self.embed_weight_grad = zeros(self.nclasses-1,self.pred_input_size)
    self.embed_helper = togpus(Tensor.arange(self.nclasses-1,dtype=dtype).reshape(1, 1, -1).realize())
    self.eval_helper = togpus(Tensor.arange(self.nclasses,dtype=dtype).realize())
    self.evalf_helper = togpus(Tensor.arange(self.maxT//2,dtype=dtype).realize())
    self.temp_eval_sum = self.zeros(self.batch_size,self.nclasses)

    self.pred_input = zeros(self.maxU, self.batch_size, self.pred_input_size)
    if not eval:
      self.pred_input_grad = zeros(self.maxU, self.batch_size, self.pred_input_size)
    self.pred_lstm = LSTM(self.batch_size,self.maxU,self.pred_input_size,self.pred_hid_size,self.pred_layers,forget_gate_bias,weights_init_scale,gpus=gpus,beam=self.beam,debug=debug,dropout=self.pred_dropout,dtype=dtype,eval=eval)

    self.pred_lin = lin_init(self.lin_outputsize,self.pred_hid_size)
    self.pred_lin_bias = lin_init(self.lin_outputsize)
    if not eval:
      self.pred_lin_bias_grad = zeros(self.lin_outputsize)
      self.pred_lin_grad = zeros(self.lin_outputsize,self.pred_hid_size)

    self.xstack_len = togpus(Tensor.zeros(batch_size,device=gpus[0],dtype=dtypes.int).contiguous().realize())

    self.joint_lin = lin_init(self.nclasses,self.lin_outputsize)
    self.joint_lin_bias = lin_init(self.nclasses)


    self.f1s = zeros(self.maxT//2,self.batch_size,2*self.enc_hid_size)
    self.f = zeros(self.maxT//2,self.batch_size,self.lin_outputsize)
    if not eval:
      self.g = zeros(self.maxU,self.batch_size,self.lin_outputsize)

      self.f2_grad = zeros(self.maxT//2,self.batch_size,self.enc_hid_size)
      self.g1_grad = zeros(self.maxU,self.batch_size,self.pred_hid_size)

      self.f1_grad = zeros(self.maxT,self.batch_size,self.enc_hid_size)

    if not eval:
      self.sum = zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
      self.logits = zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)
      self.sum_grad = zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
      self.sum2_grad = zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)
      self.joint_lin_grad = zeros(self.nclasses,self.lin_outputsize)
      self.joint_lin_bias_grad = zeros(self.nclasses)
      self.f_grad = zeros(self.maxT//2,self.batch_size,self.lin_outputsize)
      self.g_grad = zeros(self.maxU,self.batch_size,self.lin_outputsize)

    if not eval:
      self.uniform_generate = RandomUniform(self.gpus)

      self.enc1_dropouts = zeros(self.maxT,self.batch_size,self.enc_hid_size)
      self.enc2_dropouts = zeros(self.maxT//2,self.batch_size,self.enc_hid_size)
      self.pred_dropouts = zeros(self.maxU,self.batch_size,self.pred_hid_size)

    self.kenc1_dropout = runparallel(self.dropoutcell)
    self.kenc2_dropout = runparallel(self.dropoutcell)
    self.kenc1_dropout_backward = runparallel(self.dropoutcell_backward)
    self.kenc2_dropout_backward = runparallel(self.dropoutcell_backward)
    self.kpred_dropout = runparallel(self.dropoutcell)
    self.kpred_dropout_backward = runparallel(self.dropoutcell_backward)
    self.kjoint_dropout = runparallel(self.dropoutcell)
    self.kjoint_dropout_backward = runparallel(self.dropoutcell_backward)

    self.max_global_norm = max_global_norm

    self.parameters = (
      self.enc_lstm1.parameters+self.enc_lstm2.parameters+[self.embed_weight]+self.pred_lstm.parameters+
      [self.pred_lin,self.pred_lin_bias,self.enc_lin, self.enc_lin_bias,self.joint_lin,self.joint_lin_bias]
    )
    if not self.eval:
      self.ema_parameters = [[Tensor.zeros_like(el).contiguous().realize() for el in par] for par in self.parameters]
      self.copy_parameters(self.ema_parameters,self.parameters)
      self.grads = (
        self.enc_lstm1.grads+self.enc_lstm2.grads+[self.embed_weight_grad]+self.pred_lstm.grads+
        [self.pred_lin_grad,self.pred_lin_bias_grad,self.enc_lin_grad, self.enc_lin_bias_grad,self.joint_lin_grad,self.joint_lin_bias_grad]
      )
      self.grads_buffer = [[Tensor.zeros_like(el).contiguous().realize() for el in grad] for grad in self.grads]
      self.m = [[Tensor.zeros_like(el).contiguous().realize() for el in par] for par in self.parameters]
      self.v = [[Tensor.zeros_like(el).contiguous().realize() for el in par] for par in self.parameters]
      self.opt_b1 = opt_b1
      self.opt_b2 = opt_b2
      self.opt_eps = opt_eps
      self.wd = wd
      self.ema = ema
      self.tt = togpus(Tensor([0],device=gpus[0],dtype=dtype).realize())
      self.lrtens = togpus(Tensor([1],device=gpus[0],dtype=dtype).realize())
      self.lr = lr
      self.min_lr = min_lr
      self.lr_exp_gamma = lr_exp_gamma
      self.warmup_epochs = warmup_epochs
      self.hold_epochs = hold_epochs

      self.opt_parameters = self.m+self.v+[self.tt]

      self.norm = zeros(1)

      self.stepnr = 1

  @runparallel
  def apply_ema(self,*args):
    n = len(args)
    ema_parameters = args[:n//2]
    parameters = args[n//2:]
    for p1,p2 in zip(ema_parameters,parameters):
      p1.assign(self.ema*p1+(1-self.ema)*p2).realize()

  def save_parameters(self, ema=True, opt_parameters=True, filename=None):
    import pickle
    if filename is None:
      filename = "parameters_ema.dat" if ema else "parameters.dat"
    if ema:
      parameters = self.ema_parameters.copy()
    else:
      parameters = self.parameters.copy()
      if opt_parameters:
        parameters += self.opt_parameters
    with open(filename,"wb") as f:
      pickle.dump([el[0].numpy() for el in parameters],f)
    print(f"saved parameters {filename}")

  def load_parameters(self, filename="parameters.data"):
    import pickle
    with open(filename,"rb") as fp:
      loaded_params = pickle.load(fp)
    opt_parameters = len(loaded_params) == self.parameters
    parameters = self.parameters.copy()
    if opt_parameters:
      parameters += self.opt_parameters
    for el1,el2 in zip(parameters,loaded_params):
      for gpuel in el1:
        gpuel.lazydata.base.realized.copyin(el2.data)
    print(f"loaded parameters {filename}")

  def copy_parameters(self, dest, src):
    for mp1, mp2 in zip(dest,src):
      for par1, par2 in zip(mp1,mp2):
        par1.lazydata.base.realized.copyin(par2.lazydata.base.realized.as_buffer(allow_zero_copy=True))

  @runparallel
  def step_kernel(self,tt,lr,norm,*args):
    n = len(args)
    assert n%4==0
    n = int(n/4)
    params = list(args[:n])
    grads = args[n:2*n]
    m = list(args[2*n:3*n])
    v = list(args[3*n:4*n])
    tt.assign(tt + 1)
    for i, (t,tgrad) in enumerate(zip(params,grads)):
      tgrad = tgrad*norm
      m[i].assign(self.opt_b1 * m[i] + (1.0 - self.opt_b1) * tgrad)
      v[i].assign(self.opt_b2 * v[i] + (1.0 - self.opt_b2) * (tgrad * tgrad))
      m_hat = m[i] / (1.0 - self.opt_b1**tt)
      v_hat = v[i] / (1.0 - self.opt_b2**tt)
      up = (m_hat / (v_hat.sqrt() + self.opt_eps)) + self.wd * t.detach()
      if self.wd != 0:
        r1 = t.detach().square().sum().sqrt()
        r2 = up.square().sum().sqrt()
        r = Tensor.where(r1 > 0, Tensor.where(r2 > 0, r1 / r2, 1.0), 1.0)
      else:
        r = 1.0
      t.assign(t.detach() - lr * r * up)
    Tensor.corealize([tt] + m + v + params)

  @runparallel
  def global_norm_kernel(self,norm,max_global_norm,*args):
    with Context():
      res = args[0].square().sum()
      for arg in args[1:]:
        res += arg.square().sum()
      res = res.sqrt().reshape(1)
      res = (res>max_global_norm).where(max_global_norm/res,1.0)
      norm.assign(res).realize()

  def sum_grads_gpus(self):
    for grad1,grad2 in zip(self.grads,self.grads_buffer):
      for gpugrad2 in grad2:
        gpugrad2.assign(sum(gpugrad1.to(gpugrad2.device) for gpugrad1 in grad1)).realize()
    self.copy_parameters(self.grads,self.grads_buffer)

  def step(self,epoch=None,step=None):
    if epoch is not None and step is not None:
      if epoch < self.warmup_epochs:
        lr = ((step+1)/self.opt_steps_per_epoch+epoch)/self.warmup_epochs * self.lr
      elif epoch < self.warmup_epochs+self.hold_epochs:
        lr = self.lr
      else:
        lr = self.lr_exp_gamma ** (epoch+1 - self.warmup_epochs - self.hold_epochs) * self.lr
      self.lrlast = lr = max(self.min_lr,lr)
      for n,gpu in enumerate(self.gpus):
        self.lrtens[n].assign(Tensor([lr], dtype=self.lrtens[n].dtype, device=gpu)).realize()
    self.sum_grads_gpus()
    self.global_norm_kernel(self.norm,self.max_global_norm,*self.grads)
    self.step_kernel(self.tt,self.lrtens,self.norm,*self.parameters,*self.grads,*self.m,*self.v)
    self.stepnr += 1

  @runparallel
  def zero_grads_kernel(self,*args):
    for arg in args:
      arg.assign(Tensor.zeros_like(arg).contiguous()).realize()

  def zero_grads(self):
    self.zero_grads_kernel(*(self.grads))

  @runparallel
  def embed(self,embed_txt, embed_helper,embed_weight,txt,embedded):
    embed_txt.assign((embed_helper == txt.transpose(0,1).unsqueeze(2)).cast(embed_txt.dtype))
    embed_txt.realize()
    embedded.assign( (embed_txt @ embed_weight).pad(((1,0),(0,0),(0,0))).contiguous()).realize()

  @runparallel
  def embed_grad(self, embed_weight_grad, embed_txt,pred_input_grad):
    embed_weight_grad.assign(embed_txt.reshape(-1,embed_txt.shape[-1]).transpose(0,1)@pred_input_grad[1:].reshape(-1,pred_input_grad.shape[-1])).realize()

  @runparallel
  def stack(self,x,xs,x_len,xstack_len):
    xs.assign(x.transpose(0,1).reshape(x.shape[1],x.shape[0]//2,-1).transpose(0,1).contiguous()).realize()
    xstack_len.assign(((x_len+1)/2).cast(dtypes.int)).realize()

  @runparallel
  def destack(self,x,xs):
    x.assign(xs.transpose(0,1).reshape(xs.shape[1],xs.shape[0]*2,-1).transpose(0,1).contiguous()).realize()

  @runparallel
  def lin(self,f,f2,enc_lin,enc_lin_bias,g,g1,pred_lin,pred_lin_bias):
    f.assign(f2@enc_lin.T+enc_lin_bias)
    g.assign(g1@pred_lin.T+pred_lin_bias)
    f.realize()
    g.realize()

  @runparallel
  def lin1_grad(self,f2_grad,f_grad,enc_lin,g1_grad,g_grad,pred_lin,enc_lin_grad,enc_lin_bias_grad,f2,g1,pred_lin_grad,pred_lin_bias_grad):
    f2_grad.assign(f_grad@enc_lin)
    enc_lin_bias_grad += f_grad.sum(axis=[0,1])
    pred_lin_bias_grad += g_grad.sum(axis=[0,1])
    g1_grad.assign(g_grad@pred_lin)
    enc_lin_grad += f_grad.reshape((-1,f_grad.shape[-1])).T@f2.reshape((-1,f2.shape[-1]))
    pred_lin_grad += g_grad.reshape((-1,g_grad.shape[-1])).T@g1.reshape((-1,g1.shape[-1]))
    f2_grad.realize()
    g1_grad.realize()
    enc_lin_grad.realize()
    enc_lin_bias_grad.realize()
    pred_lin_grad.realize()
    pred_lin_bias_grad.realize()

  def dropoutcell(self,x,dom,dropout):
    dom.assign((dom>=dropout).cast(dom.dtype)*1.0/(1.0-dropout)).realize()
    x.assign(dom*x).realize()

  def dropoutcell_backward(self,x_grad,dom):
    x_grad.assign(x_grad*dom).realize()

  @runparallel
  def joint(self,sum,f,g,joint_lin,joint_lin_bias,logits):
    with Context(DEBUG=self.debug):
      fg = f.transpose(0,1).unsqueeze(2) + g.transpose(0,1).unsqueeze(1)
      if not self.eval:
        sum.assign((fg>0).where((sum>=self.joint_dropout)*1.0/(1.0-self.joint_dropout)*fg,0))
      else:
        sum.assign(fg.relu())

      logits.assign((sum@joint_lin.T+joint_lin_bias))
      with Context(BEAM=min(self.beam or 0,1)):
        sum.realize()
      with Context(BEAM=self.beam):
        logits.realize()
      with Context(BEAM=min(self.beam or 0,2)):
        logits.assign(logits.log_softmax())
        logits.realize()

  @runparallel
  def joint_grad(self,joint_lin_grad,joint_lin_bias_grad,joint_lin,sum,sum2_grad,sum_grad,f_grad,g_grad):
    with Context(DEBUG=self.debug, BEAM=self.beam):
      parts = math.gcd(32,math.prod(sum.shape[:-1]))
      joint_lin_bias_grad += sum2_grad.sum(axis=[0,1,2])
      joint_lin_grad += mmulsplitk(sum2_grad.reshape((-1,sum2_grad.shape[-1])).transpose(0,1), sum.reshape((-1,sum.shape[-1])), parts=parts)
      sum_grad.assign((sum>0).where((sum2_grad@joint_lin)*1.0/(1.0-self.joint_dropout),0))
      f_grad.assign(sum_grad.transpose(0,1).sum(axis=2))
      g_grad.assign(sum_grad.transpose(0,2).sum(axis=1))
      with Context(BEAM=self.beam):
        joint_lin_grad.realize()
      with Context(BEAM=min(self.beam or 0,8)):
        joint_lin_bias_grad.realize()
      with Context(BEAM=min(self.beam or 0,8)):
        sum_grad.realize()
      with Context(BEAM=min(self.beam or 0,1)):
        f_grad.realize()
        g_grad.realize()

  def transducer_forward(self,logits,txt,x_len,txt_len):
    return self.transducer.forward(logits,txt,x_len,txt_len,self.nullind)

  def transducer_backward(self,logits,sum2_grad,txt,x_len,txt_len):
    self.transducer.backward(logits,sum2_grad,txt,x_len,txt_len,self.nullind)

  def synchronize(self):
    for gpu in self.gpus:
      Device[gpu].synchronize()

  def forward(self,x,x_len,txt,txt_len,T=None,U=None):
    self.x = x
    self.x_len = x_len
    self.txt = txt
    self.txt_len = txt_len
    self.T = T if T is not None else self.maxT
    self.U = U if U is not None else self.maxU
    assert self.T%2==0

    if not self.eval:
      self.synchronize()

      self.uniform_generate(self.enc1_dropouts)
      self.uniform_generate(self.enc2_dropouts)
      self.uniform_generate(self.pred_dropouts)
      self.uniform_generate(self.sum)

      self.synchronize()

    self.f1 = self.enc_lstm1.forward(x,self.T)
    if not self.eval:
      self.kenc1_dropout(self.f1,self.enc1_dropouts,self.enc_dropout)
    self.stack(self.f1,self.f1s,self.x_len,self.xstack_len)
    self.f2 = self.enc_lstm2.forward(self.f1s,self.T//2)
    if not self.eval:
      self.kenc2_dropout(self.f2,self.enc2_dropouts,self.enc_dropout)

    self.embed(self.embed_txt,self.embed_helper,self.embed_weight,self.txt,self.pred_input)
    self.g1 = self.pred_lstm.forward(self.pred_input,self.U)
    if not self.eval:
      self.kpred_dropout(self.g1,self.pred_dropouts,self.pred_dropout)

    self.lin(self.f,self.f2,self.enc_lin,self.enc_lin_bias,self.g,self.g1,self.pred_lin,self.pred_lin_bias)
    self.joint(self.sum,self.f,self.g,self.joint_lin,self.joint_lin_bias,self.logits)
    self.synchronize()
    loss = self.transducer_forward(self.logits,self.txt,self.xstack_len,self.txt_len)
    return loss

  def backward(self):
    self.transducer_backward(self.logits,self.sum2_grad,self.txt,self.xstack_len,self.txt_len)
    self.synchronize()

    self.joint_grad(self.joint_lin_grad,self.joint_lin_bias_grad,self.joint_lin,self.sum,self.sum2_grad,self.sum_grad,self.f_grad,self.g_grad)

    self.lin1_grad(self.f2_grad,self.f_grad,self.enc_lin,self.g1_grad,self.g_grad,self.pred_lin,self.enc_lin_grad,self.enc_lin_bias_grad,self.f2,self.g1,self.pred_lin_grad,self.pred_lin_bias_grad)


    self.kpred_dropout_backward(self.g1_grad, self.pred_dropouts)
    self.pred_input_grad = self.pred_lstm.backward(self.g1_grad)
    self.embed_grad(self.embed_weight_grad,self.embed_txt,self.pred_input_grad)

    self.kenc1_dropout_backward(self.f2_grad, self.enc2_dropouts)
    self.f1s_grad = self.enc_lstm2.backward(self.f2_grad)
    self.destack(self.f1_grad, self.f1s_grad)
    self.kenc2_dropout_backward(self.f1_grad, self.enc1_dropouts)
    self.x_grad = self.enc_lstm1.backward(self.f1_grad)

  @runparallel
  def eval_embed(self, embed_helper,embed_weight,txtlabels,embedded):
    temp = (embed_helper.reshape(embed_helper.shape[-1:]) == txtlabels.unsqueeze(1)).cast(embed_weight.dtype)
    embedded.assign( (txtlabels.unsqueeze(1)!=self.nclasses-1).where(temp @ embed_weight,embedded)).realize()

  @runparallel
  def eval_embedf(self,embed_helper,embed_weight,labels,embedded):
    temp = (embed_helper.unsqueeze(1) == labels.unsqueeze(0)).cast(embed_weight.dtype)
    embedded.assign( (temp.unsqueeze(2) * embed_weight).sum(axis=0) ).realize()

  @runparallel
  def word_output_join(self,f,h,pred_lin,pred_lin_bias,joint_lin,joint_lin_bias,output,helper,sum):
    g = h@pred_lin.T+pred_lin_bias
    sum.assign((f+g).relu()@joint_lin.T+joint_lin_bias).realize()
    output.assign((sum==sum.max(axis=-1,keepdim=True)).where(helper,-1).cast(output.dtype).max(axis=-1))
    output.realize()

  def word_output(self,x_in,cp,hp,c,h,f,out):
    x = x_in
    w1 = self.pred_lstm.w1
    w2 = self.pred_lstm.w2
    b1 = self.pred_lstm.b1
    b2 = self.pred_lstm.b2
    for lay in range(self.pred_layers):
      self.pred_lstm.evaluate_forwardcell(w1[lay],w2[lay],b1[lay],b2[lay],x,cp[lay],hp[lay],c[lay],h[lay])
      x = h[lay]
    lstm_out = h[-1]
    self.word_output_join(f,lstm_out,self.pred_lin,self.pred_lin_bias,self.joint_lin,self.joint_lin_bias,out,self.eval_helper,self.temp_eval_sum)

  @runparallel
  def eval_lin(self,f,f2,enc_lin,enc_lin_bias):
    f.assign(f2@enc_lin.T+enc_lin_bias).realize()

  @runparallel
  def eval_proc_output(self,layers,do_mask,T,audio_len,prediction,U,U_withblank,*args):
    assert len(args)==4*layers
    cp = list(args[0:layers])
    hp = list(args[layers:2*layers])
    c = args[2*layers:3*layers]
    h = args[3*layers:4*layers]

    do_mask.assign((T<audio_len).cast(do_mask.dtype)).realize()
    isblank = (prediction==self.nullind).cast(do_mask.dtype)
    notblank = (prediction!=self.nullind).cast(do_mask.dtype)
    T += isblank*do_mask
    U += notblank*do_mask
    U_withblank += do_mask
    T.realize()
    U.realize()
    U_withblank.realize()
    for n in range(layers):
      cp[n].assign(cp[n]*isblank.unsqueeze(1) + c[n]*notblank.unsqueeze(1))
      hp[n].assign(hp[n]*isblank.unsqueeze(1) + h[n]*notblank.unsqueeze(1))
      cp[n].realize()
      hp[n].realize()

  @runparallel
  def eval_audio_len_stack(self,audio_len):
    audio_len.assign(((audio_len+1)/2).cast(audio_len.dtype)).realize()

  def evaluate_batch(self, audio, audio_len):
    import numpy as np
    self.x_len = audio_len
    self.f1 = self.enc_lstm1.forward(audio,self.maxT)
    self.stack(self.f1,self.f1s,self.x_len,self.xstack_len)
    self.f2 = self.enc_lstm2.forward(self.f1s,self.maxT//2)
    self.eval_lin(self.f,self.f2,self.enc_lin,self.enc_lin_bias)
    self.eval_audio_len_stack(audio_len)

    T = self.zeros(self.batch_size,dtype=dtypes.int32)
    U = self.zeros(self.batch_size,dtype=dtypes.int32)
    U_withblank = self.zeros(self.batch_size,dtype=dtypes.int32)
    do_mask = self.togpus(Tensor.ones_like(T[0],dtype=dtypes.int32).contiguous().realize())
    txt = self.zeros(self.batch_size,self.pred_input_size)
    predictions = self.zeros(self.maxT//2+300,self.batch_size,dtype=dtypes.int32)
    cp = [self.zeros(self.batch_size,self.pred_hid_size) for lay in range(self.pred_layers)]
    hp = [self.zeros(self.batch_size,self.pred_hid_size) for lay in range(self.pred_layers)]
    c = [self.zeros(self.batch_size,self.pred_hid_size) for lay in range(self.pred_layers)]
    h = [self.zeros(self.batch_size,self.pred_hid_size) for lay in range(self.pred_layers)]
    f = self.zeros(self.batch_size,self.lin_outputsize)
    self.eval_embedf(self.evalf_helper,self.f,T,f)
    prediction = [[realize_pointer(el[n]) for el in predictions] for n in range(predictions[0].shape[0])]

    n = 0
    while (work_counter:=tonumpy(do_mask).sum().item()) != 0 and n < predictions[0].shape[0]:
      self.word_output(txt,cp,hp,c,h,f,prediction[n])
      self.eval_proc_output(self.pred_layers,do_mask,T,audio_len,prediction[n],U,U_withblank,*cp,*hp,*c,*h)
      self.eval_embed(self.embed_helper,self.embed_weight,prediction[n],txt)
      self.eval_embedf(self.evalf_helper,self.f,T,f)
      n += 1
    maxu = tonumpy(U_withblank).max()
    prediction_nonull = [[el.item() for el in batch if el!=self.nclasses-1] for batch in tonumpy(predictions,axis=1)[:int(maxu)].T]
    return prediction_nonull

  def del_joint_bufs(self):
    del self.sum, self.logits, self.sum_grad, self.sum2_grad
    for gpu in self.gpus:
      Device[gpu].allocator.free_cache()

  def remake_joint_bufs(self):
    self.sum = self.zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
    self.logits = self.zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)
    self.sum_grad = self.zeros(self.batch_size,self.maxT//2,self.maxU,self.lin_outputsize)
    self.sum2_grad = self.zeros(self.batch_size,self.maxT//2,self.maxU,self.nclasses)