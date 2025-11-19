from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import cdiv, cmod, CORRECT_DIVMOD_FOLDING, unwrap

def cancel_divmod(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # simple cancel div/mod case when the range of the numerator lies within a single denominator interval
  x_min, x_max, y_min, y_max = x.vmin, x.vmax, y.vmin, y.vmax
  assert isinstance(x_min, int) and isinstance(x_max, int) and isinstance(y_min, int) and isinstance(y_max, int)
  if y_min==y_max==0: raise ZeroDivisionError(f"{'Division' if d.op is Ops.IDIV else 'Mod'} by zero trying to rewrite {x.alu(d.op, y)}")
  if y_min*y_max > 0 and (q:=cdiv(x_min,y_min)) == cdiv(x_min,y_max) == cdiv(x_max,y_min) == cdiv(x_max,y_max):
    return x - q*y if d.op is Ops.MOD else d.const_like(q)
  return None

def fold_divmod_const(d: UOp, x: UOp, y: UOp) -> UOp|None:
  if ((c := y.arg) < 0): return None
  x_peeled, const = x.pop_const()
  uops = list(x_peeled.split_uop(Ops.ADD))

  # remove_nested_mod: remove nested mod in case the inner mod is a multiple of the outer mod, example: (a%4 + b)%2 -> (a+b)%2
  if d.op is Ops.MOD and x.vmin >= 0:
    new_xs, changed = [], False
    for u in uops:
      if u.op is Ops.MOD and u.src[1].divides(c) is not None:
        new_xs.append(u.src[0])
        changed = True
      else: new_xs.append(u)
    if changed: return (UOp.sum(*new_xs) + const) % y

  # Shared decomposition for folding rules
  decomp = [(u.divides(f:=u.const_factor()),f) for u in uops]
  terms, factors = zip(*decomp)

  # fold_binary_numerator: fold if expression has one non-constant term that takes on two values
  if len(terms)==1 and (v:=terms[0]).vmax-v.vmin == 1:
    y1 = cmod(factors[0]*v.vmin+const, c) if d.op is Ops.MOD else cdiv(factors[0]*v.vmin+const, c)
    y2 = cmod(factors[0]*v.vmax+const, c) if d.op is Ops.MOD else cdiv(factors[0]*v.vmax+const, c)
    return (y2-y1)*(v-v.vmin) + y1

  # fold_divmod_congruence: fold if a is congruent to an expression whose range is between 0 and c
  if not (x.vmin<0 and CORRECT_DIVMOD_FOLDING):
    rems = [min((r:=f%c), r-c, key=abs) for f in factors]
    if (rem:=sum(r*v for r,v in zip(rems,terms))+const%c).vmin//c==rem.vmax//c:
      if d.op is Ops.MOD: return rem - rem.vmin//c*c
      return sum((f-r)//c * v for f,r,v in zip(factors,rems,terms)) + (const-const%c+rem.vmin//c*c)//c

  # gcd_with_remainder: factor out common gcd from numerator
  if x.vmin >= 0:
    gcd = UOp.gcd(*uops, y).simplify()
    if gcd.op is Ops.CONST and gcd.arg > 1:
      new_x = unwrap(x_peeled.divide_exact(gcd)).simplify() + (const%c)//gcd.arg
      if new_x.vmin >= 0:
        ret = new_x.alu(d.op, x.ufix(c//gcd.arg))
        return ret*gcd + const%gcd.arg if d.op is Ops.MOD else ret+const//c
  return None

def fold_divmod_variable(d: UOp, x: UOp, y: UOp) -> UOp|None:
  uops = list(x.split_uop(Ops.ADD))

  # 1. divide_by_gcd: x//y -> (x//gcd)//(y//gcd)  or  x%y -> gcd*(x//gcd)%(y//gcd)
  gcd = UOp.gcd(*uops, y).simplify()
  if not (gcd.op is Ops.CONST and gcd.arg==1):
    ret = unwrap(x.divide_exact(gcd)).alu(d.op, unwrap(y.divide_exact(gcd)))
    return ret*gcd if d.op is Ops.MOD else ret

  # 2. factor_remainder: (d*x+y)//d -> x+y//d  or  (d*x+y)%d
  # for mod we go further and take the remainder of all factors to reduce their size
  # These only work for floordiv (and the corresponding remainder)! Thats why we check the sign of x,y and new_x
  if y.vmin<0 or x.vmin<0: return None
  quo, rem = [], []
  for u in uops:
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    # if this is mod and y is a const, we can make the remainder factor sm
    elif d.op is Ops.MOD and y.op is Ops.CONST and (c:=u.const_factor())%y.arg!=c:
      rem.append(u.divides(c)*(c%y.arg))
      quo.append(u.const_like(0))  # we append this so we can check if something changed
    else: rem.append(u)

  if not quo: return None
  new_x = sum(rem)+x.const_like(0)
  if new_x.vmin<0: return None
  return new_x%y if d.op is Ops.MOD else new_x//y+sum(quo)

def nest_div_by_smallest_factor(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # we try and nest the div and see if it allows the numerator to be simplified
  if ((c := y.arg) < 0): return None
  factors = [u.const_factor() for u in x.split_uop(Ops.ADD) if u.op not in (Ops.CONST, Ops.VCONST)]
  div = min([y.arg]+[abs(f) for f in factors if abs(f) > 1 and (c%f)==0])
  newxs = fold_divmod_const(newx:=(x//div), x, y.const_like(div))
  if newxs is None: newxs = fold_divmod_variable(newx, x, y.const_like(div))
  if div==y.arg or newxs is None or x.vmin<0 or newx.vmin<0: return None
  return newxs//(c//div)

div_and_mod_symbolic = PatternMatcher([
  # ** 1. Fast Inline Rules **
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d)
    if c.vmin>0 and d.vmin>0 and ((x.vmin>=0 and a.vmin>=0) or (x.vmax<=0 and a.vmax<=0)) else None),  # (x//c+a)//d -> (x+a*c)//(c*d)
  (UPat.var("x", dtypes.index) // UPat.var("d"), lambda x,d: -(x//(-d)) if d.vmax < 0 else None),
  (UPat.var("x", dtypes.index) // UPat.var("d"), lambda x,d: -((-x)//d) if x.vmax <= 0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: ((x+c.arg%d.arg)//d + c.arg//d.arg) if c.arg%d.arg!=c.arg and x.vmin>=0 and n.vmin>=0 and d.arg>0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: (-(-(c.arg%d.arg + x - (d.arg-1))//d) + c.arg//d.arg) if x.vmax<=0 and n.vmin>=0 and d.arg>0 else None),

  # ** 2. Slow Rules **
  # NOTE: if you move this one below `fold_divmod_const` you get more uops in test/external/external_benchmark_schedule.py
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.var("y"))), cancel_divmod),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.cvar("y", vec=False))), fold_divmod_const),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.var("y"))), fold_divmod_variable),
  (UPat(Ops.IDIV, dtypes.index, name="d", src=(UPat.var("x"), UPat.cvar("y", vec=False))), nest_div_by_smallest_factor),

  # NOTE: these have to go at the bottom or TestSymbolicOps.test_var loops
  (UPat.var("x", dtypes.index) % UPat.var("d"), lambda x,d: -((-x)%d) if x.vmax <= 0 else None),
  (UPat.var("x", dtypes.index) % UPat.var("d"), lambda x,d: (x%(-d)) if d.vmax < 0 else None),
])