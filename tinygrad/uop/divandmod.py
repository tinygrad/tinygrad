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

def fold_binary_numerator(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # we can fold if the expression has only one non-constant term and this term can only take on two values
  if ((c := y.arg) < 0): return None
  x,const = x.pop_const()
  terms, factors = zip(*[(u.divides(f:=u.const_factor()),f) for u in x.split_uop(Ops.ADD)])
  if len(terms)==1 and (v:=terms[0]).vmax-v.vmin == 1:
    y1 = cmod(factors[0]*v.vmin+const, c) if d.op is Ops.MOD else cdiv(factors[0]*v.vmin+const, c)
    y2 = cmod(factors[0]*v.vmax+const, c) if d.op is Ops.MOD else cdiv(factors[0]*v.vmax+const, c)
    return (y2-y1)*(v-v.vmin) + y1
  return None

def fold_divmod_congruence(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # within a mod we can freely subtract multiples of c, we use this to see if a is congruent to an expression whose vmin/vmax are between 0 and c
  if (x.vmin<0 and CORRECT_DIVMOD_FOLDING) or ((c := y.arg) < 0): return None
  x,const = x.pop_const()
  terms, factors = zip(*[(u.divides(f:=u.const_factor()),f) for u in x.split_uop(Ops.ADD)])
  # a//c = (a-a%c)/c, if we can fold a%c, we can fold a//c
  rems = [min((r:=f%c), r-c, key=abs) for f in factors]
  if (rem:=sum(r*v for r,v in zip(rems,terms))+const%c).vmin//c!=rem.vmax//c: return None
  if d.op is Ops.MOD: return rem - rem.vmin//c*c
  return sum((f-r)//c * v for f,r,v in zip(factors,rems,terms)) + (const-const%c+rem.vmin//c*c)//c

def divide_by_gcd(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # x//y -> (x//gcd)//(y//gcd) or x%y -> gcd*(x//gcd)%(y//gcd)
  gcd = UOp.gcd(*x.split_uop(Ops.ADD), y).simplify()
  if gcd.op is Ops.CONST and gcd.arg==1: return None
  ret = unwrap(x.divide_exact(gcd)).alu(d.op, unwrap(y.divide_exact(gcd)))
  return ret*gcd if d.op is Ops.MOD else ret

def gcd_with_remainder(d: UOp, x: UOp, y: UOp):
  # (gcd*x+r)//(gcd*d) -> (x+(r%d)//gcd)//d + r//(gcd*d)
  # (gcd*x+r)%(gcd*d) -> gcd*(x+(r%d)//gcd)%d + r%gcd
  # These only work for floordiv (and the corresponding remainder)! Thats why we check the sign of x,y and new_x
  if ((c := y.arg) < 0) or x.vmin<0: return None
  x_no_const, const = x.pop_const()
  gcd = UOp.gcd(*x_no_const.split_uop(Ops.ADD), y).simplify()
  assert gcd.op is Ops.CONST
  if gcd.arg==1: return None
  new_x = unwrap(x_no_const.divide_exact(gcd)).simplify() + (const%c)//gcd
  if new_x.vmin<0: return None
  ret = new_x.alu(d.op, x.ufix(c//gcd.arg))
  return ret*gcd + const%gcd.arg if d.op is Ops.MOD else ret+const//c

def remove_nested_mod(m: UOp, x: UOp, y: UOp) -> UOp|None:
  # remove nested mod in case the inner mod is a multiple of the outer mod
  # example: (a%4 + b)%2 -> (a+b)%2
  if ((c := y.arg) < 0) or x.vmin<0: return None
  new_xs = []
  something_changed = False
  for u in x.split_uop(Ops.ADD):
    if u.op is Ops.MOD:
      if u.src[1].divides(c) is not None:
        something_changed = True
        u = u.src[0]
    new_xs.append(u)
  new_x: UOp = UOp.sum(*new_xs)
  if something_changed and new_x.vmin>=0: return new_x % y
  return None

def nest_div_by_smallest_factor(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # we try and nest the div and see if it allows the numerator to be simplified
  if ((c := y.arg) < 0): return None
  factors = [u.const_factor() for u in x.split_uop(Ops.ADD) if u.op not in (Ops.CONST, Ops.VCONST)]
  div = min([y.arg]+[abs(f) for f in factors if abs(f) > 1 and (c%f)==0])
  newxs = fold_divmod_congruence(newx:=(x//div), x, y.const_like(div))
  if newxs is None: newxs = factor_remainder(newx, x, y.const_like(div))
  if div==y.arg or newxs is None or x.vmin<0 or newx.vmin<0: return None
  return newxs//(c//div)

def factor_remainder(d: UOp, x: UOp, y: UOp) -> UOp|None:
  # (d*x+y)//d -> x+y//d  or  (d*x+y)%d
  # for mod we go further and take the remainder of all factors to reduce their size
  # These only work for floordiv (and the corresponding remainder)! Thats why we check the sign of x,y and new_x
  if y.vmin<0 or x.vmin<0: return None
  quo, rem = [], []
  for u in x.split_uop(Ops.ADD):
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    # if this is mod and y is a const, we can make the remainder factor sm
    elif d.op is Ops.MOD and y.op is Ops.CONST and (c:=u.const_factor())%y.arg!=c:
      rem.append(u.divides(c)*(c%y.arg))
      quo.append(u.const_like(0))  # we append this so we can check if something changed
    else: rem.append(u)
  new_x = sum(rem)+x.const_like(0)
  if len(quo)==0 or new_x.vmin<0: return None
  return new_x%y if d.op is Ops.MOD else new_x//y+sum(quo)


div_and_mod_symbolic = PatternMatcher([
  # ** div **
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d)
    if c.vmin>0 and d.vmin>0 and ((x.vmin>=0 and a.vmin>=0) or (x.vmax<=0 and a.vmax<=0)) else None),  # (x//c+a)//d -> (x+a*c)//(c*d)
  # slow div and mod rules
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.var("y"))), cancel_divmod),
  (UPat.var("x", dtypes.index) // UPat.var("d"), lambda x,d: -(x//(-d)) if d.vmax < 0 else None),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.cvar("y", vec=False))), fold_binary_numerator),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.cvar("y", vec=False))), fold_divmod_congruence),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.var("y"))), divide_by_gcd),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.cvar("y", vec=False))), gcd_with_remainder),
  (UPat(Ops.MOD, dtypes.index, name="m", src=(UPat.var("x"), UPat.cvar("y", vec=False))), remove_nested_mod),
  (UPat((Ops.IDIV), dtypes.index, name="d", src=(UPat.var("x"), UPat.cvar("y", vec=False))), nest_div_by_smallest_factor),
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d", src=(UPat.var("x"), UPat.var("y"))), factor_remainder),
  # div folding
  (UPat.var("x", dtypes.index) // UPat.var("d"), lambda x,d: -((-x)//d) if x.vmax<=0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: ((x+c.arg%d.arg)//d + c.arg//d.arg) if c.arg%d.arg!=c.arg and x.vmin>=0 and n.vmin>=0 and d.arg>0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: (-(-(c.arg%d.arg + x - (d.arg-1))//d) + c.arg//d.arg) if x.vmax<=0 and n.vmin>=0 and d.arg>0 else None),
  # ** mod **
  # mod folding
  (UPat.var("x", dtypes.index) % UPat.var("d"), lambda x,d: -((-x)%d) if x.vmax <= 0 else None),
  (UPat.var("x", dtypes.index) % UPat.var("d"), lambda x,d: (x%(-d)) if d.vmax <  0 else None),
])