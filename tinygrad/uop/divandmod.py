import math, functools
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import cdiv, cmod, CORRECT_DIVMOD_FOLDING, unwrap

# NOTE: this cache is only on index UOps and matches the cache in the old ShapeTracker in spirit
@functools.cache
def fold_divmod_general(d: UOp, correct_divmod_folding: bool) -> UOp|None:
  x, y = d.src

  # cancel_divmod: simple cancel div/mod case when the range of the numerator lies within a single denominator interval
  x_min, x_max, y_min, y_max = x.vmin, x.vmax, y.vmin, y.vmax
  assert isinstance(x_min, int) and isinstance(x_max, int) and isinstance(y_min, int) and isinstance(y_max, int)
  if y_min==y_max==0: raise ZeroDivisionError(f"{'Division' if d.op is Ops.IDIV else 'Mod'} by zero trying to rewrite {x.alu(d.op, y)}")
  # 0//y = 0 and 0%y = 0 for any y != 0
  if x_min==x_max==0: return d.const_like(0) if y_min > 0 else None
  # normalize negative numerator/denominator to positive (need x_min<0 check to avoid -0=0 loop when x is exactly 0)
  if y_max < 0: return (x%(-y)) if d.op is Ops.MOD else -(x//(-y))
  if x_max <= 0 and x_min < 0: return -((-x)%y) if d.op is Ops.MOD else -((-x)//y)
  if y_min*y_max > 0 and (q:=cdiv(x_min,y_min)) == cdiv(x_min,y_max) == cdiv(x_max,y_min) == cdiv(x_max,y_max):
    return x - q*y if d.op is Ops.MOD else d.const_like(q)

  # split uops for the rest of the processing
  x_peeled, const = x.pop_const()
  uops_no_const = tuple(x_peeled.split_uop(Ops.ADD))

  # ** Constant Denominator Rules **
  # these rules strictly require y to be a scalar constant > 0
  if y.op is Ops.CONST and (c := y.arg) > 0:
    # remove_nested_mod: remove nested mod in case the inner mod is a multiple of the outer mod, example: (a%4 + b)%2 -> (a+b)%2
    if d.op is Ops.MOD and x.vmin >= 0:
      new_xs, changed = [], False
      for u in uops_no_const:
        if u.op is Ops.MOD and u.src[1].divides(c) is not None:
          u = u.src[0]
          changed = True
        new_xs.append(u)
      if changed and (new_x:=(UOp.sum(*new_xs) + const)).vmin >= 0: return new_x % y

    # fold_binary_numerator: fold if expression has one non-constant term that takes on two values
    if len(uops_no_const)==1:
      f = uops_no_const[0].const_factor()
      if (v:=uops_no_const[0].divides(f)).vmax-v.vmin == 1:
        y1 = cmod(f*v.vmin+const, c) if d.op is Ops.MOD else cdiv(f*v.vmin+const, c)
        y2 = cmod(f*v.vmax+const, c) if d.op is Ops.MOD else cdiv(f*v.vmax+const, c)
        return (y2-y1)*(v-v.vmin) + y1

    # compute factors once for all remaining rules
    factors = tuple(u.const_factor() for u in uops_no_const)

    # fold_divmod_congruence: fold if a is congruent to an expression whose range is between 0 and c
    if not (x.vmin<0 and correct_divmod_folding):
      rems, var_min, var_max = [], 0, 0
      for u, f in zip(uops_no_const, factors):
        r = min((r0:=f%c), r0-c, key=abs)
        rems.append(r)
        if r == 0: continue
        v_min, v_max = (u.vmin//f, u.vmax//f) if f > 0 else (u.vmax//f, u.vmin//f)
        var_min, var_max = var_min + r*(v_min if r >= 0 else v_max), var_max + r*(v_max if r >= 0 else v_min)
        if var_max - var_min >= c: break
      const_rem = const % c
      if (rem_min_floor := (var_min + const_rem)//c) == (var_max + const_rem)//c and var_max - var_min < c:
        if d.op is Ops.MOD:
          terms = [(u.divides(f)*r if r not in (0,1) else u.divides(f)) for u,f,r in zip(uops_no_const,factors,rems) if r != 0]
          rem_expr = (UOp.sum(*terms) + const_rem) if (terms and const_rem) else (UOp.sum(*terms) if terms else x.const_like(const_rem))
          return rem_expr - rem_min_floor*c if rem_min_floor else rem_expr
        terms = [(u.divides(f)*q if q not in (0,1) else u.divides(f)) for u,f,r in zip(uops_no_const,factors,rems) if (q:=(f-r)//c) != 0]
        quot_sum, const_quot = (UOp.sum(*terms) if terms else None), const//c + rem_min_floor
        if quot_sum is None: return x.const_like(const_quot)
        return (quot_sum + const_quot) if const_quot else quot_sum

    # gcd_with_remainder: factor out common gcd from numerator
    if x.vmin >= 0 and factors:
      if (gcd_val := functools.reduce(math.gcd, factors, c)) > 1:
        gcd = x.const_like(gcd_val)
        new_x = unwrap(x_peeled.divide_exact(gcd)).simplify() + (const%c)//gcd.arg
        if new_x.vmin >= 0:
          ret = new_x.alu(d.op, x.ufix(c//gcd.arg))
          return ret*gcd + const%gcd.arg if d.op is Ops.MOD else ret+const//c

    # nest_div_by_smallest_factor: try and nest the div and see if it allows the numerator to be simplified
    if d.op is Ops.IDIV and x.vmin >= 0:
      div = min([c] + [abs(f) for u, f in zip(uops_no_const, factors) if u.op not in (Ops.CONST, Ops.VCONST) and abs(f) > 1 and (c%f)==0])
      # NOTE: this is recursive!
      if div < c and (newxs := fold_divmod_general(x//div, correct_divmod_folding)) is not None and newxs.vmin >= 0:
        return newxs // (c // div)

  # ** Variable Denominator / Fallback Rules **
  # These rules apply to variables OR constants that failed the checks above.

  # compute factors if not already done
  if y.op is not Ops.CONST or (c := y.arg) <= 0:
    factors = tuple(u.const_factor() for u in uops_no_const)

  # divide_by_gcd: x//y -> (x//gcd)//(y//gcd)
  if factors and y.op is Ops.CONST:
    gcd_val = functools.reduce(math.gcd, (int(f) for f in factors), y.arg)
    if const != 0: gcd_val = math.gcd(gcd_val, int(const))
    gcd = None if gcd_val == 1 else x.const_like(gcd_val)
  else:
    # Reconstruct all uops including const for these checks.
    all_uops = uops_no_const + ((x.const_like(const),) if const != 0 else ())
    gcd = UOp.gcd(*all_uops, y).simplify()
  if gcd is not None and not (gcd.op is Ops.CONST and gcd.arg==1):
    ret = unwrap(x.divide_exact(gcd)).alu(d.op, unwrap(y.divide_exact(gcd)))
    return ret*gcd if d.op is Ops.MOD else ret

  # factor_remainder: (d*x+y)//d -> x+y//d
  if y_min<0 or x_min<0: return None
  all_uops = uops_no_const if const == 0 else (*uops_no_const, x.const_like(const))
  quo, rem = [], []
  for u in all_uops:
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    elif d.op is Ops.MOD and y.op is Ops.CONST and (c:=u.const_factor())%y.arg!=c:
      rem.append(u.divides(c)*(c%y.arg))
      quo.append(u.const_like(0))
    else: rem.append(u)

  if not quo: return None
  new_x = UOp.sum(*rem) if rem else x.const_like(0)
  if new_x.vmin<0: return None
  return new_x%y if d.op is Ops.MOD else new_x//y+UOp.sum(*quo)

div_and_mod_symbolic = PatternMatcher([
  # (x//c+a)//d -> (x+a*c)//(c*d)
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d)
    if c.vmin>0 and d.vmin>0 and ((x.vmin>=0 and a.vmin>=0) or (x.vmax<=0 and a.vmax<=0)) else None),
  # (x+c)//d -> (x+c%d)//d + c//d when c >= d (extract constant quotient)
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: ((x+c.arg%d.arg)//d + c.arg//d.arg) if c.arg%d.arg!=c.arg and x.vmin>=0 and n.vmin>=0 and d.arg>0 else None),
  ((UPat.var("x", dtypes.index)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: (-(-(c.arg%d.arg + x - (d.arg-1))//d) + c.arg//d.arg) if x.vmax<=0 and n.vmin>=0 and d.arg>0 else None),
  # fold_divmod_general handles everything else
  (UPat((Ops.IDIV, Ops.MOD), dtypes.index, name="d"), lambda d: fold_divmod_general(d, bool(CORRECT_DIVMOD_FOLDING))),
])