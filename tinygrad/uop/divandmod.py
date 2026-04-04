import functools, itertools, math
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, UOp
from tinygrad.dtype import dtypes
from tinygrad.helpers import cdiv, cmod, CORRECT_DIVMOD_FOLDING, unwrap

def _uadd_terms(base: UOp, const: int, terms: list[UOp]) -> UOp:
  if not terms: return base.const_like(const)
  if const == 0:
    head, *tail = terms
    return head if not tail else head.usum(*tail)
  return base.const_like(const).usum(*terms)

# NOTE: this cache is only on index UOps
@functools.cache
def fold_divmod_general(d: UOp, correct_divmod_folding: bool) -> UOp|None:
  x, y = d.src
  # cancel_divmod: simple cancel div/mod case when the range of the numerator lies within a single denominator interval
  x_min, x_max, y_min, y_max = x.vmin, x.vmax, y.vmin, y.vmax
  assert isinstance(x_min, int) and isinstance(x_max, int) and isinstance(y_min, int) and isinstance(y_max, int)
  if y_min==y_max==0: raise ZeroDivisionError(f"{'Division' if d.op is Ops.IDIV else 'Mod'} by zero trying to rewrite {x.alu(d.op, y)}")
  if y_min*y_max > 0 and (qv:=cdiv(x_min,y_min)) == cdiv(x_min,y_max) == cdiv(x_max,y_min) == cdiv(x_max,y_max):
    return x - qv*y if d.op is Ops.MOD else d.const_like(qv)

  # split uops for the rest of the processing
  x_peeled, const = x.pop_const()
  uops_no_const = list(x_peeled.split_uop(Ops.ADD))

  # ** Constant Denominator Rules **
  # these rules strictly require y to be a scalar constant > 0
  if y.op is Ops.CONST and (c := y.arg) > 0:
    # nested_div_mod: (x%(k*c))//c -> (x//c)%k, and (x%(k*c))%c -> x%c
    if x.op is Ops.MOD and (k := x.src[1].divides(c)) is not None:
      return x.src[0] // y % k if d.op is Ops.IDIV else x.src[0] % y

    # remove_nested_mod in sum: (a%4 + b)%2 -> (a+b)%2, requires non-negative sums
    if d.op is Ops.MOD and x.vmin >= 0:
      new_xs, changed = [], False
      for u in uops_no_const:
        if u.op is Ops.MOD and u.src[1].divides(c) is not None:
          u = u.src[0]
          changed = True
        new_xs.append(u)
      if changed and (new_x:=(UOp.usum(*new_xs) + const)).vmin >= 0: return new_x % y

    # Shared decomposition for folding rules
    decomp = [(u.divides(f:=u.const_factor()),f) for u in uops_no_const]
    terms, factors = zip(*decomp)

    # fold_binary_numerator: fold if expression has one non-constant term that takes on two values
    if len(terms)==1 and (v:=terms[0]).vmax-v.vmin == 1:
      y1 = (cmod if d.op is Ops.MOD else cdiv)(factors[0]*v.vmin+const, c)
      y2 = (cmod if d.op is Ops.MOD else cdiv)(factors[0]*v.vmax+const, c)
      return (y2-y1)*(v-v.vmin) + y1

    # fold_divmod_congruence: fold if a is congruent to an expression whose range is between 0 and c
    if not (x.vmin<0 and correct_divmod_folding):
      is_mod = d.op is Ops.MOD
      fixed_rem_terms:list[UOp] = []
      fixed_rem_min, fixed_rem_max = const%c, const%c
      fixed_quo_terms:list[UOp] = []
      tie_variants_mod:list[tuple[tuple[UOp, int, int], tuple[UOp, int, int]]] = []
      tie_variants_div:list[tuple[tuple[UOp, int, int, UOp], tuple[UOp, int, int, UOp]]] = []
      for f, v in zip(factors, terms):
        vmin, vmax = v.vmin, v.vmax
        assert isinstance(vmin, int) and isinstance(vmax, int)
        r = f % c
        if r*2 == c:
          alt_r = r-c
          r_bounds = (r*vmin, r*vmax) if r >= 0 else (r*vmax, r*vmin)
          alt_bounds = (alt_r*vmin, alt_r*vmax) if alt_r >= 0 else (alt_r*vmax, alt_r*vmin)
          if is_mod:
            tie_variants_mod.append(((r*v, *r_bounds), (alt_r*v, *alt_bounds)))
          else:
            pos_variant = (r*v, *r_bounds, ((f-r)//c) * v)
            neg_variant = (alt_r*v, *alt_bounds, ((f-alt_r)//c) * v)
            tie_variants_div.append((pos_variant, neg_variant))
          continue
        r = min(r, r-c, key=abs)
        if r != 0:
          fixed_rem_terms.append(r*v)
          if r >= 0:
            fixed_rem_min += r*vmin
            fixed_rem_max += r*vmax
          else:
            fixed_rem_min += r*vmax
            fixed_rem_max += r*vmin
        if not is_mod and (q:=(f-r)//c) != 0: fixed_quo_terms.append(q*v)
      fixed_rem = _uadd_terms(x, const%c, fixed_rem_terms)
      if not is_mod: fixed_quo = _uadd_terms(x, const//c, fixed_quo_terms)

      if is_mod and not tie_variants_mod:
        if fixed_rem_min//c==fixed_rem_max//c: return fixed_rem - (fixed_rem_min//c)*c
      elif not is_mod and not tie_variants_div:
        if fixed_rem_min//c==fixed_rem_max//c: return fixed_quo + fixed_rem_min//c
      elif is_mod:
        for tie_choices in itertools.product((0, 1), repeat=len(tie_variants_mod)):
          rem = fixed_rem
          rem_min, rem_max = fixed_rem_min, fixed_rem_max
          for i,choice in enumerate(tie_choices):
            rem_term, tmin, tmax = tie_variants_mod[i][choice]
            rem += rem_term
            rem_min += tmin
            rem_max += tmax
          if rem_min//c==rem_max//c: return rem - (rem_min//c)*c
      else:
        for tie_choices in itertools.product((0, 1), repeat=len(tie_variants_div)):
          rem = fixed_rem
          rem_min, rem_max = fixed_rem_min, fixed_rem_max
          quo = fixed_quo
          for i,choice in enumerate(tie_choices):
            rem_term, tmin, tmax, quo_term = tie_variants_div[i][choice]
            rem += rem_term
            rem_min += tmin
            rem_max += tmax
            quo += quo_term
          if rem_min//c==rem_max//c: return quo + rem_min//c

    # gcd_with_remainder: factor out common gcd from numerator
    if x.vmin >= 0 and (g:=math.gcd(*factors, c)) > 1:
      new_x = unwrap(x_peeled.divides(g)).simplify() + (const//g)%(c//g)
      if new_x.vmin >= 0:
        if d.op is Ops.MOD: return new_x % (c//g) * g + const%g
        return new_x // (c//g) + const//c

    # nest_by_factor: x//c -> (x//f)//(c//f), x%c -> (x//f%(c//f))*f + b where b=x%f
    if x.vmin >= 0:
      results = []
      for div in {abs(f) for u, f in zip(uops_no_const, factors) if u.op not in (Ops.CONST, Ops.VCONST) and 1 < abs(f) < c and (c%f)==0}:
        if (newxs := fold_divmod_general(x//div, correct_divmod_folding)) is not None and newxs.vmin >= 0:
          if d.op is Ops.IDIV:
            results.append((len(newxs.backward_slice), newxs // (c // div)))
          else:
            b_parts = [f%div*t for f, t in zip(factors, terms) if f%div]
            b = _uadd_terms(x, const % div, b_parts)
            if 0 <= b.vmin and b.vmax < div:
              results.append((len((r:=(newxs % x.ufix(c//div))*div + b).backward_slice), r))
      if results: return min(results, key=lambda r: r[0])[1]

  # ** Variable Denominator / Fallback Rules **
  # These rules apply to variables OR constants that failed the checks above.
  # Reconstruct all uops including const for these checks.
  all_uops = list(x.split_uop(Ops.ADD))

  # divide_by_gcd: x//y -> (x//gcd)//(y//gcd)
  # Constant denominators cannot contribute symbolic factors, so a const-factor gcd
  # of 1 means UOp.gcd would also be 1.
  if y.op is Ops.CONST and math.gcd(*(u.const_factor() for u in all_uops), y.arg) == 1:
    gcd = y.const_like(1)
  else:
    gcd = UOp.gcd(*all_uops, y).simplify()
  if not (gcd.op is Ops.CONST and gcd.arg==1):
    ret = unwrap(x.divide_exact(gcd)).alu(d.op, unwrap(y.divide_exact(gcd)))
    return ret*gcd if d.op is Ops.MOD else ret

  # factor_remainder: (d*x+y)//d -> x+y//d
  if y.vmin<0 or x.vmin<0: return None
  quo, rem = [], []
  for u in all_uops:
    if (q:=u.divide_exact(y)) is not None: quo.append(q)
    elif y.op is Ops.CONST and (c:=u.const_factor())%y.arg!=c:
      rem.append(u.divides(c)*(c%y.arg))
      quo.append(u.divides(c)*(c//y.arg) if d.op is Ops.IDIV else u.const_like(0))
    else: rem.append(u)

  if not quo: return None
  new_x = _uadd_terms(x, 0, rem)
  if new_x.vmin<0: return None
  return new_x%y if d.op is Ops.MOD else new_x//y + _uadd_terms(x, 0, quo)

div_and_mod_symbolic = PatternMatcher([
  # ** 1. Fast Inline Rules **
  ((UPat.var("x")//UPat.cvar("c") + UPat.cvar("a"))//UPat.cvar("d"), lambda x,c,a,d: (x+a*c)//(c*d)
    if c.vmin>0 and d.vmin>0 and x.vmin>=0 and a.vmin>=0 else None),  # (x//c+a)//d -> (x+a*c)//(c*d)
  (UPat.var("x", dtypes.weakint) // UPat.var("d"), lambda x,d: -(x//(-d)) if d.vmax < 0 else None),
  (UPat.var("x", dtypes.weakint) // UPat.var("d"), lambda x,d: -((-x)//d) if x.vmax <= 0 else None),
  ((UPat.var("x", dtypes.weakint)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: ((x+c.arg%d.arg)//d + c.arg//d.arg) if c.arg%d.arg!=c.arg and x.vmin>=0 and n.vmin>=0 and d.arg>0 else None),
  ((UPat.var("x", dtypes.weakint)+UPat.cvar("c", vec=False)).named("n")//UPat.cvar("d", vec=False),
    lambda x,c,n,d: (-(-(c.arg%d.arg + x - (d.arg-1))//d) + c.arg//d.arg) if x.vmax<=0 and n.vmin>=0 and d.arg>0 else None),

  # ** 2. Slow Rules **
  (UPat((Ops.IDIV, Ops.MOD), dtypes.weakint, name="d"), lambda d: fold_divmod_general(d, bool(CORRECT_DIVMOD_FOLDING))),

  # NOTE: these have to go at the bottom or TestSymbolicOps.test_var loops
  (UPat.var("x", dtypes.weakint) % UPat.var("d"), lambda x,d: -((-x)%d) if x.vmax <= 0 else None),
  (UPat.var("x", dtypes.weakint) % UPat.var("d"), lambda x,d: (x%(-d)) if d.vmax < 0 else None),
])
