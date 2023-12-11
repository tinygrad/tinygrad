def Tuple (α : Type) (n : Nat) :=
  { as : List α // as.length = n }


structure View (n : Nat) where
  shape : Tuple Nat n
--  strides : Tuple Nat n
--  mask : Tuple Nat n
  offset : Nat

def size {n : Nat} (v : View n) : Nat :=
  List.foldl (λ x y ↦ x * y) 1 v.shape.val

def idxs {n : Nat} (v : View n) : Type :=
  { n : Nat // n >= v.offset ∧ n < v.offset + size v }


def reshapeable {n m : Nat} (v : View n) (w: View m)
  (f : idxs v → idxs w) (g : idxs w → idxs v) : Prop :=
  (size v = size w) → (f ∘ g = id) ∧ (g ∘ f = id)
