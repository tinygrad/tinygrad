def Tuple (α : Type) (n : Nat) :=
  { as : List α // as.length = n }

structure View (n : Nat) where
  offset : Nat
  shape : Tuple Nat n
  min : Tuple Nat n
  max : Tuple Nat n
  stride : Tuple Nat n


def size {n : Nat} (v : View n) : Nat :=
  List.foldl (λ x y ↦ x * y) 1 v.shape.val


def proj {n : Nat} (v : View n) (i : Fin n)  (j : Nat) : Nat :=
  (j - v.offset) % (List.get v.stride.val (Fin.mk i sorry))


def valid {n : Nat} (v : View n) (i : Fin n) (j : Nat) : Prop :=
  (proj v i j) >= List.get v.min.val (Fin.mk i sorry)
  ∧ (proj v i j) <= List.get v.max.val (Fin.mk i sorry)


def idxs {n : Nat} (v : View n) : Type :=
  { j : Nat
    // j >= v.offset
    ∧  j < v.offset + size v
    ∧  forall i : Fin n, valid v i j}

def reshapeable {n m : Nat} (v : View n) (w: View m)
  (f : idxs v → idxs w) (g : idxs w → idxs v) : Prop :=
  (f ∘ g = id) ∧ (g ∘ f = id)
