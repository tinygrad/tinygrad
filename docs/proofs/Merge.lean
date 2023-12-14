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
  let p : i.val < List.length v.stride.val := by rw [v.stride.property]; exact i.isLt
  (j - v.offset) % (List.get v.stride.val (Fin.mk i p))


def valid {n : Nat} (v : View n) (i : Fin n) (j : Nat) : Prop :=
  let p : i.val < List.length v.min.val := by rw [v.min.property]; exact i.isLt
  let q : i.val < List.length v.max.val := by rw [v.max.property]; exact i.isLt
  (proj v i j) >= List.get v.min.val (Fin.mk i p)
  ∧ (proj v i j) <= List.get v.max.val (Fin.mk i q)


def idxs {n : Nat} (v : View n) : Type :=
  { j : Nat
    // j >= v.offset
    ∧  j < v.offset + size v
    ∧  forall i : Fin n, valid v i j}

def reshapeable {n m : Nat} (v : View n) (w: View m)
  (f : idxs v → idxs w) (g : idxs w → idxs v) : Prop :=
  (f ∘ g = id) ∧ (g ∘ f = id)

def tt : Tuple Nat 2 :=
  Subtype.mk [2,2] rfl

def zz : Tuple Nat 2 :=
  Subtype.mk [0,0] rfl

def v1 : View 2 :=
  View.mk 0 tt zz tt (Subtype.mk [2,1] rfl)

def wrap (x : Nat) : Tuple Nat 1 := Subtype.mk [x] rfl

def v2 : View 1 :=
  View.mk 0 (wrap 4) (wrap 0) (wrap 3) (wrap 1)

theorem cx (f : idxs v1 → idxs v2) (g : idxs v2 → idxs v1) :
  ¬ reshapeable v1 v2 f g :=
  sorry
