import Mathlib.RingTheory.Int.Basic
import Mathlib.Data.Set.Card
import Mathlib.Tactic.Have


structure View where
  starting : ℕ
  ending : ℕ
  direction : starting ≤ ending
  step : ℕ
  step_pos : 0 < step

def View.mkOf (n : ℕ) (m : ℕ) (s : ℕ) (step_val : 0 < s) : View :=
  if hd : n ≤ m then
    View.mk n m hd s step_val
  else
    View.mk m n (Nat.le_of_not_ge hd) s step_val

@[pp_dot, simp]
def View.contains (v : View) (e : ℕ) : Prop :=
  (v.starting ≤ e ∧ e ≤ v.ending) ∧ v.step ∣ e - v.starting

def IsCorrectlyMerged (v₁ v₂ v₃ : View) : Prop :=
  ∀ x : ℕ, v₁.contains x ∨ v₂.contains x ↔ v₃.contains x

def IsMergeable (v₁ v₂ : View) : Prop := ∃ v₃ : View, IsCorrectlyMerged v₁ v₂ v₃


lemma IsCorrectlyMerged.elements_counts {v₁ v₂ v₃ : View} (hv : IsCorrectlyMerged v₁ v₂ v₃) :
    { e : ℕ | v₁.contains e}.encard + { e : ℕ | v₂.contains e}.encard ≥
    { e : ℕ | v₃.contains e}.encard := by
  sorry

lemma IsCorrectlyMerged.symmetry {v₁ v₂ v₃ : View} (hv : IsCorrectlyMerged v₁ v₂ v₃) :
    IsCorrectlyMerged v₂ v₁ v₃ := by
  sorry

lemma IsCorrectlyMerged.divisors_aux {v₁ v₂ v₃ : View} (hv : IsCorrectlyMerged v₁ v₂ v₃) :
    v₃.step ∣ v₁.step := by
  sorry

lemma IsCorrectlyMerged.divisors {v₁ v₂ v₃ : View} (hv : IsCorrectlyMerged v₁ v₂ v₃) :
    v₃.step ∣ gcd v₁.step v₂.step :=
  dvd_gcd hv.divisors_aux hv.symmetry.divisors_aux

lemma IsCorrectlyMerged.startings {v₁ v₂ v₃ : View} (hv : IsCorrectlyMerged v₁ v₂ v₃) :
    v₃.starting ≤ min v₁.starting v₂.starting := by
  by_contra contr
  push_neg at contr
  specialize hv (min v₁.starting v₂.starting)
  simp [View.contains] at hv
  by_cases hs : v₁.starting ≤ v₂.starting
  · rw [Nat.min_eq_left hs] at *
    sorry
  · push_neg at hs
    have hs'' : min v₁.starting v₂.starting = v₂.starting
    · exact Nat.min_eq_right hs.le
    sorry

-- TODO state correctly!
lemma IsCorrectlyMerged.endings {v₁ v₂ v₃ : View} (hv : IsCorrectlyMerged v₁ v₂ v₃) :
    max v₁.ending v₂.ending ≤ v₃.ending :=
  sorry
/-
Does not hold exactly like this!
Counterexample: `v₁ = ⟨0, 7, Nat.zero_le 7, 2, two_pos⟩ = v₂` but `v₃ = ⟨0, 6, Nat.zero_le 6, 2, two_pos⟩`
-/

-- Views `[0, 2, 4, 6]` and `[1, 3, 5, 7]` are mergeable (into `[0, 1, 2, 3, 4, 5, 6, 7]`).
example {v₁ v₂ : View}
    (v₁is : v₁ = ⟨0, 6, Nat.zero_le 6, 2, two_pos⟩) (v₂is : v₂ = ⟨1, 7, NeZero.one_le, 2, two_pos⟩) :
    IsMergeable v₁ v₂ := by
  use ⟨0, 7, Nat.zero_le 7, 1, one_pos⟩
  intro x
  constructor
  · intro hx
    cases hx with
    | inl hx₁ =>
      constructor
      · constructor
        · simp_all
        · have h7 : x ≠ 7
          · aesop
          have hx7 : x ≤ 7
          · simp_all
            linarith
          aesop
      · simp
    | inr hx₂ =>
      constructor
      · aesop
      · simp
  · intro hx
    if h2 : 2 ∣ x then
      left
      simp [v₁is, h2]
      have h7 : x ≤ 7
      · exact hx.left.right
      omega
    else
      right
      simp [v₂is, h2]
      have h1 : 1 ≤ x
      · omega
      have h2dvd : 2 ∣ x - 1
      · clear * - h2 h1
        omega
      aesop

-- Views `[0, 2, 4, 6]` and `[0, 3, 6]` are not mergeable.
example {v₁ v₂ : View}
    (v₁is : v₁ = ⟨0, 6, Nat.zero_le 6, 2, two_pos⟩) (v₂is : v₂ = ⟨0, 6, Nat.zero_le 6, 3, three_pos⟩) :
    ¬ IsMergeable v₁ v₂ := by
  intro ⟨v₃, merged⟩
  have hg : gcd v₁.step v₂.step = 1
  · simp only [v₁is, v₂is]
    decide
  have step_v₃ : v₃.step = 1
  · apply Nat.eq_one_of_dvd_one
    rw [←hg]
    exact IsCorrectlyMerged.divisors merged
  have hv₁ : ¬ v₁.contains 1
  · aesop
  have hv₂ : ¬ v₂.contains 1
  · aesop
  have hv₃ : v₃.contains 1
  · have hs : v₃.starting ≤ 0
    · convert merged.startings
      simp_all
    have he : 1 ≤ v₃.ending
    · have h6 : v₃.contains 2
      · rw [← merged 2]
        left
        rw [v₁is]
        simp
      have stronger : 2 ≤ v₃.ending
      · exact h6.left.right
      linarith
    simp_all
  specialize merged 1
  tauto


-- TODO
-- implement a alogrithm that merges two views
-- show that result matches specification
-- proof by construction
