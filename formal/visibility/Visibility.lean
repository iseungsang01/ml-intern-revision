import Std

/- Discrete index contract only. `eligible` is exported by a separate bridge;
   neither instrument arrival nor Python/CUDA execution is assumed verified. -/
namespace Visibility

/-- Greatest eligible index strictly below `n`. -/
def latest (eligible : Nat → Bool) : Nat → Option Nat
  | 0 => none
  | n + 1 => if eligible n then some n else latest eligible n

theorem latest_sound (e : Nat → Bool) (n i : Nat)
    (h : latest e n = some i) : i < n ∧ e i = true := by
  induction n with
  | zero => simp [latest] at h
  | succ n ih =>
    simp only [latest] at h
    split at h
    next he =>
      have hi : n = i := Option.some.inj h
      subst i
      exact ⟨Nat.lt_succ_self n, he⟩
    next _ =>
      obtain ⟨hb, he⟩ := ih h
      exact ⟨Nat.lt_trans hb (Nat.lt_succ_self n), he⟩

theorem latest_none_iff (e : Nat → Bool) (n : Nat) :
    latest e n = none ↔ ∀ i, i < n → e i = false := by
  induction n with
  | zero => simp [latest]
  | succ n ih =>
    cases he : e n with
    | false =>
      simp only [latest, he, Bool.false_eq_true, ↓reduceIte, ih]
      constructor
      · intro h i hi
        by_cases hn : i = n
        · simpa [hn] using he
        · exact h i (by omega)
      · intro h i hi
        exact h i (by omega)
    | true =>
      constructor
      · intro h
        simp [latest, he] at h
      · intro h
        have bad := h n (Nat.lt_succ_self n)
        simp [he] at bad

theorem latest_greatest (e : Nat → Bool) (n i : Nat)
    (h : latest e n = some i) : ∀ j, j < n → e j = true → j ≤ i := by
  induction n with
  | zero => simp [latest] at h
  | succ n ih =>
    simp only [latest] at h
    split at h
    next _ =>
      have hi : n = i := Option.some.inj h
      subst i
      intro j hj _
      omega
    next he =>
      intro j hj hej
      have hne : j ≠ n := by
        intro hjn
        subst j
        exact he hej
      exact ih h j (by omega) hej

theorem latest_prefix (e f : Nat → Bool) (n : Nat)
    (h : ∀ i, i < n → e i = f i) : latest e n = latest f n := by
  induction n with
  | zero => rfl
  | succ n ih =>
    simp only [latest, h n (Nat.lt_succ_self n)]
    split
    · rfl
    · exact ih (fun i hi => h i (by omega))

/-- Eligibility agrees if metadata agree and observation validity agrees at
    every metadata-visible row. Hidden validity flags need not agree. -/
theorem eligibility_noninterference (visible observed changed : Nat → Bool)
    (n : Nat) (h : ∀ i, i < n → visible i = true → observed i = changed i) :
    latest (fun i => visible i && observed i) n =
    latest (fun i => visible i && changed i) n := by
  apply latest_prefix
  intro i hi
  cases hv : visible i with
  | false => simp
  | true => simp [h i hi hv]

def readout (e : Nat → Bool) (values : Nat → α) (n : Nat) : Option α :=
  (latest e n).map values

/-- Equal eligibility and equal values on eligible prefix rows suffice;
    values at current/future/ineligible rows are unrestricted. This fixes each
    query's metadata; it says nothing about changing current timestamps. -/
theorem readout_noninterference (e f : Nat → Bool) (x y : Nat → α) (n : Nat)
    (he : ∀ i, i < n → e i = f i)
    (hx : ∀ i, i < n → e i = true → x i = y i) :
    readout e x n = readout f y n := by
  unfold readout
  rw [← latest_prefix e f n he]
  cases hs : latest e n with
  | none => rfl
  | some i =>
    obtain ⟨hi, hei⟩ := latest_sound e n i hs
    simp [hx i hi hei]

#print axioms latest_sound
#print axioms latest_none_iff
#print axioms latest_greatest
#print axioms latest_prefix
#print axioms eligibility_noninterference
#print axioms readout_noninterference

end Visibility
