"""
2D channel/couette flow.
"""

; Solve with time integration
; Solve in frequency domain

; Energy balance between all pathways in frequency domain
; and drag = dissipation.
; Does system look quasi-Hamiltonian in frequency domain?
; Energy at frequency must be at a stationary point -
; otherwise 'virtual force' occurs (like d'Alembert's principle).
;
; integrating using forces in time domain is essentially following energy
; gradients with time as the iteration variable

; Pipe flow?


(defn dfdy [f y]
  None)

(defn d2fdy2 [f y]
  None)

(defn d4fdy4 [f y]
  None)
