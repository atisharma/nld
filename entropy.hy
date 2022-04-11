; entropy production is related to attractor dimension
; Kaplan-Yorke dimension - Renyi entropy - Lyapunov dimension

; http://www.scholarpedia.org/article/Attractor_dimensions
; https://digital.library.unt.edu/ark:/67531/metadc4559/m2/1/high_res_d/thesis.pdf
; https://en.wikipedia.org/wiki/Entropy_coding
; https://en.wikipedia.org/wiki/Kaplan%E2%80%93Yorke_conjecture
; https://arxiv.org/abs/2108.05928

(require [hy.contrib.walk [let]])

(import [jax.numpy :as jnp])
(import [jax [jit]])

; also see lz4, zlib, blosc
(import [zstandard :as zstd])

(import [numerics [truncate]])


; express dynamics as a string of bytes

; calculate Lyapunov dimension
; calculate information dimension, D1
; calculate correltaion dimension, D2
; calculate Renyi dimension, Dq

; show information dimension / rate related to dimension of inertial manifold (Floryan & Graham)


(defn quantize [data [box 1e-3] [steps None] [T None] [order "C"]]
  """
  Quantize a trajectory of states into boxes.
  Implemented by dividing by box size as casting as 32-bit int.
  Default to row-major (C) order so that the loop over state iterates
  faster than the time. This is default anyway.
  """
  (let [trunc-data (truncate data :steps steps :T T)
        trajectory (:trajectory data)
        traj (if (-> trajectory jnp.iscomplex (.any))
               (jnp.concatenate [(jnp.real trajectory) (jnp.imag trajectory)] 1)
               trajectory)]
    (-> traj
        (/ box)
        (.astype jnp.int32)
        (jnp.ravel order))))


(defn compress [data [box 1e-3] [steps -1] [T None] [order "C"]]
  """
  Quantize a trajectory and then compress the resulting string of bytes.
  """
  (-> data
      (quantize :box box :steps steps :T T :order order)
      (.tobytes)
      zstd.compress))


(defn entropy-vs-box [data [steps -1] [T None]] 
  """
  Calculate length of compressed trajectory as it varies with box size.
  """
  (let [box (jnp.geomspace 1e-6 10.0 200)]
    {"entropy" (lfor dx box (len (compress data :box dx :steps steps :T T)))
     "box" box}))



; entropy vs box size (relate to Lyapunov dim)
; entropy vs T (entropy rate)
; entropy vs kappa (KS)
