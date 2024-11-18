; entropy production is related to attractor dimension
; Kaplan-Yorke dimension - Renyi entropy - Lyapunov dimension

; http://www.scholarpedia.org/article/Attractor_dimensions
; https://digital.library.unt.edu/ark:/67531/metadc4559/m2/1/high_res_d/thesis.pdf
; https://en.wikipedia.org/wiki/Entropy_coding
; https://en.wikipedia.org/wiki/Kaplan%E2%80%93Yorke_conjecture
; https://arxiv.org/abs/2108.05928

(require hyrule.argmove [-> ->> as->])
(require hyrule.control [unless])

(import io)
(import random [randbytes])

(import jax.numpy :as jnp)
(import jax [jit])

; also see lz4, zlib, blosc
(import zstandard :as zstd)

(import .numerics [truncate])


; express dynamics as a string of bytes

; calculate Lyapunov dimension
; calculate information dimension, D1
; calculate correlation dimension, D2
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
        trajectory (:trajectory trunc-data)
        traj (if (-> trajectory jnp.iscomplex (.any))
               (jnp.concatenate [(jnp.real trajectory) (jnp.imag trajectory)] 1)
               trajectory)]
    (-> traj
        (/ box)
        (.astype jnp.int32)
        (jnp.ravel order))))

(defn compress [data [box 1e-3] [steps None] [T None] [order "C"] [compressor zstd]]
  """
  Quantize a trajectory and then compress the resulting string of bytes.
  You can specify a pre-trained compressor derived from a dictionary, like
    :compressor (.ZstdCompressor zstd :dict-data dict-data)
  """
  (-> data
      (quantize :box box :steps steps :T T :order order)
      (.tobytes)
      compressor.compress))

(defn dictionary [data [box 1e-3] [steps None] [T None] [order "C"] [level 22]]
  """
  Train a compression dictionary on some data. Return the dictionary.
  """
  (let [d (-> data
             (quantize :box box :steps steps :T T :order order)
             (.tobytes)
             (zstd.ZstdCompressionDict))])
  (.precompute-compress d :level level)
  d)

(defn synthesise [dictionary [length 1000]]
  """
  Synthetic data based on a pre-computed dictionary.
  """
  ; FIXME: this does not work yet as compressed data has frame descriptors.
  ; These would need to be replicated.
  (let [buffer (.BytesIO io)
        random-bytes (randbytes length)
        decompressor (.ZstdDecompressor zstd :dict-data dictionary)]
    (with [stream (.stream-writer decompressor buffer)]
      (.write stream random-bytes))))

(defn vs-box [data [steps None] [T None]] 
  """
  Calculate length of compressed trajectory as it varies with box size.
  """
  (let [box (jnp.geomspace 1e-6 10.0 200)]
    {"entropy" (lfor dx box (len (compress data :box dx :steps steps :T T)))
     "box" box}))

(defn vs-time [data [box 1e-3] [T0 1e-10] [T 100]] 
  """
  Calculate length of compressed trajectory as it varies with trajectory duration T.
  """
  (let [time (jnp.linspace T0 T 200)]
    {"entropy" (lfor t time (len (compress data :box box :T t)))
     "T" time}))

; entropy vs box size (relate to Lyapunov dim)
; entropy vs T (entropy rate)
; entropy vs kappa (KS)
; synthetic data using dictionary
