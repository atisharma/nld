"""
Minimal implementation of Runge-Kutta integration with fixed timestep.
"""

; TODO: consider integrating with an adaptive timestep,
;       returning a trajectory aliased to fixed timestep

(require [hy.contrib.walk [let]])

(import [itertools [accumulate]])
(import [jax.numpy :as jnp])
(import [jax [jit]])


(defn rk4 [f v * h [t 0] #** kwargs]
  """
  4th-order Runge-Kutta integration.

  v:        state vector
  f:        dv/dt
  h:        timestep
  t:        time t
  returns:  v(t+h)

  k_1	        = h f(v(n),          t(n))
  k_2	        = h f(v(n) + k_1/2,  t(n) + h/2)
  k_3	        = h f(v(n) + k_2/2,  t(n) + h/2)
  k_4	        = h f(v(n) + k_3,    t(n) + h)
  v(n+1) - v(n) = + k_1/6 + k_2/3 + k_3/3 + k_4/6 + O(h^5)	
  """
  (let [k1 (f v                   :t t #** kwargs)
        k2 (f (+ v (* h 0.5 k1))  :t (+ t (* 0.5 h)) #** kwargs)
        k3 (f (+ v (* h 0.5 k2))  :t (+ t (* 0.5 h)) #** kwargs)
        k4 (f (+ v (* h k3))      :t (+ t h) #** kwargs)]
    (+ v
       (* (/ h 6)
          (+ k1 (* k2 2) (* k3 2) k4)))))


(defn integrate [f v0 * [h 1e-4] [steps None] [T None] [as-iterator False] #** kwargs]
  """
  Integrate the 1D PDE.
    T is overriden by steps.
    Returns a dict.
    The trajectory is a device array, or (if specified) an unrealised iterator,
    of shape steps x states.
  """
  ; N.B. though rk4 cannot be jit compiled because it accepts a function
  ; as an argument, its partial application to f can be.
  (let [rk4-step (jit (fn [vn t] (rk4 f vn :h h :t t #** kwargs)))
        steps-to-run (or steps (round (/ T h)))
        result (accumulate (range steps-to-run) :func rk4-step :initial v0)
        t (* (jnp.arange 0 (+ 1 steps-to-run)) h)]
    {"trajectory" (if as-iterator result (-> result list jnp.array)) 
     "t" t
     "h" h
     "steps" steps-to-run
     "T" (get t -1)
     "system" f.__name__}))


(defn truncate [data [steps -1] [T None]]
  """
  Truncate a trajectory to a final time, or number of steps.
  """
  (let [h (:h data)
        steps-to-use (or steps (+ 1 (round (/ T h))))
        t (get (:t data) (slice 0 steps-to-use))]
    {#** data
     "trajectory" (get (:trajectory data) (slice 0 steps-to-use) (slice None))
     "t" t
     "steps" steps-to-use
     "T" (get t -1)}))
