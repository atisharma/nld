"""
A library of dynamical systems including 1D PDEs.

See 'Elegant Chaos' by J C Sprott for inspiration.
"""

;TODO: get GPGPU working

(require [hy.contrib.walk [let]])

(import jax)
;(jax.config.update "jax_enable_x64" True)

(import [jax.numpy :as jnp])
(import [jax.numpy.fft [rfft irfft rfftfreq]])
(import [jax.numpy.fft [fft ifft fftfreq]])
(import [jax.numpy [pi]])
(import [jax [random jit]])

(import [numerics [integrate]])


(setv rkey (random.PRNGKey 10))

(defn initial-condition [N]
  """
  Random initial condition of dimension N.
  """
  (* (random.normal rkey :shape (, N)) 1e-1))


;;;;;;;;;;;
;; PDEs ;;;
;;;;;;;;;;;

; hardcoded PDE discretisation parameters
(setv N 32)
(setv x (/ (* 2 pi (jnp.arange N)) N))


; for rfft
(setv rDxh (* 1j N (rfftfreq N)))
; for fft
(setv Dxh (* 1j N (fftfreq N)))

(setv v0 (* (random.normal rkey :shape (, N)) 1e-1))


(with-decorator jit
  (defn ks [v * [t 0] [kappa (/ (* 4 pi pi) 1521)]]
    """
    Fourier representation of Kuramoto-Sivashinsky.
      dv(x,t)/dt = d/dx (vv) - d^2/dx^2 v - d^4/dx^4 v
      x in [0, 2pi]
   kappa:
      (2pi/39)^2  - Davide Lasagna case
      (pi / 11)^2 - Mike Graham chaotic case
      16/337      - Floryan & Graham standing wave
      4/87        - Floryan & Graham beating travelling wave (2-torus)
      16/71       - Floryan & Graham bursting, two saddle points, four heteroclinic orbits
    """
    (let [v-hat       (rfft v)
          v-hat-xx    (* rDxh rDxh v-hat)
          v-hat-xxxx  (* rDxh rDxh v-hat-xx)
          vv-hat-x    (* rDxh (rfft (* v v)))]
      (irfft (- vv-hat-x
                v-hat-xx
                (* kappa v-hat-xxxx))))))
         

(with-decorator jit
  (defn ks-cubic [u * [t 0] [L 100]]
    """
    Kuramoto-Sivashinsky with cubic nonlinearity.
      See Chapter 8 of 'Elegant Chaos'.
    du(x,t)/dt = -uu u_x - u_xx - u_xxxx
    x in [0, 2pi]
    """
    (let [u-hat       (rfft u)
          u-hat-x     (/ (* rDxh u-hat) L)
          u-hat-xx    (/ (* rDxh u-hat-x) L)
          u-hat-xxxx  (/ (* rDxh rDxh u-hat-xx) L L)
          u-x         (irfft u-hat-x)]
      (- (+ (* u u u-x)
            (irfft (+ u-hat-xx
                      u-hat-xxxx)))))))
         

(with-decorator jit
  (defn pd13 [u * [t 0] [L 100]]
    """
    PD13 from table 8.1 in 'Elegant Chaos'.
    du(x,t)/dt = 1 + (1 - u^3 u_xxx) u_xxx
    x in [0, 2pi]
    """
    (let [u-hat       (rfft u)
          u-hat-xxx   (/ (* rDxh rDxh rDxh u-hat) L)
          u-xxx       (irfft u-hat-xxx)]
      (+ 1
         (* u-xxx
            (- 1 (* u u u u-xxx)))))))
         

(with-decorator jit
  (defn nlse [phi * [t 0] [kappa -0.5]]
    """
    Nonlinear Schroedinger equation.
    dphi(x,t)/dt = -i/2 d^2/dx^2 phi - i kappa phi |phi|^2
    x in [0, 2pi]
    """
    (let [phi-hat       (fft phi)
          phi-hat-xx    (* Dxh Dxh phi-hat)
          phi-xx (ifft phi-hat-xx)]
      (+ (* -0.5j phi-xx) (* -j kappa phi (jnp.abs phi) (jnp.abs phi))))))


(with-decorator jit
  (defn diffusion [u * [t 0] [nu 1]]
    """
    Diffusion equation.
    """
    (let [u-hat       (rfft u)
          u-hat-xx    (* rDxh rDxh u-hat)]
      (* nu (irfft u-hat-xx)))))
  

(with-decorator jit
  (defn neutral [u * [t 0]]
    """
    dv/dt = 0.
    """
    (* u 0)))


(with-decorator jit
  (defn stable [u * [t 0]]
    """
    dv/dt = -u
    """
    (- v)))



;;;;;;;;;;;;;;
;; 3D ODEs ;;;
;;;;;;;;;;;;;;

(with-decorator jit
  (defn lorenz [v * [t 0] [beta (/ 8.0 3.0)] [rho 30] [sigma 10]]
    """
    Lorenz system.
    v(t) = (x, y, z)(t)
    dx/dt = sigma (x - y)
    dy/dt = rx - y - xz
    dz/dt = xy - bz
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(* sigma (- y x))
                  (- (* x (- rho z)) y)
                  (- (* x y) (* beta z))]))))


(with-decorator jit
  (defn roessler [v * [t 0] [a 0.1] [b 0.1] [c 14]]
    """
    Roessler system.
    v(t) = (x, y, z)(t)
    dx/dt = - (y + z)
    dy/dt = x + ay
    dz/dt = b + z(x-c)
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(* -1 (+ y z))
                  (+ x (* a y))
                  (+ b (* z (- x c)))]))))


(with-decorator jit
  (defn roessler-p4 [v * [t 0] [a 0.5] [b 0.5]]
    """
    Roessler prototype-4 system.
    v(t) = (x, y, z)(t)
    dx/dt = - (y + z)
    dy/dt = x
    dz/dt = a(y-yy) - bz
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(* -1 (+ y z))
                  x
                  (- (* a y) (* a y y) (* b z))]))))


(with-decorator jit
  (defn sqd [v * [t 0]]
    """
    Sprott SQ_D system.
      Time reversible, although dissipative.
    v(t) = (x, y, z)(t)
    dx/dt = - y
    dy/dt = x + z
    dz/dt = xz + 3yy
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(- y)
                  (+ x z)
                  (+ (* x z) (* 2 y y))]))))


(with-decorator jit
  (defn multiscroll [v * [t 0]]
    """
    Multiscroll system.
    v(t) = (x, y, z)(t)
    dx/dt = x - yz
    dy/dt = xz - y
    dz/dt = xy - 2z
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(- x (* y z))
                  (- (* x z) y)
                  (- (* x y) (* 2 z))]))))


(with-decorator jit
  (defn chua [v * [t 0] [a 3]]
    """
    Simplified ('elegant') Chua's system.
    v(t) = (x, y, z)(t)
    dx/dt = ay - x + |x + 1| - |x - 1|
    dy/dt = z - x
    dz/dt = y
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(- (+ (* a y) (jnp.abs (+ x 1))) x (jnp.abs (- x 1)))
                  (- z x)
                  y]))))


(with-decorator jit
  (defn rikitake [v * [t 0] [alpha 1][ mu 1]]
    """
    Rikitake dynamo.
    v(t) = (x, y, z)(t)
    dx/dt = −µx + yz
    dy/dt = −µy + x(z − α)
    dz/dt = 1 − xy
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(- (* y z) (* mu x))
                  (- (* x (- z alpha))
                     (* mu y))
                  (- 1 (* x y))]))))


(with-decorator jit
  (defn nose-hoover [v * [t 0]]
    """
    Nosé–Hoover Oscillator
    v(t) = (x, y, z)(t)
    dx/dt = y
    dy/dt = yz - x
    dz/dt = 1 − yy
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [y
                  (- (* y z) x)
                  (- 1 (* y y))]))))


(with-decorator jit
  (defn labyrinth [v * [t 0]]
    """
    Labyrinth chaos of René Thomas
    v(t) = (x, y, z)(t)
    dx/dt = sin(y)
    dy/dt = sin(z)
    dz/dt = sin(x)
    """
    (let [x (get v 0)
          y (get v 1)
          z (get v 2)]
      (jnp.array [(jnp.sin y)
                  (jnp.sin z)
                  (jnp.sin x)]))))


(with-decorator jit
  (defn duffing [v * [t 0] [alpha -1] [beta 1] [delta 0.3] [gamma 0.29] [omega 1.2]]
    """
    Duffing equation.
      δ controls the amount of damping,
      α controls the linear stiffness,
      β controls the amount of non-linearity in the restoring force;
      if β=0, the Duffing equation describes a damped and driven simple harmonic oscillator,
      γ is the amplitude of the periodic driving force;
      if γ=0 the system is without a driving force, and
      ω is the angular frequency of the periodic driving force.
    dx/dt = y
    dy/dt = gamma cos(omega t) - delta y - alpha x - beta x^3
    """
    (let [x (get v 0)
          y (get v 1)]
      (jnp.array [y
                  (- (* gamma (jnp.cos (* omega t)))
                     (* delta y)
                     (* alpha x)
                     (* beta x x x))]))))
                 


;;;;;;;;;;;;;;
;; 2D ODEs ;;;
;;;;;;;;;;;;;;

(with-decorator jit
  (defn lv [v * [t 0] [alpha 0.67] [beta 1.33] [gamma 1] [delta 1]]
    """
    Lokta-Volterra (predator-prey) equations.
    dx/dt = αx - βxy
    dy/dt = δxy - γy
    """
    (let [x (get v 0)
          y (get v 1)]
      (jnp.array [(- (* alpha  x) (* beta x y))
                  (- (* delta x y) (* gamma y))]))))


(with-decorator jit
  (defn vdp [v * [t 0] [mu 3.0]]
    """
    Van der Pol oscillator.
    dx/dt = y
    dy/dt = y mu (1 - x^2) - x
    """
    (let [x (get v 0)
          y (get v 1)]
      (jnp.array [y
                  (- (* y
                        mu
                        (- 1 (* x x)))
                     x)]))))


(with-decorator jit
  (defn dixon [v * [t 0] [alpha 0] [beta 0.7]]
    """
    Dixon system.
    dx/dt = xy / (xx - yy) - alpha x
    dy/dt = yy / (xx + yy) - beta y + beta - 1
    """
    (let [x (get v 0)
          y (get v 1)
          vsq (+ (* x x) (* y y))]
      (jnp.array [(- (/ (* x y) vsq)
                     (* alpha x))
                  (- (+ (/ (* y y) vsq)
                        beta)
                     (* beta y)
                     1)]))))



;;;;;;;;;;;;;;;;;;;
;; ND ODEs, N>3 ;;;
;;;;;;;;;;;;;;;;;;;

(with-decorator jit
  (defn coupled-pendulum [v * [t 0] [k 1]]
    """
    Coupled pendulum.
    d^2x/dt^2 = k(y-x) - sin(x)
    d^2y/dt^2 = k(x-y) - sin(y)
    """
    (let [x (get v 0)
          dxdt (get v 1)
          y (get v 2)
          dydt (get v 3)]
      (jnp.array [dxdt
                  (- (* k (- y x))
                     (jnp.sin x))
                  dydt
                  (- (* k (- x y))
                     (jnp.sin y))]))))
                  
                 
