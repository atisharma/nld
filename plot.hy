"""
Integrate systems with RK4 and then plot them.
"""

(require [hy.contrib.walk [let]])

(import [matplotlib [pyplot :as plt]])

(import [jax.numpy :as jnp])
(import [jax.numpy [pi]])

(import [systems [x]])


; default plotting style
(setv plotstyle {"linewidth" 0.5
                 "color" "#00000088"
                 "antialiased" True})

(setv default-cmap "viridis")


(defn contourf [data * [silent True] [x x] [plot-abs False] [colorbar False] [cmap default-cmap] [num-contours 16] #** kwargs]
  """
  Integrate and plot a 1D PDE.
  Plotting using countourf is very slow -- use pcolor
  """
  (let [[fig ax] (plt.subplots 1 1)
        t (:t data)
        trajectory (:trajectory data)
        plot-data (if plot-abs (jnp.abs trajectory.T) trajectory.T)
        mappable (ax.contourf t x plot-data num-contours :cmap cmap #** kwargs)
        cbar (when colorbar (plt.colorbar :mappable mappable))]
    (ax.set_xlabel "$t$")
    (ax.set_ylabel "$x$")
    (ax.set_title f"{(:system data)} system, $h$={(:h data)}, $N$={(len x)}")
    (unless silent
            {"fig" fig
             "axes" ax
             "x" x
             "colorbar" cbar})))


(defn pcolor [data * [silent True] [x x] [plot-abs False] [colorbar False] [cmap default-cmap] #** kwargs]
  """
  Integrate and plot a 1D PDE.
  """
  (let [[fig ax] (plt.subplots 1 1)
        t (:t data)
        trajectory (:trajectory data)
        ; e.g. NLSE is complex, would plot the abs
        plot-data (if plot-abs (jnp.abs trajectory.T) trajectory.T)
        mappable (ax.pcolor t x plot-data :cmap cmap :shading "auto" #** kwargs)
        cbar (when colorbar (plt.colorbar :mappable mappable))]
    (ax.set_xlabel "$t$")
    (ax.set_ylabel "$x$")
    (ax.set_title f"{(:system data)} system, $h$={(:h data)}, $N$={(len x)}")
    (unless silent
            {"fig" fig
             "axes" ax
             "x" x
             "colorbar" cbar})))


(defn plot1d [data * [silent True] #** kwargs]
  """
  Integrate and plot the first two states.
  """
  (let [fig (.figure plt)
        ax (.axes plt)]
    (ax.plot (:t data) (:trajectory data) #** {#** plotstyle #** kwargs})
    (ax.set_xlabel "$t$")
    (ax.set_ylabel "$x$")
    (ax.set_title f"{(:system data)} system, $T$={(:T data)}, $h$={(:h data)}")
    (unless silent
            {"fig" fig
             "axes" ax})))


(defn plot2d [data * [silent True] #** kwargs]
  """
  Integrate and plot the first two states.
  """
  (let [fig (.figure plt)
        ax (.axes plt)
        trajectory (:trajectory data)
        xs (get trajectory (, Ellipsis 0))
        ys (get trajectory (, Ellipsis 1))]
    (ax.plot xs ys #** {#** plotstyle #** kwargs})
    (ax.set_xlabel "$x$")
    (ax.set_ylabel "$y$")
    (ax.set_title f"{(:system data)} system, $T$={(:T data)}, $h$={(:h data)}")
    (unless silent
            {"fig" fig
             "axes" ax})))


(defn plot3d [data * [silent True] #** kwargs]
  """
  Integrate and plot the first three states.
  """
  (let [fig (.figure plt)
        ax (.axes plt :projection "3d")
        trajectory (:trajectory data)
        xs (get trajectory (, Ellipsis 0))
        ys (get trajectory (, Ellipsis 1))
        zs (get trajectory (, Ellipsis 2))]
    (ax.plot xs ys zs #** {#** plotstyle #** kwargs})
    (ax.set_xlabel "$x$")
    (ax.set_ylabel "$y$")
    (ax.set_zlabel "$z$")
    (ax.set_title f"{(:system data)} system, $T$={(:T data)}, $h$={(:h data)}")
    (unless silent
            {"fig" fig
             "axes" ax})))


(defn plot [data #** kwargs]
  """
  Integrate and plot.
  """
  (let [l (-> (:trajectory data)
              (. shape)
              (get 1))]
    (cond [(= l 2) (plot2d   data #** kwargs)]
          [(= l 3) (plot3d   data #** kwargs)]
          [:else   (contourf data #** kwargs)])))
