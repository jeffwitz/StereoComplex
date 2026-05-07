# Physical CMO Model

This page defines the compact Common Main Objective (CMO) model implemented in
`stereocomplex.physics.CMOPhysicalStereoModel`.

It is not a full lens-design simulator. It is a paraxial ray-space model whose
goal is narrower: test whether a measured non-central rayfield is compatible
with a shared-objective stereo microscope geometry.

## Optical Assumption

A CMO stereo microscope is modeled as:

1. one common main objective;
2. two effective left/right sub-pupils separated by a baseline $b$;
3. two tube-lens/sensor channels with identical pixel pitch;
4. optional per-channel Brown-Conrady distortion at the angular level.

The model enforces a structural CMO constraint: the left and right chief rays
converge toward the working point on the main optical axis. This is different
from the polynomial surrogate, where the two channels have independent
effective origins and independent angular polynomial fields.

## Parameters

The shared rig parameters are:

| Parameter | Meaning |
|---|---|
| $f_{\mathrm{obj}}$ | effective focal length of the common main objective |
| $Z_w$ | working distance / chief-ray crossover plane |
| $b$ | left/right sub-pupil separation |
| $f_{\mathrm{tube}}$ | effective tube-lens focal length |
| $c_x,c_y$ | shared principal point in pixels |
| $p$ | pixel pitch in mm |
| $\theta_y$ | small global tilt around the vertical axis |

Each channel also has Brown-Conrady coefficients

```{math}
(k_1,k_2,p_1,p_2,k_3)_L,
\qquad
(k_1,k_2,p_1,p_2,k_3)_R.
```

The default vector therefore has 18 scalars:

```{math}
\theta =
\left[
f_{\mathrm{obj}},
Z_w,
b,
f_{\mathrm{tube}},
c_x,
c_y,
p,
\theta_y,
\mathbf d_L,
\mathbf d_R
\right].
```

## Ray Construction

For a pixel $(u,v)$ in channel $c\in\{L,R\}$, first convert to tube-lens
angular coordinates:

```{math}
\alpha_x^d = \frac{(u-c_x)p}{f_{\mathrm{tube}}},
\qquad
\alpha_y^d = \frac{(v-c_y)p}{f_{\mathrm{tube}}}.
```

The distorted angular coordinates are undistorted with the channel's
Brown-Conrady coefficients:

```{math}
(\alpha_x,\alpha_y)
=
D_{\mathrm{Brown}}^{-1}
\left(
\alpha_x^d,\alpha_y^d;
k_1,k_2,p_1,p_2,k_3
\right).
```

Let $s_L=-1$ and $s_R=+1$. The effective sub-pupil point is

```{math}
S_c=
\left(
s_c\frac{b}{2},
0,
Z_w-f_{\mathrm{obj}}
\right)^T.
```

The pixel selects a point on the working plane:

```{math}
P_c(u,v)=
\left(
Z_w\alpha_x,
Z_w\alpha_y,
Z_w
\right)^T.
```

The ray is the line through $S_c$ and $P_c$:

```{math}
\mathcal R_c(u,v)=
\left(
S_c,
\frac{P_c(u,v)-S_c}{\|P_c(u,v)-S_c\|}
\right).
```

For the central pixel, $P_c=(0,0,Z_w)^T$. The chief-ray angle is therefore

```{math}
\gamma=\arctan\left(\frac{b}{2f_{\mathrm{obj}}}\right),
```

and both channels cross the optical axis at the working plane. This chief-ray
constraint is the main geometric difference between the physical CMO and a
pair of independent non-central polynomial channels.

## Identifiability

From ray geometry alone, $f_{\mathrm{tube}}$ and pixel pitch $p$ appear through
the ratio $p/f_{\mathrm{tube}}$. They are not separately identifiable unless
one of them is fixed by external information. StereoComplex therefore reports
and tests the identifiable angular scale $p/f_{\mathrm{tube}}`.

Similarly, strong Brown radial coefficients can be correlated if the observed
field of view is narrow. The robust validation quantities are:

- ray-space RMS against the measured Zernike field;
- chief-ray convergence and effective CMO baseline;
- recovered working plane and sub-pupil geometry;
- pose consistency in the CMO bundle-adjustment benchmark.

## Relation To The Polynomial Surrogate

The existing `CMOPolynomialChannelModel` is better described as a generic
non-central polynomial channel surrogate. It is useful because it can fit many
smooth rayfields, including CMO-like ones, but it does not encode the shared
main-objective constraints.

The physical CMO model is less flexible but more interpretable. On a true CMO
oracle, it should achieve comparable ray-space residuals with fewer effective
degrees of freedom, so BIC should prefer it over the polynomial surrogate.

## References

- Olympus US 7,564,619, "Stereoscopic microscope", 2009.
- Wang et al., "Calibration of a stereo microscope based on non-coplanar
  feature points", *Optics and Lasers in Engineering*, 134, 2020.
- Schreier, Garcia and Sutton, "Advances in light microscope stereo vision",
  *Experimental Mechanics*, 44(3), 278-288, 2004.
- Pan, Wang and Cheng, "High-accuracy 3D shape and deformation measurements
  with a CMO stereo microscope", *Optics Express*, 22(15), 18373-18387, 2014.
