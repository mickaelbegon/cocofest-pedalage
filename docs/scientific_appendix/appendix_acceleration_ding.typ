#set page(paper: "us-letter", margin: (top: 0.75in, bottom: 0.72in, x: 0.78in))
#set text(font: "Libertinus Serif", size: 10pt)
#set heading(numbering: "1.")
#show heading: set text(weight: "bold")

= Technical appendix Computational acceleration for FES cycling with the Ding model

This appendix documents the two principal reductions used in the FES-cycling work: projection of the constrained multibody mechanics onto the crank manifold, and local elimination/reconstruction of linear Ding states. It distinguishes exact identities of a chosen discrete transcription from continuous-model equivalence and from observed solver speed. The objective is to reduce runtime without representing a numerical approximation as a physiological validation; measurement-corrected NMPC remains necessary for use with a participant.

== A. Historical baseline and notation

The historical reference attributed to Kevin is commit `e9a211ed2eb1af30683a324159a6fcd6e899fc05` (21 November 2025). It already uses RHO NMPC with a primal warm start, IPOPT/MA57 on Linux, degree-3 Radau collocation, the fatigable Ding model, and force-length, force-velocity, and passive-force relationships. Its main configuration is 30 Hz, a two-cycle window, a one-cycle shift, and five requested cycles. These are baseline elements, not later innovations.

For $M$ muscles, the full state is $x=(C_m,F_m,A_m,T_m,K_m)_(m=1)^M,q,dot q$. The symbols $C,F,A,T,K$ denote normalized calcium, force, force-scale capacity, $Tau1$, and $K_m$, respectively. The crank coordinate is $theta$ and its angular velocity is $omega=dot theta$. Pulse width is $u$.

== B. Starting point: the Ding force-fatigue system

The five-state phenomenological model is consistent with the Ding family of force and fatigue models [1--3]. Let

$rho(u)=1-exp(-(u-u_0)/u_t), quad sigma(C,K)=C/(K+C),$

and let $g(theta,omega)=g_l(theta)g_v(theta,omega)+g_p(theta)$ collect the active force-length, force-velocity, and passive factors. The implementation used here has

$dot C=(H(t)-C)/tau_c,$

$dot F=g(theta,omega)[A rho(u)sigma(C,K)-F/(T+tau_2 sigma(C,K))],$

$dot A=-(A-A_r)/tau_f+alpha_A F,$

$dot T=-(T-T_r)/tau_f+alpha_T F, quad dot K=-(K-K_r)/tau_f+alpha_K F.$

The force equation is nonlinear and remains coupled to mechanics. In contrast, the $A,T,K$ equations have the same scalar recovery operator and the same forcing $F$. This structure is the basis of the local state reduction; it is not an assumption that force itself is linear.

== C. Reduced crank mechanics

=== C.1 Constraint manifold and projection

The full model has generalized coordinates $q in R^n$ and bilateral or unilateral hand--crank contact constraints. Offline, the physical contact problem is solved over one crank revolution to obtain the admissible configuration curve

$q=Q(theta), quad Q(theta+2pi)=Q(theta).$

The reduced model is therefore not a freely fitted torque curve. It parameterizes the contact-consistent configuration manifold by the physical crank angle. Differentiation gives

$dot q=Q'(theta)omega, quad ddot q=Q'(theta)dot omega+Q''(theta)omega^2.$

Starting with the full forward dynamics $M(q)ddot q+h(q,dot q)=tau_m+tau_ext$, premultiply by the tangent $Q'(theta)^T$. Define

$I(theta)=Q'^T M(Q)Q',$

$G(theta)=Q'^T h(Q,0),$

$V(theta)=Q'^T[h(Q,Q')-h(Q,0)+M(Q)Q''],$

where the last expression is evaluated with unit crank speed. If $eta_m(theta)$ is the tangent-projected effectiveness of muscle $m$ and $eta_e(theta)$ the corresponding effectiveness of the external crank torque, the two-state dynamics are

$dot theta=omega,$

$I(theta)dot omega=sum_(m=1)^M eta_m(theta)F_m+eta_e(theta)tau_ext-G(theta)-V(theta)omega^2.$

This is a Lagrange/tangent projection of the original constrained dynamics. It removes the redundant generalized coordinates while retaining the crank coordinate and speed. It is valid on the sampled contact branch; it does not reproduce departures from that branch.

=== C.2 Offline construction of the numerical reduced profile

At sampled angles, the implementation evaluates the full Biorbd mass matrix, nonlinear effects, muscle-length Jacobian, and contact-consistent $Q,Q',Q''$. It computes

$I=Q'^T M Q', quad eta_m=-J_{ell,m}Q', quad eta_e=Q'_r,$

as well as normalized fibre length $ell_m(theta)$ and velocity-per-speed $nu_m(theta)=J_{ell,m}Q'$. Smooth periodic Fourier series are fitted to $I,G,V,eta_m,eta_e,ell_m,nu_m$. Online evaluation is consequently a small symbolic expression in $(theta,omega,F)$ rather than a full multibody/contact solve. The original Hill relations are retained using the fitted geometry:

$tilde l_m=ell_m(theta), quad tilde v_m=nu_m(theta)omega/10.$

The profile must be regenerated and audited whenever the source biomechanical model, contact geometry, selected muscles, or fitting order changes. Validation compares projected acceleration, muscle geometry, and passive/active Hill factors with the full constrained model at independent samples.

=== C.3 What is gained and what is not

For four muscles, the full forward model has 20 Ding states plus six mechanical states; the reduced dynamic model keeps the 20 Ding states and $(theta,omega)$, reducing 26 states to 22. In the controlled 100-RHO IPOPT/Radau-3 campaign, hot median time changed from 4.595 to 1.031 s and total wall time from 615.4 to 260.9 s; the reported fatigue difference was 0.097 percent. These values are campaign-specific and do not multiply with gains from other changes. Reduced mechanics is a model transformation, not merely a compiler optimization; full-versus-reduced mechanical agreement remains a required gate.

== D. Local reduction of Ding states

=== D.1 Continuous identities

For any $z in {A,T,K}$ with rest value $z_r$ and coefficient $alpha_z$, the integrating-factor solution over an interval $[t_k,t_k+s]$ is

$z(t_k+s)=z_r+exp(-s/tau_f)(z_k-z_r)+alpha_z int_0^s exp(-(s-v)/tau_f)F(t_k+v) d v.$

All three fatigue variables share the same convolution of $F$. Thus, with $J(s)=int_0^s exp(v/tau_f)F(t_k+v) d v$,

$z(t_k+s)=z_r+exp(-s/tau_f)[z_k-z_r+alpha_z J(s)].$

Equivalently, when $alpha_A != 0$, define

$beta_T=alpha_T/alpha_A, quad beta_K=alpha_K/alpha_A,$

$d_T=(T-T_r)-beta_T(A-A_r), quad d_K=(K-K_r)-beta_K(A-A_r).$

Then $dot d_T=-d_T/tau_f$ and $dot d_K=-d_K/tau_f$. Keeping $(F,A)$ and the true incoming offsets yields

$T=T_r+beta_T(A-A_r)+d_{T,k}exp(-s/tau_f),$

$K=K_r+beta_K(A-A_r)+d_{K,k}exp(-s/tau_f).$

For the periodic-node calcium input $H(t_k+s)=H_k exp(-s/tau_c)$, calcium is also reconstructible on the interval:

$C(t_k+s)=exp(-s/tau_c)(C_k+H_k s/tau_c).$

The calcium state is continuous at a stimulation; the forcing amplitude changes. Neither periodic steady state nor zero fatigue offsets may replace the physical initial state inherited from the previous RHO window.

=== D.2 Discrete equivalence takes priority in an OCP

The continuous formulas above do *not* reproduce a finite-step Radau or Gauss-Legendre map exactly. To preserve the actual transcription, eliminate only the linear stage equations after discretization. For a Runge--Kutta tableau $(mathcal A,b,c)$ and step $h$, stage values obey

$bold X=bold 1 x_k+h mathcal A bold f(bold X).$

For five-stage Radau IIA,

$bold C=(I+h mathcal A/tau_c)^(-1)[bold 1 C_k+(h/tau_c)mathcal A bold H],$

$bold z=z_r bold 1+(I+h mathcal A/tau_f)^(-1)[bold 1(z_k-z_r)+h alpha_z mathcal A bold F].$

After substitution, the nonlinear muscle-stage residual contains the force stages, while calcium and fatigue stages are reconstructed. This is algebraically identical to the full Radau discrete equations, subject to linear-solve and NLP tolerances. Radau IIA is stiffly accurate, so its final stage is the interval endpoint [4,5]. A full five-state, five-stage muscle has 25 stage variables; this local condensation leaves five nonlinear force stages, while physical node states may still be kept for RHO transfer and constraints.

For Acados IRK Gauss--Legendre 4-by-5, the eliminated $C,d_T,d_K$ profiles are computed with its own tableau, substep by substep, and supplied to the reduced model. The endpoint must be calculated as

$v_(k+1)=v_k+h b^T[r(t_k+hc)-bold V/tau],$

because the last Gauss--Legendre stage is not the endpoint. Reusing Radau profiles in Gauss--Legendre would change the discrete problem.

=== D.3 Dimension, constraints, and derivatives

With four muscles and reduced mechanics, retaining only $(F,A)$ gives 10 physical states instead of 22; $C,T,K$ are reconstructed locally. This count applies only when the stimulation schedule, calcium parameters, and incoming state are fixed for a window. It is not the same as Radau stage condensation, which can remove internal variables while retaining all physical node states.

Every objective and constraint that uses a removed variable must be evaluated on its reconstruction at the same nodes or stages as the full problem. For scaled variables, $bar y=beta bar A+delta$, a bound on $y$ becomes an interval on $A$ only after correct scaling and sign handling. The physical domain still requires $K+C>0$ and $T+tau_2 sigma(C,K)>0$. Reconstructed quantities must remain in the automatic-differentiation graph: treating state-dependent profiles as constants would give wrong sensitivities. This warning is consistent with the wider direct-collocation literature, where formulation can change both robustness and computational cost [6].

=== D.4 Evidence and limitations

On isolated, converged maps, the local ACADOS proof produced maximum reconstruction errors of 1.33e-16 for GL4-by-5 and 3.61e-15 for Radau IIA five-stage; corresponding selected sensitivity errors were 3.24e-12 and 9.48e-12. These are map-equivalence checks, not OCP performance results. In the IPOPT/Radau-5 reduced RHO A/B, 50 propagated windows succeeded in both variants and hot median time changed from 0.798 to 0.551 s (-30.9 percent). For Acados, the experimental local formulation completed 76 valid cycles before stopping at cycle 77; over the first 75 hot cycles, the median changed from 0.455 to 0.167 s (-63.3 percent). The latter is promising but is not a 100/100 robustness claim.

== E. Fidelity, solver comparison, and benchmark protocol

IPOPT/MA57 with five-stage Radau IIA is the fidelity reference. Acados uses multiple shooting and IRK Gauss--Legendre 4-by-5:

$x_(k+1)=Phi_h(x_k,u_k).$

A common initial state, seed, Ding parameters, mechanical profile, load, objective, and bounds are essential. Both solutions must be replayed with DOP853 under fixed controls. Define

$epsilon_omega=max_t max(0,omega_min-omega(t),omega(t)-omega_max).$

The controlled pair had DOP853 costs 0.01295477 (IPOPT) and 0.01296194 (Acados), but both had small between-node speed excursions. With a common tightened fast margin, costs were 0.01383435 and 0.01384597 (Acados +0.084 percent), with DOP853 excursions 0.00515 and 0.00772 rad/s. A strict ranking requires zero excursion; similar small excursions support only a qualified numerical comparison.

Each ablation should report build/compilation time, hot-solve time, wall time, iterations, valid-cycle fraction, common DOP853 cost, $epsilon_omega$, phase error, pulse-width bounds, KKT dimensions/nonzeros, and control distance. RHO--FHO comparisons require equal horizon and separately reported per-cycle and total time. Do not combine individual speedup factors.

== F. References

[1] J. Ding, A. S. Wexler, and S. A. Binder-Macleod. “A predictive model of fatigue in human skeletal muscles.” *Journal of Applied Physiology*, 89(4):1322--1332, 2000. doi:10.1152/jappl.2000.89.4.1322.

[2] J. Ding, A. S. Wexler, and S. A. Binder-Macleod. “A mathematical model that predicts the force-frequency relationship of human skeletal muscle.” *Muscle & Nerve*, 26(4):477--485, 2002. doi:10.1002/mus.10198.

[3] J. Ding, A. S. Wexler, and S. A. Binder-Macleod. “Mathematical models for fatigue minimization during functional electrical stimulation.” *Journal of Electromyography and Kinesiology*, 13(6):575--588, 2003. doi:10.1016/S1050-6411(03)00102-0.

[4] E. Hairer and G. Wanner. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer, 1996. doi:10.1007/978-3-642-05221-7.

[5] E. Hairer and G. Wanner. “Stiff differential equations solved by Radau methods.” *Journal of Computational and Applied Mathematics*, 111(1--2):93--111, 1999. doi:10.1016/S0377-0427(99)00134-X.

[6] F. De Groote, A. L. Kinney, A. V. Rao, and B. J. Fregly. “Evaluation of direct collocation optimal control problem formulations for solving the muscle redundancy problem.” *Annals of Biomedical Engineering*, 44(10):2922--2936, 2016. doi:10.1007/s10439-016-1591-9.
