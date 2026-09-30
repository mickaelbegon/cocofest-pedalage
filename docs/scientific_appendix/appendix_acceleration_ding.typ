#set page(paper: "us-letter", margin: (top: 0.75in, bottom: 0.72in, x: 0.78in))
#set text(font: "Libertinus Serif", size: 10pt)
#set heading(numbering: "1.")
#show heading: set text(weight: "bold")

= Technical appendix Computational acceleration for FES cycling with the Ding model

This appendix documents the two principal reductions used in the FES-cycling work: projection of the constrained multibody mechanics onto the crank manifold, and local elimination/reconstruction of linear Ding states. It distinguishes exact identities of a chosen discrete transcription from continuous-model equivalence and from observed solver speed. The objective is to reduce runtime without representing a numerical approximation as a physiological validation; measurement-corrected NMPC remains necessary for use with a participant.

== A. Historical baseline and notation

The historical reference is the FES hand-cycling formulation described by Co et al. [7]. It already uses RHO NMPC with a primal warm start, IPOPT/MA57 on Linux, degree-3 Radau collocation, the fatigable Ding model, and force-length, force-velocity, and passive-force relationships. Its main configuration is 30 Hz, a two-cycle window, a one-cycle shift, and five requested cycles. These are baseline elements, not later innovations.

For $M$ muscles, the full state is $x=(C_m,F_m,A_m,T_m,K_m)_(m=1)^M,q,dot(q)$. The symbols $C,F,A,T,K$ denote normalized calcium, force, force-scale capacity, $tau_1$, and $K_m$, respectively. The crank coordinate is $theta$ and its angular velocity is $omega=dot(theta)$. Pulse width is $u$.

== B. Starting point: the Ding force-fatigue system

The five-state phenomenological model is consistent with the Ding family of force and fatigue models [1--3]. Let

$rho(u)=1-exp(-(u-u_0)/u_t), quad sigma(C,K)=C/(K+C),$

and let $g(theta,omega)=g_l(theta)g_v(theta,omega)+g_p(theta)$ collect the active force-length, force-velocity, and passive factors. The implementation used here has

$dot(C)=(H(t)-C)/tau_c,$

$dot(F)=g(theta,omega)[A rho(u)sigma(C,K)-F/(T+tau_2 sigma(C,K))],$

$dot(A)=-(A-A_r)/tau_f+alpha_A F,$

$dot(T)=-(T-T_r)/tau_f+alpha_T F, quad dot(K)=-(K-K_r)/tau_f+alpha_K F.$

The force equation is nonlinear and remains coupled to mechanics. In contrast, the $A,T,K$ equations have the same scalar recovery operator and the same forcing $F$. This structure is the basis of the local state reduction; it is not an assumption that force itself is linear.

== C. Reduced crank mechanics

=== C.1 Constraint manifold and projection

The full model has generalized coordinates $q in R^n$ and bilateral or unilateral hand--crank contact constraints. Offline, the physical contact problem is solved over one crank revolution to obtain the admissible configuration curve

$q=Q(theta), quad Q(theta+2pi)=Q(theta).$

The reduced model is therefore not a freely fitted torque curve. It parameterizes the contact-consistent configuration manifold by the physical crank angle. Differentiation gives

$dot(q)=Q'(theta)omega, quad dot.double(q)=Q'(theta)dot(omega)+Q''(theta)omega^2.$

Starting with the full forward dynamics $M(q)dot.double(q)+h(q,dot(q))=tau_m+tau_e$, premultiply by the tangent $Q'(theta)^T$. Define

$I(theta)=Q'^T M(Q)Q',$

$G(theta)=Q'^T h(Q,0),$

$V(theta)=Q'^T[h(Q,Q')-h(Q,0)+M(Q)Q''],$

where the last expression is evaluated with unit crank speed. If $eta_m(theta)$ is the tangent-projected effectiveness of muscle $m$ and $eta_e(theta)$ the corresponding effectiveness of the external crank torque, the two-state dynamics are

$dot(theta)=omega,$

$I(theta)dot(omega)=sum_(m=1)^M eta_m(theta)F_m+eta_e(theta)tau_e-G(theta)-V(theta)omega^2.$

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

$z(t_k+s)=z_r+exp(-s/tau_f)(z_k-z_r)+alpha_z integral_0^s exp(-(s-v)/tau_f)F(t_k+v) d v.$

All three fatigue variables share the same convolution of $F$. Thus, with $J(s)=integral_0^s exp(v/tau_f)F(t_k+v) d v$,

$z(t_k+s)=z_r+exp(-s/tau_f)[z_k-z_r+alpha_z J(s)].$

Equivalently, when $alpha_A != 0$, define

$beta_T=alpha_T/alpha_A, quad beta_K=alpha_K/alpha_A,$

$d_T=(T-T_r)-beta_T(A-A_r), quad d_K=(K-K_r)-beta_K(A-A_r).$

Then $dot(d_T)=-d_T/tau_f$ and $dot(d_K)=-d_K/tau_f$. Keeping $(F,A)$ and the true incoming offsets yields

$T=T_r+beta_T(A-A_r)+d_{T,k}exp(-s/tau_f),$

$K=K_r+beta_K(A-A_r)+d_{K,k}exp(-s/tau_f).$

For the periodic-node calcium input $H(t_k+s)=H_k exp(-s/tau_c)$, calcium is also reconstructible on the interval:

$C(t_k+s)=exp(-s/tau_c)(C_k+H_k s/tau_c).$

The calcium state is continuous at a stimulation; the forcing amplitude changes. Neither periodic steady state nor zero fatigue offsets may replace the physical initial state inherited from the previous RHO window.

=== D.2 Discrete equivalence takes priority in an OCP

The continuous formulas above do *not* reproduce a finite-step Radau or Gauss-Legendre map exactly. To preserve the actual transcription, eliminate only the linear stage equations after discretization. For a Runge--Kutta tableau $(cal(A),b,c)$ and step $h$, stage values obey

$bold(X)=bold(1) x_k+h cal(A) bold(f)(bold(X)).$

For five-stage Radau IIA,

$bold(C)=(I+h cal(A)/tau_c)^(-1)[bold(1) C_k+(h/tau_c)cal(A) bold(H)],$

$bold(z)=z_r bold(1)+(I+h cal(A)/tau_f)^(-1)[bold(1)(z_k-z_r)+h alpha_z cal(A) bold(F)].$

After substitution, the nonlinear muscle-stage residual contains the force stages, while calcium and fatigue stages are reconstructed. This is algebraically identical to the full Radau discrete equations, subject to linear-solve and NLP tolerances. Radau IIA is stiffly accurate, so its final stage is the interval endpoint [4,5]. A full five-state, five-stage muscle has 25 stage variables; this local condensation leaves five nonlinear force stages, while physical node states may still be kept for RHO transfer and constraints.

For Acados IRK Gauss--Legendre 4-by-5, the eliminated $C,d_T,d_K$ profiles are computed with its own tableau, substep by substep, and supplied to the reduced model. The endpoint must be calculated as

$v_(k+1)=v_k+h b^T[r(t_k+h c)-bold(V)/tau],$

because the last Gauss--Legendre stage is not the endpoint. Reusing Radau profiles in Gauss--Legendre would change the discrete problem.

=== D.3 Dimension, constraints, and derivatives

With four muscles and reduced mechanics, retaining only $(F,A)$ gives 10 physical states instead of 22; $C,T,K$ are reconstructed locally. This count applies only when the stimulation schedule, calcium parameters, and incoming state are fixed for a window. It is not the same as Radau stage condensation, which can remove internal variables while retaining all physical node states.

Every objective and constraint that uses a removed variable must be evaluated on its reconstruction at the same nodes or stages as the full problem. For scaled variables, $bar(y)=beta bar(A)+delta$, a bound on $y$ becomes an interval on $A$ only after correct scaling and sign handling. The physical domain still requires $K+C>0$ and $T+tau_2 sigma(C,K)>0$. Reconstructed quantities must remain in the automatic-differentiation graph: treating state-dependent profiles as constants would give wrong sensitivities. This warning is consistent with the wider direct-collocation literature, where formulation can change both robustness and computational cost [6].

=== D.4 Evidence and limitations

On isolated, converged maps, the local ACADOS proof produced maximum reconstruction errors of 1.33e-16 for GL4-by-5 and 3.61e-15 for Radau IIA five-stage; corresponding selected sensitivity errors were 3.24e-12 and 9.48e-12. These are map-equivalence checks, not OCP performance results. In the IPOPT/Radau-5 reduced RHO A/B, 50 propagated windows succeeded in both variants and hot median time changed from 0.798 to 0.551 s (-30.9 percent). For Acados, the experimental local formulation completed 76 valid cycles before stopping at cycle 77; over the first 75 hot cycles, the median changed from 0.455 to 0.167 s (-63.3 percent). The latter is promising but is not a 100/100 robustness claim.

== E. Fidelity, solver comparison, and benchmark protocol

IPOPT/MA57 with five-stage Radau IIA is the fidelity reference. Acados uses multiple shooting and IRK Gauss--Legendre 4-by-5:

$x_(k+1)=Phi_h(x_k,u_k).$

A common initial state, seed, Ding parameters, mechanical profile, load, objective, and bounds are essential. Both solutions must be replayed with DOP853 under fixed controls. Define

$epsilon_"omega"=max_t max(0,omega_"min"-omega(t),omega(t)-omega_"max").$

The controlled pair had DOP853 costs 0.01295477 (IPOPT) and 0.01296194 (Acados), but both had small between-node speed excursions. With a common tightened fast margin, costs were 0.01383435 and 0.01384597 (Acados +0.084 percent), with DOP853 excursions 0.00515 and 0.00772 rad/s. A strict ranking requires zero excursion; similar small excursions support only a qualified numerical comparison.

Each ablation should report build/compilation time, hot-solve time, wall time, iterations, valid-cycle fraction, common DOP853 cost, $epsilon_"omega"$, phase error, pulse-width bounds, KKT dimensions/nonzeros, and control distance. RHO--FHO comparisons require equal horizon and separately reported per-cycle and total time. Do not combine individual speedup factors.

=== E.1 Terminal half-step guard at the RHO seam

In free-cadence reduced mechanics, the state at the end of one RHO window is the *fixed incoming state* of the following window. A control in that following window cannot repair a violation that occurs immediately after the seam. The optional terminal guard therefore reuses the already active reduced internal cadence predictor at the terminal node. If $h$ is one shooting interval and $f_omega$ is the reduced angular acceleration, it constrains

$omega_"half" = omega_T + h/2 f_omega(theta_T, omega_T, F_T, tau_"ext")$

to the same cadence interval as the internal shooting-node guard. It is intentionally an Euler half-step predictor: it is cheap, differentiable, and uses no new decision variable or new NLP compilation. It is only available with dynamic reduced mechanics and an active internal reduced-speed guard. It is neither a DOP853 replay nor a certificate that every constraint in the next full cycle is feasible.

The matched 100-window IPOPT/MA57/Radau-5 reduced-dynamics A/B completed all 100 certified windows in both variants. The reported times are the solver-loop measurements; cold construction, compilation, and post-solve replay are excluded.

#table(
  columns: 5,
  [*Terminal seam guard*], [*Certified windows*], [*Hot median / P90 (s)*], [*Validated loop wall (s)*], [*Interpretation*],
  [off], [100 / 100], [0.650 / 0.898], [81.172], [reference internal guard only],
  [Euler half-step on], [100 / 100], [0.680 / 0.871], [80.961], [same 100-window certification; timing-neutral within this small A/B]
)

The A/B shows that the local seam constraint did not destroy the observed one-second-scale RHO operation. It does *not* establish a clinical or continuous-time safety gain. A separate cross-solver diagnostic found an Euler prediction at 0.9h more conservative than the observed continuous minimum velocity; consequently, extra Euler samples and a tighter margin are not presented as a continuous feasibility certificate. The required acceptance test remains a common dense DOP853 replay, including the seam and the next applied cycle.

== F. Isokinetic and isoresistive formulations

The two mechanical formulations answer different questions. In the isokinetic formulation, the crank speed is constrained, $omega(t)=omega_"ref"$, and the instantaneous resistance follows from the projected mechanical balance. The terminal work state provides a transparent per-turn target,

$E_"prod"(T)-E_"prod"(0)=W=2 pi bar(tau).$

Consequently, changing $bar(tau)$ is a numeric terminal-bound update; it does not prescribe an instantaneous external torque. This is the appropriate formulation for comparing numerical transcriptions at a common cadence and for the two-independent-arm protocol below.

In the isoresistive (free-speed) formulation, the applied resistance is prescribed, $tau_e(t)=tau_R(t)$, while $omega$ is a state governed by the projected dynamics. Work and cadence are outputs rather than imposed values. It is useful for studying acceleration, coast phases, and a physical common crank, but it cannot be compared to isokinetic results using only solver time: the trajectories, constraints, and control problem differ. A fair comparison fixes the initial state, pulse constraints, horizon, objective, resistance convention, and DOP853 replay protocol, then reports cadence and work as outcomes.

#table(
  columns: 4,
  [*Property*], [*Isokinetic*], [*Isoresistive*], [*Required audit*],
  [Mechanical input], [$omega=omega_"ref"$], [$tau_e=tau_R$], [sign convention and units],
  [Mechanical output], [inferred instantaneous load], [$omega(t)$ and $theta(t)$], [phase and speed trace],
  [Turn work], [terminal target $E_"prod"(T)$], [trajectory outcome], [DOP853 work integral],
  [Two independent arms], [well defined with a common clock], [not a physical shared-crank model], [state the architecture]
)

== G. Two independent arms and paced allocation

Two independent unilateral isokinetic OCPs are used when the scientific question is allocation of a prescribed bilateral work budget, not bilateral contact mechanics. Each arm has four Ding muscle models, its own state history, pulse-width controls, and fatigue cost. The process-isolated coordinator imposes, at every certified cycle $k$,

$W_R^(k)+W_L^(k)=2 pi bar(tau)_"total".$

No left-arm state is inserted in the right OCP or conversely. The coordinator only supplies the next numeric terminal-work target after both unilateral OCPs certify. Thus it preserves two compiled unilateral NLPs and is not a coupled eight-muscle OCP.

=== G.1 Cold allocation of total work between arms

The manual cold split is the declared initial pair $bar(tau)_R,bar(tau)_L$. The capacity--fatigability cold alternative is deliberately deferred until one common certified cycle has supplied an admissible state for each arm. From a maximum-PW isokinetic envelope, let $c_(s,m)$ be the positive work opportunity of muscle $m$ on side $s$, and let $d_(s,m)$ be the exact prescribed-force Ding capacity decrement ratio over the reference cycle. The side summaries are

$C_s=sum_m c_(s,m), quad q_(s,m)=c_(s,m)/C_s, quad D_s=sum_m q_(s,m)d_(s,m), quad E_s=W_"ref"/D_s.$

The initial right fraction is $p_R=E_R/(E_R+E_L)$, then clipped only by an explicitly declared minimum arm torque. $C_s$ is an opportunity measure, not a feasibility certificate; $D_s$ contains the fatigability comparison on a common reference work. This one-time initialisation is recorded separately from subsequent feedback and is applied only to the next cycle.

=== G.2 Slow causal updates of the bilateral work split

After both sides have certified a block boundary, a capacity-only supervisory alternative computes

$p_R^*=r_R^g/(r_R^g+r_L^g), quad bar(tau)_R=bar(tau)_"total" Pi(p_R), quad bar(tau)_L=bar(tau)_"total"-bar(tau)_R.$

Here $r_s$ is a terminal reserve ratio, $g$ is a declared gain, and $Pi$ denotes geometric smoothing, a maximum fraction step, and the minimum arm torque. Updates occur only every $K$ certified cycles (currently $K=10$ by default). The law is causal and work-conserving, but does not predict the endurance optimum.

=== G.3 Cold and iterative muscle-weight policies

The unilateral fatigue objective retains the form $sum_m w_(s,m)(1-A_(s,m)/A_(s,m),"rest")^2$. All four-component weight vectors are normalized to geometric mean one, bounded in log space, and changed by a bounded log step so that a parameter update does not rebuild the compiled objective. The explicit ablation set is:

#table(
  columns: 4,
  [*Policy*], [*Cold state*], [*Slow update at a certified block*], [*Scientific role*],
  [Unit], [$w_m=1$], [none], [neutral reference],
  [Physio-U], [unit; no FHO/BO information], [mechanical Shapley work credit times state-conditioned prescribed-force Ding fatigue challenge], [article-inspired state-conditioned hypothesis],
  [Mechanical-sensitivity-squared v1], [unit; first update waits for a live certified envelope], [$w_m$ targets are proportional to $b_m^2$, where $b_m$ is normalized mechanical Shapley work credit], [mechanics-only ablation of Physio-U],
  [Capacity feedback], [unit weights], [inverse local capacity-ratio feedback], [legacy causal reference]
)

The mechanics-only condition intentionally does not multiply $alpha_A$ or a fatigue decrement into $b_m^2$: Ding fatigue is already propagated in $A_m$, and the RHO cost already squares the normalized fatigue term. Adding it again would confound the ablation by effectively increasing the fatigue power. Physio-U remains distinct because it does use the state-conditioned force challenge. For either iterative policy, if the envelope is absent, contains a zero/invalid mechanical credit, is outside the Ding challenge domain, or comes from an uncertified cycle, the incumbent weights are held. This is fail-closed behaviour, not an inferred physiological weight.

Numerically, the live envelope is assembled only at a due certified boundary from the complete local Ding state $(C_"n",F,A,Tau_1,K_"m")$, pulse-width limits, reduced isokinetic geometry, and the certified terminal-work demand. The target is projected to relative bounds $[w_"min",w_"max"]$, then smoothed and capped by $|Delta log w_m|<=Delta_"max"$. These numeric parameter updates preserve a single compiled NLP; the envelope and supervisory calculations are outside the within-cycle OCP.

The 200-cycle process-isolated IPOPT/MA57/Radau-5 validation used a total equivalent mean torque of 0.3 N m. All 200 windows per arm were certified and the maximum absolute mismatch between requested and realised terminal work was zero. The work split evolved from 0.10/0.20 N m to approximately 0.150/0.150 N m; the minimum terminal capacities were 0.941 (right) and 0.938 (left). This is evidence of correct software operation for this protocol, not evidence of clinical endurance benefit. In particular, an independent-arm architecture is valid directly in isokinetics; in isoresistance, two independent cranks are a modelling assumption and a single physical crank requires a coupled bilateral mechanical model.

#table(
  columns: 5,
  [*Campaign*], [*Validated windows*], [*Median / P90*], [*Wall time*], [*Interpretation*],
  [Independent arms, fixed targets], [100 + 100], [0.784 / 0.966 s (R); 0.800 / 1.020 s (L)], [about 141 s parallel], [two process-isolated unilateral RHO sessions],
  [Paced independent arms, 0.3 N m total], [200 + 200], [0.773 / 0.917 s (R); 0.774 / 0.940 s (L)], [384.1 s], [weights and work split updated every 10 cycles]
)

== H. Accuracy and timing evidence

#table(
  columns: 5,
  [*Question*], [*Method*], [*Accuracy result*], [*Timing result*], [*Status*],
  [Local Ding map], [GL4-by-5 local reconstruction], [max state error $1.33 times 10^(-16)$; selected sensitivity error $3.24 times 10^(-12)$], [not an OCP benchmark], [discrete-map check],
  [Local Ding map], [Radau IIA five-stage reconstruction], [max state error $3.61 times 10^(-15)$; selected sensitivity error $9.48 times 10^(-12)$], [not an OCP benchmark], [discrete-map check],
  [IPOPT local Ding], [50 propagated Radau-5 RHO windows], [same certified propagated windows], [median 0.798 to 0.551 s (-30.9%)], [promising],
  [Acados local Ding], [first 75 hot windows], [76 valid cycles; stopped at 77], [median 0.455 to 0.167 s (-63.3%)], [not a 100-cycle claim],
  [Cross-solver replay], [common DOP853 replay], [cost 0.01383435 IPOPT; 0.01384597 Acados (+0.084%)], [$epsilon_"omega"$: 0.00515 and 0.00772 rad/s], [qualified ranking]
)

=== H.1 Cumulative 30 Hz ablation with IPOPT MA57

The following campaign is a cumulative rather than factorial ablation: each row retains the preceding changes. Every listed IPOPT result used one-cycle RHO windows, reduced dynamic mechanics, 30 uniformly spaced stimulations per crank revolution, Radau IIA degree five, MA57, and 100 requested windows. A result is called *validated* only when all windows converged, were feasible below $10^(-5)$, and passed the final reduced-mechanics audit. Cold construction, compilation, and audit time are deliberately not folded into the hot percentiles; the latter describe cycles 2--100.

#table(
  columns: 6,
  [*Case*], [*Incremental change*], [*Validated cycles*], [*Hot median (s)*], [*Hot P90 (s)*], [*Interpretation*],
  [K0], [historical full mechanics, MX, Radau-3], [0 / 2], [51.399 (2 windows)], [51.399], [numerically solved, but no physically valid cycle],
  [K1], [full mechanics, MX, Radau-5], [0 / 100], [29.739], [43.501], [100 NLP solves; absolute crank-progress audit failed],
  [K2], [corrected generalized wheel torque], [0 / 100], [27.562], [34.947], [100 NLP solves; absolute crank-progress audit failed],
  [K3], [reduced mechanics], [100 / 100], [5.943], [6.918], [mechanical reduction alone does not make the MX graph fast],
  [K4], [periodic-node calcium], [100 / 100], [5.740], [6.584], [small timing change before symbolic compilation],
  [K5], [SX graph], [100 / 100], [0.678], [0.798], [dominant observed acceleration],
  [K6], [compiled exact Lagrangian Hessian callback], [100 / 100], [0.449], [0.521], [fastest validated 30 Hz configuration],
  [K7], [compact RHO output], [100 / 100], [0.465], [0.563], [small runtime cost for bounded output retention],
  [K8], [hard 100 microsecond $Delta P_"W"$ bound], [100 / 100], [0.920], [1.151], [regularity constraint increases solve effort],
  [K9], [normalized $Delta P_"W"$ penalty, weight 0.01], [100 / 100], [0.789], [0.892], [best regularized configuration in this campaign]
)

The main conclusion is not that speedup factors should be multiplied. K3--K9 alter both the symbolic graph and the constrained NLP. Nevertheless, the ordered experiment identifies SX as the largest timing change in this implementation and shows that the compiled Hessian is beneficial after the symbolic graph has been reduced. The compact-output option has negligible hot-time effect relative to K6. The hard slew bound increases the P90 above one second; adding the normalized penalty improves both the median and P90 while retaining the bound.

=== H.2 Isokinetic replication and dynamic-consistency records

The following independent isokinetic replication starts at K5 because reduced mechanics, periodic-node Ding calcium, Radau IIA degree five, and SX are already required by that branch. The prescribed cadence was $omega=-2 pi$ rad/s and the target work was equivalent to a mean torque of 0.2 N m, i.e. $E_"prod"(T)=1.256637$ J per turn. Every row solved 100 one-cycle IPOPT/MA57 windows on CPU 12--15 with single-thread numerical libraries. All 100 windows in every row converged, had primal infeasibility at most $10^(-5)$, and passed the dense isokinetic speed/load audit.

#table(
  columns: 7,
  [*Case*], [*Change from K5*], [*Valid*], [*Hot median / P90 (s)*], [*Solver sum (s)*], [*RHO solve-loop wall (s)*], [*Executed fatigue; min A*],
  [K5], [SX reference], [100 / 100], [0.804 / 0.998], [85.205], [99.121], [1502.703; 0.940],
  [K6], [compiled exact $"nlp"_"hess_l"$], [100 / 100], [0.644 / 0.779], [68.545], [219.117], [1502.703; 0.940],
  [K7], [compact output], [100 / 100], [0.639 / 0.781], [68.190], [217.812], [1502.703; 0.940],
  [K8], [hard 100 microsecond $Delta P_"W"$ lifting bound], [100 / 100], [0.719 / 0.814], [74.534], [222.063], [4164.343; 0.911],
  [K9], [K8 plus normalized $Delta P_"W"$ penalty 0.01], [100 / 100], [0.693 / 0.829], [73.407], [223.673], [3167.562; 0.917],
)

The hot K6 median is 19.9 percent below K5, while K7 is statistically indistinguishable from K6 on this hardware. The column *RHO solve-loop wall* includes the one-off graph construction and C compilation before and within the RHO loop, but excludes the separate post-solve export/audit stage; it is not an end-to-end latency nor a per-cycle control latency. For K6--K9 it is about 218--224 s because each case used a distinct compilation cache, whereas K5 was 99.1 s without that generated-Hessian compilation. The exact cold compilation component is consequently not inferred by subtracting two different NLPs. K8 and K9 are not speed-only variants: their hard slew constraint changes the admissible controls, raising the executed fatigue and reducing the terminal capacity compared with K5--K7.

Dynamic consistency was requested as an observed quantity rather than a rejection gate. A two-cycle K5 DOP853 smoke replay gave a maximum optimized-trajectory versus DOP853 state discrepancy of 0.155110, while the independently constructed RK4 map versus DOP853 discrepancy was $1.63 times 10^(-4)$. The dense isokinetic speed audit itself had zero bound violation. A 100-cycle DOP853 post-processing attempt completed the 100 native NLP windows but terminated before writing its replay result; it is recorded as a post-processing failure, not as a DOP853 validation or as an NLP failure. K5--K9 native archives and their short DOP853 smoke record are retained under `local-results/isokinetic-k5-k9-dynamic-consistency-20260922`; a bounded, milestone-only DOP853 replay is required before making a cross-formulation cost ranking.

=== H.3 Radau s=3 at 50 Hz versus Radau s=5 at 30 Hz

This comparison is deliberately *not factorial*. The 50 Hz case is the historical full-mechanics, MX, finite-history Ding baseline; K5 at 30 Hz uses reduced mechanics, periodic-node Ding calcium, and SX. It answers whether the historical configuration remains dynamically viable at its native 50 Hz setting, not the isolated effect of changing collocation order.

#table(
  columns: 6,
  [*Case*], [*Native timing evidence*], [*Certified cycles*], [*Trajectory--DOP853 discrepancy*], [*RK4--DOP853 discrepancy*], [*Conclusion*],
  [Radau IIA s=3, 50 Hz, full/MX], [51.393 s hot median in the historical 2-window run], [0 / 2], [max 0.487 in the new map-audit smoke], [max $7.75 times 10^(-5)$], [not dynamically viable under the current physical gate],
  [Radau IIA s=5, 30 Hz, K5], [0.804 / 0.998 s hot median / P90 over 100 cycles], [100 / 100], [max 0.155 in a 2-window smoke], [$1.63 times 10^(-4)$], [numerically and mechanically viable; 100-cycle DOP853 replay remains pending]
)

The new 50 Hz Radau s=3 smoke attempted two windows but stopped after the first physical failure, so it provides no valid new hot percentile. Its independent map audit gave trajectory--DOP853 discrepancies from 0.362 to 0.487, whereas the RK4--DOP853 discrepancy remained below $7.8 times 10^(-5)$. Thus the large mismatch is attributable to the collocated trajectory/initial-state contract rather than insufficient DOP853 accuracy. The historical 51.393 s timing is retained only as a failed-baseline cost; it must not be ranked against K5 as an integrator-only speed ratio.

=== H.4 Four-muscle pulse-width patterns

The control archives from the five validated 30 Hz IPOPT/MA57 runs K5--K9 were retained. The following plot extracts the *applied* pulse widths for the four muscles, without interpolation, at cycles 10, 55, and 100. Each column contains the 30 evenly spaced stimulations of one crank revolution. K5--K7 differ principally in numerical representation and output retention, whereas K8 introduces the hard increment bound and K9 retains that bound with the normalized increment penalty. Thus the plot is a reproducible descriptive comparison of returned controls, not evidence that all five problems have the same admissible set or DOP853 cost.

#figure(
  image("figures/pulse_width_patterns_k5_k9.png", width: 100%),
  caption: [Applied four-muscle pulse-width patterns at cycles 10, 55, and 100 for the validated 30 Hz K5--K9 IPOPT/MA57 archives. Values are the archived applied pulse widths in microseconds; the sample index is the stimulation within the one-second crank cycle.]
)

The source data and plotting rules are recorded in `scripts/plot_pulse_width_patterns.py`. It fails when a trajectory does not retain all requested cycles or the expected four applied-pulse-width channels, preventing a partial archive from being compared silently.

K0 degree-three full mechanics was not run as a 30 Hz / 100-cycle case because the current executable refuses that transcription after a prior independent DOP853 disagreement. This is a software safety gate, not a statement that its discrepancy should be hidden: an audit-only override and a milestone DOP853 replay are required to report its 30 Hz dynamic discrepancy explicitly. Its only available measurement is a two-window 50 Hz smoke run: both NLPs solved, neither was physically validated, and its single hot timing was 51.399 s (end-to-end 163.184 s). K1 (full mechanics with degree-five Radau) and K2 (corrected generalized external torque) each completed 100 numerical NLP solves at 30 Hz, but no cycle passed the absolute crank-progress validation. Their hot median / P90 times were 29.739 / 43.501 s and 27.562 / 34.947 s, respectively (end-to-end 3447.233 s and 3084.242 s). These figures document convergence effort and the audit failure; they are not efficacy comparisons. The issue is formulation/audit validity, not evidence that a solver is intrinsically slower.

=== H.5 Frequency, calcium periodicity, and solver checkpoints

For a one-second crank cycle with $N_s$ evenly spaced pulses, the periodic-node model uses $h=1/N_s$ and recomputes the calcium decay, post-pulse amplitude, and periodic fixed point at every tested frequency:

$d=exp(-h/tau_c), quad C_n^*=d H_0(h) (h/tau_c)/(1-d).$

Thus 33, 35, 40, and 50 Hz are fixed-frequency periodic-calcium experiments, not reuses of a 30 Hz calcium trace. The model represents the steady periodic regime after warm-up. It does not represent the transient immediately after changing frequency. A true 30-to-35 Hz continuation must phase-resample the controls and initialize calcium with the fixed point for the new $h$, or explicitly simulate the transition.

#table(
  columns: 6,
  [*Frequency and case*], [*Solver*], [*Validated cycles*], [*Hot median (s)*], [*Hot P90 (s)*], [*Status*],
  [30 Hz K5], [MadNLP 0.10.1 + MA57], [100 / 100], [0.925], [1.193], [valid native comparator],
  [33 Hz K7], [IPOPT + MA57], [100 / 100], [0.558], [0.681], [valid; below one second],
  [35 Hz K5], [IPOPT + MA57], [100 / 100], [0.875], [1.451], [valid; tail exceeds one second],
  [35 Hz K5], [MadNLP 0.10.1 + MA57], [100 / 100], [1.148], [1.399], [valid but not real-time],
  [40 Hz K5], [IPOPT + MA57], [100 / 100], [1.043], [1.319], [valid; beyond one-second median],
  [40 Hz K5], [MadNLP 0.10.1 + MA57], [0 / 100], [--], [--], [native continuity tolerance failure],
  [50 Hz K10], [IPOPT + MA57], [100 / 100], [1.488], [1.716], [robustness point, not real-time]
)

The 33 Hz K7 result is the present conservative high-frequency reference: it preserves the periodic-calcium contract, passes the mechanical audit, and remains below one second at the P90. The 50 Hz K10 result demonstrates numerical robustness under the regularized pulse-width policy, but it is not suitable for a one-second controller budget. The original IPOPT/MUMPS runs did not produce a native valid 100-cycle trajectory under an otherwise identical K3--K5 protocol; consequently, no MA57-versus-MUMPS speed ratio is reported. The new MadNLP stack uses MA57 through its compiled interface. It is a useful independent NLP implementation, but it is not faster than IPOPT/MA57 at the tested 30--35 Hz checkpoints.

=== H.4 Acados and cross-solver interpretation

Acados is not assigned an IPOPT-equivalent internal objective or timing interpretation. Its reduced-dynamics IRK route can complete 100 windows, but the retained experiment has a velocity excursion between shooting nodes. This is a transcription-specific violation that is accepted only as a diagnostic status, never as an IPOPT-equivalent physical certificate. The local Ding reduction remains encouraging for Acados because the discrete-map reconstruction checks are near machine precision, but the 76-cycle stopping point prevents a 100-cycle performance claim.

For any final solver ranking, replay selected certified solutions with DOP853 at 30, 33, 35, and 40 Hz, using fixed controls and the same physiological states. Report the replayed objective, maximum speed excursion, phase/work error, and pulse-width feasibility separately from each solver's internal objective. This is particularly important for Acados and at high stimulation frequency, where an inter-node error can remain invisible to a node-only constraint check.

=== H.5 Sequential validation policy

An exhaustive product of frequencies, solvers, graph options, and regularizers is neither necessary nor informative. The efficient policy is: first identify a certified IPOPT/MA57 reference at a frequency; next test the next frequency only when the reference is feasible; then test an independent solver at 30 Hz, at the last IPOPT point whose P90 is below one second, and at the first point beyond that threshold. MUMPS is evaluated only as a controlled linear-solver A/B within IPOPT. Acados is evaluated at matched mechanical/calcium contracts and is retained only after its DOP853 audit passes. This staged design preserves falsifiability while restricting expensive 100-cycle experiments to decision-relevant boundaries.

== I. Runtime reductions outside the nonlinear optimization

The timing reported by an NLP solver is not the total control latency. The following engineering changes reduce work outside the mathematical optimization and must be measured separately from iteration time:

#table(
  columns: 3,
  [*Change*], [*What it removes*], [*Scientific safeguard*],
  [Offline reduced-profile construction and cache], [repeated full Biorbd/contact evaluation during each NLP call], [regenerate and audit the profile after any biomechanical-model change],
  [Persistent RHO workers], [rebuilding Python/CasADi/IPOPT objects every cycle], [retain the native transfer and certify every window],
  [Process isolation], [unsafe shared IPOPT/MA57/CasADi threaded state], [IPC barrier before an adaptive block is applied],
  [Compact RHO output and bounded history], [retention of unnecessary solution/model graphs], [preserve per-window audit fields and terminal state],
  [Warm starts and transferred bounds], [cold-start iterations], [same seed and transfer rule in all solver comparisons],
  [Blockwise weight updates], [cost reconstruction every cycle], [update only after both arms are certified; report boundary overhead]
)

The 0.3 N m paced two-arm campaign had a median pair wall time of 1.447 s, P90 2.412 s, and P95 5.673 s. The upper tail is dominated by the ten-cycle boundaries where objectives are rewritten. This overhead is part of the real controller budget and should never be hidden by quoting only IPOPT's internal solve time.

== J. External wheel moments and generalized torque convention

The historical load is a global external moment $M=e_z tau_e$ applied to the `wheel` segment. Its virtual work is not, in general, represented by placing $tau_e$ in one relative joint coordinate. The mechanically equivalent generalized load is

$Q_e(q)=J_(omega, "wheel")(q)^T e_z tau_e,$

where $J_(omega, "wheel")$ maps generalized velocity to the wheel angular velocity in the global frame. In the unilateral Wu model the wheel is downstream of shoulder, elbow, and crank rotation; the measured projection is approximately $[1,1,1]^T tau_e$. Thus the external resistance loads all three coordinates. It is physically appropriate for a brake or environmental resistance acting on the wheel.

By contrast, a torque applied internally at the crank bearing is a distinct model and can act only on the relative crank coordinate. Treating these two loads as equivalent changes the mechanical RHS and is not a solver speedup. The reduced equation must use $eta_e(theta)=q'(theta)^T J_(omega,"wheel")^T e_z$, not merely the tangent component of the crank coordinate. The projection is generated from the biomechanical model, verified for constancy, and cached profiles are versioned anew after this correction. A 100-state direct-dynamics audit of the corrected full representations gave a maximum mismatch of $1.14 times 10^(-13)$; this audit is required before comparing solver timings.

== K. References

[1] J. Ding, A. S. Wexler, and S. A. Binder-Macleod. “A predictive model of fatigue in human skeletal muscles.” *Journal of Applied Physiology*, 89(4):1322--1332, 2000. doi:10.1152/jappl.2000.89.4.1322.

[2] J. Ding, A. S. Wexler, and S. A. Binder-Macleod. “A mathematical model that predicts the force-frequency relationship of human skeletal muscle.” *Muscle & Nerve*, 26(4):477--485, 2002. doi:10.1002/mus.10198.

[3] J. Ding, A. S. Wexler, and S. A. Binder-Macleod. “Mathematical models for fatigue minimization during functional electrical stimulation.” *Journal of Electromyography and Kinesiology*, 13(6):575--588, 2003. doi:10.1016/S1050-6411(03)00102-0.

[4] E. Hairer and G. Wanner. *Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems*, 2nd ed. Springer, 1996. doi:10.1007/978-3-642-05221-7.

[5] E. Hairer and G. Wanner. “Stiff differential equations solved by Radau methods.” *Journal of Computational and Applied Mathematics*, 111(1--2):93--111, 1999. doi:10.1016/S0377-0427(99)00134-X.

[6] F. De Groote, A. L. Kinney, A. V. Rao, and B. J. Fregly. “Evaluation of direct collocation optimal control problem formulations for solving the muscle redundancy problem.” *Annals of Biomedical Engineering*, 44(10):2922--2936, 2016. doi:10.1007/s10439-016-1591-9.

[7] K. Co, P. Puchaud, F. Moissenet, and M. Begon. “Maximizing Task Endurance through Muscle Fatigue Minimization with Consideration of Muscle Fatigability and Task Contribution: an in-Silico FES Study of Handcycling.” Manuscript, 2026.
