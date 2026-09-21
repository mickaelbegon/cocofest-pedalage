from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = "docs/scientific_appendix/appendix_acceleration_ding.docx"


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:fill"), fill); tc_pr.append(shd)


def border(cell):
    tc_pr = cell._tc.get_or_add_tcPr(); borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        line = OxmlElement(f"w:{edge}"); line.set(qn("w:val"), "single")
        line.set(qn("w:sz"), "4"); line.set(qn("w:color"), "D9D9D9"); borders.append(line)
    tc_pr.append(borders)


def paragraph(doc, text="", style=None):
    result = doc.add_paragraph(style=style); result.add_run(text)
    result.paragraph_format.space_after = Pt(6); result.paragraph_format.line_spacing = 1.12
    return result


def equation(doc, text):
    result = doc.add_paragraph(); result.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = result.add_run(text); run.font.name = "Cambria Math"
    run._element.rPr.rFonts.set(qn("w:hAnsi"), "Cambria Math"); run.font.size = Pt(10.5)
    result.paragraph_format.space_after = Pt(8)


doc = Document(); section = doc.sections[0]
section.top_margin = Inches(.75); section.bottom_margin = Inches(.72)
section.left_margin = Inches(.78); section.right_margin = Inches(.78)
styles = doc.styles; styles["Normal"].font.name = "Aptos"; styles["Normal"].font.size = Pt(10)
for name, size in (("Title", 17), ("Heading 1", 13.5), ("Heading 2", 11.5)):
    styles[name].font.name = "Aptos"; styles[name].font.size = Pt(size); styles[name].font.color.rgb = RGBColor(0, 0, 0)

title = paragraph(doc, "Technical appendix Computational acceleration for FES cycling with the Ding model", "Title")
title.alignment = WD_ALIGN_PARAGRAPH.CENTER
properties = title._p.get_or_add_pPr(); existing = properties.find(qn("w:pBdr"))
if existing is not None: properties.remove(existing)
title_border = OxmlElement("w:pBdr"); bottom = OxmlElement("w:bottom"); bottom.set(qn("w:val"), "nil")
title_border.append(bottom); properties.append(title_border)
subtitle = paragraph(doc, "Mathematical construction of reduced mechanics and local Ding-state reduction")
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
paragraph(doc, "This appendix explains the two main model reductions used for FES cycling: projection of the constrained multibody model onto the crank manifold, and local reconstruction of linear Ding states. It distinguishes an exact identity of a selected discrete transcription from continuous-model equivalence and from observed solver speed. Measurement-corrected NMPC remains necessary for use with a participant.")

paragraph(doc, "1. Historical baseline and notation", "Heading 1")
paragraph(doc, "The historical reference attributed to Kevin is commit e9a211ed2eb1af30683a324159a6fcd6e899fc05, dated 21 November 2025. It already includes RHO NMPC with primal warm start, IPOPT with MA57 on Linux, degree-3 Radau collocation, a fatigable Ding model, and force-length, force-velocity, and passive-force relationships. The main configuration is 30 Hz, a two-cycle window, a one-cycle shift, and five requested cycles. These are baseline elements, not later innovations.")
paragraph(doc, "For M muscles, the full state contains five Ding states per muscle, x = (Cn, F, A, Tau1, Km), and the mechanical generalized coordinates q and qdot. The physical crank coordinate is theta and omega = theta-dot. Pulse width is denoted u.")

paragraph(doc, "2. Starting point The Ding force fatigue system", "Heading 1")
paragraph(doc, "The five-state phenomenological model follows the Ding family of force and fatigue models [1-3]. Define the pulse-width recruitment and calcium sensitivity terms:")
equation(doc, "ρ(u) = 1 − exp(−(u − u₀)/uₜ),     σ(C,K) = C/(K + C)")
paragraph(doc, "Let g(theta, omega) collect the active force-length, force-velocity, and passive-force factors. The implementation used in the OCP is:")
equation(doc, "Ċ = (H(t) − C)/τc")
equation(doc, "Ḟ = g(θ,ω)[A ρ(u) σ(C,K) − F/(T + τ₂ σ(C,K))]")
equation(doc, "Ȧ = −(A − Ar)/τf + αA F,     Ṫ = −(T − Tr)/τf + αT F,     K̇ = −(K − Kr)/τf + αK F")
paragraph(doc, "The force equation remains nonlinear and coupled to mechanics. The A, Tau1, and Km equations, however, share the same recovery operator and the same forcing F. This exact structural property is what enables local state reduction; it does not make the force dynamics linear.")

paragraph(doc, "3. Reduced crank mechanics", "Heading 1")
paragraph(doc, "3.1 Contact manifold and tangent projection", "Heading 2")
paragraph(doc, "The full model has generalized coordinates q and hand-to-crank contact constraints. Offline, the physical contact problem is solved over one crank revolution, yielding a periodic contact-consistent configuration curve q = Q(theta). The reduced model therefore parameterizes the admissible contact manifold by the physical crank angle rather than fitting an arbitrary torque curve.")
equation(doc, "q = Q(θ),     q̇ = Q′(θ)ω,     q̈ = Q′(θ)ω̇ + Q″(θ)ω²")
paragraph(doc, "Starting from M(q)q-double-dot + h(q,q-dot) = tau_m + tau_ext and premultiplying by Q-prime-transpose defines the effective inertia I(theta), gravitational term G(theta), quadratic velocity term V(theta), muscle effectiveness eta_m(theta), and external-torque effectiveness eta_e(theta). The online dynamics are:")
equation(doc, "θ̇ = ω")
equation(doc, "I(θ)ω̇ = Σm ηm(θ)Fm + ηe(θ)τext − G(θ) − V(θ)ω²")
paragraph(doc, "This is a Lagrange/tangent projection of the constrained multibody equations. It removes redundant generalized coordinates while retaining the crank coordinate and speed. It is valid on the sampled contact branch and does not model departures from it.")

paragraph(doc, "3.2 Construction of the reduced numerical profile", "Heading 2")
paragraph(doc, "At sampled crank angles, the implementation evaluates the Biorbd mass matrix, nonlinear effects, muscle-length Jacobian, and Q, Q-prime, and Q-double-prime. It computes I = Q-prime-transpose M Q-prime, eta_m = minus J-length,m Q-prime, and eta_e = Q-prime,r. It also obtains normalized fibre length and fibre velocity per unit crank speed. Smooth periodic Fourier series are fitted to I, G, V, the effectiveness coefficients, and muscle geometry. Online dynamics are therefore small symbolic functions of theta, omega, and muscle forces instead of a full contact and multibody solve.")
equation(doc, "l̃m = lm(θ),     ṽm = νm(θ)ω/10")
paragraph(doc, "The original Hill force-length, force-velocity, and passive laws are evaluated from these fitted geometry profiles. The reduced profile must be regenerated and audited whenever the source biomechanical model, contact geometry, selected muscles, or Fourier order changes. Independent samples must compare projected acceleration and muscle geometry against the full constrained model.")

paragraph(doc, "3.3 Measured effect and scope", "Heading 2")
paragraph(doc, "For four muscles, the full forward model has 20 Ding states plus six mechanical states. Dynamic reduced mechanics keeps the 20 Ding states and theta, omega, reducing 26 states to 22. In the controlled 100-RHO IPOPT/Radau-3 campaign, hot median time changed from 4.595 to 1.031 s and total wall time from 615.4 to 260.9 s; the fatigue difference was 0.097 percent. These are campaign-specific results and must not be multiplied with gains from other modifications. Reduced mechanics is a model transformation, not merely a code optimization.")

paragraph(doc, "4. Local reduction of Ding states", "Heading 1")
paragraph(doc, "4.1 Continuous identities", "Heading 2")
paragraph(doc, "For z in {A, T, K}, with rest value zr and fatigue coefficient alpha-z, an integrating factor gives:")
equation(doc, "z(tk+s) = zr + exp(−s/τf)(zk − zr) + αz ∫₀ˢ exp(−(s−v)/τf)F(tk+v)dv")
paragraph(doc, "All three states share the same filtered-force convolution. If J(s) = integral from 0 to s of exp(v/tau-f) F(tk+v) dv, then z(tk+s) = zr + exp(−s/tau-f)[zk − zr + alpha-z J(s)]. This permits a continuous formulation using F and J, with A, Tau1, and Km reconstructed.")
paragraph(doc, "A more convenient local formulation keeps F and A. When alpha-A is nonzero, set beta-T = alpha-T/alpha-A, beta-K = alpha-K/alpha-A, dT = (T−Tr)−beta-T(A−Ar), and dK = (K−Kr)−beta-K(A−Ar). The force terms cancel exactly:")
equation(doc, "ḋT = −dT/τf,     ḋK = −dK/τf")
equation(doc, "T = Tr + βT(A − Ar) + dT,k exp(−s/τf),     K = Kr + βK(A − Ar) + dK,k exp(−s/τf)")
paragraph(doc, "For the periodic-node calcium input H(tk+s) = Hk exp(−s/tau-c), calcium is also reconstructible on an interval:")
equation(doc, "C(tk+s) = exp(−s/τc)[Ck + Hk s/τc]")
paragraph(doc, "A stimulation changes the forcing amplitude but not the calcium state. The reconstruction must use the real incoming RHO state, including nonzero fatigue offsets; replacing it by a periodic fixed point changes the physical initial condition.")

paragraph(doc, "4.2 Preserve the discrete transcription", "Heading 2")
paragraph(doc, "Continuous identities do not reproduce a finite-step Radau or Gauss-Legendre map exactly. For a Runge-Kutta tableau (A,b,c) and step h, preserve the OCP by eliminating only the linear equations after discretization. For five-stage Radau IIA:")
equation(doc, "C⃗ = (I + h A/τc)⁻¹[1⃗ Ck + (h/τc)A H⃗]")
equation(doc, "z⃗ = zr1⃗ + (I + h A/τf)⁻¹[1⃗(zk − zr) + h αz A F⃗]")
paragraph(doc, "After this substitution, the nonlinear muscle residual contains force-stage variables, while calcium and fatigue stages are reconstructed. This is algebraically identical to the full Radau discrete equations up to linear-solve and NLP tolerances. Radau IIA is stiffly accurate, so the last stage is the endpoint [4,5]. A five-state, five-stage muscle has 25 stage variables; this local condensation leaves five nonlinear force stages, while the physical node states may be retained for RHO transfer and constraints.")
paragraph(doc, "For Acados IRK Gauss-Legendre 4x5, eliminated C, dT, and dK profiles must be built with the Gauss-Legendre tableau, substep by substep. The endpoint is not the final stage and must be evaluated as:")
equation(doc, "vk+1 = vk + h bᵀ[r(tk + hc) − V⃗/τ]")
paragraph(doc, "Reusing Radau profiles in Gauss-Legendre would change the discrete problem.")

paragraph(doc, "4.3 Dimension, constraints, and derivatives", "Heading 2")
paragraph(doc, "With four muscles and reduced mechanics, retaining only F and A gives 10 physical states instead of 22; C, Tau1, and Km are reconstructed locally. This differs from Radau-stage condensation, which can keep all physical states at nodes while eliminating internal stage variables. The 10-state count requires fixed stimulation scheduling, calcium parameters, and incoming state in a window.")
paragraph(doc, "Every cost or constraint using a removed variable must be evaluated on its reconstruction at the same nodes or stages as in the full problem. For scaled variables, a relation y-bar = beta A-bar + delta converts bounds only after correct scaling and sign handling. The physical domain K+C > 0 and T + tau-2 sigma(C,K) > 0 remains mandatory. Reconstructed expressions must remain in the automatic-differentiation graph; treating a state-dependent profile as constant gives incorrect sensitivities. Direct-collocation formulations are known to change both robustness and computational cost [6].")

paragraph(doc, "4.4 Evidence and limitations", "Heading 2")
paragraph(doc, "On isolated, converged maps, the local ACADOS proof produced maximum reconstruction errors of 1.33e−16 for GL4x5 and 3.61e−15 for Radau IIA five-stage; selected sensitivity errors were 3.24e−12 and 9.48e−12. These establish map equivalence, not OCP performance. In the IPOPT/Radau-5 reduced RHO A/B, both variants completed 50 propagated windows; hot median time changed from 0.798 to 0.551 s (−30.9 percent). For ACADOS, the experimental local formulation completed 76 valid cycles before stopping at cycle 77; over the first 75 hot cycles the median changed from 0.455 to 0.167 s (−63.3 percent). This is promising but not a 100/100 robustness claim.")

paragraph(doc, "5. Solver comparison and benchmark protocol", "Heading 1")
paragraph(doc, "IPOPT/MA57 with five-stage Radau IIA is the fidelity reference. Acados uses multiple shooting with an IRK Gauss-Legendre 4x5 integrator. Common incoming state, seed, Ding parameters, mechanical profile, load, objective, and bounds are prerequisites to comparison. Both control sequences must be replayed with DOP853 under fixed controls. Define the continuous speed excursion:")
equation(doc, "εω = maxₜ max(0, ωmin − ω(t), ω(t) − ωmax)")
paragraph(doc, "The controlled pair had DOP853 costs of 0.01295477 for IPOPT and 0.01296194 for Acados, but both had small between-node excursions. With a common tightened fast margin, costs were 0.01383435 and 0.01384597 (Acados +0.084 percent), with excursions 0.00515 and 0.00772 rad/s. Strict ranking requires zero excursion; similar small excursions support only a qualified numerical comparison.")
paragraph(doc, "Every ablation should report build and compilation time, hot-solve time, wall time, iterations, valid-cycle fraction, common DOP853 cost, epsilon-omega, phase error, pulse-width bounds, KKT dimensions/nonzeros, and control distance. RHO-FHO tests require equal horizons and separate per-cycle and total times. Individual speedup factors must not be multiplied.")

paragraph(doc, "6. References", "Heading 1")
for reference in [
    "[1] Ding J, Wexler AS, Binder-Macleod SA. A predictive model of fatigue in human skeletal muscles. Journal of Applied Physiology. 2000;89(4):1322-1332. doi:10.1152/jappl.2000.89.4.1322.",
    "[2] Ding J, Wexler AS, Binder-Macleod SA. A mathematical model that predicts the force-frequency relationship of human skeletal muscle. Muscle & Nerve. 2002;26(4):477-485. doi:10.1002/mus.10198.",
    "[3] Ding J, Wexler AS, Binder-Macleod SA. Mathematical models for fatigue minimization during functional electrical stimulation. Journal of Electromyography and Kinesiology. 2003;13(6):575-588. doi:10.1016/S1050-6411(03)00102-0.",
    "[4] Hairer E, Wanner G. Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems. 2nd ed. Springer; 1996. doi:10.1007/978-3-642-05221-7.",
    "[5] Hairer E, Wanner G. Stiff differential equations solved by Radau methods. Journal of Computational and Applied Mathematics. 1999;111(1-2):93-111. doi:10.1016/S0377-0427(99)00134-X.",
    "[6] De Groote F, Kinney AL, Rao AV, Fregly BJ. Evaluation of direct collocation optimal control problem formulations for solving the muscle redundancy problem. Annals of Biomedical Engineering. 2016;44(10):2922-2936. doi:10.1007/s10439-016-1591-9.",
]:
    paragraph(doc, reference)

doc.save(OUT)
