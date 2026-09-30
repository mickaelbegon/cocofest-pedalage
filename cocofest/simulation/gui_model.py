"""Headless form conversion and scientific summaries for the Tk application."""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping

from .config import SimulationConfig
from .capabilities import CapabilityRegistry


@dataclass(frozen=True)
class FieldSpec:
    name: str
    label: str
    group: str
    kind: str = "text"


FORM_FIELDS = (
    FieldSpec("mode", "Stratégie", "Problème", "choice"),
    FieldSpec("solver", "Solveur", "Problème", "choice"),
    FieldSpec("mechanics", "Mécanique", "Problème", "choice"),
    FieldSpec("bilateral_reduced", "Deux bras combinés : manivelle mécanique unique", "Problème", "bool"),
    FieldSpec("formulation", "Dynamique (dynamic = cadence libre)", "Problème", "choice"),
    FieldSpec("isokinetic_kinematics", "Cinématique isocinétique (prescribed = θ/ω éliminés)", "Isocinétique", "choice"),
    FieldSpec("cycles", "Cycles exécutés", "Problème"),
    FieldSpec("cycles_per_window", "Cycles par fenêtre RHO", "Problème"),
    FieldSpec("stimulations_per_cycle", "Contrôles / stimulations par cycle", "Problème"),
    FieldSpec("signed_crank_torque", "Couple signé (N.m)", "Problème"),
    FieldSpec("pulse_width_max_step_us", "ΔPW entre contrôles successifs (µs, vide = aucun)", "Contraintes avancées"),
    FieldSpec("pulse_width_slew_formulation", "Formulation ΔPW (direct = sans états ajoutés)", "Contraintes avancées", "choice"),
    FieldSpec("pulse_width_slew_weight", "Poids quadratique ΔPW intra-fenêtre (0 = désactivé)", "Contraintes avancées"),
    FieldSpec("pulse_width_slew_reference_us", "Référence de normalisation ΔPW (µs)", "Contraintes avancées"),
    FieldSpec(
        "reduced_internal_crank_velocity_guard",
        "Guard vitesse interne réduite",
        "Contraintes avancées",
        "choice",
    ),
    FieldSpec(
        "reduced_terminal_half_step_velocity_guard",
        "Cadence au demi-pas suivant (prédiction terminale RHO)",
        "Contraintes avancées",
        "bool",
    ),
    FieldSpec("terminal_q_slack", "Tolérance angle terminal (rad)", "Contraintes avancées"),
    FieldSpec("integration", "Transcription / intégration", "Solveur", "choice"),
    FieldSpec("collocation_degree", "Degré Radau / IRK IPOPT", "Solveur"),
    FieldSpec("ipopt_enforce_start_constraints", "IPOPT : contraintes historiques au nœud initial", "Solveur", "bool"),
    FieldSpec("ipopt_linear_solver", "Solveur linéaire IPOPT / warmup", "Solveur", "choice"),
    FieldSpec("madnlp_linear_solver", "Solveur linéaire MadNLP", "Solveur", "choice"),
    FieldSpec("acados_qp_solver", "Backend QP ACADOS", "Solveur", "choice"),
    FieldSpec("acados_sim_stages", "Étages IRK ACADOS", "Solveur"),
    FieldSpec("acados_sim_steps", "Sous-pas IRK ACADOS", "Solveur"),
    FieldSpec(
        "ipopt_ding_local_reduction",
        "IPOPT : reconstruire Cn/Tau1/Km Ding (expérimental)",
        "Solveur",
        "bool",
    ),
    FieldSpec(
        "acados_ding_local_reduction",
        "ACADOS : reconstruire Cn/Tau1/Km Ding (expérimental)",
        "Solveur",
        "bool",
    ),
    FieldSpec("compile_evaluators", "Compiler les évaluateurs", "Solveur", "bool"),
    FieldSpec("compile_hessian_only", "Compiler seulement la Hessienne IPOPT", "Solveur", "bool"),
    FieldSpec("compact_rho_output", "Sortie RHO compacte", "Exécution", "bool"),
    FieldSpec("madnlp_recovery", "Recovery IPOPT pour MadNLP", "Solveur", "bool"),
    FieldSpec("madnlp_hot_max_iterations", "MadNLP : itérations chaudes max.", "Solveur"),
    FieldSpec("madnlp_hot_max_wall_time", "MadNLP : durée chaude max. (s)", "Solveur"),
    FieldSpec("threads", "Threads solveur", "Exécution"),
    FieldSpec("numeric_threads", "Threads bibliothèques numériques", "Exécution"),
    FieldSpec("output_root", "Dossier de sortie", "Exécution", "directory"),
    FieldSpec("acados_ipopt_cycle1_seed", "ACADOS : seed IPOPT exact du cycle 1", "Exécution", "file"),
    FieldSpec("common_initial_solution", "Seed commun certifié (.npz, IPOPT / MadNLP)", "Exécution", "file"),
    FieldSpec("model_config", "Configuration musculaire JSON (optionnelle)", "Exécution", "file"),
    FieldSpec("weights_config", "Poids Physio / PACE et provenance (JSON)", "Exécution", "file"),
    FieldSpec("dry_run", "Prévisualisation seulement", "Exécution", "bool"),
    FieldSpec("isokinetic_omega", "Vitesse imposée (rad/s, négative)", "Isocinétique"),
    FieldSpec("energy_equivalent_torque", "Couple équivalent énergétique (N.m)", "Isocinétique"),
    FieldSpec("load_torque_min", "Couple de charge minimum (N.m)", "Isocinétique"),
    FieldSpec("load_torque_max", "Couple de charge maximum (N.m)", "Isocinétique"),
    FieldSpec("extra_arguments", "Arguments avancés : tableau JSON", "Avancé", "json"),
)


def form_values(config: SimulationConfig) -> dict[str, str | bool]:
    result = {}
    for spec in FORM_FIELDS:
        value = getattr(config, spec.name)
        if spec.kind == "bool":
            result[spec.name] = value
        elif spec.kind == "json":
            result[spec.name] = json.dumps(value, ensure_ascii=False)
        else:
            result[spec.name] = "" if value is None else str(value)
    return result


def config_from_form(values: Mapping[str, str | bool], base: SimulationConfig | None = None) -> SimulationConfig:
    """Convert widgets explicitly, retaining any non-present configuration fields."""
    base = base or SimulationConfig()
    data = base.to_dict()
    defaults = SimulationConfig()
    for spec in FORM_FIELDS:
        if spec.name not in values:
            continue
        value = values[spec.name]
        default = getattr(defaults, spec.name)
        try:
            if spec.kind == "bool":
                if type(value) is not bool:
                    raise ValueError("booléen attendu")
            elif spec.kind == "json":
                value = json.loads(str(value))
            elif default is None:
                value = str(value).strip() or None
                if spec.name == "pulse_width_max_step_us" and value is not None:
                    value = float(value)
            elif type(default) is int:
                value = int(str(value).strip())
            elif type(default) is float:
                value = float(str(value).strip())
            else:
                value = str(value).strip()
        except (TypeError, ValueError) as error:
            raise ValueError(f"{spec.label} : {error}") from error
        data[spec.name] = value
    config = SimulationConfig.from_dict(data)
    return CapabilityRegistry.require_valid(config)


def scientific_summary(config: SimulationConfig) -> str:
    window = config.cycles if config.mode == "fho" else config.cycles_per_window
    dynamics = "cadence libre" if config.formulation == "dynamic" else f"ω = {config.isokinetic_omega:g} rad/s"
    if config.formulation == "isokinetic":
        dynamics += f" · cinématique {config.isokinetic_kinematics}"
    slew = "aucune borne ΔPW" if config.pulse_width_max_step_us is None else f"|u[k+1] − u[k]| ≤ {config.pulse_width_max_step_us:g} µs"
    reduction = (
        "\nRéduction Ding ACADOS expérimentale activée : F,A conservés ; Cn,Tau1,Km reconstruits avec IRK Gauss-Legendre 4×5. "
        "États initiaux fixés requis ; multiplicateurs du problème complet non reconstruits."
        if config.acados_ding_local_reduction else ""
    )
    ipopt_reduction = (
        "\nRéduction Ding IPOPT expérimentale activée : F,A restent des états de collocation ; "
        "Cn,Tau1,Km sont reconstruits par l'opérateur discret Radau-5, avec audit dans le NLP complet. "
        "Cette voie SX interprétée est validée pour le RHO unilatéral ; le smoke bilatéral passe, "
        "mais une campagne longue reste requise avant un usage de référence."
        if config.ipopt_ding_local_reduction else ""
    )
    sides = " · bilatéral" if config.bilateral_reduced else ""
    guard = "actif" if CapabilityRegistry.effective_reduced_velocity_guard(config) else "inactif"
    terminal_guard = (
        "\nRaccord RHO : borne de cadence sur une prédiction Euler au demi-pas suivant ; "
        "ce contrôle local ne certifie pas toutes les contraintes du cycle suivant."
        if config.reduced_terminal_half_step_velocity_guard else ""
    )
    common_seed = (
        f"\nComparaison : seed commun certifié {config.common_initial_solution}. "
        "Il doit correspondre au modèle, à la transcription, aux contrôles et aux contraintes ; "
        "sa provenance et sa compatibilité sont revérifiées par le moteur."
        if config.common_initial_solution and config.solver != "acados" else ""
    )
    return (f"{config.mode.upper()} · {config.solver.upper()} · mécanique {config.mechanics}{sides} · {dynamics}\n"
            f"{config.cycles} cycles exécutés · fenêtre de {window} cycle(s) · "
            f"{config.stimulations_per_cycle} contrôles par cycle\n"
            f"Couple signé {config.signed_crank_torque:g} N.m (positif = résistance si ω < 0)\n"
            f"{slew} ; contrôles successifs, y compris le raccord entre RHO\n"
            f"Formulation ΔPW : {config.pulse_width_slew_formulation} ; guard vitesse interne {guard}\n"
            f"Coût ΔPW intra-fenêtre : poids {config.pulse_width_slew_weight:g}, référence {config.pulse_width_slew_reference_us:g} µs ; raccord RHO seulement borné\n"
            "ACADOS : seed IPOPT du cycle 1 obligatoire ; provenance et compatibilité revérifiées par le moteur."
            + terminal_guard + reduction + ipopt_reduction + common_seed)


def result_summary(payload, requested_cycles: int) -> str:
    """Report the engine's declarations without inferring them from process exit."""
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list) or not payload["results"]:
        return "Statut scientifique indisponible : format de résultat non reconnu."
    summaries = []
    for result in payload["results"]:
        if not isinstance(result, dict):
            return "Statut scientifique indisponible : résultat incomplet."
        solver = result.get("solver", "moteur")
        cycles = result.get("validated_cycles")
        success = result.get("success")
        if success is True and type(cycles) is int and cycles >= requested_cycles:
            state = "réussite déclarée par le moteur"
        elif success is False:
            state = "échec déclaré par le moteur"
        else:
            state = "résultat partiel ou non certifié"
        summaries.append(f"{solver} : {state} ; cycles validés {cycles if cycles is not None else '?'} / {requested_cycles}.")
    return "\n".join(summaries)
