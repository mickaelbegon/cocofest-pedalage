"""Tkinter simulation workbench; run with python -m cocofest.simulation.gui.

The form and subprocess controller are separate, headless-testable modules.
Scientific dependencies are only loaded inside the child solver process.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
import queue
import threading
import time

from .config import SimulationConfig
from .capabilities import CapabilityRegistry
from .launch import ROOT, build_launch_plan
from .gui_model import FORM_FIELDS, config_from_form, form_values, result_summary, scientific_summary
from .execution import SimulationProcess, runtime_helpers
from .bayesian_model import BayesianCampaignConfig, campaign_summary
from .async_bayesian_model import (
    AsyncBayesianCampaignConfig, configured_muscle_names, muscle_weight_coordinate_name,
    muscle_weight_reference, relative_weight_bounds,
)
from .launch import LaunchPlan
from .cross_rollout_model import CrossRolloutConfig, build_cross_rollout_plan, cross_rollout_summary, cross_rollout_report
from .independent_arms_gui_model import IndependentArmsGuiConfig, independent_arms_summary, independent_solver_choices
from .cpu_affinity import available_cpu_ids, reserved_cpu_ids
from .campaign_history import CampaignHistory
from .gui_analysis import LiveArtifacts, generate_analysis_figures
from .bilateral_endurance_campaign import BilateralEnduranceCampaignConfig, BilateralEnduranceCandidate, BilateralTorqueBracketConfig
from .gui_problem_type import (
    BILATERAL_COMBINED, INDEPENDENT_ARMS, PROBLEM_TYPE_DESCRIPTIONS,
    PROBLEM_TYPE_BY_LABEL, PROBLEM_TYPE_LABELS, UNILATERAL, main_problem_type,
)


_MUSCLE_WEIGHT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def has_positive_wraplength(value) -> bool:
    """Return whether a Tk label has an explicitly enabled numeric wrap length.

    ``ttk.Label.cget("wraplength")`` is an empty string for labels created
    without that option.  It must not make the canvas resize callback fail.
    """
    try:
        return float(value or 0) > 0
    except (TypeError, ValueError):
        return False


def muscle_weight_search_space(raw_names, low, high):
    """Build a non-redundant centred-log coordinate domain for muscle BO.

    ``low`` and ``high`` are the controller's effective relative-weight box,
    not an arbitrary multiplier range.  One muscle fixes the common-scale
    gauge (Biceps when present); the remaining coordinates lie in [-1, 1] and
    are transformed by the runner into weights with geometric mean exactly 1.
    """
    raw = str(raw_names).split(",") if isinstance(raw_names, str) else raw_names
    names = tuple(str(name).strip() for name in raw if str(name).strip())
    if not names:
        raise ValueError("Indiquez au moins un muscle à pondérer.")
    if len(set(names)) != len(names):
        raise ValueError("Chaque muscle ne peut apparaître qu'une seule fois.")
    invalid = [name for name in names if not _MUSCLE_WEIGHT_NAME.fullmatch(name)]
    if invalid:
        raise ValueError(f"Nom musculaire invalide : {', '.join(invalid)}.")
    low, high = float(low), float(high)
    if not (0 < low < 1 < high):
        raise ValueError("Les bornes effectives doivent vérifier 0 < min < 1 < max.")
    reference = muscle_weight_reference(names)
    return {muscle_weight_coordinate_name(name): {"type": "float", "low": -1.0, "high": 1.0}
            for name in names if name != reference}


def muscle_weight_campaign(base_config, *, output_root, study_name, muscle_names,
                           scale_min, scale_max, trials, workers, startup_trials,
                           seed, max_cycles, timeout_s):
    """Create the restart-safe asynchronous controller campaign from GUI values."""
    if base_config.mode not in ("rho-physio", "rho-pace"):
        raise ValueError("Le BO des poids musculaires nécessite le mode rho-physio ou rho-pace.")
    if base_config.solver != "ipopt":
        raise ValueError("Le BO des poids musculaires est actuellement raccordé au RHO IPOPT uniquement.")
    if not base_config.model_config:
        raise ValueError("Le BO des poids musculaires requiert une configuration musculaire JSON.")
    if not base_config.weights_config:
        raise ValueError("Le BO des poids musculaires requiert le JSON de poids de référence.")
    declared_names = configured_muscle_names(base_config.model_config)
    requested_names = tuple(name.strip() for name in str(muscle_names).split(",") if name.strip())
    if tuple(requested_names) != declared_names:
        raise ValueError("Les muscles BO doivent suivre exactement l'ordre déclaré par model_config.")
    configured_low, configured_high = relative_weight_bounds(base_config.weights_config)
    if (float(scale_min), float(scale_max)) != (configured_low, configured_high):
        raise ValueError("Les bornes BO doivent reproduire la boîte effective déclarée dans weights_config.policy "
                         f"({configured_low:g}, {configured_high:g}).")
    return AsyncBayesianCampaignConfig(
        base_config=base_config.to_dict(), output_root=str(output_root), study_name=str(study_name).strip(),
        study_kind="controller", phase="screening", max_cycles=int(max_cycles),
        metric="continuous_endurance", n_trials=int(trials), workers=int(workers),
        n_startup_trials=int(startup_trials), seed=int(seed), timeout_s=float(timeout_s),
        search_space=muscle_weight_search_space(declared_names, configured_low, configured_high),
    ).validate()


class SimulationApp:
    def __init__(self, window, config=None, prefix=None):
        import tkinter as tk
        from tkinter import ttk
        from tkinter.scrolledtext import ScrolledText
        self.tk, self.ttk = tk, ttk
        self.window = window
        self.config = config or SimulationConfig(output_root=f"gui-results/{datetime.now():%Y%m%d-%H%M%S}")
        self.process = SimulationProcess()
        self.active_plan = None
        self.active_config = None
        self.active_campaign = None
        self.active_rollout = None
        self.active_independent_arms = None
        self.active_history_id = None
        self.campaign_history = CampaignHistory(ROOT / "gui-results" / "campaign-history.json")
        self._completion_shown = False
        self._closing = False
        self._live_artifacts = None
        self._live_started_at = None
        self._live_finished_at = None
        self._next_live_update = 0
        self._analysis_results = queue.Queue()
        self._analysis_busy = False
        window.title("Cocofest · Simulation de pédalage")
        window.geometry("1120x900")
        window.minsize(860, 720)
        style = ttk.Style(window)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Title.TLabel", font=("Sans", 17, "bold"))
        style.configure("Error.TLabel", foreground="#a42b29")
        style.configure("Success.TLabel", foreground="#24603e")

        outer = ttk.Frame(window, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Simulation de pédalage", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Configurer le problème, vérifier les compatibilités, puis lancer le solveur.").pack(anchor="w", pady=(3, 12))
        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 10))
        ttk.Button(toolbar, text="Ouvrir une configuration…", command=self.load).pack(side="left")
        ttk.Button(toolbar, text="Enregistrer JSON…", command=self.save).pack(side="left", padx=6)
        ttk.Button(toolbar, text="Valeurs par défaut", command=self.reset).pack(side="left")
        problem_row = ttk.LabelFrame(outer, text="Type de problème", padding=(10, 6))
        problem_row.pack(fill="x", pady=(0, 10))
        ttk.Label(problem_row, text="Architecture").grid(row=0, column=0, sticky="w", padx=(0, 10))
        self.problem_type = tk.StringVar(value=PROBLEM_TYPE_LABELS[main_problem_type(self.config)])
        self.problem_type_selector = ttk.Combobox(
            problem_row,
            textvariable=self.problem_type,
            values=tuple(PROBLEM_TYPE_LABELS.values()),
            state="readonly",
            width=42,
        )
        self.problem_type_selector.grid(row=0, column=1, sticky="w")
        self.problem_type_label = ttk.Label(problem_row, wraplength=790)
        self.problem_type_label.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        self.problem_type.trace_add("write", self.select_problem_type)
        self.update_problem_type_description()
        self.scientific_status = tk.StringVar(value="Statut scientifique : aucun résultat pour cette session.")
        ttk.Label(outer, textvariable=self.scientific_status, wraplength=1040).pack(side="bottom", anchor="w", pady=(8, 0))
        # Bound the long configuration area so launch actions and live output
        # remain reachable even at the minimum supported window size.
        configuration_area = ttk.Frame(outer)
        configuration_area.pack(fill="x")
        configuration_canvas = tk.Canvas(configuration_area, height=330, highlightthickness=0)
        configuration_scroll = ttk.Scrollbar(configuration_area, orient="vertical", command=configuration_canvas.yview)
        configuration_canvas.configure(yscrollcommand=configuration_scroll.set)
        configuration_scroll.pack(side="right", fill="y")
        configuration_canvas.pack(side="left", fill="x", expand=True)
        configuration_content = ttk.Frame(configuration_canvas)
        configuration_item = configuration_canvas.create_window((0, 0), window=configuration_content, anchor="nw")
        configuration_content.bind("<Configure>", lambda event: configuration_canvas.configure(
            scrollregion=configuration_canvas.bbox("all")))
        def resize_configuration(event):
            configuration_canvas.itemconfigure(configuration_item, width=event.width)
            pending = list(configuration_content.winfo_children())
            while pending:
                widget = pending.pop()
                pending.extend(widget.winfo_children())
                if isinstance(widget, ttk.Label) and has_positive_wraplength(widget.cget("wraplength")):
                    widget.configure(wraplength=max(300, event.width - 50))

        configuration_canvas.bind("<Configure>", resize_configuration)
        self.configuration_canvas = configuration_canvas
        section_row = ttk.Frame(configuration_content)
        section_row.pack(fill="x", pady=(0, 6))
        ttk.Label(section_row, text="Section de configuration").pack(side="left", padx=(0, 10))
        self.configuration_section = tk.StringVar()
        section_selector = ttk.Combobox(section_row, textvariable=self.configuration_section, state="readonly", width=38)
        section_selector.pack(side="left", fill="x", expand=True)
        self.form_notebook = ttk.Notebook(configuration_content)
        self.form_notebook.pack(fill="x")
        def select_section(event):
            self.form_notebook.select(section_selector.current())

        def selected_section_changed(event):
            self.configuration_section.set(self.form_notebook.tab(self.form_notebook.select(), "text"))
            configuration_canvas.yview_moveto(0)

        section_selector.bind("<<ComboboxSelected>>", select_section)
        self.form_notebook.bind("<<NotebookTabChanged>>", selected_section_changed)
        groups = {}
        self.variables = {}
        self.controls = {}
        initial = form_values(self.config)
        for spec in FORM_FIELDS:
            if spec.group not in groups:
                frame = ttk.Frame(self.form_notebook, padding=12)
                frame.columnconfigure(1, weight=1)
                self.form_notebook.add(frame, text=spec.group)
                groups[spec.group] = (frame, 0)
                if spec.group == "Problème":
                    self.main_problem_tab = frame
            frame, row = groups[spec.group]
            ttk.Label(frame, text=spec.label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=4)
            variable = tk.BooleanVar(value=initial[spec.name]) if spec.kind == "bool" else tk.StringVar(value=initial[spec.name])
            self.variables[spec.name] = variable
            if spec.kind == "bool":
                control = ttk.Checkbutton(frame, variable=variable)
            elif spec.kind == "choice":
                control = ttk.Combobox(frame, textvariable=variable, values=CapabilityRegistry.choices(spec.name), state="readonly", width=36)
            else:
                control = ttk.Entry(frame, textvariable=variable, width=55)
            control.grid(row=row, column=1, sticky="ew", pady=4)
            self.controls[spec.name] = control
            if spec.kind in ("file", "directory"):
                ttk.Button(frame, text="Parcourir…", command=lambda field=spec.name, kind=spec.kind: self.browse(field, kind)).grid(row=row, column=2, padx=(6, 0))
            groups[spec.group] = (frame, row + 1)
        replay = ttk.Frame(self.form_notebook, padding=12)
        replay.columnconfigure(1, weight=1)
        self.form_notebook.add(replay, text="Rejeu croisé / DOP853")
        ttk.Label(replay, text="Rejeu ouvert : commandes sauvegardées, état initial commun, aucune réoptimisation.\n"
                  "DOP853 est la référence ; Radau-5 et GL4×5 utilisent leurs tableaux IRK.\n"
                  "Le coût physique commun et les diagnostics sont enregistrés dans matrix.json.",
                  wraplength=930).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        replay_defaults = {"sources": "[]", "reduced_profile": "benchmark-seed/reduced-cycling-fourier12.npz",
                           "model_config": "", "cost_config": "", "evaluators": "dop853, radau5, gauss-legendre4x5", "start_cycle": "0",
                           "cycles": "1", "cycle_duration_s": "1.0",
                           "dop853_rtol": "1e-11", "dop853_atol": "1e-13", "samples_per_interval": "65",
                           "output_root": f"gui-results/cross-rollout-{datetime.now():%Y%m%d-%H%M%S}"}
        self.rollout_variables = {key: tk.StringVar(value=value) for key, value in replay_defaults.items()}
        labels = (("Solutions NPZ (liste JSON)", "sources"), ("Profil mécanique réduit NPZ", "reduced_profile"),
                  ("Modèle musculaire JSON (archives historiques)", "model_config"),
                  ("Coût commun JSON (optionnel)", "cost_config"), ("Évaluateurs", "evaluators"),
                  ("Cycle initial (index 0)", "start_cycle"), ("Cycles rejoués", "cycles"),
                  ("Durée d'un cycle (s)", "cycle_duration_s"),
                  ("DOP853 : tolérance relative", "dop853_rtol"), ("DOP853 : tolérance absolue mise à l'échelle", "dop853_atol"),
                  ("Échantillons de contraintes / intervalle", "samples_per_interval"), ("Dossier de résultats", "output_root"))
        for row, (label, name) in enumerate(labels, 1):
            ttk.Label(replay, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 16))
            ttk.Entry(replay, textvariable=self.rollout_variables[name], width=55).grid(row=row, column=1, sticky="ew", pady=3)
            if name in ("sources", "reduced_profile", "model_config", "cost_config", "output_root"):
                ttk.Button(replay, text="Parcourir…", command=lambda field=name: self.browse_rollout(field)).grid(row=row, column=2, padx=(6, 0))
        self.rollout_validation = ttk.Label(replay, wraplength=980, style="Error.TLabel")
        self.rollout_validation.grid(row=len(labels)+1, column=0, columnspan=3, sticky="w", pady=5)
        replay_buttons = ttk.Frame(replay)
        replay_buttons.grid(row=len(labels)+2, column=0, columnspan=3, sticky="w")
        self.preview_rollout_button = ttk.Button(replay_buttons, text="Prévisualiser le rejeu", command=self.preview_rollout)
        self.preview_rollout_button.pack(side="left")
        self.start_rollout_button = ttk.Button(replay_buttons, text="Lancer le rejeu ouvert", command=self.start_rollout)
        self.start_rollout_button.pack(side="left", padx=6)
        ttk.Button(replay_buttons, text="Charger requête…", command=self.load_rollout).pack(side="left")
        ttk.Button(replay_buttons, text="Enregistrer requête…", command=self.save_rollout).pack(side="left", padx=6)
        ttk.Button(replay_buttons, text="Configurer un RHO réoptimisé", command=self.select_reoptimized_rho).pack(side="left")
        arms = ttk.Frame(self.form_notebook, padding=12)
        self.independent_arms_tab = arms
        arms.columnconfigure(1, weight=1)
        self.form_notebook.add(arms, text="Deux bras indépendants")
        ttk.Label(
            arms,
            text=("Deux problèmes unilatéraux partagent la même cadence imposée, mais ne sont pas synchronisés cycle par cycle par défaut. "
                  "Ils ne partagent ni états Ding/fatigue, ni commandes, ni couple mécanique. Cette voie est exclusivement "
                  "isocinétique. Chaque cible est un travail positif par tour ou son couple moyen équivalent : W = 2π·τ̄.\n\n"
                  "Le couple instantané reste déduit du bilan isocinétique par le moteur; ce formulaire ne prescrit pas un couple "
                  "externe instantané. La politique expérimentale « retour capacité » installe une barrière après chaque cycle, "
                  "puis répartit le travail total du cycle suivant entre les bras; elle exige des métriques de réserve certifiées."),
            wraplength=970,
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        arms_defaults = {
            "solver": "ipopt", "cycles": "100", "cycles_per_window": "1", "stimulations_per_cycle": "30",
            "isokinetic_omega": str(-2.0 * 3.141592653589793),
            "right_work_j_per_cycle": "", "right_equivalent_mean_torque_nm": "0.1",
            "left_work_j_per_cycle": "", "left_equivalent_mean_torque_nm": "0.1",
            "resistance_pace_policy": "fixed", "resistance_pace_initial_split_policy": "manual",
            "resistance_pace_update_every_cycles": "10", "resistance_pace_capacity_gain": "1.0",
            "resistance_pace_smoothing": "0.25", "resistance_pace_max_fraction_step": "0.10",
            "resistance_pace_minimum_arm_torque_nm": "0.0",
            "endurance_torque_upper_nm": "0.6", "endurance_torque_tolerance_nm": "0.02",
            "parametric_fatigue_weights": "false",
            "muscle_weight_policy": "unit", "physio_update_every_cycles": "20",
            "physio_update_smoothing": "0.2", "physio_update_max_log_step": "0.09531017980432493",
            "physio_update_deadband_log": "0.009950330853168083",
            "physio_update_min_relative_weight": "0.25", "physio_update_max_relative_weight": "4.0",
            "right_solver_cpu": "Automatique", "left_solver_cpu": "Automatique",
            "factory": "cocofest.simulation.independent_arms_process:build_process_independent_arms", "right_runner_config": "", "left_runner_config": "",
            "output_root": f"gui-results/independent-arms-{datetime.now():%Y%m%d-%H%M%S}",
        }
        self.independent_arms_variables = {name: tk.StringVar(value=value) for name, value in arms_defaults.items()}
        arm_sections = (
            ("Tâche", "Même solveur IPOPT et même cadence imposée pour les deux OCPs.", (
                ("Solveur (même choix pour les deux bras)", "solver"),
                ("Cycles exécutés", "cycles"), ("Cycles par fenêtre RHO", "cycles_per_window"),
                ("Contrôles / stimulations par cycle", "stimulations_per_cycle"),
                ("Vitesse imposée ω (rad/s, négative)", "isokinetic_omega"),
                ("Bras droit : travail cible / cycle (J, optionnel)", "right_work_j_per_cycle"),
                ("Bras droit : couple moyen équivalent τ̄ (N.m, optionnel)", "right_equivalent_mean_torque_nm"),
                ("Bras gauche : travail cible / cycle (J, optionnel)", "left_work_j_per_cycle"),
                ("Bras gauche : couple moyen équivalent τ̄ (N.m, optionnel)", "left_equivalent_mean_torque_nm"),
            )),
            ("Répartition D/G", "La rétroaction capacité ne modifie que le travail demandé au prochain cycle; elle ne couple pas les deux OCPs.", (
                ("Répartition de résistance", "resistance_pace_policy"),
                ("Initialisation D/G (manuel ou capacité-fatigabilité après cycle 1)", "resistance_pace_initial_split_policy"),
                ("Mise à jour de répartition tous les N cycles", "resistance_pace_update_every_cycles"),
                ("Retour capacité : gain", "resistance_pace_capacity_gain"),
                ("Retour capacité : lissage [0,1]", "resistance_pace_smoothing"),
                ("Retour capacité : pas max. de fraction [0,1]", "resistance_pace_max_fraction_step"),
                ("Retour capacité : τ̄ minimal / bras (N.m)", "resistance_pace_minimum_arm_torque_nm"),
            )),
            ("Coût musculaire", "Physio-U et l'ablation mécanique nécessitent des poids paramétriques et les mettent à jour entre cycles, sans reconstruire le NLP.", (
                ("Poids de fatigue paramétriques (préserve le NLP compilé)", "parametric_fatigue_weights"),
                ("Politique de poids musculaires", "muscle_weight_policy"),
                ("Physio-U : mise à jour tous les N cycles", "physio_update_every_cycles"),
                ("Physio-U : lissage [0,1]", "physio_update_smoothing"),
                ("Physio-U : pas maximal log", "physio_update_max_log_step"),
                ("Physio-U : seuil mort log", "physio_update_deadband_log"),
                ("Physio-U : poids relatif minimal", "physio_update_min_relative_weight"),
                ("Physio-U : poids relatif maximal", "physio_update_max_relative_weight"),
            )),
            ("Exécution & campagnes", "La recherche d'endurance est un outil de campagne; la fabrique runtime reste un réglage avancé.", (
                ("CPU solveur bras droit", "right_solver_cpu"),
                ("CPU solveur bras gauche", "left_solver_cpu"),
                ("Endurance : borne haute τ total (N.m)", "endurance_torque_upper_nm"),
                ("Endurance : tolérance τ total (N.m)", "endurance_torque_tolerance_nm"),
                ("Fabrique runtime (module:fonction, optionnel si les configs la déclarent)", "factory"),
                ("Bras droit : configuration du modèle/worker (chemin ou référence)", "right_runner_config"),
                ("Bras gauche : configuration du modèle/worker (chemin ou référence)", "left_runner_config"),
                ("Dossier de résultats", "output_root"),
            )),
        )
        arm_form = ttk.Notebook(arms)
        arm_form.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.independent_arms_controls = {}
        for section, description, arm_labels in arm_sections:
            page = ttk.Frame(arm_form, padding=10)
            page.columnconfigure(1, weight=1)
            arm_form.add(page, text=section)
            ttk.Label(page, text=description, wraplength=930).grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 7))
            for row, (label, name) in enumerate(arm_labels, 1):
                ttk.Label(page, text=label).grid(row=row, column=0, sticky="w", pady=3, padx=(0, 16))
                if name == "solver":
                    control = ttk.Combobox(page, textvariable=self.independent_arms_variables[name], state="readonly", width=34,
                                           values=independent_solver_choices(arms_defaults["factory"]))
                    self.independent_solver_control = control
                elif name == "resistance_pace_policy":
                    control = ttk.Combobox(page, textvariable=self.independent_arms_variables[name], state="readonly", width=34,
                                           values=("fixed", "capacity_feedback"))
                elif name == "resistance_pace_initial_split_policy":
                    control = ttk.Combobox(page, textvariable=self.independent_arms_variables[name], state="readonly", width=34,
                                           values=("manual", "capacity_fatigability_after_first_cycle"))
                elif name == "parametric_fatigue_weights":
                    control = ttk.Combobox(page, textvariable=self.independent_arms_variables[name], state="readonly", width=34,
                                           values=("false", "true"))
                elif name == "muscle_weight_policy":
                    control = ttk.Combobox(page, textvariable=self.independent_arms_variables[name], state="readonly", width=34,
                                           values=("unit", "physio_u", "mechanical_sensitivity_squared_v1_experimental"))
                elif name in ("right_solver_cpu", "left_solver_cpu"):
                    control = tk.OptionMenu(page, self.independent_arms_variables[name], "Automatique")
                    self._solver_cpu_menus = getattr(self, "_solver_cpu_menus", {})
                    self._solver_cpu_menus["right" if name.startswith("right") else "left"] = control
                    control.configure(width=52)
                else:
                    control = ttk.Entry(page, textvariable=self.independent_arms_variables[name], width=55)
                control.grid(row=row, column=1, sticky="ew", pady=3)
                self.independent_arms_controls[name] = control
                if name == "output_root":
                    ttk.Button(page, text="Parcourir…", command=lambda: self.browse_independent_arms("output_root")).grid(row=row, column=2, padx=(6, 0))
        self.independent_arms_validation = ttk.Label(arms, wraplength=980, style="Error.TLabel")
        self.independent_arms_validation.grid(row=2, column=0, columnspan=3, sticky="w", pady=(7, 3))
        arm_buttons = ttk.Frame(arms)
        arm_buttons.grid(row=3, column=0, columnspan=3, sticky="w")
        self.preview_independent_arms_button = ttk.Button(arm_buttons, text="Prévisualiser les deux bras", command=self.preview_independent_arms)
        self.preview_independent_arms_button.grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.start_independent_arms_button = ttk.Button(arm_buttons, text="Lancer les deux bras", command=self.start_independent_arms)
        self.start_independent_arms_button.grid(row=0, column=1, sticky="w", padx=4, pady=3)
        ttk.Button(arm_buttons, text="Actualiser les CPU", command=self.refresh_solver_cpu_menus).grid(row=0, column=2, sticky="w", padx=4, pady=3)
        self.preview_endurance_button = ttk.Button(arm_buttons, text="Prévisualiser endurance 10 min", command=self.preview_endurance)
        self.preview_endurance_button.grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.start_endurance_button = ttk.Button(arm_buttons, text="Lancer endurance 10 min", command=self.start_endurance)
        self.start_endurance_button.grid(row=1, column=1, sticky="w", padx=4, pady=3)
        self.start_endurance_search_button = ttk.Button(arm_buttons, text="Chercher τ maximal", command=self.start_endurance_search)
        self.start_endurance_search_button.grid(row=1, column=2, sticky="w", padx=4, pady=3)
        self.apply_independent_adjustment_button = ttk.Button(arm_buttons, text="Reporter la proposition", command=self.apply_independent_adjustment, state="disabled")
        self.apply_independent_adjustment_button.grid(row=2, column=0, sticky="w", padx=4, pady=3)
        ttk.Button(arm_buttons, text="Charger requête…", command=self.load_independent_arms).grid(row=2, column=1, sticky="w", padx=4, pady=3)
        ttk.Button(arm_buttons, text="Enregistrer requête…", command=self.save_independent_arms).grid(row=2, column=2, sticky="w", padx=4, pady=3)
        self.refresh_solver_cpu_menus()
        campaign = ttk.Frame(self.form_notebook, padding=10)
        self.form_notebook.add(campaign, text="BO solveurs")
        campaign.columnconfigure(1, weight=1)
        campaign_defaults = {
            "bo_solvers": "ipopt, madnlp, fatrop", "bo_calls": "16", "bo_initial": "6",
            "bo_seed": "42", "bo_degree_min": "3", "bo_degree_max": "5", "bo_threads": "1, 2, 4",
            "bo_metric": "solver_time", "bo_output": f"gui-results/bayesian-{datetime.now():%Y%m%d-%H%M%S}",
        }
        self.campaign_variables = {key: tk.StringVar(value=value) for key, value in campaign_defaults.items()}
        labels = (("Solvers", "bo_solvers"), ("Budget / initiaux", "bo_calls"), ("Graine", "bo_seed"),
                  ("Degré collocation min / max", "bo_degree_min"), ("Threads candidats", "bo_threads"),
                  ("Métrique", "bo_metric"), ("Dossier de campagne", "bo_output"))
        for row, (label, name) in enumerate(labels):
            ttk.Label(campaign, text=label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=3)
            if name == "bo_calls":
                pair = ttk.Frame(campaign)
                pair.grid(row=row, column=1, sticky="ew", pady=3)
                ttk.Entry(pair, textvariable=self.campaign_variables[name], width=8).pack(side="left")
                ttk.Label(pair, text="  /  ").pack(side="left")
                ttk.Entry(pair, textvariable=self.campaign_variables["bo_initial"], width=8).pack(side="left")
            elif name == "bo_degree_min":
                pair = ttk.Frame(campaign)
                pair.grid(row=row, column=1, sticky="ew", pady=3)
                ttk.Entry(pair, textvariable=self.campaign_variables[name], width=8).pack(side="left")
                ttk.Label(pair, text="  /  ").pack(side="left")
                ttk.Entry(pair, textvariable=self.campaign_variables["bo_degree_max"], width=8).pack(side="left")
            elif name == "bo_metric":
                ttk.Combobox(campaign, textvariable=self.campaign_variables[name], state="readonly", width=32,
                             values=("solver_time", "hot_time", "p90_time", "continuous_endurance")).grid(row=row, column=1, sticky="w", pady=3)
            else:
                ttk.Entry(campaign, textvariable=self.campaign_variables[name], width=55).grid(row=row, column=1, sticky="ew", pady=3)
        self.campaign_validation = ttk.Label(campaign, wraplength=1000, style="Error.TLabel")
        self.campaign_validation.grid(row=len(labels), column=0, columnspan=2, sticky="w", pady=(5, 0))
        campaign_buttons = ttk.Frame(campaign)
        campaign_buttons.grid(row=len(labels) + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.preview_campaign_button = ttk.Button(campaign_buttons, text="Prévisualiser la campagne", command=self.preview_campaign)
        self.preview_campaign_button.pack(side="left")
        self.start_campaign_button = ttk.Button(campaign_buttons, text="Lancer la BO", command=self.start_campaign)
        self.start_campaign_button.pack(side="left", padx=7)
        muscle_campaign = ttk.Frame(self.form_notebook, padding=10)
        self.form_notebook.add(muscle_campaign, text="BO poids musculaires")
        muscle_campaign.columnconfigure(1, weight=1)
        muscle_defaults = {
            "muscle_bo_names": "Delt_ant, Delt_post, Biceps, Triceps",
            "muscle_bo_min": "0.25", "muscle_bo_max": "4", "muscle_bo_trials": "32",
            "muscle_bo_workers": "4", "muscle_bo_startup": "8", "muscle_bo_seed": "42",
            "muscle_bo_cycles": "500", "muscle_bo_timeout": "7200",
            "muscle_bo_output": f"gui-results/muscle-weights-bo-{datetime.now():%Y%m%d-%H%M%S}",
        }
        self.muscle_campaign_variables = {key: tk.StringVar(value=value) for key, value in muscle_defaults.items()}
        muscle_labels = (
            ("Muscles (noms exacts, séparés par virgule)", "muscle_bo_names"),
            ("Boîte effective poids relatifs min / max", "muscle_bo_min"),
            ("Essais / workers", "muscle_bo_trials"),
            ("Initialisation aléatoire / graine", "muscle_bo_startup"),
            ("Cycles maximum / timeout essai (s)", "muscle_bo_cycles"),
            ("Dossier de campagne", "muscle_bo_output"),
        )
        for row, (label, name) in enumerate(muscle_labels):
            ttk.Label(muscle_campaign, text=label).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=3)
            if name in ("muscle_bo_min", "muscle_bo_trials", "muscle_bo_startup", "muscle_bo_cycles"):
                companion = {"muscle_bo_min": "muscle_bo_max", "muscle_bo_trials": "muscle_bo_workers",
                             "muscle_bo_startup": "muscle_bo_seed", "muscle_bo_cycles": "muscle_bo_timeout"}[name]
                pair = ttk.Frame(muscle_campaign)
                pair.grid(row=row, column=1, sticky="ew", pady=3)
                ttk.Entry(pair, textvariable=self.muscle_campaign_variables[name], width=10).pack(side="left")
                ttk.Label(pair, text="  /  ").pack(side="left")
                ttk.Entry(pair, textvariable=self.muscle_campaign_variables[companion], width=10).pack(side="left")
            else:
                ttk.Entry(muscle_campaign, textvariable=self.muscle_campaign_variables[name], width=55).grid(row=row, column=1, sticky="ew", pady=3)
        ttk.Label(muscle_campaign,
                  text="Disponible pour rho-physio / rho-pace avec modèle et poids de référence. Chaque essai écrit une copie traçable des poids ; ACADOS n'est pas pris en charge pour cette étude.",
                  wraplength=980).grid(row=len(muscle_labels), column=0, columnspan=2, sticky="w", pady=(5, 0))
        self.muscle_campaign_validation = ttk.Label(muscle_campaign, wraplength=1000, style="Error.TLabel")
        self.muscle_campaign_validation.grid(row=len(muscle_labels)+1, column=0, columnspan=2, sticky="w", pady=(5, 0))
        muscle_buttons = ttk.Frame(muscle_campaign)
        muscle_buttons.grid(row=len(muscle_labels)+2, column=0, columnspan=2, sticky="w", pady=(6, 0))
        self.preview_muscle_campaign_button = ttk.Button(muscle_buttons, text="Prévisualiser le BO musculaire", command=self.preview_muscle_campaign)
        self.preview_muscle_campaign_button.pack(side="left")
        self.start_muscle_campaign_button = ttk.Button(muscle_buttons, text="Lancer le BO musculaire", command=self.start_muscle_campaign)
        self.start_muscle_campaign_button.pack(side="left", padx=7)
        section_selector.configure(values=tuple(self.form_notebook.tab(tab, "text") for tab in self.form_notebook.tabs()))
        self.configuration_section.set(self.form_notebook.tab(self.form_notebook.select(), "text"))
        runtime = ttk.Frame(outer)
        runtime.pack(fill="x", pady=(12, 6))
        ttk.Label(runtime, text="Environnement du solveur").pack(side="left", padx=(0, 12))
        self.prefix = tk.StringVar(value=str(prefix or self.default_prefix(self.config.solver)))
        ttk.Entry(runtime, textvariable=self.prefix).pack(side="left", fill="x", expand=True)
        ttk.Button(runtime, text="Détecter", command=self.detect_runtime).pack(side="left", padx=6)
        self.validation = ttk.Label(outer, wraplength=1040, style="Error.TLabel")
        self.validation.pack(anchor="w", pady=(4, 6))
        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=(2, 10))
        self.preview_button = ttk.Button(buttons, text="Prévisualiser la commande", command=self.preview)
        self.preview_button.pack(side="left")
        self.start_button = ttk.Button(buttons, text="Lancer", command=self.start)
        self.start_button.pack(side="left", padx=7)
        self.stop_button = ttk.Button(buttons, text="Arrêter", command=self.stop, state="disabled")
        self.stop_button.pack(side="left")
        self.process_status = tk.StringVar(value="Processus : inactif")
        ttk.Label(buttons, textvariable=self.process_status).pack(side="right")
        lower = ttk.Notebook(outer)
        lower.pack(fill="both", expand=True)
        self.summary_text = ScrolledText(lower, height=7, wrap="word", font=("Sans", 10))
        self.command_text = ScrolledText(lower, height=7, wrap="word", font=("Monospace", 9))
        self.log_text = ScrolledText(lower, height=7, wrap="word", font=("Monospace", 9))
        self.rollout_text = ScrolledText(lower, height=7, wrap="none", font=("Monospace", 9))
        self.live_text = ScrolledText(lower, height=7, wrap="word", font=("Sans", 10))
        self.set_text(self.live_text, "Aucune simulation lancée dans cette session.")
        history = ttk.Frame(lower, padding=6)
        history.columnconfigure(0, weight=1)
        history.rowconfigure(0, weight=1)
        history_columns = ("started_at", "kind", "solver", "formulation", "cycles", "controls", "cpu", "status", "output_root")
        self.campaign_history_table = ttk.Treeview(history, columns=history_columns, show="headings", height=10)
        headings = {
            "started_at": "Lancé", "kind": "Type", "solver": "Solveur", "formulation": "Formulation",
            "cycles": "Cycles", "controls": "Contrôles", "cpu": "CPU", "status": "État", "output_root": "Dossier",
        }
        widths = {"started_at": 145, "kind": 145, "solver": 75, "formulation": 105, "cycles": 65,
                  "controls": 90, "cpu": 105, "status": 90, "output_root": 360}
        for column in history_columns:
            self.campaign_history_table.heading(column, text=headings[column])
            self.campaign_history_table.column(column, width=widths[column], anchor="w", stretch=column == "output_root")
        history_scroll_y = ttk.Scrollbar(history, orient="vertical", command=self.campaign_history_table.yview)
        history_scroll_x = ttk.Scrollbar(history, orient="horizontal", command=self.campaign_history_table.xview)
        self.campaign_history_table.configure(yscrollcommand=history_scroll_y.set, xscrollcommand=history_scroll_x.set)
        self.campaign_history_table.grid(row=0, column=0, sticky="nsew")
        history_scroll_y.grid(row=0, column=1, sticky="ns")
        history_scroll_x.grid(row=1, column=0, sticky="ew")
        ttk.Button(history, text="Actualiser", command=self.refresh_campaign_history).grid(row=2, column=0, sticky="w", pady=(6, 0))
        history_actions = ttk.Frame(history)
        history_actions.grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Button(history_actions, text="Détails de la campagne", command=self.show_campaign_details).pack(side="left")
        ttk.Button(history_actions, text="Figures de la campagne sélectionnée", command=self.analyze_history).pack(side="left", padx=6)
        analysis = ttk.Frame(lower, padding=8)
        ttk.Label(analysis, text="Figures descriptives : temps de résolution, capacité/fatigue, travail et PW/forces lorsque les exports sont disponibles.\n"
                  "Les figures n'effectuent pas de certification dynamique. Les échecs de résolution sont marqués en rouge.",
                  wraplength=970).pack(anchor="w")
        ttk.Button(analysis, text="Analyser la simulation courante", command=self.analyze_active_run).pack(anchor="w", pady=6)
        ttk.Button(analysis, text="Analyser un dossier de résultats…", command=self.analyze_folder).pack(anchor="w")
        self.analysis_status = tk.StringVar(value="Les images PNG seront enregistrées dans gui-analysis/ du dossier choisi.")
        ttk.Label(analysis, textvariable=self.analysis_status, wraplength=970).pack(anchor="w", pady=6)
        lower.add(self.summary_text, text="Résumé scientifique")
        lower.add(self.live_text, text="Suivi de simulation")
        lower.add(self.command_text, text="Commande et environnement")
        lower.add(self.log_text, text="Journal en direct")
        lower.add(self.rollout_text, text="Matrice des replays")
        lower.add(history, text="Campagnes lancées")
        lower.add(analysis, text="Figures des solutions")
        self.lower = lower
        for variable in self.variables.values():
            variable.trace_add("write", self.validate)
            variable.trace_add("write", self.validate_muscle_campaign)
        self.variables["bilateral_reduced"].trace_add("write", self.sync_problem_type_from_main_form)
        for variable in self.campaign_variables.values():
            variable.trace_add("write", self.validate_campaign)
        for variable in self.muscle_campaign_variables.values():
            variable.trace_add("write", self.validate_muscle_campaign)
        for variable in self.rollout_variables.values():
            variable.trace_add("write", self.validate_rollout)
        for variable in self.independent_arms_variables.values():
            variable.trace_add("write", self.validate_independent_arms)
        self.prefix.trace_add("write", self.validate)
        window.protocol("WM_DELETE_WINDOW", self.close)
        self.validate()
        self.validate_campaign()
        self.validate_muscle_campaign()
        self.validate_rollout()
        self.validate_independent_arms()
        self.refresh_campaign_history()
        window.after(100, self.poll)

    @staticmethod
    def default_prefix(solver):
        suite = "madnlp32" if solver == "madnlp" else "rho32"
        return runtime_helpers().conda_env_prefix(suite) or Path.home() / "miniforge3/envs" / f"cocofest-{suite}"

    def detect_runtime(self):
        self.prefix.set(str(self.default_prefix(self.variables["solver"].get())))

    def current_config(self):
        return config_from_form({key: variable.get() for key, variable in self.variables.items()}, self.config)

    def selected_problem_type(self):
        """Return the architecture identifier selected in the top-level menu."""
        try:
            return PROBLEM_TYPE_BY_LABEL[self.problem_type.get()]
        except KeyError as error:
            raise ValueError("Type de problème GUI inconnu.") from error

    def update_problem_type_description(self):
        problem_type = PROBLEM_TYPE_BY_LABEL.get(self.problem_type.get(), UNILATERAL)
        self.problem_type_label.configure(text=PROBLEM_TYPE_DESCRIPTIONS[problem_type])

    def select_problem_type(self, *_):
        """Navigate without pretending that the three architectures are interchangeable."""
        problem_type = self.selected_problem_type()
        self.update_problem_type_description()
        if problem_type == INDEPENDENT_ARMS:
            if hasattr(self, "independent_arms_tab"):
                self.form_notebook.select(self.independent_arms_tab)
            return
        # The main-form representations share the same request schema.  Only
        # their explicit mechanical architecture flag changes here; no muscle
        # weights, objective, seed, or independent-arm request is copied.
        if hasattr(self, "variables"):
            self.variables["bilateral_reduced"].set(problem_type == BILATERAL_COMBINED)
            if problem_type == BILATERAL_COMBINED and self.variables["formulation"].get() == "isokinetic":
                self.variables["formulation"].set("dynamic")
            if hasattr(self, "main_problem_tab"):
                self.form_notebook.select(self.main_problem_tab)

    def sync_problem_type_from_main_form(self, *_):
        """Keep the architecture menu honest when the advanced checkbox is edited."""
        if self.selected_problem_type() == INDEPENDENT_ARMS:
            return
        selected = BILATERAL_COMBINED if self.variables["bilateral_reduced"].get() else UNILATERAL
        label = PROBLEM_TYPE_LABELS[selected]
        if self.problem_type.get() != label:
            self.problem_type.set(label)

    def current_rollout(self):
        fields = self.rollout_variables
        sources = json.loads(fields["sources"].get())
        if not isinstance(sources, list):
            raise ValueError("Sources : une liste JSON de chemins NPZ est requise.")
        return CrossRolloutConfig(sources=tuple(sources), reduced_profile=fields["reduced_profile"].get().strip(),
                                  model_config=fields["model_config"].get().strip() or None,
                                  cost_config=fields["cost_config"].get().strip() or None,
                                  evaluators=tuple(value.strip() for value in fields["evaluators"].get().split(",") if value.strip()),
                                  start_cycle=int(fields["start_cycle"].get()), cycles=int(fields["cycles"].get()),
                                  cycle_duration_s=float(fields["cycle_duration_s"].get()),
                                  dop853_rtol=float(fields["dop853_rtol"].get()), dop853_atol=float(fields["dop853_atol"].get()),
                                  samples_per_interval=int(fields["samples_per_interval"].get()),
                                  output_root=fields["output_root"].get().strip()).validate()

    def current_independent_arms(self):
        """Build the two-arm request without leaking its semantics into the main form."""
        fields = self.independent_arms_variables
        def optional_float(name):
            raw = fields[name].get().strip()
            return None if not raw else float(raw)
        def optional_text(name):
            raw = fields[name].get().strip()
            return raw or None
        def boolean(name):
            raw = fields[name].get().strip().lower()
            if raw not in {"true", "false"}:
                raise ValueError(f"{name} doit être true ou false.")
            return raw == "true"
        def solver_cpu(name):
            raw = fields[name].get().strip()
            if raw == "Automatique":
                return None
            if not raw.startswith("CPU "):
                raise ValueError(f"{name} doit être 'Automatique' ou 'CPU N'.")
            return int(raw[4:])
        return IndependentArmsGuiConfig(
            solver=fields["solver"].get().strip(), cycles=int(fields["cycles"].get()),
            cycles_per_window=int(fields["cycles_per_window"].get()),
            stimulations_per_cycle=int(fields["stimulations_per_cycle"].get()),
            isokinetic_omega=float(fields["isokinetic_omega"].get()),
            right_work_j_per_cycle=optional_float("right_work_j_per_cycle"),
            left_work_j_per_cycle=optional_float("left_work_j_per_cycle"),
            right_equivalent_mean_torque_nm=optional_float("right_equivalent_mean_torque_nm"),
            left_equivalent_mean_torque_nm=optional_float("left_equivalent_mean_torque_nm"),
            resistance_pace_policy=fields["resistance_pace_policy"].get().strip(),
            resistance_pace_initial_split_policy=fields["resistance_pace_initial_split_policy"].get().strip(),
            resistance_pace_update_every_cycles=int(fields["resistance_pace_update_every_cycles"].get()),
            resistance_pace_capacity_gain=float(fields["resistance_pace_capacity_gain"].get()),
            resistance_pace_smoothing=float(fields["resistance_pace_smoothing"].get()),
            resistance_pace_max_fraction_step=float(fields["resistance_pace_max_fraction_step"].get()),
            resistance_pace_minimum_arm_torque_nm=float(fields["resistance_pace_minimum_arm_torque_nm"].get()),
            parametric_fatigue_weights=boolean("parametric_fatigue_weights"),
            muscle_weight_policy=fields["muscle_weight_policy"].get().strip(),
            physio_update_every_cycles=int(fields["physio_update_every_cycles"].get()),
            physio_update_smoothing=float(fields["physio_update_smoothing"].get()),
            physio_update_max_log_step=float(fields["physio_update_max_log_step"].get()),
            physio_update_deadband_log=float(fields["physio_update_deadband_log"].get()),
            physio_update_min_relative_weight=float(fields["physio_update_min_relative_weight"].get()),
            physio_update_max_relative_weight=float(fields["physio_update_max_relative_weight"].get()),
            right_solver_cpu=solver_cpu("right_solver_cpu"),
            left_solver_cpu=solver_cpu("left_solver_cpu"),
            factory=optional_text("factory"),
            right_runner_config=optional_text("right_runner_config"),
            left_runner_config=optional_text("left_runner_config"),
            output_root=fields["output_root"].get().strip(),
        ).validate()

    def refresh_solver_cpu_menus(self):
        """Refresh per-arm CPU menus and disable dedicated CPUs used elsewhere."""
        menus = getattr(self, "_solver_cpu_menus", None)
        if not menus:
            return
        reserved = reserved_cpu_ids()
        selected = {}
        for side in ("right", "left"):
            raw = self.independent_arms_variables[f"{side}_solver_cpu"].get().strip()
            selected[side] = None if raw == "Automatique" else int(raw.removeprefix("CPU "))
        # A CPU selected for one arm is unavailable to the other arm immediately.
        for side, widget in menus.items():
            variable = self.independent_arms_variables[f"{side}_solver_cpu"]
            other = selected["left" if side == "right" else "right"]
            menu = self.window.nametowidget(widget["menu"])
            menu.delete(0, "end")
            menu.add_command(label="Automatique", command=self.tk._setit(variable, "Automatique", self._on_solver_cpu_change))
            disabled = reserved | ({other} if other is not None else set())
            for cpu in available_cpu_ids():
                label = f"CPU {cpu}"
                menu.add_command(label=label, command=self.tk._setit(variable, label, self._on_solver_cpu_change),
                                 state="disabled" if cpu in disabled else "normal")
            menu.configure(postcommand=self.refresh_solver_cpu_menus)

    def _on_solver_cpu_change(self, _value):
        self.refresh_solver_cpu_menus()
        self.validate_independent_arms()

    def record_campaign_launch(self, *, kind, solver, formulation, cycles, controls, cpu, output_root):
        """Persist a concise launch receipt only after the child process exists."""
        self.active_history_id = self.campaign_history.add(
            kind=str(kind), solver=str(solver), formulation=str(formulation), cycles=str(cycles),
            controls=str(controls), cpu=str(cpu), output_root=str(output_root),
        )
        self.refresh_campaign_history()
        root = Path(output_root).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        self._live_artifacts = LiveArtifacts(root)
        self._live_started_at = time.monotonic()
        self._live_finished_at = None
        self._next_live_update = 0

    def refresh_campaign_history(self):
        table = getattr(self, "campaign_history_table", None)
        if table is None:
            return
        table.delete(*table.get_children())
        for entry in reversed(self.campaign_history.entries()):
            table.insert("", "end", iid=entry.get("id"), values=tuple(str(entry.get(column, "")) for column in table["columns"]))

    def selected_campaign(self):
        selected = self.campaign_history_table.selection()
        return next((entry for entry in self.campaign_history.entries() if selected and entry.get("id") == selected[0]), None)

    def show_campaign_details(self):
        from tkinter import messagebox
        entry = self.selected_campaign()
        if entry is None:
            messagebox.showinfo("Campagne", "Sélectionnez une campagne dans le tableau.", parent=self.window)
            return
        self.set_text(self.summary_text, json.dumps(entry, indent=2, ensure_ascii=False))
        self.lower.select(self.summary_text)

    def analyze_history(self):
        entry = self.selected_campaign()
        if entry is not None:
            self.request_analysis(Path(entry["output_root"]))

    def analyze_active_run(self):
        if self.active_plan is not None and self.active_plan.result_json is not None:
            self.request_analysis(self.active_plan.result_json.parent)
        else:
            self.analysis_status.set("Lancez une simulation ou choisissez un dossier de résultats.")

    def analyze_folder(self):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.window, title="Dossier de résultats de simulation")
        if selected:
            self.request_analysis(Path(selected))

    def request_analysis(self, root):
        if self._analysis_busy:
            return
        root = Path(root).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        self._analysis_busy = True
        self.analysis_status.set("Génération des figures en cours…")
        def work():
            try:
                self._analysis_results.put((generate_analysis_figures(root), None))
            except Exception as error:
                self._analysis_results.put(([], f"{type(error).__name__}: {error}"))
        threading.Thread(target=work, name="gui-analysis", daemon=True).start()

    def poll_analysis(self):
        try:
            paths, error = self._analysis_results.get_nowait()
        except queue.Empty:
            return
        self._analysis_busy = False
        if error:
            self.analysis_status.set(error)
            return
        self.analysis_status.set("Figures enregistrées : " + ", ".join(str(path) for path in paths))
        popup = self.tk.Toplevel(self.window)
        popup.title("Analyse des solutions")
        notebook = self.ttk.Notebook(popup)
        notebook.pack(fill="both", expand=True)
        popup._figure_images = []
        for path in paths:
            image = self.tk.PhotoImage(file=str(path))
            factor = max(1, (image.width() + 999) // 1000, (image.height() + 649) // 650)
            image = image.subsample(factor, factor)
            popup._figure_images.append(image)
            frame = self.ttk.Frame(notebook)
            self.ttk.Label(frame, image=image).pack()
            self.ttk.Label(frame, text=str(path), wraplength=1000).pack()
            notebook.add(frame, text=path.stem)

    def refresh_independent_arms_sections(self):
        """Expose only the controls applicable to the selected two-arm policy."""
        fields = self.independent_arms_variables
        controls = self.independent_arms_controls
        adaptive_weights = fields["muscle_weight_policy"].get().strip() != "unit"
        if adaptive_weights and fields["parametric_fatigue_weights"].get() != "true":
            # Physio-U rewrites numeric objective parameters between cycles;
            # without this contract it would rebuild the compiled NLP.
            fields["parametric_fatigue_weights"].set("true")
        controls["parametric_fatigue_weights"].configure(state="disabled" if adaptive_weights else "readonly")
        for name in (
            "physio_update_every_cycles", "physio_update_smoothing", "physio_update_max_log_step",
            "physio_update_deadband_log", "physio_update_min_relative_weight", "physio_update_max_relative_weight",
        ):
            controls[name].configure(state="normal" if adaptive_weights else "disabled")
        capacity_feedback = fields["resistance_pace_policy"].get().strip() == "capacity_feedback"
        for name in (
            "resistance_pace_update_every_cycles", "resistance_pace_capacity_gain",
            "resistance_pace_smoothing", "resistance_pace_max_fraction_step",
            "resistance_pace_minimum_arm_torque_nm",
        ):
            controls[name].configure(state="normal" if capacity_feedback else "disabled")

    def validate_independent_arms(self, *_):
        self.refresh_independent_arms_sections()
        self.independent_solver_control.configure(values=independent_solver_choices(
            self.independent_arms_variables["factory"].get().strip() or None))
        try:
            config = self.current_independent_arms()
            occupied = reserved_cpu_ids()
            selected = {cpu for cpu in (config.right_solver_cpu, config.left_solver_cpu) if cpu is not None}
            conflict = sorted(selected & occupied)
            if conflict:
                raise ValueError(f"CPU déjà réservé par un processus actif : {', '.join(map(str, conflict))}.")
            right, left = config.target_work_j("right"), config.target_work_j("left")
            self.independent_arms_validation.configure(
                text=(f"Deux problèmes isocinétiques indépendants · Wdroite={right:.6g} J/tour · "
                      f"Wgauche={left:.6g} J/tour · exécution indépendante. "
                      f"Poids musculaires : {config.muscle_weight_policy}. "
                      "Le couple instantané est calculé par le coordinateur, non prescrit ici."),
                style="Success.TLabel")
            valid = not self.process.running
        except (ValueError, TypeError) as error:
            self.independent_arms_validation.configure(text=str(error), style="Error.TLabel")
            valid = False
        self.preview_independent_arms_button.configure(state="normal" if valid else "disabled")
        self.start_independent_arms_button.configure(state="normal" if valid else "disabled")
        endurance_valid = (valid and config.solver == "ipopt" and config.cycles_per_window == 1
                           and config.resistance_pace_policy == "capacity_feedback"
                           and config.muscle_weight_policy == "unit")
        self.preview_endurance_button.configure(state="normal" if endurance_valid else "disabled")
        self.start_endurance_button.configure(state="normal" if endurance_valid else "disabled")
        self.start_endurance_search_button.configure(state="normal" if endurance_valid else "disabled")

    def current_endurance_campaign(self):
        """Translate the two-arm form into one certified 600-cycle candidate."""
        config = self.current_independent_arms()
        if (config.solver != "ipopt" or config.cycles_per_window != 1
                or config.resistance_pace_policy != "capacity_feedback"
                or config.muscle_weight_policy != "unit"):
            raise ValueError("L'endurance 10 min exige IPOPT, fenêtres RHO d'un cycle, retour capacité et poids unitaires ; Physio-U se lance actuellement via « Lancer les deux bras ».")
        runtime = config.to_runtime_dict()
        runtime["runtime_prefix"] = str(Path(self.prefix.get()).expanduser())
        total = runtime["right_equivalent_mean_torque_nm"] + runtime["left_equivalent_mean_torque_nm"]
        names = ("Delt_ant", "Delt_post", "Biceps", "Triceps")
        candidate = BilateralEnduranceCandidate(total, runtime["right_equivalent_mean_torque_nm"] / total,
                                                 dict.fromkeys(names, 1.), dict.fromkeys(names, 1.)).validate()
        root = Path(config.output_root).expanduser()
        if not root.is_absolute():
            root = ROOT / root
        campaign = BilateralEnduranceCampaignConfig(
            base_payload=runtime, output_root=str(root) + "-endurance-600",
            fidelities=(30, 120, 300, 600), target_cycles=600,
            update_every_cycles=config.resistance_pace_update_every_cycles,
            adaptation_enabled=False,
        ).validate()
        return campaign, candidate

    def preview_endurance(self):
        from tkinter import messagebox
        try:
            campaign, candidate = self.current_endurance_campaign()
            path = Path(campaign.output_root).parent / f"{Path(campaign.output_root).name}.input.json"
            command = [str(Path(self.prefix.get()).expanduser() / "bin/python"), "-m",
                       "cocofest.simulation.bilateral_endurance_runner", "--campaign", str(path)]
            manifest = {"campaign": campaign.to_dict(), "candidate": candidate.canonical_dict()}
            self.set_text(self.command_text, " ".join(command) + "\n\n" + json.dumps(manifest, indent=2, ensure_ascii=False))
            self.lower.select(self.command_text)
            return campaign, candidate, path, command
        except (ValueError, OSError) as error:
            messagebox.showerror("Endurance bilatérale invalide", str(error), parent=self.window)
            return None

    def current_endurance_bracket(self):
        campaign, candidate = self.current_endurance_campaign()
        fields = self.independent_arms_variables
        bracket = BilateralTorqueBracketConfig(candidate.total_equivalent_mean_torque_nm,
            float(fields["endurance_torque_upper_nm"].get()),
            float(fields["endurance_torque_tolerance_nm"].get())).validate()
        return campaign, candidate, bracket

    def start_endurance(self):
        from tkinter import messagebox
        preview = self.preview_endurance()
        if preview is None:
            return
        campaign, candidate, path, command = preview
        try:
            root = Path(campaign.output_root)
            if root.exists():
                raise FileExistsError(f"Dossier de campagne existant : {root}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"campaign": campaign.to_dict(), "candidate": candidate.canonical_dict()}, indent=2) + "\n", encoding="utf-8")
            prefix = Path(self.prefix.get()).expanduser()
            plan = LaunchPlan(tuple(command), ROOT, {}, "rho32", None)
            self.process.start(plan, runtime_helpers().base_environment(prefix, "rho32", 1, 1), path.with_suffix(".log"))
            config = self.current_independent_arms()
            self.record_campaign_launch(kind="Endurance bilatérale", solver="ipopt", formulation="isocinétique",
                                        cycles="600", controls=config.stimulations_per_cycle, cpu="automatique",
                                        output_root=campaign.output_root)
            self.active_plan, self.active_config, self.active_campaign = plan, None, campaign
            self.active_rollout, self.active_independent_arms = None, None
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"Endurance bilatérale : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("Promotion stricte 30 → 120 → 300 → 600 cycles ; seul le palier 600 certifie le couple.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Lancement endurance impossible", str(error), parent=self.window)

    def start_endurance_search(self):
        """Launch a bounded scalar torque search; technical stops remain visible."""
        from tkinter import messagebox
        try:
            campaign, candidate, bracket = self.current_endurance_bracket()
            root = Path(campaign.output_root)
            if root.exists():
                raise FileExistsError(f"Dossier de campagne existant : {root}")
            path = root.parent / f"{root.name}.bracket-input.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"campaign": campaign.to_dict(), "candidate": candidate.canonical_dict(),
                                        "bracket": asdict(bracket)}, indent=2) + "\n", encoding="utf-8")
            prefix = Path(self.prefix.get()).expanduser()
            command = (str(prefix / "bin/python"), "-m", "cocofest.simulation.bilateral_endurance_runner", "--campaign", str(path))
            plan = LaunchPlan(command, ROOT, {}, "rho32", None)
            self.process.start(plan, runtime_helpers().base_environment(prefix, "rho32", 1, 1), path.with_suffix(".log"))
            config = self.current_independent_arms()
            self.record_campaign_launch(kind="Recherche endurance", solver="ipopt", formulation="isocinétique",
                                        cycles="≤600", controls=config.stimulations_per_cycle, cpu="automatique",
                                        output_root=campaign.output_root)
            self.active_plan, self.active_config, self.active_campaign = plan, None, campaign
            self.active_rollout, self.active_independent_arms = None, None
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"Recherche endurance : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("Dichotomie de τ : une erreur technique arrête la campagne, elle n'est jamais assimilée à de la fatigue.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Recherche endurance impossible", str(error), parent=self.window)

    def browse_independent_arms(self, field):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.window) if field == "output_root" else filedialog.askopenfilename(parent=self.window)
        if selected:
            self.independent_arms_variables[field].set(selected)

    def load_independent_arms(self):
        from tkinter import filedialog, messagebox
        selected = filedialog.askopenfilename(parent=self.window, filetypes=[("Requête JSON", "*.json")])
        if not selected:
            return
        try:
            config = IndependentArmsGuiConfig.from_json(Path(selected).read_text(encoding="utf-8"))
            for name, variable in self.independent_arms_variables.items():
                value = getattr(config, name)
                if name in ("right_solver_cpu", "left_solver_cpu"):
                    variable.set("Automatique" if value is None else f"CPU {value}")
                else:
                    variable.set("" if value is None else str(value))
            self.refresh_solver_cpu_menus()
        except (ValueError, TypeError, OSError) as error:
            messagebox.showerror("Requête deux-bras illisible", str(error), parent=self.window)

    def save_independent_arms(self):
        from tkinter import filedialog, messagebox
        try:
            config = self.current_independent_arms()
            selected = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json", filetypes=[("Requête JSON", "*.json")])
            if selected:
                Path(selected).write_text(config.to_json(), encoding="utf-8")
        except (ValueError, TypeError, OSError) as error:
            messagebox.showerror("Requête deux-bras invalide", str(error), parent=self.window)

    def independent_arms_plan(self, config):
        output = Path(config.output_root).expanduser()
        if not output.is_absolute():
            output = ROOT / output
        request = output / "independent-arms-request.json"
        prefix = Path(self.prefix.get()).expanduser()
        command = (str(prefix / "bin/python"), str(ROOT / "scripts/run_independent_isokinetic_arms.py"),
                   "--config", str(request), "--output-dir", str(output), "--prefix", str(prefix))
        return request, LaunchPlan(command, ROOT, {}, "rho32", output / "summary.json")

    def preview_independent_arms(self):
        from tkinter import messagebox
        try:
            config = self.current_independent_arms()
            request, plan = self.independent_arms_plan(config)
            self.set_text(self.command_text, plan.command + "\n\nRequête GUI :\n" + config.to_json() +
                          "\nContrat envoyé au coordinateur :\n" + json.dumps(config.to_runtime_dict(), indent=2, ensure_ascii=False))
            self.lower.select(self.command_text)
            return config, request, plan
        except (ValueError, OSError) as error:
            messagebox.showerror("Deux bras indépendants : configuration invalide", str(error), parent=self.window)
            return None

    def start_independent_arms(self):
        from tkinter import messagebox
        preview = self.preview_independent_arms()
        if preview is None:
            return
        config, request, plan = preview
        try:
            prefix = Path(self.prefix.get()).expanduser()
            if not (prefix / "bin/python").is_file():
                raise FileNotFoundError(f"Environnement absent : {prefix}")
            if request.exists():
                raise FileExistsError(f"Requête existante : choisissez un nouveau dossier ({request})")
            request.parent.mkdir(parents=True, exist_ok=True)
            request.write_text(json.dumps(config.to_runtime_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            environment = runtime_helpers().base_environment(prefix, plan.suite, 1, 1)
            self.process.start(plan, environment, plan.result_json.parent / "gui-run.log")
            self.record_campaign_launch(kind="Deux bras indépendants", solver=config.solver, formulation=config.formulation,
                                        cycles=config.cycles, controls=config.stimulations_per_cycle,
                                        cpu=(f"D:{config.right_solver_cpu} G:{config.left_solver_cpu}"
                                             if config.right_solver_cpu is not None else "automatique"),
                                        output_root=config.output_root)
            self.active_plan, self.active_config, self.active_campaign, self.active_rollout = plan, None, None, None
            self.active_independent_arms = config
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"Deux bras : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("Deux problèmes indépendants : attente de la synthèse du coordinateur.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
            self.validate()
            self.validate_campaign()
            self.validate_muscle_campaign()
            self.validate_rollout()
            self.validate_independent_arms()
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Lancement deux-bras impossible", str(error), parent=self.window)

    def apply_independent_adjustment(self):
        from tkinter import messagebox
        if self.active_plan is None or self.active_plan.result_json is None:
            return
        try:
            payload = json.loads(self.active_plan.result_json.read_text(encoding="utf-8"))
            proposal = payload.get("recommended_adjustment")
            if not isinstance(proposal, dict):
                raise ValueError("Aucune proposition de travail/couple équivalent n'est déclarée par le coordinateur.")
            for side in ("right", "left"):
                value = proposal.get(f"{side}_equivalent_mean_torque_nm")
                if not isinstance(value, (int, float)) or isinstance(value, bool):
                    raise ValueError(f"Proposition {side} absente ou invalide.")
                self.independent_arms_variables[f"{side}_work_j_per_cycle"].set("")
                self.independent_arms_variables[f"{side}_equivalent_mean_torque_nm"].set(f"{value:.12g}")
            self.form_notebook.select(self.independent_arms_tab)
            self.scientific_status.set("Proposition reportée. Lancez une nouvelle simulation : le coordinateur déclarera si les artefacts compilés sont réutilisés.")
        except (ValueError, OSError, TypeError) as error:
            messagebox.showerror("Proposition indisponible", str(error), parent=self.window)

    def validate_rollout(self, *_):
        try:
            config = self.current_rollout()
            self.rollout_validation.configure(text=f"{len(config.sources)} source(s) × {len(config.evaluators)} évaluateurs ; {config.cycles} cycle(s). "
                                              "Modèle, commandes, horizon et état initial seront vérifiés au chargement.", style="Success.TLabel")
            valid = not self.process.running
        except (ValueError, TypeError) as error:
            self.rollout_validation.configure(text=str(error), style="Error.TLabel")
            valid = False
        self.preview_rollout_button.configure(state="normal" if valid else "disabled")
        self.start_rollout_button.configure(state="normal" if valid else "disabled")

    def browse_rollout(self, field):
        from tkinter import filedialog
        if field == "sources":
            selected = filedialog.askopenfilenames(parent=self.window, filetypes=[("Solutions NPZ", "*.npz")])
            if selected:
                self.rollout_variables[field].set(json.dumps(list(selected), ensure_ascii=False))
            return
        selected = filedialog.askdirectory(parent=self.window) if field == "output_root" else filedialog.askopenfilename(parent=self.window)
        if selected:
            self.rollout_variables[field].set(selected)

    def load_rollout(self):
        from tkinter import filedialog, messagebox
        selected = filedialog.askopenfilename(parent=self.window, filetypes=[("Requête JSON", "*.json")])
        if not selected:
            return
        try:
            config = CrossRolloutConfig.from_json(Path(selected).read_text())
            for name, variable in self.rollout_variables.items():
                value = getattr(config, name)
                variable.set(json.dumps(list(value), ensure_ascii=False) if name == "sources" else
                             ", ".join(value) if name == "evaluators" else "" if value is None else str(value))
        except (ValueError, TypeError, OSError) as error:
            messagebox.showerror("Requête de rejeu illisible", str(error), parent=self.window)

    def save_rollout(self):
        from tkinter import filedialog, messagebox
        try:
            config = self.current_rollout()
            selected = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json", filetypes=[("Requête JSON", "*.json")])
            if selected:
                Path(selected).write_text(config.to_json(), encoding="utf-8")
        except (ValueError, TypeError, OSError) as error:
            messagebox.showerror("Requête de rejeu invalide", str(error), parent=self.window)

    def preview_rollout(self):
        from tkinter import messagebox
        try:
            config = self.current_rollout()
            plan = build_cross_rollout_plan(config, Path(self.prefix.get()).expanduser())
            self.set_text(self.command_text, plan.command+"\n\nRejeu ouvert ; score physique commun explicitement configuré.\n"
                          "DOP853 référence ; schémas IRK indépendants, sans réoptimisation ni appel NLP.\n"+config.to_json())
            self.lower.select(self.command_text)
            return config, plan
        except (ValueError, TypeError, OSError) as error:
            messagebox.showerror("Rejeu invalide", str(error), parent=self.window)

    def start_rollout(self):
        from tkinter import messagebox
        preview = self.preview_rollout()
        if preview is None:
            return
        config, plan = preview
        try:
            prefix = Path(self.prefix.get()).expanduser()
            if not (prefix / "bin/python").is_file():
                raise FileNotFoundError(f"Environnement absent : {prefix}")
            environment = runtime_helpers().base_environment(prefix, plan.suite, 1, 1)
            self.process.start(plan, environment, plan.result_json.parent / "gui-run.log")
            self.record_campaign_launch(kind="Rejeu croisé", solver="sans NLP", formulation="propagation",
                                        cycles=config.cycles, controls="—", cpu="automatique", output_root=config.output_root)
            self.active_plan, self.active_config, self.active_campaign, self.active_rollout = plan, None, None, config
            self.active_independent_arms = None
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"Rejeu ouvert : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("Évaluation croisée : référence DOP853, commandes fixes, propagation continue.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
            self.validate()
            self.validate_campaign()
            self.validate_muscle_campaign()
            self.validate_rollout()
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Lancement du rejeu impossible", str(error), parent=self.window)

    def select_reoptimized_rho(self):
        self.variables["mode"].set("rho")
        self.form_notebook.select(0)
        self.scientific_status.set("RHO réoptimisé : choisir le solveur et sa formulation, puis Lancer. "
                                   "Les décisions sont recalculées à chaque fenêtre ; les sources du rejeu ouvert ne sont pas utilisées.")

    @staticmethod
    def set_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def validate(self, *_):
        try:
            config = self.current_config()
            if not self.prefix.get().strip():
                raise ValueError("Choisissez un environnement du solveur.")
            self.validation.configure(text="Configuration compatible. Les fichiers et bibliothèques seront vérifiés au lancement.", style="Success.TLabel")
            self.set_text(self.summary_text, scientific_summary(config))
            valid = True
        except (ValueError, TypeError) as error:
            self.validation.configure(text=str(error), style="Error.TLabel")
            valid = False
        self.preview_button.configure(state="normal" if valid else "disabled")
        self.start_button.configure(state="normal" if valid and not self.process.running else "disabled")

    def preview(self):
        from tkinter import messagebox
        try:
            config = self.current_config()
            plan = build_launch_plan(config, Path(self.prefix.get()).expanduser())
            text = (plan.command + "\n\nRépertoire : " + str(plan.cwd) + "\n\nVariables :\n" +
                    json.dumps(plan.environment_updates, indent=2, ensure_ascii=False))
            if plan.resolved_config is not None:
                text += "\n\nConfiguration SHA-256 : " + plan.resolved_config.config_hash
            self.set_text(self.command_text, text)
            self.lower.select(self.command_text)
            return config, plan
        except (ValueError, OSError) as error:
            messagebox.showerror("Configuration invalide", str(error), parent=self.window)
            return None

    def current_campaign(self):
        fields = self.campaign_variables
        return BayesianCampaignConfig(
            base_config=self.current_config().to_dict(), output_root=fields["bo_output"].get().strip(),
            solvers=tuple(s.strip() for s in fields["bo_solvers"].get().split(",") if s.strip()),
            n_calls=int(fields["bo_calls"].get()), n_initial_points=int(fields["bo_initial"].get()),
            random_state=int(fields["bo_seed"].get()), collocation_degree_min=int(fields["bo_degree_min"].get()),
            collocation_degree_max=int(fields["bo_degree_max"].get()),
            thread_choices=tuple(int(v.strip()) for v in fields["bo_threads"].get().split(",") if v.strip()),
            metric=fields["bo_metric"].get(),
        ).validate()

    def validate_campaign(self, *_):
        try:
            config = self.current_campaign()
            self.campaign_validation.configure(text=campaign_summary(config), style="Success.TLabel")
            valid = not self.process.running
        except (ValueError, TypeError) as error:
            self.campaign_validation.configure(text=str(error), style="Error.TLabel")
            valid = False
        self.preview_campaign_button.configure(state="normal" if valid else "disabled")
        self.start_campaign_button.configure(state="normal" if valid else "disabled")

    def preview_campaign(self):
        from tkinter import messagebox
        try:
            campaign = self.current_campaign()
            root = Path(campaign.output_root).expanduser()
            config_path = root.parent / f"{root.name}.campaign-input.json"
            command = [str(Path(self.prefix.get()).expanduser() / "bin/python"), "-m", "cocofest.simulation.bayesian_runner",
                       "--campaign", str(config_path), "--prefix", str(Path(self.prefix.get()).expanduser())]
            self.set_text(self.command_text, " ".join(command) + "\n\n" + campaign_summary(campaign))
            self.lower.select(self.command_text)
            return campaign, root, config_path, command
        except (ValueError, OSError) as error:
            messagebox.showerror("Campagne invalide", str(error), parent=self.window)
            return None

    def start_campaign(self):
        from tkinter import messagebox
        preview = self.preview_campaign()
        if preview is None:
            return
        campaign, root, config_path, command = preview
        try:
            prefix = Path(self.prefix.get()).expanduser()
            if not (prefix / "bin/python").is_file():
                raise FileNotFoundError(f"Environnement absent : {prefix}")
            if root.exists():
                raise FileExistsError(f"Dossier de campagne existant : {root}")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("x", encoding="utf-8") as stream:
                stream.write(campaign.to_json())
            plan = LaunchPlan(tuple(command), ROOT, {}, "rho32", None)
            environment = runtime_helpers().base_environment(prefix, "rho32", 1, 1)
            self.process.start(plan, environment, config_path.with_suffix(".log"))
            base = self.current_config()
            self.record_campaign_launch(kind="BO solveur", solver=", ".join(campaign.solvers), formulation=base.formulation,
                                        cycles=base.cycles, controls=base.stimulations_per_cycle, cpu="selon essai",
                                        output_root=campaign.output_root)
            self.active_plan, self.active_config, self.active_campaign = plan, None, campaign
            self.active_rollout = None
            self.active_independent_arms = None
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"Campagne BO : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("Campagne : les essais certifiés et les échecs seront écrits dans campaign.jsonl.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
            self.validate()
            self.validate_campaign()
            self.validate_muscle_campaign()
            self.validate_rollout()
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Lancement de campagne impossible", str(error), parent=self.window)

    def current_muscle_campaign(self):
        fields = self.muscle_campaign_variables
        trials = int(fields["muscle_bo_trials"].get())
        startup = int(fields["muscle_bo_startup"].get())
        if startup >= trials:
            raise ValueError("Les essais initiaux doivent être strictement inférieurs au budget.")
        output = fields["muscle_bo_output"].get().strip()
        return muscle_weight_campaign(
            self.current_config(), output_root=output,
            study_name=f"muscle-weights-{Path(output).name}",
            muscle_names=fields["muscle_bo_names"].get(),
            scale_min=fields["muscle_bo_min"].get(), scale_max=fields["muscle_bo_max"].get(),
            trials=trials, workers=int(fields["muscle_bo_workers"].get()), startup_trials=startup,
            seed=int(fields["muscle_bo_seed"].get()), max_cycles=int(fields["muscle_bo_cycles"].get()),
            timeout_s=float(fields["muscle_bo_timeout"].get()),
        )

    def validate_muscle_campaign(self, *_):
        try:
            campaign = self.current_muscle_campaign()
            names = [name.removeprefix("muscle_weight_coordinate__") for name in campaign.search_space]
            reference = next(name for name in configured_muscle_names(self.current_config().model_config)
                             if name not in names)
            self.muscle_campaign_validation.configure(
                text=(f"BO Optuna parallèle · {campaign.n_trials} essais, {campaign.workers} workers · "
                      f"{campaign.max_cycles} cycles max. Coordonnées log centrées : {', '.join(names)} "
                      f"(référence {reference}).\n"
                      "Les horizons terminés sont censurés ; seuls les arrêts fatigue certifiés deviennent des observations."),
                style="Success.TLabel")
            valid = not self.process.running
        except (ValueError, TypeError) as error:
            self.muscle_campaign_validation.configure(text=str(error), style="Error.TLabel")
            valid = False
        self.preview_muscle_campaign_button.configure(state="normal" if valid else "disabled")
        self.start_muscle_campaign_button.configure(state="normal" if valid else "disabled")

    def preview_muscle_campaign(self):
        from tkinter import messagebox
        try:
            campaign = self.current_muscle_campaign()
            root = Path(campaign.output_root).expanduser()
            config_path = root.parent / f"{root.name}.campaign.json"
            command = [str(Path(self.prefix.get()).expanduser() / "bin/python"),
                       "-m", "cocofest.simulation.async_bayesian_runner",
                       "--campaign", str(config_path), "--prefix", str(Path(self.prefix.get()).expanduser())]
            self.set_text(self.command_text, " ".join(command) + "\n\n" + campaign.to_json())
            self.lower.select(self.command_text)
            return campaign, root, config_path, command
        except (ValueError, OSError) as error:
            messagebox.showerror("BO musculaire invalide", str(error), parent=self.window)
            return None

    def start_muscle_campaign(self):
        from tkinter import messagebox
        preview = self.preview_muscle_campaign()
        if preview is None:
            return
        campaign, root, config_path, command = preview
        try:
            prefix = Path(self.prefix.get()).expanduser()
            if not (prefix / "bin/python").is_file():
                raise FileNotFoundError(f"Environnement absent : {prefix}")
            if root.exists():
                raise FileExistsError(f"Dossier de campagne existant : {root}")
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with config_path.open("x", encoding="utf-8") as stream:
                stream.write(campaign.to_json())
            plan = LaunchPlan(tuple(command), ROOT, {}, "rho32", None)
            environment = runtime_helpers().base_environment(prefix, "rho32", 1, 1)
            self.process.start(plan, environment, config_path.with_suffix(".log"))
            base = self.current_config()
            self.record_campaign_launch(kind="BO poids musculaires", solver=base.solver, formulation=base.formulation,
                                        cycles=campaign.max_cycles, controls=base.stimulations_per_cycle,
                                        cpu=f"{campaign.workers} workers", output_root=campaign.output_root)
            self.active_plan, self.active_config, self.active_campaign = plan, None, campaign
            self.active_rollout = None
            self.active_independent_arms = None
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"BO musculaire : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("BO musculaire : poids copiés par essai avec provenance ; résultats censurés séparés des échecs.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
            self.validate()
            self.validate_campaign()
            self.validate_muscle_campaign()
            self.validate_rollout()
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Lancement du BO musculaire impossible", str(error), parent=self.window)

    def browse(self, field, kind):
        from tkinter import filedialog
        selected = filedialog.askdirectory(parent=self.window) if kind == "directory" else filedialog.askopenfilename(parent=self.window)
        if selected:
            self.variables[field].set(selected)

    def apply_config(self, config):
        self.config = config
        # Loading a main-form JSON must leave the independent-arm page: that
        # JSON has no independent-arm semantics or per-side target state.
        self.problem_type.set(PROBLEM_TYPE_LABELS[main_problem_type(config)])
        for name, value in form_values(config).items():
            self.variables[name].set(value)
        self.detect_runtime()

    def reset(self):
        self.apply_config(SimulationConfig(output_root=f"gui-results/{datetime.now():%Y%m%d-%H%M%S}"))

    def save(self):
        from tkinter import filedialog, messagebox
        try:
            config = self.current_config()
            filename = filedialog.asksaveasfilename(parent=self.window, defaultextension=".json", filetypes=[("Configuration JSON", "*.json")])
            if filename:
                Path(filename).write_text(config.to_json(), encoding="utf-8")
        except (ValueError, OSError) as error:
            messagebox.showerror("Enregistrement impossible", str(error), parent=self.window)

    def load(self):
        from tkinter import filedialog, messagebox
        filename = filedialog.askopenfilename(parent=self.window, filetypes=[("Configuration JSON", "*.json")])
        if not filename:
            return
        try:
            config = SimulationConfig.from_json(Path(filename).read_text(encoding="utf-8"))
            CapabilityRegistry.require_valid(config)
            self.apply_config(config)
        except (ValueError, OSError, TypeError) as error:
            messagebox.showerror("Configuration illisible", str(error), parent=self.window)

    def start(self):
        from tkinter import messagebox
        preview = self.preview()
        if preview is None:
            return
        config, plan = preview
        if config.dry_run:
            self.process_status.set("Processus : prévisualisation seulement")
            return
        try:
            prefix = Path(self.prefix.get()).expanduser()
            if not (prefix / "bin/python").is_file():
                raise FileNotFoundError(f"Environnement absent : {prefix}")
            input_fields = ["model_config"]
            if config.solver == "acados":
                input_fields.append("acados_ipopt_cycle1_seed")
            if config.mode in ("rho-physio", "rho-pace"):
                input_fields.append("weights_config")
            for name in input_fields:
                value = getattr(config, name)
                if value:
                    path = Path(value).expanduser()
                    if not path.is_absolute():
                        path = ROOT / path
                    if not path.is_file():
                        raise FileNotFoundError(f"{name} : fichier absent ({path})")
            environment = runtime_helpers().base_environment(prefix, plan.suite, config.threads, config.numeric_threads)
            environment.update(plan.environment_updates)
            self.process.start(plan, environment, plan.result_json.parent / "gui-run.log")
            self.record_campaign_launch(kind=config.mode.upper(), solver=config.solver, formulation=config.formulation,
                                        cycles=config.cycles, controls=config.stimulations_per_cycle,
                                        cpu=f"{config.threads} thread(s)", output_root=plan.result_json.parent)
            self.active_plan, self.active_config, self.active_campaign = plan, config, None
            self.active_rollout = None
            self.active_independent_arms = None
            self._completion_shown = False
            self.set_text(self.log_text, "")
            self.process_status.set(f"Processus : en cours (PID {self.process.process.pid})")
            self.scientific_status.set("Statut scientifique : en attente du résultat de cette exécution.")
            self.stop_button.configure(state="normal")
            self.lower.select(self.log_text)
            self.validate()
            self.validate_campaign()
            self.validate_muscle_campaign()
            self.validate_rollout()
        except (ValueError, OSError, RuntimeError) as error:
            messagebox.showerror("Lancement impossible", str(error), parent=self.window)

    def stop(self):
        self.process.stop()
        self.campaign_history.set_status(self.active_history_id, "arrêt demandé")
        self.refresh_campaign_history()
        self.process_status.set("Processus : arrêt demandé…")
        self.stop_button.configure(state="disabled")

    def poll(self):
        self.poll_analysis()
        if self._live_artifacts is not None and time.monotonic() >= self._next_live_update:
            self._next_live_update = time.monotonic() + 1.0
            if not self.process.running and self._live_finished_at is None:
                self._live_finished_at = time.monotonic()
            elapsed = (self._live_finished_at or time.monotonic()) - self._live_started_at
            pid = getattr(self.process.process, "pid", "—")
            state = "en cours" if self.process.running else "inactif"
            self.set_text(self.live_text, f"Processus {state} · PID {pid} · temps écoulé {elapsed:.1f} s\n\n" +
                          self._live_artifacts.update() + "\n\nLes métriques sont celles publiées par le moteur ; la durée écoulée inclut initialisation et compilation.")
        lines = self.process.drain()
        if lines:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", "".join(lines))
            # The full log remains on disk; keep the widget responsive in long campaigns.
            if int(self.log_text.index("end-1c").split(".")[0]) > 6000:
                self.log_text.delete("1.0", "1001.0")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")
        returncode = self.process.poll()
        if returncode is not None and not self.process.running and not self._completion_shown:
            self._completion_shown = True
            self.process_status.set(f"Processus : terminé, code {returncode}")
            self.stop_button.configure(state="disabled")
            status = "processus terminé (validation à consulter)" if returncode == 0 else ("arrêtée" if self.process.stop_requested_at is not None else f"échec ({returncode})")
            self.campaign_history.set_status(self.active_history_id, status)
            self.refresh_campaign_history()
            try:
                if self.active_campaign is not None:
                    summary = "Campagne terminée : consultez summary.json et le dossier trials de la campagne."
                elif self.active_independent_arms is not None:
                    payload = json.loads(self.active_plan.result_json.read_text(encoding="utf-8"))
                    summary = independent_arms_summary(payload, self.active_independent_arms.cycles)
                    self.set_text(self.summary_text, summary)
                    self.lower.select(self.summary_text)
                    self.apply_independent_adjustment_button.configure(
                        state="normal" if isinstance(payload.get("recommended_adjustment"), dict) else "disabled"
                    )
                elif self.active_rollout is not None:
                    payload = json.loads(self.active_plan.result_json.read_text(encoding="utf-8"))
                    summary = cross_rollout_summary(payload)
                    self.set_text(self.rollout_text, cross_rollout_report(payload))
                    self.lower.select(self.rollout_text)
                else:
                    payload = json.loads(self.active_plan.result_json.read_text(encoding="utf-8"))
                    summary = result_summary(payload, self.active_config.cycles)
            except (OSError, ValueError):
                summary = "Statut scientifique indisponible : aucun résultat JSON exploitable pour cette exécution."
            self.scientific_status.set(summary)
            self.validate()
            self.validate_campaign()
            self.validate_muscle_campaign()
            self.validate_rollout()
            self.validate_independent_arms()
        if self._closing and not self.process.running:
            self.window.destroy()
            return
        self.window.after(100, self.poll)

    def close(self):
        # Closing this task's GUI also stops its child process tree.
        if self._closing:
            return
        if self.process.running:
            from tkinter import messagebox
            if not messagebox.askyesno(
                "Simulation en cours",
                "Une simulation est en cours. Fermer la fenêtre arrêtera cette simulation.\n\n"
                "Arrêter la simulation et fermer ?",
                parent=self.window,
            ):
                return
            self._closing = True
            self.stop()
        else:
            self._closing = True
            self.window.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--prefix", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print the launch plan without opening a window or starting a solver")
    args = parser.parse_args(argv)
    config = SimulationConfig.from_json(args.config.read_text()) if args.config else SimulationConfig()
    CapabilityRegistry.require_valid(config)
    prefix = args.prefix or SimulationApp.default_prefix(config.solver)
    if args.dry_run:
        print(build_launch_plan(config, prefix).command)
        return 0
    import tkinter as tk
    try:
        window = tk.Tk()
    except tk.TclError as error:
        parser.exit(2, f"Affichage graphique indisponible : {error}\nUtilisez --dry-run pour prévisualiser sans écran.\n")
    SimulationApp(window, config if args.config else None, prefix)
    window.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
