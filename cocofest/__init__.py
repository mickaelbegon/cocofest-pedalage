"""Cocofest public API, with scientific dependencies loaded on first use.

Historical names remain available through from-cocofest imports. Lightweight
setup tools such as cocofest.simulation can run before solvers are installed.
"""
from importlib import import_module as _import_module

_EXPORTS = {
    "CustomObjective": (".custom_objectives", "CustomObjective"),
    "CustomConstraint": (".custom_constraints", "CustomConstraint"),
    "FesModel": (".models.fes_model", "FesModel"),
    "DingModelFrequency": ("cocofest.models.ding2003.ding2003", "DingModelFrequency"),
    "DingModelFrequencyWithFatigue": ("cocofest.models.ding2003.ding2003_with_fatigue", "DingModelFrequencyWithFatigue"),
    "DingModelPulseWidthFrequency": ("cocofest.models.ding2007.ding2007", "DingModelPulseWidthFrequency"),
    "DingModelPulseWidthFrequencyWithFatigue": ("cocofest.models.ding2007.ding2007_with_fatigue", "DingModelPulseWidthFrequencyWithFatigue"),
    "DingModelPulseWidthFrequencyWithFatiguePeriodic": ("cocofest.models.ding2007.ding2007_with_fatigue_periodic", "DingModelPulseWidthFrequencyWithFatiguePeriodic"),
    "DingModelPulseWidthFrequencyWithFatiguePeriodicNode": ("cocofest.models.ding2007.ding2007_with_fatigue_periodic_node", "DingModelPulseWidthFrequencyWithFatiguePeriodicNode"),
    "Marion2009ModelFrequency": ("cocofest.models.marion2009.marion2009", "Marion2009ModelFrequency"),
    "Marion2009ModelFrequencyWithFatigue": ("cocofest.models.marion2009.marion2009_with_fatigue", "Marion2009ModelFrequencyWithFatigue"),
    "Marion2009ModelPulseWidthFrequency": ("cocofest.models.marion2009.marion2009_modified", "Marion2009ModelPulseWidthFrequency"),
    "Marion2009ModelPulseWidthFrequencyWithFatigue": ("cocofest.models.marion2009.marion2009_modified_with_fatigue", "Marion2009ModelPulseWidthFrequencyWithFatigue"),
    "DingModelPulseIntensityFrequency": ("cocofest.models.hmed2018.hmed2018", "DingModelPulseIntensityFrequency"),
    "DingModelPulseIntensityFrequencyWithFatigue": ("cocofest.models.hmed2018.hmed2018_with_fatigue", "DingModelPulseIntensityFrequencyWithFatigue"),
    "VeltinkModelPulseIntensity": ("cocofest.models.veltink1992.veltink1992", "VeltinkModelPulseIntensity"),
    "VeltinkRienerModelPulseIntensityWithFatigue": ("cocofest.models.veltink1992.veltink1992_and_riener1998", "VeltinkRienerModelPulseIntensityWithFatigue"),
    "FesMskModel": (".models.dynamical_model", "FesMskModel"),
    "ReducedFesCyclingModel": (".models.reduced_cycling_model", "ReducedFesCyclingModel"),
    "duplicate_bilateral_muscles": (".models.reduced_cycling_model", "duplicate_bilateral_muscles"),
    "make_bilateral_reduced_dynamics": (".models.reduced_cycling_model", "make_bilateral_reduced_dynamics"),
    "ModelMaker": (".models.model_maker", "ModelMaker"),
    "OcpFes": (".optimization.fes_ocp", "OcpFes"),
    "OcpFesId": (".optimization.fes_id_ocp", "OcpFesId"),
    "OcpFesMsk": (".optimization.fes_ocp_multibody", "OcpFesMsk"),
    "FesNmpc": (".optimization.fes_nmpc", "FesNmpc"),
    "FesNmpcMsk": (".optimization.fes_nmpc_multibody", "FesNmpcMsk"),
    "IvpFes": (".integration.ivp_fes", "IvpFes"),
    "FourierSeries": (".fourier_approx", "FourierSeries"),
    "get_circle_coord": (".dynamics.inverse_kinematics_and_dynamics", "get_circle_coord"),
    "inverse_kinematics_cycling": (".dynamics.inverse_kinematics_and_dynamics", "inverse_kinematics_cycling"),
    "inverse_dynamics_cycling": (".dynamics.inverse_kinematics_and_dynamics", "inverse_dynamics_cycling"),
    "PeriodicFourierSeries": (".dynamics.reduced_cycling", "PeriodicFourierSeries"),
    "ReducedCyclingDynamics": (".dynamics.reduced_cycling", "ReducedCyclingDynamics"),
    "ReducedCyclingKinematics": (".dynamics.reduced_cycling", "ReducedCyclingKinematics"),
    "benchmark_reduced_casadi_mechanical_kernel": (".dynamics.reduced_cycling", "benchmark_reduced_casadi_mechanical_kernel"),
    "build_reduced_cycling_dynamics": (".dynamics.reduced_cycling", "build_reduced_cycling_dynamics"),
    "solve_cycling_contact_kinematics": (".dynamics.reduced_cycling", "solve_cycling_contact_kinematics"),
    "solve_bilateral_cycling_kinematics": (".dynamics.reduced_cycling", "solve_bilateral_cycling_kinematics"),
    "validate_reduced_cycling_dynamics": (".dynamics.reduced_cycling", "validate_reduced_cycling_dynamics"),
    "PlotCyclingResult": (".result.plot", "PlotCyclingResult"),
    "SolutionToPickle": (".result.pickle", "SolutionToPickle"),
    "FES_plot": (".result.graphics", "FES_plot"),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, symbol = target
    value = getattr(_import_module(module, __name__), symbol)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
