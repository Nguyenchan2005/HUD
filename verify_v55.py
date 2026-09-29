"""Tự kiểm tra cấu hình, cấu trúc, công thức trọng yếu và đầu ra của pipeline v5.5."""

from __future__ import annotations

import json
import copy
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from core import (PolySurface, build_final_scorecard, build_rays, chief_centered_geometric_spot_rms,
                  fan_imaging_metrics, fit_surface,
                  load_inputs, make_virtual_image, mirror_shape_gate,
                  polar_pattern, primary_pupils, ray_based_mtf)
from pipeline_v55 import (_actual_optics, _chief_index, _forward_distortion, _o2_mirror_gate_view, MAX_STEP, STAGES,
                          STEP_EXPLANATIONS_VI, validate_config)

ROOT = Path(__file__).resolve().parent
checks = []
def check(name: str, condition: bool) -> None:
    """Ghi điều kiện kiểm thử và dừng ngay nếu không đạt."""
    checks.append({"check": name, "pass": bool(condition)})
    if not condition: raise AssertionError(name)

config = json.loads((ROOT / "config_v55.json").read_text(encoding="utf-8")); validate_config(config)
multistart_source = (ROOT / "run_multistart_v55.py").read_text(encoding="utf-8")
def comment_count(value: object) -> int:
    """Đếm khóa _comment trong config JSON lồng nhau."""
    if isinstance(value, dict):
        return sum(1 for key in value if key.startswith("_comment")) + sum(comment_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(comment_count(v) for v in value)
    return 0
check("VIETNAMESE_CONFIG_COMMENTS_PRESENT", comment_count(config) >= 50)
scorecard_numerical = {
    "distortion": {"D_max_percent": 6.5, "D_partial_max_percent_diagnostic": 6.5,
                   "evaluation_status": "COMPLETE", "valid_bundle_count": 45,
                   "required_bundle_count": 45},
    "packaging_constraint_enabled": True, "packaging_lambda": 1.1,
    "visor_footprint": {"converged_VISOR_u_min_mm": -10.0,
                        "converged_VISOR_u_max_mm": 12.0,
                        "converged_VISOR_v_min_mm": -8.0,
                        "converged_VISOR_v_max_mm": 9.0},
    "visor_footprint_inside_66x42": True,
    "ray_count": 100, "forward_converged": 100, "forward_physical_valid": 100,
    "physical_sequential_validity": True, "fermat_all_converged": True,
    "fermat_convergence_history": [{"solve": "VERIFY", "success_count": 100,
                                    "gradient_pass_count": 100, "all_converged": True}],
    "MTF": {"complete": True, "minimum_MTF_at_max_frequency": .15},
    "final_mf1": {"MF1_Fan": 1.0, "active_ray_count": 100,
                  "physical_valid_ray_count": 100,
                  "physical_valid_global_RMS_diagnostic_mm": .2,
                  "certified_physical_MF1_Fan": 1.0},
    "final_mf2": {"S_AQP_signed_mm2": .25, "MF2": 0.0},
    "optics": {"achieved_mean": {"FOV_H_deg": 10.0, "FOV_V_deg": 4.0,
                                   "VID_center_mm": 8000.0, "D6_center_deg": 3.0,
                                   "azimuth_center_deg": 0.0}},
}
scorecard = build_final_scorecard(config, scorecard_numerical)
scorecard_by_name = {row["item"]: row for row in scorecard}
check("STEP25_SCORECARD_STORES_DYNAMIC_RULE_AND_REASON",
      scorecard_by_name["USER_FIXED_GRID_VECTOR_DISTORTION"]["operator"] == "<"
      and scorecard_by_name["USER_FIXED_GRID_VECTOR_DISTORTION"]["threshold"] ==
      config["distortion"]["hard_limit_percent"]
      and bool(scorecard_by_name["USER_FIXED_GRID_VECTOR_DISTORTION"]["reason"]))
check("STEP25_HARD_TOLERANCES_AND_MF2_ARE_REAL_GATES",
      scorecard_by_name["FOV_H_deg"]["operator"] == "<="
      and scorecard_by_name["FOV_H_deg"]["threshold"] == config["hard_tolerances"]["fov_h_deg"]
      and scorecard_by_name["VID_mm"]["status"] == "FAIL"
      and scorecard_by_name["SIGNED_MF2_OBSCURATION"]["status"] == "PASS")
obscured_numerical = copy.deepcopy(scorecard_numerical)
obscured_numerical["final_mf2"]["S_AQP_signed_mm2"] = -0.42
obscured_scorecard = {row["item"]: row for row in
                      build_final_scorecard(config, obscured_numerical)}
check("STEP25_FINAL_MF2_OBSCURATION_CANNOT_ESCAPE_HARD_GATE",
      obscured_scorecard["SIGNED_MF2_OBSCURATION"]["status"] == "FAIL"
      and "0.42" in obscured_scorecard["SIGNED_MF2_OBSCURATION"]["reason"])
stricter_scorecard_config = copy.deepcopy(config)
stricter_scorecard_config["packaging_lambda_max"] = 1.05
stricter_scorecard_config["mtf"]["minimum_at_max_frequency"] = .2
stricter_scorecard_config["hard_tolerances"]["fov_h_deg"] = .1
stricter_scorecard = {row["item"]: row for row in
                      build_final_scorecard(stricter_scorecard_config, scorecard_numerical)}
check("STEP25_SCORECARD_UPDATES_WHEN_CONFIG_THRESHOLD_CHANGES",
      stricter_scorecard["PACKAGING_LAMBDA"]["status"] == "FAIL"
      and "1.05" in stricter_scorecard["PACKAGING_LAMBDA"]["reason"]
      and stricter_scorecard["NUMERICAL_MTF_AT_MAX_FREQUENCY"]["status"] == "FAIL"
      and "0.2" in stricter_scorecard["NUMERICAL_MTF_AT_MAX_FREQUENCY"]["reason"]
      and stricter_scorecard["FOV_H_deg"]["status"] == "FAIL"
      and "0.1" in stricter_scorecard["FOV_H_deg"]["reason"])
wrapped_azimuth_config = copy.deepcopy(config)
wrapped_azimuth_config["azimuth_deg"] = 179.0
wrapped_azimuth_config["hard_tolerances"]["azimuth_deg"] = 3.0
wrapped_azimuth_numerical = copy.deepcopy(scorecard_numerical)
wrapped_azimuth_numerical["optics"]["achieved_mean"]["azimuth_center_deg"] = -179.0
wrapped_azimuth_scorecard = {row["item"]: row for row in
                             build_final_scorecard(wrapped_azimuth_config,
                                                   wrapped_azimuth_numerical)}
check("STEP25_AZIMUTH_ERROR_USES_CIRCULAR_SHORTEST_DISTANCE",
      wrapped_azimuth_scorecard["azimuth_deg"]["status"] == "PASS"
      and abs(float(wrapped_azimuth_scorecard["azimuth_deg"]["result"]) - 2.0) < 1e-12)
spot_verify = chief_centered_geometric_spot_rms(
    np.array([
        [0., 0., 0.],
        [2., 0., 0.],
        [0., 0., 0.],
        [0., 4., 0.],
    ]),
    np.ones(4, bool),
    np.array([0, 0, 1, 1]),
    np.zeros(4, int),
    np.array([
        True,
        False,
        True,
        False,
    ]),
    np.eye(3),
)

check(
    "CHIEF_CENTERED_PLANAR_SPOT_RMS_IS_FIELD_PUPIL_CENTERED",
    (
        abs(
            float(
                spot_verify[
                    "RMS_spot_radius_mm"
                ]
            )
            - np.sqrt(
                5.0
            )
        )
        < 1e-12
        and
        spot_verify[
            "valid_ray_count"
        ]
        == 4
    ),
)
sx, sy = np.meshgrid(np.linspace(-3.0, 3.0, 9), np.linspace(-2.0, 2.0, 7))
sx, sy = sx.ravel(), sy.ravel(); sphere_radius = 40.0
sz = sphere_radius-np.sqrt(sphere_radius**2-sx*sx-sy*sy)
sphere_points = np.column_stack([sx, sy, sz])
sphere_normals = np.column_stack([-sx/sphere_radius, -sy/sphere_radius,
                                  np.sqrt(1.0-(sx*sx+sy*sy)/sphere_radius**2)])
sphere_seed = PolySurface.plane("M1", np.zeros(3), np.array([0.0, 0.0, 1.0]), (5.0, 5.0))
sphere_model, sphere_stats = fit_surface(
    sphere_points, sphere_normals, sphere_seed, 2, True, config["fit_weights"],
    len(sphere_points)//2, config["surface_fit"], sphere_points+10.0*sphere_normals,
    sphere_points+10.0*sphere_normals)
check("CHIEF_VERTEX_CONSTRAINED_SPHERE_AND_GAUGED_VARIABLE_PROJECTION",
      abs(float(sphere_stats["base_sphere_radius_mm"])-sphere_radius) < 1e-6
      and abs(float(sphere_model.curvature)-1.0/sphere_radius) < 1e-8
      and abs(float(sphere_stats["chief_vertex_radial_residual_to_geometric_sphere_mm"])) < 1e-12
      and float(sphere_stats["polynomial_gauge"]["maximum_constraint_residual"]) < 1e-12
      and sphere_stats["joint_variable_projection"]["enabled"] is True
      and float(sphere_stats["sag_rms_mm"]) < 1e-9)
check("SINGLE_CANONICAL_CONFIG", (ROOT / "config_v55.json").exists()

      and not (ROOT / "manual_flow_v5_5" / "STEP_00" / "config.json").exists())
      # Kiem thu hoi quy: sphere dung, va hai truong hop nghiem bi ep cham bien.
check("CHIEF_SPHERE_USES_HARD_RADIUS_BOUNDS",
      sphere_stats["constrained_sphere_fit"]["bound_enforcement"]
      == "SCIPY_LEAST_SQUARES_BOUNDS"
      and sphere_stats["constrained_sphere_fit"]["radius_within_bounds"])

for case_name, factor_bounds, expected_factor in (
        ("UPPER", [0.25, 0.8], 0.8),
        ("LOWER", [1.2, 4.0], 1.2)):
    bounded_options = copy.deepcopy(config["surface_fit"])
    bounded_options["sphere_radius_factor_bounds"] = factor_bounds

    _, bounded_stats = fit_surface(
        sphere_points, sphere_normals, sphere_seed, 2, True,
        config["fit_weights"], len(sphere_points)//2, bounded_options)

    fit_diag = bounded_stats["constrained_sphere_fit"]
    bounded_radius = float(bounded_stats["base_sphere_radius_mm"])
    radius_lo, radius_hi = fit_diag["radius_bounds_mm"]
    chief_error = float(bounded_stats[
        "chief_vertex_radial_residual_to_geometric_sphere_mm"])

    check(f"CHIEF_SPHERE_{case_name}_BOUND_IS_ENFORCED",
          fit_diag["success"] and fit_diag["radius_within_bounds"]
          and radius_lo*(1.0-1e-6) <= bounded_radius <= radius_hi*(1.0+1e-6)
          and abs(bounded_radius - expected_factor*sphere_radius) < 1e-5
          and abs(chief_error) < 1e-10)
check("MULTISTART_RUNNER_PRESERVES_SINGLE_RUN", "run_multistart_v55.py" not in
      (ROOT / "run_pipeline_v55.py").read_text(encoding="utf-8"))
check(
    "MULTISTART_DEEP_RUNS_ONLY_SELECTED_BEST_GEOMETRY",
    "_select_step7_deep_run_schedule(" in multistart_source
    and "ctx = copy.deepcopy(base)" in multistart_source
    and 'ctx.data["step7_selected_geometry_rank"] = 1' in multistart_source
    and (
        'ctx.data["step7_deep_run_policy"] = "BEST_GEOMETRY_ONLY"' in multistart_source
        or "BEST_FIRST_WITH_RAW_CONVEX_M2_FALLBACK" in multistart_source
    )
    and "for step in range(8, 27)" in multistart_source,
)
check("MULTISTART_FAILURE_ISOLATION_AND_RANKING",
      "FAILED_DURING_PIPELINE" in multistart_source
      and "except Exception as exc" in multistart_source
      and '("COMPLETED", "FAILED_DURING_PIPELINE")' in multistart_source
      and "all hard gates pass" in multistart_source
      and "hard_gate_fail_count" in multistart_source
      and "MULTISTART_COMPARISON.json" in multistart_source)
deferred_renderer = (
    ROOT
    / "render_multistart_saved_v55.py"
)

check(
    "MULTISTART_DEFERS_ALL_VISUALIZATION",
    "from visualization_v55 import"
        not in multistart_source
    and "render_step(ctx"
        not in multistart_source
    and "render_surface_ray_view(ctx"
        not in multistart_source
    and "render_spot_evolution(ctx)"
        not in multistart_source
    and "render_planar_seed_precheck("
        not in multistart_source
    and "DEFERRED_VISUALIZATION_PLAN.json"
        in multistart_source
    and deferred_renderer.exists()
)

check(
    "MULTISTART_DEFERRED_RENDER_STEPS_ARE_CHECKPOINTED",
    "steps.update("
    "_deferred_render_steps(ctx)"
    ")" in multistart_source
)

check(
    "MULTISTART_INTERNAL_IMAGE_PATHS_DISABLED",
    config["execution"].get(
        "multistart_inline_rendering"
    ) is False
    and config["execution"].get(
        "render_step23_phase_images"
    ) is False
    and config["live_monitor"].get(
        "images_enabled"
    ) is False
)

deferred_source = (
    deferred_renderer.read_text(
        encoding="utf-8"
    )
)

check(
    "DEFERRED_RENDERER_DOES_NOT_RERUN_OPTICS",
    "_call_stage_algorithm"
        not in deferred_source
    and "execution_session("
        not in deferred_source
    and "solve_fermat_m2("
        not in deferred_source
    and "forward_shoot("
        not in deferred_source
)

check(
    "DEFERRED_RENDERER_REUSES_CANONICAL_RENDERERS",
    all(
        name in deferred_source
        for name in (
            "render_planar_seed_precheck",
            "render_step",
            "render_surface_ray_view",
            "render_spot_evolution",
        )
    )
)
refinement = config["geometry_refinement"]
check("GEOMETRY_REFINEMENT_CONFIGURATION_IS_EXPLICIT",
      isinstance(refinement["enabled"], bool))
check("GEOMETRY_REFINEMENT_POSE_ONLY_SCOPE",
      set(refinement["surfaces"]) == {"M1", "M2", "DISPLAY"}
      and refinement["surfaces"]["DISPLAY"]["enabled"] is False
      and not any(key in refinement for key in ("aperture", "curvature", "conic")))
check("MULTISTART_GEOMETRY_ONLY_FOR_TOP_K_FROM_STEP_22",
      "baseline_ranking[:min(int(refinement[\"top_k\"])" in multistart_source
      and '"CHECKPOINTS" / "STEP_22" / "state.pkl.gz"' in multistart_source
      and 'ctx.data["_geometry_refinement_runtime_enabled"] = False' in multistart_source
      and "for step in range(23, 27)" in multistart_source)
clean_input = load_inputs(Path(config["input_dir"]), config["hud_geometry_authority"])
expected_clean_names = {
    "Visor2_016_ULTRA_SMOOTH_CLEAN_CANONICAL.json",
    "Visor2_016_ULTRA_SMOOTH_CLEAN_VERIFICATION.json",
}
check("ONLY_TWO_CLEAN_INPUT_FILES_ARE_LOADED",
      {path.name for path in clean_input["files"]} == expected_clean_names
      and len(clean_input["files"]) == 2)
check("CLEAN_CANONICAL_IS_CERTIFIED_AND_COMPLETE",
      clean_input["clean_verification"]["status"] == "PASS"
      and len(clean_input["visor_inner"]) == 44785
      and clean_input["visor_grid_shape"] == (169, 265))
check("P1_P8_PACKAGING_DISABLED_IN_CANONICAL_CONFIG",
      config["hud_geometry_authority"]["packaging_vertices_mm"] is None
      and clean_input["packaging_vertices"] is None
      and clean_input["packaging_constraint_enabled"] is False)
free_packaging = copy.deepcopy(config)
free_packaging["hud_geometry_authority"]["packaging_vertices_mm"] = None
validate_config(free_packaging)
free_input = load_inputs(Path(free_packaging["input_dir"]), free_packaging["hud_geometry_authority"])
check("P1_P8_NULL_DISABLES_PACKAGING_WITHOUT_FALLBACK",
      free_input["packaging_vertices"] is None
      and free_input["packaging_constraint_enabled"] is False)
omitted_packaging = copy.deepcopy(config)
del omitted_packaging["hud_geometry_authority"]["packaging_vertices_mm"]
validate_config(omitted_packaging)
omitted_input = load_inputs(Path(omitted_packaging["input_dir"]), omitted_packaging["hud_geometry_authority"])
check("P1_P8_OMITTED_DISABLES_PACKAGING_WITHOUT_FALLBACK",
      omitted_input["packaging_vertices"] is None
      and omitted_input["packaging_constraint_enabled"] is False)
check("DISPLAY_REFERENCE_AUTO_AUTHORITY",
      config["fan_reference"]["mode"] == "AUTO_FROM_DISPLAY_SEED_AND_TARGET_VI_EXTENT"
      and config["fan_reference"]["M_x"] is None and config["fan_reference"]["M_y"] is None)
check("FIELD_GRID_CONFIG_PRESENT", set(config["field_grid"]) == {"horizontal_count", "vertical_count"})
check("ALL_RUNNERS_USE_CANONICAL_CONFIG",
      'ROOT / "config_v55.json"' in (ROOT / "manual_stage_runner_v55.py").read_text(encoding="utf-8")
      and 'default=ROOT / "config_v55.json"' in multistart_source)
grid3 = copy.deepcopy(config); grid3["field_grid"] = {"horizontal_count": 3, "vertical_count": 3}
validate_config(grid3)
vi3 = make_virtual_image(grid3)
pupil_count3 = int(grid3["primary_pupil_grid"]["horizontal_count"] * grid3["primary_pupil_grid"]["vertical_count"])
rays3 = build_rays(vi3["fields"], primary_pupils(
                   tuple(grid3["eyebox_y_mm"]), tuple(grid3["eyebox_z_mm"]),
                   (grid3["primary_pupil_grid"]["horizontal_count"],
                    grid3["primary_pupil_grid"]["vertical_count"])),
                   polar_pattern(49, grid3["pupil_diameter_mm"] / 2.0))
check("CONFIGURABLE_PRIMARY_PUPIL_GRID_SELECTS_TRUE_CENTER", len(vi3["fields"]) == 9
      and vi3["central_field_index"] == 4 and rays3["central_pupil_index"] == pupil_count3 // 2
      and int(rays3["pupil_index"][_chief_index(rays3)]) == pupil_count3 // 2
      and len(rays3["rows"]) == 9 * pupil_count3 * 49)

grid15 = copy.deepcopy(config); grid15["primary_pupil_grid"] = {"horizontal_count": 5, "vertical_count": 3}
validate_config(grid15)
pupils15 = primary_pupils(tuple(grid15["eyebox_y_mm"]), tuple(grid15["eyebox_z_mm"]), (5, 3))
check("CONFIGURABLE_5X3_PRIMARY_PUPIL_GRID_HAS_CENTER", len(pupils15) == 15
      and pupils15[7]["pupil_id"] == "C")

grid5 = copy.deepcopy(config); grid5["field_grid"] = {"horizontal_count": 5, "vertical_count": 3}
validate_config(grid5)
vi5 = make_virtual_image(grid5)
pupil_count5 = int(grid5["primary_pupil_grid"]["horizontal_count"] * grid5["primary_pupil_grid"]["vertical_count"])
pupils5 = primary_pupils(
    tuple(grid5["eyebox_y_mm"]), tuple(grid5["eyebox_z_mm"]),
    (grid5["primary_pupil_grid"]["horizontal_count"],
     grid5["primary_pupil_grid"]["vertical_count"]))
rays5 = build_rays(vi5["fields"], pupils5, polar_pattern(49, grid5["pupil_diameter_mm"] / 2.0))
check("CONFIGURABLE_5X3_FIELD_GRID_WITH_CONFIGURED_PUPILS", len(vi5["fields"]) == 15
      and vi5["central_field_index"] == 7 and len(rays5["rows"]) == 15 * pupil_count5 * 49)
ideal5 = np.asarray([f["vi_point"] for f in vi5["fields"]])
virtual5 = np.repeat(ideal5[:, None, :], rays5["pupil_count"], axis=1)
ctx5 = SimpleNamespace(data={"vi": vi5, "fixed_vi_grid": ideal5, "rays": rays5,
                             "pupils": pupils5}, config=grid5)
valid5 = np.ones((rays5["field_count"], rays5["pupil_count"]), bool)
forward_dist5 = _forward_distortion(ctx5, virtual5, valid5)
actual5 = _actual_optics(ctx5, virtual5, valid5)
check("DYNAMIC_CENTER_AND_DISTORTION_COUNTS",
      forward_dist5["evaluation_status"] == "COMPLETE"
      and forward_dist5["required_count"] == 14 * pupil_count5
      and actual5["evaluation_status"] == "COMPLETE"
      and abs(float(actual5["per_pupil"][len(pupils5)//2]["VID_center_mm"]) - float(grid5["vid_mm"])) < 1e-9)
invalid5 = valid5.copy(); invalid5[0, 0] = False
check("INCOMPLETE_FORWARD_BUNDLE_CANNOT_CERTIFY_DISTORTION_OR_OPTICS",
      _forward_distortion(ctx5, virtual5, invalid5)["D_max_percent"] is None
      and _actual_optics(ctx5, virtual5, invalid5)["achieved_mean"]["FOV_H_deg"] is None)

invalid_grid = copy.deepcopy(config); invalid_grid["field_grid"] = {"horizontal_count": 4, "vertical_count": 3}
try:
    validate_config(invalid_grid)
    even_grid_rejected = False
except ValueError:
    even_grid_rejected = True
check("EVEN_FIELD_GRID_REJECTED_TO_PRESERVE_CENTER", even_grid_rejected)
check("27_STAGES_00_TO_26", MAX_STEP == 26 and set(STAGES) == set(range(27)))
manual = ROOT / "manual_flow_v5_5"
check("27_MANUAL_FOLDERS", len(list(manual.glob("STEP_[0-9][0-9]"))) == 27)
check("CANONICAL_CONFIG_V55", config["schema"].endswith("V5_5"))
check("CLEAN_V5_5_ONLY_MANUAL_FLOW", not (ROOT / "manual_flow").exists() and manual.exists())
check("27_VISUALIZATION_WRAPPERS", len(list(manual.glob("STEP_*/visualization.py"))) == 27)
check("27_SURFACE_RAY_WRAPPERS", len(list(manual.glob("STEP_*/surface_rays.py"))) == 27)
algorithm_wrappers = list(manual.glob("STEP_*/algorithm.py"))
step_readmes = list(manual.glob("STEP_*/README.md"))
check("27_VIETNAMESE_STEP_EXPLANATIONS", set(STEP_EXPLANATIONS_VI) == set(range(27))
      and all(STEP_EXPLANATIONS_VI[i] for i in range(27)))
check("ALGORITHM_FILES_HAVE_VIETNAMESE_COMMENTS", len(algorithm_wrappers) == 27
      and all("# Thuật toán STEP_" in p.read_text(encoding="utf-8") for p in algorithm_wrappers))
check("STEP_READMES_HAVE_ALGORITHM_EXPLANATIONS", len(step_readmes) == 27
      and all("**Thuật toán:**" in p.read_text(encoding="utf-8") for p in step_readmes))
pipeline_source = (
    ROOT
    / "pipeline_v55.py"
).read_text(
    encoding="utf-8"
)


def pipeline_step_source(number: int) -> str:
    """Lấy đúng thân một STEP đã được hợp nhất trong pipeline_v55.py."""
    tree = ast.parse(pipeline_source)
    name = f"step_{number:02d}"
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == name
    )
    source_segment = ast.get_source_segment(pipeline_source, node)
    if source_segment is None:
        raise RuntimeError(f"PIPELINE_STEP_SOURCE_UNAVAILABLE:{name}")
    return source_segment


source = pipeline_source
step23_source = pipeline_step_source(23)
check(
    "STEP23_OPTIONAL_23A_23S_23G_23C_ORDER",
    step23_source.index(
        '_coefficient_dlsq_phase(ctx, "23A_COEFFICIENT_DLSQ")'
    )
    < step23_source.index(
        "_bounded_surface_parameter_refine(ctx)"
    )
    < step23_source.index(
        "_bounded_geometry_refine(ctx)"
    )
    < step23_source.index(
        '_coefficient_dlsq_phase(ctx, "23C_COEFFICIENT_DLSQ")'
    ),
)
check(
    "STEP23_WRITES_PHASE_IMAGES_AND_AUDIT_FILES",
    all(
        name in step23_source
        for name in (
            "PHASE_23A",
            "PHASE_23S",
            "PHASE_23G",
            "PHASE_23C",
            "23S_SURFACE_PARAMETER_HISTORY.csv",
            "23S_CURVATURE_CONIC_REFINED_SURFACES.json",
            "23G_GEOMETRY_HISTORY.csv",
            "23G_POSE_REFINED_SURFACES.json",
        )
    ),
)
step24_source = pipeline_step_source(24)
check(
    "STEP24_STORES_FINAL_REVERSE_TRACE_PROVENANCE",
    "final_reverse_trace" in step24_source
    and
    "final_reverse_trace_fingerprint"
    in step24_source
    and
    "final_reverse_trace_provenance"
    in step24_source,
)
step25_source = pipeline_step_source(25)
check(
    "STEP25_REUSES_STEP24_FINAL_REVERSE_TRACE",
    "final_reverse_trace" in step25_source
    and
    "final_reverse_trace_fingerprint"
    in step25_source
    and
    "trace_reverse(" not in step25_source,
)
check("NO_729_OR_3969_CORE_FLOW", "rays729" not in source and "rays3969" not in source)
check("NO_FIXED_NINE_FIELD_PIPELINE", "range(9)" not in source and "2205/2205" not in source
      and "03_FIELDS_9.csv" not in source)
check("DYNAMIC_FIELD_EXPORT_FILENAMES", 'fields_filename = f"03_FIELDS_{field_count}.csv"' in source
      and 'rays_filename = f"05_CHARACTERISTIC_RAYS_{ray_count}.csv"' in source)
visualization_source = (ROOT / "visualization_v55.py").read_text(encoding="utf-8")
manual_runner_source = (ROOT / "manual_stage_runner_v55.py").read_text(encoding="utf-8")
check("VISUALIZATION_FIELD_GRID_IS_DYNAMIC", "reshape(3, 3, 3)" not in visualization_source
      and "range(9)" not in visualization_source and "_field_shape" in visualization_source)
check("STEP17_WRITES_LIVE_BRANCH_IMAGES",
      "render_step17_branch_progress" in source
      and "render_step17_branch_progress" in visualization_source
      and "LATEST_BRANCH_PROGRESS.png" in visualization_source
      and "SOLVING_FERMAT" in source and "FERMAT_PASS" in source)
check("SPOT_EVOLUTION_FROM_THREE_REAL_SNAPSHOTS",
      all(f'_capture_spot_snapshot(ctx, "{label}", {step})' in source
          for label, step in (("PLANAR", 8), ("ORDER_2", 12), ("ORDER_5", 17)))
      and "render_spot_evolution(ctx)" in source
      and "render_spot_evolution(ctx)" in manual_runner_source
      and "20_SPOT_EVOLUTION_PLANAR_ORDER2_ORDER5.png" in visualization_source
      and '"paper_values_reused": False' in visualization_source)
check("STEP20_SINGLE_PAPER_STYLE_CENTROIDED_SPOT_OUTPUT",
      "HUD_FAN_V5_5_PAPER_STYLE_CENTROIDED_SPOT_EVOLUTION" in visualization_source
      and '"single_step20_spot_output": True' in visualization_source
      and "CENTROIDED_GEOMETRIC_SPOT_RMS_RADIUS_BY_FIELD_AND_PUPIL" in visualization_source
      and "PAPER_FAN_FIGURE8_SPOT_RMS_UM" in visualization_source
      and '(folder / obsolete).unlink(missing_ok=True)' in visualization_source)
check("STEP20_PUPIL_RMS_PLOT_IS_DYNAMIC",
      "range(len(pupil_rms))" in visualization_source
      and "Five outer-pupil RMS terms" not in visualization_source)
check("PRIMARY_PUPIL_GRID_HAS_NO_FIXED_FIVE_PUPIL_FLOW_ASSUMPTION",
      "field_count * 5 * 49" not in source
      and "outer sum of five pupil" not in visualization_source
      and "8 mm footprints" not in visualization_source
      and "get(\"pupil_count\", 5)" not in visualization_source)
check("STEP20_MANUAL_SPOT_WRAPPER", (manual / "STEP_20" / "spot_evolution.py").exists())
check("EVERY_STEP_WRITES_RECEIVING_SURFACE_RAY_VIEW",
      "render_surface_ray_view(ctx, step)" in manual_runner_source
      and "render_surface_ray_view(ctx, number)" in source
      and "CURRENT_STEP_CONTEXT_NO_INVENTED_RAYS" in visualization_source)
check("SEPARATE_REFERENCE_FILES", "08A_FAN_DYNAMIC_REFERENCE_GRID_FINAL.csv" in source and "08B_DISTORTION_FIXED_IDEAL_GRID.csv" in source)
check("FORWARD_FINAL", "forward_shoot(" in source and "reconstruct_virtual_points(" in source)
core_source = (ROOT / "core.py").read_text(encoding="utf-8")
check("LOADER_HAS_NO_LEGACY_INPUT_DEPENDENCY",
      "Toa_do_HUD_Trang_tinh1_2.json" not in core_source
      and "Untitled_spreadsheet_Sheet1_2.json" not in core_source)
check("CONFIG_DRIVEN_FIELD_GENERATION", 'config["field_grid"]' in core_source
      and "np.linspace" in core_source and 'rays["central_field_index"]' in core_source)
check(
    "STEP7_AUTO_GEOMETRY_PHYSICAL_UNOBSCURED_OPTIONAL_PACKAGING_COMPACTNESS_RANKED",
    "_step7_auto_seed_descriptors" in source
    and "_planar_seed_rank_key" in source
    and "PACKAGING_LAMBDA_EXCEEDS_LIMIT" in source
    and '"rms_is_hard_gate": False' in source
    and "PLANAR_CHIEF_CENTERED_SPOT_RMS_NOT_STRICTLY_BELOW_LIMIT" not in source
    and "chief_centered_geometric_spot_rms" in source
    and 'orientation_result_flip": False' in source
    and "packaging_raw is None or packaging_raw == []" in core_source
)
check("FERMAT_FAILURE_IS_HARD", "_accept_fermat_domain" not in source
      and "FERMAT_STATIONARY_POINT_NOT_CONVERGED_FOR_ALL_RAYS" in source
      and 'if not audit["all_converged"]:' in source
      and '"partial_fermat_accepted": False' in source)
check("FAILED_LATER_FERMAT_RETAINS_ONLY_FULLY_VALID_INCUMBENT",
      "TERMINATED_RETAIN_LAST_FULLY_VALID_STATE_NO_ADMISSIBLE_NEXT_CYCLE" in source
      and "accepted_cycles_this_order > 0" in source
      and "incumbent_shape_changed_by_rejected_cycle\": False" in source
      and "ACCEPTED_CONSTRUCTION_PATH_ONLY" in source)
check("SIGNED_RHO_BRANCHES", any(x < 0 for x in config["solver"]["rho_candidates"])
      and any(x > 0 for x in config["solver"]["rho_candidates"]))
check(
    "STEP15_16_COARSE_TO_FINE_FULL_PHYSICAL_RHO_SEARCH",
    config["solver"]["rho_o2_search"]["mode"]
    == "COARSE_TO_FINE_FULL_PHYSICAL"
    and int(config["solver"]["rho_o2_search"]["grid_points"]) >= 5
    and int(config["solver"]["rho_o2_search"]["grid_points"]) % 2 == 1
    and "FIXED_POST_STEP14_ORDER2_CONTEXT_FOR_EVERY_RHO" in source
    and "RESTORATION_ONLY_NOT_ELIGIBLE_FOR_STEP16_BEST" in source
    and "HARD_FULL_PHYSICAL_THEN_FAN_THEN_BUNDLE_P95" in source
    and "16_RHO_COARSE_TO_FINE_SUMMARY.json" in source
    and 'ctx.data["_rho_frontier"] = [' in source
    and '"retained_branch_count": 1' in source
)
check(
    "OPTICAL_CONVERGENCE_DIAGNOSTIC_IS_PRESENT_AND_NONAUTHORITATIVE_BEFORE_STEP24",
    bool(
        config["execution"][
            "optical_convergence"
        ]["enabled"]
    )
    and "def _record_optical_convergence(" in source
    and "OPTICAL_CONVERGENCE_HISTORY.csv" in source
    and '"STEP12 O2"' in source
    and '"STEP16 rho="' in source
    and 'f"STEP17 O{int(order)}"' in source
    and '"STEP23 FINAL OPT"' in source
    and '"STEP24 AUTHORITY"' in source
    and "INTERMEDIATE_DIAGNOSTIC_NOT_GATE" in source
    and "STEP24_NUMERICAL_AUTHORITY" in source
    and "authority=True" in source
)
check("FAN_ENGINEERING_SEPARATED", "engineering_constraints_outside_Fan_core" in source
      and "J_Fan_dimensionless" in source and "J_engineering_dimensionless" in source)
check("FORWARD_ENGINEERING_ACCEPTANCE_GATE_OUTSIDE_FAN_JACOBIAN",
      "_forward_engineering_eval" in source
      and "evaluate_engineering=lambda" in source
      and "EXTERNAL_FORWARD_MONITOR_ACCEPTANCE_GATE_NOT_FAN_RESIDUAL" in source)
check("FORWARD_MONITOR_CANNOT_IMPROVE_BY_DROPPING_EVALUATED_PAIRS",
      "_forward_distortion_evaluated_pair_mask" in source
      and "candidate_mask[base_mask]" in core_source)
check("FOOTPRINT_USES_CONVERGED_RAYS_AND_PHYSICAL_GATE_REQUIRES_ALL",
      'valid_visor_loc = visor_loc[fwd["converged"]]' in source
      and 'physical = bool(converged == len(r["rows"]))' in source)
check("ALL_ORDER5_NON_PISTON_TERMS_OPTIMIZED", 'if t != (0, 0)' in source)
check("CLEAR_APERTURE_SURFACE_SANITY", "CONVEX_CLEAR_APERTURE_POLYGON_PLUS_BOUNDARY_VERTICES" in core_source)
shape_policy_verify = {
    "maximum_freeform_departure_mm": 1.0,
    "maximum_normal_departure_deg": 45.0,
    "maximum_principal_curvature_per_mm": 0.2,
}
bowl_verify = PolySurface(
    "VERIFY_BOWL", np.zeros(3), np.eye(3), np.array([5.0, 5.0]),
    [(2, 0), (0, 2)], np.array([0.1, 0.1]), np.array([5.0, 5.0]))
tolerated_kg_verify = PolySurface(
    "VERIFY_TOLERATED_KG", np.zeros(3), np.eye(3), np.array([5.0, 5.0]),
    [(2, 0), (0, 2)], np.array([0.01, -0.00078125]), np.array([5.0, 5.0]))
rejected_kg_verify = PolySurface(
    "VERIFY_REJECTED_KG", np.zeros(3), np.eye(3), np.array([5.0, 5.0]),
    [(2, 0), (0, 2)], np.array([0.01, -0.003125]), np.array([5.0, 5.0]))
saddle_verify = PolySurface(
    "VERIFY_SADDLE", np.zeros(3), np.eye(3), np.array([5.0, 5.0]),
    [(2, 0), (0, 2)], np.array([0.1, -0.1]), np.array([5.0, 5.0]))
bowl_gate_verify = mirror_shape_gate(bowl_verify, shape_policy_verify)
tolerated_kg_gate_verify = mirror_shape_gate(
    tolerated_kg_verify,
    shape_policy_verify,
    orientation_sign=1.0,
)
rejected_kg_gate_verify = mirror_shape_gate(
    rejected_kg_verify,
    shape_policy_verify,
    orientation_sign=1.0,
)
saddle_gate_verify = mirror_shape_gate(saddle_verify, shape_policy_verify)
check("MIRROR_SHAPE_GATE_SINGLE_BOWL_NUMERICAL",
      bowl_gate_verify["pass"]
      and bowl_gate_verify["KG_min_per_mm2"] > 0.0
      and bowl_gate_verify["curvature_sign_flip_count"] == 0
      and tolerated_kg_gate_verify["pass"]
      and -1e-7 <= tolerated_kg_gate_verify["KG_min_per_mm2"] < 0.0
      and tolerated_kg_gate_verify["oriented_H_min_per_mm"] > 0.0
      and not rejected_kg_gate_verify["pass"]
      and rejected_kg_gate_verify["KG_min_per_mm2"] < -1e-7
      and not saddle_gate_verify["pass"]
      and saddle_gate_verify["KG_min_per_mm2"] < 0.0)

o2_warn_quality = copy.deepcopy(
    config[
        "surface_fit"
    ][
        "step12_quality_gates"
    ]
)

o2_warn_quality[
    "maximum_sag_rms_mm"
] = 1e-12

o2_bowl_view = _o2_mirror_gate_view(
    bowl_gate_verify,
    o2_warn_quality,
)

o2_saddle_view = _o2_mirror_gate_view(
    saddle_gate_verify,
    o2_warn_quality,
)

check(
    "O2_TOPOLOGY_HARD_SHAPE_QUALITY_WARN_SPLIT",
    o2_bowl_view[
        "topology_pass"
    ]
    and
    o2_bowl_view[
        "quality_status"
    ]
    == "WARN"
    and
    not o2_saddle_view[
        "topology_pass"
    ],
)
check("STEP11_SYSTEM_AWARE_O2_PAIR_SEARCH_ACTIVE",
      "SYSTEM_AWARE_O2_PAIR_CONSTRUCTION" in source
      and "11_M1_PHYSICAL_DOF_SCALING.csv" in source
      and "11_O2_SEARCH_HISTORY.csv" in source
      and "STEP11_NO_FEASIBLE_O2_PAIR" in source
      and "FEASIBLE_FIRST_THEN_ACTUAL_OPTICAL_OBJECTIVE_THEN_M2_REPRESENTABILITY" in source
      and "_step11_shape_curvature_record_fields" in source
      and "M1_KG_min_per_mm2" in source
      and "M2_KG_min_per_mm2" in source)
check(
    "STEP11_TOPOLOGY_CONFIGURABLE_QUALITY_SOFT_ARCHITECTURE",
    "M1_MIRROR_SHAPE_GATE" in source
    and "M2_MIRROR_SHAPE_GATE" in source
    and '"topology_is_hard_constraint_not_merit":' in source
    and '"shape_quality_enforcement":' in source
    and '"ci_trust_enforcement":' in source
    and '"physical_quality_status":' in source
    and "SOFT_PREFERRED_THRESHOLD_FOR_RANK_BUCKET" in source,
)
check("RAY_FACING_MIRROR_TOPOLOGY_AUTHORITY_ACTIVE",
      config["surface_fit"]["mirror_topology_authority"].get("enforcement")
      in ({"M1": "HARD", "M2": "WARN"}, {"M1": "HARD", "M2": "HARD"})
      and config["surface_fit"]["mirror_topology_authority"].get("M1")
      == "CONCAVE_RAY_FACING"
      and config["surface_fit"]["mirror_topology_authority"].get("M2")
      == "CONVEX_RAY_FACING"
      and "def _mirror_topology_gate(" in pipeline_source
      and "MIRROR_TOPOLOGY_CHIEF_RAY_GRAZING" in pipeline_source
      and pipeline_source.count("_mirror_topology_gate(") >= 7
      and "candidate_surface_valid" in pipeline_source)
check("STEP16_17_FEASIBLE_FIRST_RESTORATION_AND_FINAL_FULL_PHYSICAL_GATE",
      "PHYSICAL_FIRST_HIT_INCOMPLETE" in source
      and ("CI_NO_ADMISSIBLE_ORDER5_START_BEFORE_STEP_18" in source
           or "CI_RESTORATION_DID_NOT_RECOVER_FULL_PHYSICAL_TRACE_BEFORE_STEP_18" in source)
      and "FEASIBLE_FIRST_THEN_LIMITED_RESTORATION" in source
      and "SIGNED_MF2_OBSCURATION_FAIL" in source
      and config["solver"]["require_full_physical_ci"] is True
      and config["solver"]["restoration_branch_enabled"] is True
      and config["solver"]["require_unobscured_ci"] is True)
check("RHO_BEAM_AND_PAPER_SPOT_RANKING_ACTIVE",
      int(config["solver"]["rho_beam_width"]) > 1
      and "chief_centered_spot_RMS_mm" in source
      and "FEASIBLE_FIRST_THEN_LIMITED_RESTORATION_THEN_FAN_MERIT_THEN_PAPER_SPOT" in source)
check("STEP12_ACTUAL_O2_REFINEMENT_ACTIVE",
      "12_O2_REFINEMENT_HISTORY.csv" in source
      and "12_O2_ACTUAL_SURFACE_METRICS.json" in source
      and "12_O2_ACTUAL_SURFACE_GATES.csv" in source
      and "ORDER2_ACTUAL_O2_GATE_FAIL" in source
      and config["surface_fit"]["step12_o2_refinement"]["enabled"] is True
      and config["surface_fit"]["step12_quality_gates"]["enforcement"] in ("WARN", "HARD"))
check("STEP12_CI_CLOUD_IS_DIAGNOSTIC_ONLY",
      "12_CLOUD_DIAGNOSTICS_ROLE.json" in source
      and '"acceptance_gate":' in source
      and '"DIAGNOSTIC_ONLY"' in source
      and '"step12_effective_integrability_enforcement":' in source)
check("BUNDLE_INTEGRABILITY_DIAGNOSTICS_AND_GATES_ACTIVE",
      "FIELD_PUPIL_BUNDLE" in core_source
      and "local_geometry_normal_rms_deg" in core_source
      and "loop_abs_circulation_p95_mm" in core_source
      and "edge_gradient_height_residual_p95_mm" in core_source
      and "17_SELECTED_CI_CLOUD_INTEGRABILITY.json" in source)
check("CI_PATCH_V3_PRODUCTION_POLICY",
      config["ci_construction"]["mode"] == "SURFACE_COMPATIBLE_PATCH_V3")
check("CI_INTEGRABILITY_HARD_RETAINED_FOR_RECONSTRUCTION",
      config["surface_fit"]["integrability_gates"]["enforcement"] == "HARD"
      and "ORDER_{order}_CI_CLOUD_INTEGRABILITY_DIAGNOSTIC_GATE_FAIL" in source)
check("UNIT_NORMAL_TANGENT_METRIC_AND_CONIC_GAUGE_ACTIVE",
      "TARGET_UNIT_NORMAL_DOT_SURFACE_TANGENTS" in core_source
      and "A20+A02=0_BASE_CONIC_OWNS_SYMMETRIC_QUADRATIC_POWER" in core_source
      and "raw_slope_euclidean_objective_used\": False" in core_source)
check("K_PROFILE_AND_SOURCE_PROVENANCE_ACTIVE",
      "k_profile_scan" in core_source and "00_ALGORITHM_SOURCE_MANIFEST.json" in source
      and "algorithm_source_manifest_sha256" in visualization_source)
check(
    "STEP17_O4_O5_SPOT_REGRESSION_GUARD_ACTIVE",
    config["solver"]["step17_spot_guard"]["enabled"] is True
    and int(
        config["solver"]["step17_spot_guard"][
            "minimum_order"
        ]
    ) == 4
    and float(
        config["solver"]["step17_spot_guard"][
            "maximum_regression_mm"
        ]
    ) >= 0.0
    and "SPOT_RMS_REGRESSION_EXCEEDS_TOLERANCE"
    in source
    and "spot_RMS_guard_pass" in source,
)
check(
    "STEP17_O4_O5_ZERO_PAD_CONTINUATION_ACTIVE",
    "_promote_ci_state_without_shape_change"
    in source
    and "eligible_target_orders" in source
    and "basis_promotion_orders" in source
    and "order4_zero_pad_promotion" in source,
)
check("ORDER5_ZERO_PAD_IS_NOT_LABELLED_AS_SHAPE_IMPROVEMENT",
      "ORDER5_BASIS_ZERO_PAD_ONLY_" in source
      and "SHAPE_UNCHANGED_FROM_ORDER" in source
      and '"accepted_shape_update": False' in source
      and '"basis_promoted": True, "shape_optimized": False' in source
      and "shape_optimized_through_order" in source)
check("CI_ORDER_LEVELS_ARE_NESTED_AND_RETAIN_A22",
      "preserve_fan_axis_order2" in core_source
      and "FAN_AXIS_ORDER2_UNION_TOTAL_XY_ORDER_3" in source
      and "active_terms_never_dropped_during_order_transition" in source)
check("DLSQ_FILTER_RESTORATION_AND_REJECTION_AUDIT_ACTIVE",
      "ENGINEERING_FILTER_RESTORATION_STEP" in core_source
      and "ENGINEERING_COORDINATE_POLL_RESTORATION_STEP" in core_source
      and "BLOCKED_NO_ADMISSIBLE_DLSQ_TRIAL" in core_source
      and "23_DLSQ_TRIALS.csv" in source
      and "23_DLSQ_TERMINATION.json" in source)
check("MTF_DECLARED_NUMERICAL_METHOD", "ray_based_mtf(" in source
      and "NUMERICAL_APPROXIMATION_NOT_COMMERCIAL_CODE_DIFFRACTION_MTF" in core_source
      and "np.fft" not in source and "np.fft" not in core_source)
check("NO_FAKE_ZEMAX_FILE", ".zmx" not in source and ".zos" not in source)

multistart_source = (ROOT / "run_multistart_v55.py").read_text(encoding="utf-8")
render_saved_source = (ROOT / "render_multistart_saved_v55.py").read_text(encoding="utf-8")

check("MULTISTART_SAVES_RENDER_STATE_FOR_EVERY_COMPLETED_STEP",
      "_save_render_state(" in multistart_source
      and 'execution_status="ALGORITHM_COMPLETED"' in multistart_source
      and "RENDER_STATES" in multistart_source)
check("MULTISTART_SAVES_PARTIAL_RENDER_STATE_ON_FAILURE",
      'execution_status="ALGORITHM_FAILED"' in multistart_source
      and "PARTIAL_FAILURE_STATE" in multistart_source)
check("DEFERRED_RENDERER_SCANS_ALL_27_STEPS",
      "range(start_step, end_step + 1)" in render_saved_source
      and "end_step: int = 26" in render_saved_source)
check("DEFERRED_RENDERER_DOES_NOT_FILTER_BY_RENDER_STEPS",
      "render_steps" not in render_saved_source)
check("DEFERRED_RENDERER_HANDLES_FAILED_SEED",
      "RENDERED_FROM_PARTIAL_FAILURE_STATE" in render_saved_source
      and "_discover_design_runs" in render_saved_source)
check("DEFERRED_RENDERER_DOES_NOT_RERUN_OPTICS",
      '"reran_optical_algorithms": False' in render_saved_source
      and "_load_state" in render_saved_source)
check("HARD_GATE_POLICY_UNCHANGED_BY_VISUALIZATION",
      config["surface_fit"]["integrability_gates"]["enforcement"] == "HARD")

# Mọi file Python phải tự giải thích module và từng class/hàm bằng docstring tiếng Việt.
documented_sources = (
    list(ROOT.glob("*.py"))
    + list(
        manual.glob(
            "STEP_*/*.py"
        )
    )
)
missing_docs = []
damaged_docs = []
for path in documented_sources:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    nodes = [("<module>", tree)] + [
        (node.name, node) for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    for name, node in nodes:
        doc = ast.get_docstring(node)
        if not doc:
            missing_docs.append(f"{path}:{name}")
        elif "?" in doc:
            damaged_docs.append(f"{path}:{name}")
check("ALL_PYTHON_MODULES_CLASSES_FUNCTIONS_DOCUMENTED", not missing_docs)
check("VIETNAMESE_DOCSTRINGS_NOT_ENCODING_DAMAGED", not damaged_docs)

rays = {
    "pupil_index":
        np.array([
            0, 0, 0, 0,
            1, 1, 1, 1,
        ]),

    "field_index":
        np.array([
            0, 0, 1, 1,
            0, 0, 1, 1,
        ]),

    "chief":
        np.array([
            True, False,
            True, False,
            True, False,
            True, False,
        ]),
}

trace = {
    "landing":
        np.array([
            [0., 0., 0.],
            [1., 0., 0.],
            [10., 0., 0.],
            [11., 0., 0.],
            [5., 0., 0.],
            [7., 0., 0.],
            [15., 0., 0.],
            [17., 0., 0.],
        ]),

    "valid":
        np.ones(
            8,
            bool,
        ),
}

refs = np.array([
    [0., 0., 0.],
    [10., 0., 0.],
])

m = fan_imaging_metrics(
    trace,
    rays,
    refs,
    2.0,
)

expected = (
    np.sqrt(
        0.5
    )
    +
    np.sqrt(
        2.0
    )
)

check(
    "MF1_CHIEF_CENTERED_ALL_RAYS_OUTER_PUPIL_SUM",
    (
        abs(
            m[
                "E1_mm"
            ]
            - expected
        )
        < 1e-12
        and
        abs(
            m[
                "MF1_Fan"
            ]
            - 2
            * expected
        )
        < 1e-12
        and
        abs(
            m[
                "chief_pupil_spread_RMS_mm"
            ]
            - 2.5
        )
        < 1e-12
        and
        abs(
            m[
                "chief_field_bias_RMS_mm"
            ]
            - 2.5
        )
        < 1e-12
    ),
)

trace_partial = {
    "landing":
        trace[
            "landing"
        ].copy(),

    "valid":
        np.array([
            True,
            False,
            True,
            True,
            True,
            True,
            True,
            True,
        ]),
}

m_partial = fan_imaging_metrics(
    trace_partial,
    rays,
    refs,
    2.0,
)

check(
    "MF1_INVALID_RAYS_RETAINED_BUT_NOT_FALSELY_CERTIFIED",
    (
        abs(
            m_partial[
                "MF1_Fan"
            ]
            - 2
            * expected
        )
        < 1e-12
        and
        m_partial[
            "physical_valid_ray_count"
        ]
        == 7
        and
        m_partial[
            "certified_physical_MF1_Fan"
        ]
        is None
        and
        m_partial[
            "physical_valid_global_RMS_diagnostic_mm"
        ]
        is not None
    ),
)

mtf_trace = {"display_local": np.array([[0., 0., 0.], [0., 0., 0.]]),
             "directions": [np.array([[1., 0., 0.], [1., .1, 0.]])],
             "valid": np.ones(2, bool)}
mtf_rays = {"field_index": np.zeros(2, int), "pupil_index": np.zeros(2, int),
            "sample_index": np.arange(2), "chief": np.array([True, False])}
mtf_pattern = [{"dy_mm": 0., "dz_mm": 0.}, {"dy_mm": 1., "dz_mm": 0.}]
mtf = ray_based_mtf(mtf_trace, mtf_rays, mtf_pattern, [0., 10.], [550.], [1.])
check("MTF_ZERO_FREQUENCY_NORMALIZED", mtf["complete"] and abs(mtf["rows"][0]["MTF_u"] - 1.0) < 1e-12)

step11_diag_path = ROOT / "render_step11_diagnostic_v55.py"
check("STEP11_DIAGNOSTIC_RENDERER_EXISTS", step11_diag_path.exists())
if step11_diag_path.exists():
    step11_src = step11_diag_path.read_text(encoding="utf-8")
    check(
        "STEP11_DIAGNOSTIC_RENDERER_SCHEMA_AND_FUNCTIONS",
        "HUD_FAN_V5_5_STEP11_SEARCH_DIAGNOSTIC_V2" in step11_src
        and "render_step11_search_diagnostic" in step11_src
        and "parse_step11_search_history" in step11_src,
    )

active = manual / "ACTIVE_RUN.json"
if active.exists():
    a = json.loads(active.read_text(encoding="utf-8")); run = Path(a["run_dir"])
    check("LATEST_MANUAL_RUN_COMPLETE", a["status"] == "COMPLETE" and a["completed_step"] == 26)
    if config["mtf_requested"]:
        check("MTF_FILES_WHEN_REQUESTED", (run / "24N_MTF.csv").exists()
              and (run / "24N_MTF_SUMMARY.json").exists())
    else:
        check("NO_MTF_FILES_WHEN_NOT_REQUESTED", not any(run.glob("24N_MTF*")) and not any(run.glob("24N_PSF*")))
    check("NO_ZEMAX_FILES_WHEN_SKIPPED", not any(run.glob("*.zmx")) and not any(run.glob("*.zos")))
    check("FINAL_FORWARD_OUTPUT_EXISTS", (run / "24N_FORWARD_SHOOTING_SUMMARY.csv").exists())
    check("FINAL_STATUS_EXISTS", (run / "FINAL_RUN_STATUS.txt").exists())
    final_report = json.loads((run / "25_FINAL_OPTICAL_REPORT.json").read_text(encoding="utf-8"))
    final_objective = json.loads((run / "STEP_23" / "23_OBJECTIVE_FINAL.json").read_text(encoding="utf-8"))
    final_fan = (float(final_report["raw_Fan"]["MF1"]["MF1_Fan"])
                 + float(final_report["raw_Fan"]["MF2"]["MF2"]))
    check("FINAL_REPORT_USES_POST_DLSQ_FAN", abs(final_fan - float(final_objective["J_Fan_dimensionless"])) < 1e-12)
    if config["mtf_requested"] and float(config["mtf"]["minimum_at_max_frequency"]) <= 0.0:
        mtf_summary = json.loads((run / "24N_MTF_SUMMARY.json").read_text(encoding="utf-8"))
        check("MTF_UNGRADED_WITHOUT_USER_TOLERANCE", mtf_summary["grading"] == "UNGRADED_NO_USER_TOLERANCE"
              and mtf_summary["passes_declared_minimum"] is None)
    views = list(run.glob("STEP_*/*_3D_SPATIAL_VIEW.png"))
    metadata = list(run.glob("STEP_*/*_3D_VIEW_METADATA.json"))
    check("27_STEP_3D_IMAGES", len(views) == 27 and all(p.stat().st_size > 0 for p in views))
    check("27_STEP_3D_METADATA", len(metadata) == 27)
    view_records = [json.loads(p.read_text(encoding="utf-8")) for p in metadata]
    check("STEP_SPECIFIC_VISUALIZATION_SCHEMA",
          all(x.get("schema") == "HUD_FAN_V5_5_STEP_ALGORITHM_VIEW" for x in view_records))
    check("27_UNIQUE_ALGORITHM_FOCUSES",
          len({x.get("algorithm_focus") for x in view_records}) == 27)
    check("ALGORITHM_EVIDENCE_PRESENT",
          all(bool(x.get("algorithm_purpose")) and bool(x.get("evidence"))
              and len(x.get("panels", [])) == 3 for x in view_records))
    spot_png = run / "STEP_20" / "20_SPOT_EVOLUTION_PLANAR_ORDER2_ORDER5.png"
    spot_json = run / "STEP_20" / "20_SPOT_EVOLUTION_METADATA.json"
    spot_csv = run / "STEP_20" / "20_SPOT_EVOLUTION_METRICS.csv"
    check("CURRENT_RUN_SPOT_EVOLUTION_FILES",
          all(path.exists() and path.stat().st_size > 0
              for path in (spot_png, spot_json, spot_csv)))
    spot_record = json.loads(spot_json.read_text(encoding="utf-8"))
    check("SPOT_EVOLUTION_USES_CURRENT_RUN_DATA",
          spot_record["data_source"] == "CURRENT_RUN_RAY_LANDINGS"
          and spot_record["paper_values_reused"] is False
          and len(spot_record["stages"]) == 3)

report = {"schema": "HUD_FAN_V5_5_VERIFICATION", "checks": checks, "pass": all(x["pass"] for x in checks)}
(ROOT / "V5_5_VERIFICATION.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
