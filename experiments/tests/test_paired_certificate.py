from __future__ import annotations

import math
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

import numpy as np
from PIL import Image

from src.paired_certificate import (
    build_pair_certificate,
    build_spatial_redundancy_certificate,
    directional_signal,
    pair_must_receive_for_state,
    spatial_detection_count_for_state,
)
from src.geometry import minimum_enclosing_circle, point_in_convex_polygon
from src.gain_aware_verifier import (
    certify_gain_aware_multiplicity,
    sampled_gain_aware_multiplicity,
)
from src.policies import POLICIES, run_policy
from src.rmdirectional_adapter import (
    decode_path_gain_db,
    discover_sample_ids,
    load_sample,
    nearest_path_gain_db,
)
from src.simulator import (
    CorrelatedShadowingField,
    IdealDirectionalSensor,
    ModelConfig,
    Scenario,
    antenna_range_factor,
    generate_scenario,
)
from scripts.run_rmdirectional_external_audit import geographically_stratified_sample_ids


class PairCertificateTests(unittest.TestCase):
    def setUp(self):
        # 位于中心方向 0 度、半角约 10 度内的凸四边形。
        self.polygon = np.array(
            [[100.0, -10.0], [900.0, -100.0], [900.0, 100.0], [100.0, 10.0]]
        )
        self.origin = np.zeros(2)

    def valid_certificate(self):
        return build_pair_certificate(
            self.polygon,
            self.origin,
            bearing_deg=0.0,
            bearing_error_deg=10.0,
            longitudinal_offset=400.0,
            transverse_offset=100.0,
            reliable_radius=999.0,
        )

    def test_valid_continuous_certificate(self):
        certificate = self.valid_certificate()
        self.assertTrue(certificate.valid, certificate.reason)
        self.assertGreater(len(certificate.far_polygon), 0)
        self.assertLessEqual(certificate.far_max_distance, 999.0)
        self.assertGreater(100.0, 400.0 * math.tan(math.radians(10.0)))

    def test_compatible_emission_directions_always_hit_one_probe(self):
        certificate = self.valid_certificate()
        sources = [
            np.array([400.0, 0.0]),
            np.array([600.0, 50.0]),
            np.array([900.0, -100.0]),
            np.array([900.0, 100.0]),
        ]
        for source in sources:
            for beta_deg in np.linspace(0.0, 360.0, 1441):
                emission = np.array(
                    [math.cos(math.radians(beta_deg)), math.sin(math.radians(beta_deg))]
                )
                # 只检查与历史成功接收 S 相容的真实朝向。
                if directional_signal(source, emission, self.origin, 1000.0):
                    self.assertTrue(
                        pair_must_receive_for_state(
                            certificate, source, emission, reception_radius=1000.0
                        ),
                        msg=f"source={source.tolist()}, beta={beta_deg}",
                    )

    def test_spatial_q1_certificate_guarantees_two_distinct_visible_locations(self):
        certificate = build_spatial_redundancy_certificate(
            self.polygon,
            self.origin,
            bearing_deg=0.0,
            bearing_error_deg=10.0,
            longitudinal_offset=400.0,
            inner_transverse_offset=100.0,
            transverse_spacing=25.0,
            reliable_radius=999.0,
            tolerated_false_negatives=1,
        )
        self.assertTrue(certificate.valid, certificate.reason)
        sources = [
            np.array([400.0, 0.0]),
            np.array([600.0, 50.0]),
            np.array([900.0, -100.0]),
            np.array([900.0, 100.0]),
        ]
        for source in sources:
            for beta_deg in np.linspace(0.0, 360.0, 1441):
                emission = np.array(
                    [math.cos(math.radians(beta_deg)), math.sin(math.radians(beta_deg))]
                )
                if directional_signal(source, emission, self.origin, 1000.0):
                    count = spatial_detection_count_for_state(
                        certificate, source, emission, reception_radius=1000.0
                    )
                    self.assertGreaterEqual(
                        count, 2, msg=f"source={source.tolist()}, beta={beta_deg}"
                    )
                    visible = [
                        directional_signal(source, emission, point, 1000.0)
                        for point in certificate.points
                    ]
                    for missed_index in range(len(visible)):
                        self.assertTrue(
                            any(
                                value
                                for index, value in enumerate(visible)
                                if index != missed_index
                            ),
                            msg=(
                                "一次任意漏检后仍应至少保留一个理想成功点："
                                f"source={source.tolist()}, beta={beta_deg}, "
                                f"missed_index={missed_index}"
                            ),
                        )

    def test_persistent_miss_is_stable_at_the_same_location(self):
        scenario = Scenario(
            seed=19,
            source=(100.0, 0.0),
            emission_direction=(-1.0, 0.0),
            reception_radius_m=1000.0,
            initial_position=(0.0, 0.0),
            initial_bearing_deg=0.0,
        )
        sensor = IdealDirectionalSensor(
            scenario,
            ModelConfig(
                miss_probability=0.5,
                miss_mode="persistent_location",
                miss_correlation_m=25.0,
            ),
        )
        kinds = [sensor.measure((0.0, 0.0), occurrence=index).kind for index in range(5)]
        self.assertEqual(len(set(kinds)), 1)

    def test_continuous_shadowing_is_persistent_and_spatially_correlated(self):
        same_point = np.array([125.0, -80.0])
        near_point = same_point + np.array([2.5, 0.0])
        far_point = same_point + np.array([250.0, 0.0])
        same_values = []
        near_values = []
        far_values = []
        for seed in range(128):
            field = CorrelatedShadowingField(seed, 25.0, features=512)
            first = field.at(same_point)
            self.assertEqual(first, field.at(same_point))
            same_values.append(first)
            near_values.append(field.at(near_point))
            far_values.append(field.at(far_point))
        near_correlation = float(np.corrcoef(same_values, near_values)[0, 1])
        far_correlation = float(np.corrcoef(same_values, far_values)[0, 1])
        self.assertGreater(near_correlation, 0.75)
        self.assertLess(abs(far_correlation), 0.25)

    def test_parabolic_pattern_reduces_edge_detection_range(self):
        config = ModelConfig(
            antenna_mode="parabolic_halfplane",
            antenna_half_power_beamwidth_deg=65.0,
            antenna_max_attenuation_db=30.0,
            path_loss_exponent=2.5,
        )
        boresight = antenna_range_factor(
            (100.0, 0.0), (0.0, 0.0), (1.0, 0.0), config
        )
        edge = antenna_range_factor(
            (0.0, 100.0), (0.0, 0.0), (1.0, 0.0), config
        )
        expected_attenuation = 12.0 * (90.0 / 65.0) ** 2
        expected_edge = 10 ** (
            -expected_attenuation / (10.0 * config.path_loss_exponent)
        )
        self.assertAlmostEqual(boresight, 1.0)
        self.assertAlmostEqual(edge, expected_edge)
        self.assertLess(edge, 0.13)

    def test_uniform_halfplane_certificate_can_fail_under_parabolic_pattern(self):
        scenario = Scenario(
            seed=901,
            source=(400.0, 0.0),
            emission_direction=(-1.0, 0.0),
            reception_radius_m=1000.0,
            initial_position=(0.0, 0.0),
            initial_bearing_deg=0.0,
        )
        sensor = IdealDirectionalSensor(
            scenario,
            ModelConfig(
                miss_probability=0.0,
                antenna_mode="parabolic_halfplane",
                antenna_half_power_beamwidth_deg=65.0,
                antenna_max_attenuation_db=30.0,
                path_loss_exponent=2.5,
            ),
        )
        self.assertEqual(sensor.measure((0.0, 0.0)).kind, "direction")
        self.assertEqual(sensor.measure((400.0, 200.0)).kind, "no_signal")
        self.assertEqual(sensor.measure((400.0, -200.0)).kind, "no_signal")

    def test_gain_aware_audit_rejects_nonideal_counterexample(self):
        polygon = np.array(
            [[400.0, -1.0], [401.0, -1.0], [401.0, 1.0], [400.0, 1.0]]
        )
        config = ModelConfig(
            antenna_mode="parabolic_halfplane",
            antenna_half_power_beamwidth_deg=65.0,
            antenna_max_attenuation_db=30.0,
            path_loss_exponent=2.5,
        )
        audit = sampled_gain_aware_multiplicity(
            polygon,
            history_point=(0.0, 0.0),
            probe_points=((400.0, 200.0), (400.0, -200.0)),
            required_multiplicity=1,
            config=config,
            position_subdivisions=4,
            orientation_step_deg=2.0,
        )
        self.assertFalse(audit.valid_on_sampled_states)
        self.assertEqual(audit.minimum_guaranteed_multiplicity, 0)

    def test_gain_aware_audit_accepts_repeated_history_location(self):
        polygon = np.array(
            [[300.0, -20.0], [500.0, -20.0], [500.0, 20.0], [300.0, 20.0]]
        )
        config = ModelConfig(
            antenna_mode="parabolic_halfplane",
            antenna_half_power_beamwidth_deg=65.0,
            antenna_max_attenuation_db=30.0,
            path_loss_exponent=2.5,
        )
        audit = sampled_gain_aware_multiplicity(
            polygon,
            history_point=(0.0, 0.0),
            probe_points=((0.0, 0.0),),
            required_multiplicity=1,
            config=config,
            position_subdivisions=4,
            orientation_step_deg=2.0,
        )
        self.assertTrue(audit.valid_on_sampled_states)
        self.assertEqual(audit.minimum_guaranteed_multiplicity, 1)

    def test_continuous_gain_certificate_finds_counterexample(self):
        polygon = np.array(
            [[399.0, -1.0], [401.0, -1.0], [401.0, 1.0], [399.0, 1.0]]
        )
        config = ModelConfig(
            antenna_mode="parabolic_halfplane",
            antenna_half_power_beamwidth_deg=65.0,
            antenna_max_attenuation_db=30.0,
            path_loss_exponent=2.5,
        )
        certificate = certify_gain_aware_multiplicity(
            polygon,
            history_point=(0.0, 0.0),
            probe_points=((400.0, 200.0), (400.0, -200.0)),
            required_multiplicity=1,
            config=config,
            counterexample_position_subdivisions=4,
            counterexample_orientation_step_deg=2.0,
        )
        self.assertEqual(certificate.status, "counterexample")
        self.assertEqual(certificate.counterexample_detection_count, 0)

    def test_continuous_gain_certificate_uses_geometric_edge_handoff(self):
        polygon = np.array(
            [[400.0, -1.0], [401.0, -1.0], [401.0, 1.0], [400.0, 1.0]]
        )
        config = ModelConfig(
            antenna_mode="parabolic_halfplane",
            antenna_half_power_beamwidth_deg=90.0,
            antenna_max_attenuation_db=30.0,
            path_loss_exponent=2.5,
        )
        certificate = certify_gain_aware_multiplicity(
            polygon,
            history_point=(0.0, 0.0),
            probe_points=((400.0, 200.0), (400.0, -200.0)),
            required_multiplicity=1,
            config=config,
            geometric_visibility_multiplicity=1,
            position_tolerance_m=0.5,
            orientation_tolerance_deg=0.1,
            counterexample_position_subdivisions=6,
            counterexample_orientation_step_deg=1.0,
        )
        self.assertEqual(certificate.status, "certified")

    def test_continuous_gain_certificate_accepts_history_location(self):
        polygon = np.array(
            [[300.0, -20.0], [500.0, -20.0], [500.0, 20.0], [300.0, 20.0]]
        )
        config = ModelConfig(
            antenna_mode="parabolic_halfplane",
            antenna_half_power_beamwidth_deg=65.0,
            antenna_max_attenuation_db=30.0,
            path_loss_exponent=2.5,
        )
        certificate = certify_gain_aware_multiplicity(
            polygon,
            history_point=(0.0, 0.0),
            probe_points=((0.0, 0.0),),
            required_multiplicity=1,
            config=config,
            position_tolerance_m=5.0,
            orientation_tolerance_deg=1.0,
            counterexample_position_subdivisions=4,
            counterexample_orientation_step_deg=5.0,
        )
        self.assertEqual(certificate.status, "certified")

    def test_spatial_q2_certificate_guarantees_three_visible_locations(self):
        certificate = build_spatial_redundancy_certificate(
            self.polygon,
            self.origin,
            bearing_deg=0.0,
            bearing_error_deg=10.0,
            longitudinal_offset=400.0,
            inner_transverse_offset=100.0,
            transverse_spacing=25.0,
            reliable_radius=999.0,
            tolerated_false_negatives=2,
        )
        self.assertTrue(certificate.valid, certificate.reason)
        for source in (
            np.array([400.0, 0.0]),
            np.array([600.0, 50.0]),
            np.array([900.0, -100.0]),
        ):
            for beta_deg in np.linspace(0.0, 360.0, 721):
                emission = np.array(
                    [math.cos(math.radians(beta_deg)), math.sin(math.radians(beta_deg))]
                )
                if directional_signal(source, emission, self.origin, 1000.0):
                    count = spatial_detection_count_for_state(
                        certificate, source, emission, reception_radius=1000.0
                    )
                    self.assertGreaterEqual(
                        count, 3, msg=f"source={source.tolist()}, beta={beta_deg}"
                    )

    def test_insufficient_transverse_offset_is_rejected(self):
        certificate = build_pair_certificate(
            self.polygon,
            self.origin,
            bearing_deg=0.0,
            bearing_error_deg=10.0,
            longitudinal_offset=400.0,
            transverse_offset=60.0,
            reliable_radius=999.0,
        )
        self.assertFalse(certificate.valid)
        self.assertIn("横向间隔", certificate.reason)

    def test_insufficient_reliable_radius_is_rejected(self):
        certificate = build_pair_certificate(
            self.polygon,
            self.origin,
            bearing_deg=0.0,
            bearing_error_deg=10.0,
            longitudinal_offset=400.0,
            transverse_offset=100.0,
            reliable_radius=300.0,
        )
        self.assertFalse(certificate.valid)
        self.assertIn("接收距离", certificate.reason)

    def test_empty_far_slice_is_not_a_certificate_action(self):
        certificate = build_pair_certificate(
            self.polygon,
            self.origin,
            bearing_deg=0.0,
            bearing_error_deg=10.0,
            longitudinal_offset=950.0,
            transverse_offset=200.0,
            reliable_radius=999.0,
        )
        self.assertFalse(certificate.valid)
        self.assertIn("远侧区域为空", certificate.reason)

    def test_minimum_enclosing_circle_contains_all_vertices(self):
        center, radius = minimum_enclosing_circle(self.polygon)
        distances = np.linalg.norm(self.polygon - center, axis=1)
        self.assertTrue(np.all(distances <= radius + 1e-8))

    def test_generated_initial_state_contains_truth_and_all_policies_finish(self):
        model = ModelConfig(max_pair_rounds=6)
        for seed in range(3):
            scenario = generate_scenario(seed, model)
            for policy in POLICIES:
                result = run_policy(scenario, policy, model=model)
                self.assertTrue(result["completed"], (seed, policy))
                self.assertFalse(result["unsafe_exclusion"], (seed, policy))

    def test_temporal_repetition_does_not_fix_persistent_location_misses(self):
        model = ModelConfig(
            miss_probability=0.25,
            miss_mode="persistent_location",
            miss_correlation_m=25.0,
        )
        for seed in range(40, 55):
            scenario = generate_scenario(seed, model)
            pair = run_policy(scenario, "full_pair", model=model)
            repeated = run_policy(scenario, "temporal_repeat_q2", model=model)
            self.assertEqual(pair["completed"], repeated["completed"])
            self.assertEqual(pair["unsafe_exclusion"], repeated["unsafe_exclusion"])
            self.assertGreaterEqual(repeated["time_s"], pair["time_s"])

    def test_temporal_repetition_does_not_fix_correlated_shadowing_misses(self):
        model = ModelConfig(
            miss_probability=0.25,
            miss_mode="correlated_shadowing",
            miss_correlation_m=25.0,
        )
        for seed in range(55, 65):
            scenario = generate_scenario(seed, model)
            pair = run_policy(scenario, "full_pair", model=model)
            repeated = run_policy(scenario, "temporal_repeat_q2", model=model)
            self.assertEqual(pair["completed"], repeated["completed"])
            self.assertEqual(pair["unsafe_exclusion"], repeated["unsafe_exclusion"])
            self.assertGreaterEqual(repeated["time_s"], pair["time_s"])

    def test_convex_boundary_membership(self):
        self.assertTrue(point_in_convex_polygon([100.0, -10.0], self.polygon))
        self.assertFalse(point_in_convex_polygon([50.0, 0.0], self.polygon))

    def test_rmdirectional_path_gain_decoding(self):
        image = np.array([[0, 255], [127, 64]], dtype=np.uint8)
        decoded = decode_path_gain_db(image, -127.0, -50.0)
        self.assertAlmostEqual(decoded[0, 0], -127.0)
        self.assertAlmostEqual(decoded[0, 1], -50.0)
        self.assertAlmostEqual(decoded[1, 0], -127.0 + 127 / 255 * 77.0)

    def test_rmdirectional_selection_spreads_across_tiles(self):
        ids = [
            ("a", "1", "ll", "0"),
            ("a", "1", "ll", "1"),
            ("b", "2", "ur", "0"),
            ("b", "2", "ur", "1"),
            ("c", "3", "lr", "0"),
            ("c", "3", "lr", "1"),
        ]
        selected = geographically_stratified_sample_ids(ids, 3)
        self.assertEqual(len(selected), 3)
        self.assertEqual(len({sample_id[:3] for sample_id in selected}), 3)
        self.assertEqual(selected, geographically_stratified_sample_ids(ids, 3))

    def test_rmdirectional_minimal_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("path_gain", "tx_antennas", "antenna_patterns", "gis"):
                (root / name).mkdir()
            (root / "max_power.json").write_text(
                json.dumps({"pg_trnc": -127.0, "pg_max": -50.0}),
                encoding="utf-8",
            )
            image = np.array([[0, 64], [127, 255]], dtype=np.uint8)
            Image.fromarray(image, mode="L").save(root / "path_gain" / "pg_a_b_c_0.png")
            Image.fromarray(np.zeros_like(image), mode="L").save(
                root / "gis" / "nbuildings_a_b_c.png"
            )
            parameters = [
                {
                    "tx_coords": [1.0, 1.0, 12.0],
                    "phi": 0.25,
                    "theta": -0.1,
                    "antenna": 7,
                }
            ]
            (root / "tx_antennas" / "a_b_c_txparams.json").write_text(
                json.dumps(parameters),
                encoding="utf-8",
            )
            np.save(root / "antenna_patterns" / "pattern_7.npy", np.ones((2, 2)))

            self.assertEqual(discover_sample_ids(root), [("a", "b", "c", "0")])
            sample = load_sample(root, ("a", "b", "c", "0"))
            self.assertEqual(sample.antenna_id, 7)
            self.assertTrue(np.allclose(sample.tx_coords, [1.0, 1.0, 12.0]))
            self.assertAlmostEqual(nearest_path_gain_db(sample, [1, 1]), -50.0)
            self.assertIsNone(nearest_path_gain_db(sample, [-1, 0]))

            archive = root / "fixture.zip"
            with ZipFile(archive, "w") as bundle:
                for relative in (
                    "max_power.json",
                    "path_gain/pg_a_b_c_0.png",
                    "tx_antennas/a_b_c_txparams.json",
                    "antenna_patterns/pattern_7.npy",
                    "gis/nbuildings_a_b_c.png",
                ):
                    bundle.write(root / relative, relative)
            self.assertEqual(discover_sample_ids(archive), [("a", "b", "c", "0")])
            zipped_sample = load_sample(archive, ("a", "b", "c", "0"))
            self.assertEqual(zipped_sample.antenna_id, 7)
            self.assertAlmostEqual(nearest_path_gain_db(zipped_sample, [1, 1]), -50.0)


if __name__ == "__main__":
    unittest.main()
