"""Simulation 门面的无窗口回归测试。

覆盖核心行为不变量：数量守恒、速度上限、无 NaN、
多种群切换、障碍物增删、捕食重生与重置。
"""

import unittest

import numpy as np

from boids import Params, Simulation


class TestSimulation(unittest.TestCase):
    def setUp(self):
        self.sim = Simulation(Params(), rng=np.random.default_rng(0))

    def test_boid_count_conserved(self):
        for _ in range(200):
            self.sim.step()
        self.assertEqual(self.sim.boid_positions.shape, (50, 2))

    def test_no_nan_and_speed_limits_with_all_features(self):
        self.sim.toggle_predators()
        for _ in range(300):
            self.sim.step(np.array([0.0, 0.0]))
        self.assertFalse(np.isnan(self.sim.boid_positions).any())
        self.assertFalse(np.isnan(self.sim.boid_velocities).any())
        boid_speeds = np.linalg.norm(self.sim.boid_velocities, axis=1)
        self.assertLessEqual(boid_speeds.max(), self.sim.params.max_speed + 1e-9)
        pred_speeds = np.linalg.norm(self.sim.predator_velocities, axis=1)
        self.assertLessEqual(pred_speeds.max(), self.sim.params.predator_speed + 1e-9)

    def test_predation_keeps_total_constant(self):
        self.sim.toggle_predators()
        before = self.sim.eaten_total
        for _ in range(300):
            self.sim.step()
        self.assertGreaterEqual(self.sim.eaten_total, before)
        self.assertEqual(self.sim.boid_positions.shape[0], 50)

    def test_multi_species_toggle(self):
        self.assertTrue(self.sim.multi_species)
        self.assertEqual(set(np.unique(self.sim.boid_species).tolist()), {0, 1, 2})
        self.sim.toggle_multi_species()
        self.assertFalse(self.sim.multi_species)
        self.assertEqual(set(np.unique(self.sim.boid_species).tolist()), {0})

    def test_obstacle_add_remove(self):
        n0 = len(self.sim.obstacles)
        self.sim.add_obstacle(100.0, 100.0)
        self.assertEqual(len(self.sim.obstacles), n0 + 1)
        i = self.sim.remove_obstacle(100.0, 100.0)
        self.assertIsNotNone(i)
        self.assertEqual(len(self.sim.obstacles), n0)

    def test_obstacles_disabled_still_stable(self):
        self.sim.toggle_obstacles()  # 关闭障碍物
        for _ in range(50):
            self.sim.step()
        self.assertFalse(np.isnan(self.sim.boid_positions).any())

    def test_reset_rebounds_and_clears_counter(self):
        self.sim.toggle_predators()
        for _ in range(100):
            self.sim.step()
        self.sim.reset()
        hw = self.sim.params.width / 2.0
        hh = self.sim.params.height / 2.0
        self.assertTrue((np.abs(self.sim.boid_positions[:, 0]) <= hw).all())
        self.assertTrue((np.abs(self.sim.boid_positions[:, 1]) <= hh).all())
        self.assertEqual(self.sim.eaten_total, 0)


if __name__ == "__main__":
    unittest.main()
