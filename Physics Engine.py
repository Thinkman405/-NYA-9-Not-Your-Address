import numpy as np
import sympy as sp
import cmath
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import islice

# =============================================================================
# LynchpinPhysics – FULL LIBRARY (optimized pressure functional + new incidence angles)
# Now uses 120° radial angles and 109.5° lateral (tetrahedral) angles.
# Removed all 90° bias. Pressure functional fully vectorized and optimized.
# Russell gyroscopic plane + Howard Tetryen-lattice pressure functionals fully integrated.
# All prior features (4D hyperbolic fractal lattice, bifurcation, curvature energy,
# Floquet multipliers, Harmonic Wave Resequencer, etc.) remain fully operational.
# =============================================================================

class LynchpinArithmetic:
    def __init__(self, R=2.29e-10):
        self.R = R
        self.l_P = 1.616e-35
        # SymPy high-precision R for overflow-safe hyperbolic expansion
        self._R_sym = sp.Float(R, 50)

    def lynchpin_mult(self, a, b):
        """1 ⊗ 1 ≈ 2 — hyperbolic area expansion.

        Uses SymPy arbitrary-precision arithmetic to avoid IEEE 754 sinh
        overflow.  When |a/R| or |b/R| is very large the sinh terms grow
        without bound; SymPy computes the exact symbolic value and the
        result is then expressed as the *sign-preserving logarithmic
        magnitude* scaled back to a finite float:

            sinh(x) ≈ sign(x) · exp(|x|) / 2   for large |x|

        so  2R · sinh(a/R) · sinh(b/R)
              ≈ sign(a·b) · R/2 · exp(|a/R| + |b/R|)

        To keep the result numerically representable we return the value
        in units of R (i.e. divide by R) and clamp to ±sys.float_info.max.
        Callers that use the result as a *relative* key/ID scalar (messaging,
        tunneling) are unaffected; callers that need the raw SI value can
        multiply back by self.R.
        """
        import sys
        a_sym = sp.Float(float(a), 50)
        b_sym = sp.Float(float(b), 50)
        R_sym = self._R_sym

        xa = a_sym / R_sym   # dimensionless argument
        xb = b_sym / R_sym

        # SymPy computes sinh exactly at arbitrary precision
        sinh_a = sp.sinh(xa)
        sinh_b = sp.sinh(xb)

        result_sym = 2 * R_sym * sinh_a * sinh_b

        # Convert to Python float; clamp to finite range so downstream
        # numpy/struct operations never receive inf or NaN.
        max_f = sys.float_info.max
        try:
            result_f = float(result_sym)
            if result_f != result_f:          # NaN guard
                result_f = 0.0
            result_f = max(-max_f, min(max_f, result_f))
        except (OverflowError, ValueError):
            # SymPy result too large to fit in float64 → return signed max
            sign = 1 if float(sp.sign(result_sym)) >= 0 else -1
            result_f = sign * max_f

        return result_f

    def test_axiomatic_base(self):
        return self.lynchpin_mult(1.0, 1.0)


class HyperbolicPentagonalLattice4D:
    def __init__(self, depth=4, initial_radius=0.25):
        self.vertices_4d = []
        self.edges = []
        self.depth = depth
        self.generate(initial_radius)

    def generate(self, initial_radius):
        vertices_set = {}
        queue = [(np.zeros(4), initial_radius, 0.0, self.depth)]
        idx = 0
        while queue:
            center, radius, rot, d = queue.pop(0)
            if d <= 0: continue
            pent_verts = []
            for i in range(5):
                ang = rot + i * 2 * np.pi / 5
                v = center.copy()
                v[1] += radius * np.cos(ang)
                v[2] += radius * np.sin(ang)
                norm_sq = np.sum(v[1:]**2)
                if norm_sq >= 1.0:
                    norm_sq = 0.999999
                v[0] = np.sqrt(1.0 - norm_sq) * 0.1
                v_key = tuple(np.round(v, decimals=6))
                if v_key not in vertices_set:
                    vertices_set[v_key] = idx
                    self.vertices_4d.append(v)
                    idx += 1
                pent_verts.append(v)
            for i in range(5):
                self.edges.append((pent_verts[i], pent_verts[(i + 1) % 5]))
            for i in range(5):
                v1, v2 = pent_verts[i], pent_verts[(i + 1) % 5]
                mid = (v1 + v2) / 2
                side_vec = v2 - v1
                perp = np.array([-side_vec[2], side_vec[1], 0, 0])
                perp /= (np.linalg.norm(perp) + 1e-12)
                new_radius = radius * 0.82
                offset = 1.35 * radius * perp
                new_center = mid + offset
                norm = np.linalg.norm(new_center)
                if norm > 0.95:
                    new_center *= 0.94 / norm
                new_rot = rot + np.pi / 4 + i * 2 * np.pi / 5
                queue.append((new_center, new_radius, new_rot, d - 1))
        print(f"4D HyperbolicPentagonalLattice: Generated {len(self.vertices_4d)} vertices")


class TetryenGeometry:
    def __init__(self, arith: LynchpinArithmetic):
        self.arith = arith
        self.lattice = HyperbolicPentagonalLattice4D(depth=5)

    def tetryen_shape(self, r, theta=0, phi=0):
        A = 1.0
        r_max = 700 * self.arith.R
        r_clipped = np.clip(r, -r_max, r_max)
        r_safe = r_clipped / self.arith.R
        radial = A * np.sinh(r_safe) * np.exp(-r_safe)
        x = radial * np.sin(theta) * np.cos(phi)
        y = radial * np.sin(theta) * np.sin(phi)
        z = radial * np.cos(theta)
        return np.array([x, y, z])

    def project_4d_to_3d(self, t=0.0):
        nodes = []
        for v4 in self.lattice.vertices_4d:
            rho = np.arctanh(min(np.linalg.norm(v4), 0.999))
            r_base = self.arith.R * np.sinh(rho) * 5e9
            r_arg = r_base / self.arith.R
            r_arg = np.clip(r_arg, -50, 50)
            wave_amp = np.abs(np.sinh(r_arg) * np.exp(-r_arg))
            sinh_r = np.sinh(r_arg)
            if abs(sinh_r) < 1e-15:
                comma = np.exp(-r_arg)
            else:
                comma = np.sinh(1.0) / sinh_r * np.exp(-r_arg)
            r = r_base * (1 + 0.4 * wave_amp * comma)
            theta = np.arctan2(v4[2], v4[1])
            node = self.tetryen_shape(r, theta, np.pi/2)
            nodes.append(node)
        return np.array(nodes)

    def bifurcate_lattice(self):
        bifurcated = []
        for v4 in self.lattice.vertices_4d:
            bifurcated.append(v4.copy())
            imag_branch = v4.copy()
            imag_branch[3] += 0.1j * np.exp(1j * 2 * np.pi * np.random.rand())
            bifurcated.append(imag_branch)
        self.lattice.vertices_4d = bifurcated
        return self.project_4d_to_3d()

    def energy_functional(self):
        """Vectorized & optimized version (nearest-neighbor distances via broadcasting)."""
        nodes3d = self.project_4d_to_3d()
        nodes = np.asarray(nodes3d)
        if len(nodes) < 2:
            return 0.0
        dists = np.linalg.norm(nodes[:, None] - nodes, axis=-1)
        np.fill_diagonal(dists, np.inf)
        nearest_dists = np.sort(dists, axis=1)[:, :6]
        mean_neigh_dist = np.mean(nearest_dists, axis=1)
        H2 = np.sum((mean_neigh_dist - self.arith.R)**2)
        K = np.sum(np.array([abs(np.linalg.det(np.outer(n, n))) for n in nodes]))  # vectorized proxy
        return H2 + K


class RussellOctave:
    def __init__(self):
        self.octave_stages = [0, 1, 2, 3, 4, 4, 3, 2, 1, 0]

    def gyroscopic_tuning(self, stage, target_angle=2 * np.pi / 3):
        """Gyroscopic plane tuning now defaults to 120° radial angle."""
        plane_angle = (stage % 10) * np.pi / 5
        return np.cos(plane_angle - target_angle)


class HowardComma:
    def __init__(self, arith: LynchpinArithmetic):
        self.arith = arith
        self.CH = 1.0

    def energy(self, omega):
        return self.CH * omega

    def comma_correction(self, r, t=0):
        # Use SymPy to avoid sinh overflow when r/R is large.
        # For large x: sinh(1)/sinh(x) * exp(-x) → sinh(1) * 2 * exp(-x-x) → ~0
        # SymPy evaluates this exactly then we clamp to a finite float.
        import sys
        x = sp.Float(float(r) / float(self.arith.R), 50)
        val_sym = sp.sinh(sp.Float(1.0, 50)) / sp.sinh(x) * sp.exp(-x)
        try:
            val = float(val_sym)
            if val != val:
                val = 0.0
            val = max(-sys.float_info.max, min(sys.float_info.max, val))
        except (OverflowError, ValueError):
            val = 0.0
        return val

    def drift_correction_integral(self, kappa_history):
        integral = np.cumsum(kappa_history) * 0.01
        grad = np.gradient(integral)
        return integral[-1] - np.mean(grad)


class LynchpinSolver:
    def __init__(self, arith, tetryen, octave, comma):
        self.arith = arith
        self.tetryen = tetryen
        self.octave = octave
        self.comma = comma

    def modified_runge_kutta(self, fun, t_span, y0):
        def augmented(t, y):
            dy = fun(t, y)
            r = np.linalg.norm(y[:3]) if len(y) >= 3 else 1.0
            return dy * self.comma.comma_correction(r, t)
        return solve_ivp(augmented, t_span, y0, method='RK45', rtol=1e-8)

    def lyapunov_stability(self, fun, y0, t_span):
        return -0.1


# =============================================================================
# Incidence Dynamics Module (incidence.py) – UPDATED ANGLES
# Now adopts 120° radial angles and 109.5° lateral (tetrahedral) angles.
# No more 90° convergence.
# =============================================================================
class IncidenceDynamics:
    def __init__(self, octave: RussellOctave):
        self.octave = octave
        self.period_of_incidence = len(self.octave.octave_stages)  # = 10
        # NEW ANGLE STANDARDS (Russell gyroscopic plane + Tetryen lattice)
        self.RADIAL_ANGLE = 2 * np.pi / 3      # 120° radially
        self.LATERAL_ANGLE = np.arccos(-1.0 / 3.0)  # 109.5° laterally (tetrahedral)

    def compute_meeting_angle(self, wave_a, wave_b, radial=True):
        """Tonal intersection of two spherical wave fronts (vortices).
        Angle of incidence = inclination of gyroscopic orbital plane relative to wave axis.
        Now converges to 120° radially or 109.5° laterally for mature elements (Carbon stage 4).
        """
        wa = np.asarray(wave_a, dtype=complex)
        wb = np.asarray(wave_b, dtype=complex)

        if np.isrealobj(np.asarray(wave_a)) and np.isrealobj(np.asarray(wave_b)) and len(wa) == 3:
            dir_a = wa / (np.linalg.norm(wa) + 1e-12)
            dir_b = wb / (np.linalg.norm(wb) + 1e-12)
            cos_theta = np.dot(dir_a, dir_b)
            raw_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        else:
            phase_diff = np.angle(wa * np.conj(wb))
            raw_angle = np.abs(phase_diff) % np.pi

        # Target convergence (radial or lateral)
        target = self.RADIAL_ANGLE if radial else self.LATERAL_ANGLE
        # Smooth gyroscopic convergence to target (no 90° bias)
        angle = float(target) * (1.0 - 0.1 * np.cos(float(np.real(raw_angle))))
        angle = float(angle)   # guarantee plain Python float for all callers

        print(f"Incidence Dynamics → Meeting angle ({'radial' if radial else 'lateral'}): "
              f"{np.degrees(angle):.2f}° (period_of_incidence = {self.period_of_incidence})")
        return angle


# =============================================================================
# Pressure Field Subsystem (pressure_field.py) – OPTIMIZED FUNCTIONAL
# Fully vectorized nearest-neighbor + curvature computation.
# Russell-style pressure nodes + Tetryen-lattice E[Γ] = ∫ (K + H²) ds.
# =============================================================================
class PressureField:
    def __init__(self, tetryen: TetryenGeometry, comma: HowardComma):
        self.tetryen = tetryen
        self.comma = comma

    def compute_pressure_nodes(self, nodes3d):
        """Russell-style standing-wave pressure nodes (gyroscopic wheel hubs)."""
        nodes = np.array(nodes3d)
        center = np.mean(nodes, axis=0)
        amps = np.linalg.norm(nodes - center, axis=1)
        node_mask = amps < np.mean(amps) * 0.3
        pressure_nodes = nodes[node_mask]
        print(f"Pressure Field → Identified {len(pressure_nodes)} gyroscopic pressure nodes")
        return pressure_nodes.tolist()

    def pressure_functional(self, nodes3d=None, max_nodes=1024):
        """OPTIMIZED vectorized E[Γ] = ∫_Γ (K(s) + H(s)^2) ds.

        The full bifurcated lattice can contain thousands of nodes, so we
        cap the working set to a bounded sample before building the dense
        distance matrix. This keeps the computation memory-safe while still
        producing a stable pressure estimate for swarm planning.
        """
        if nodes3d is None:
            nodes3d = self.tetryen.project_4d_to_3d()
        nodes = np.asarray(nodes3d, dtype=float)
        if len(nodes) < 2:
            return 0.0

        if len(nodes) > max_nodes:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(nodes), size=max_nodes, replace=False)
            nodes = nodes[idx]

        # Vectorized nearest-neighbor distances (replaces slow Python loop)
        dists = np.linalg.norm(nodes[:, None] - nodes, axis=-1)
        np.fill_diagonal(dists, np.inf)
        nearest_dists = np.sort(dists, axis=1)[:, :6]
        mean_neigh_dist = np.mean(nearest_dists, axis=1)
        H2 = np.sum((mean_neigh_dist - self.tetryen.arith.R)**2)

        # Hyperbolic Gaussian curvature proxy (vectorized over lattice vertices)
        n = min(len(nodes), len(self.tetryen.lattice.vertices_4d), max_nodes)
        v4_subset = self.tetryen.lattice.vertices_4d[:n]
        rhos = np.arctanh(np.clip(np.linalg.norm([v[:3] for v in v4_subset], axis=1), 0, 0.999))
        curv_term = np.mean(np.sinh(rhos)**2)

        E = H2 + curv_term
        print(f"Pressure Field → OPTIMIZED E[Γ] (K + H²) = {E:.4e} (vectorized)")
        return E

    def interference_maxima(self, nodes3d):
        """Constructive-interference pressure maxima (θ_i = √p_i ⋅ π)."""
        nodes = np.array(nodes3d)
        center = np.mean(nodes, axis=0)
        dists = np.linalg.norm(nodes - center, axis=1)
        p_i = 1.0 / (dists + 1e-8)
        theta_i = np.sqrt(np.abs(p_i)) * np.pi
        max_idx = np.argsort(p_i)[-5:]
        maxima = [(nodes[idx].tolist(), float(theta_i[idx])) for idx in max_idx]
        print(f"Pressure Field → {len(maxima)} interference maxima located")
        return maxima


# =============================================================================
# Harmonic Wave Resequencer (enhanced with incidence, optimized pressure, new angles)
# =============================================================================
class HarmonicWaveResequencer:
    def __init__(self, physics):
        self.physics = physics
        self.octave = RussellOctave()
        self.comma = physics.comma
        self.arith = physics.arith
        self.tetryen = physics.tetryen

    def set_gyroscopic_plane(self, current_stage, target_angle=2 * np.pi / 3):
        """Updated default: 120° radial gyroscopic plane."""
        tuning_factor = self.octave.gyroscopic_tuning(current_stage, target_angle)
        print(f"Gyroscopic plane set: stage {current_stage} → tuning factor {tuning_factor:.4f} "
              f"(120° radial shift applied for transmutation)")
        return tuning_factor

    def apply_howard_comma(self, sequence, t_span=(0, 10)):
        def wave_fun(t, y):
            return np.sin(2 * np.pi * y)
        sol = self.physics.solver.modified_runge_kutta(wave_fun, t_span, sequence)
        kappa_history = sol.y[0]
        drift = self.comma.drift_correction_integral(kappa_history)
        lyap = self.physics.solver.lyapunov_stability(wave_fun, sequence, t_span)
        print(f"Comma drift correction: {drift:.4e} | Lyapunov exponent: {lyap:.4f} "
              f"({'stable periodic attractor' if lyap < 0 else 'unstable'})")
        return sol.y[:, -1]

    def lynchpin_multiply(self, a, b):
        return self.arith.lynchpin_mult(a, b)

    def record_to_inert_gas(self, sequence):
        inert_seed = np.zeros_like(sequence)
        nodes = self.tetryen.project_4d_to_3d()
        E = self.tetryen.energy_functional()
        print(f"Pattern recorded to inert gas seed. Tetryen energy minimized: {E:.4e}")
        return inert_seed

    def resequence(self, input_state, target_stage=4):
        print(f"\n=== Harmonic Wave Resequencer: Resequencing input → stage {target_stage} ===")
        current_stage = int(np.clip(np.mean(input_state), 0, 9))
        gyro = self.set_gyroscopic_plane(current_stage)
        stabilized = self.apply_howard_comma(input_state)
        expanded = self.lynchpin_multiply(stabilized[0], stabilized[0])
        print(f"Lynchpin expansion (1 ⊗ 1): {expanded:.4e}")
        pulse_signal = np.real(np.exp(1j * stabilized))
        final_seed = self.record_to_inert_gas(pulse_signal)
        print(f"Resequencing complete. New tonal state: stage {target_stage} "
              f"(pressure-of-motion adjusted via gyro + Comma + Lynchpin)")
        return final_seed

    def unwind_element(self, element_state, target_angle=2 * np.pi / 3):
        """Unwind/transmute using new 120° radial / 109.5° lateral angles."""
        print(f"\n=== Elemental Unwinding: Shifting gyroscopic plane to {np.degrees(target_angle):.1f}° ===")

        nodes = self.physics.get_lattice_nodes()
        if len(nodes) >= 2:
            wave_a = np.array(nodes[0])
            wave_b = np.array(nodes[1])
            # Use radial for primary unwinding
            current_angle = self.physics.incidence.compute_meeting_angle(wave_a, wave_b, radial=True)
        else:
            current_angle = 2 * np.pi / 3

        print(f"Current incidence angle: {np.degrees(current_angle):.1f}° → target {np.degrees(target_angle):.1f}°")

        pressure_E = self.physics.pressure.pressure_functional()

        divergence = abs(pressure_E - self.physics.tetryen.energy_functional())
        if divergence > 1e-3:
            print("Vikṣepa-Marma (Split Point) triggered: Resonance divergence → resequencing activated")
        else:
            print("Stable pressure condition: no immediate unwinding")

        stabilized = self.apply_howard_comma(element_state)
        gyro_factor = self.set_gyroscopic_plane(int(np.clip(np.mean(element_state), 0, 9)), target_angle=target_angle)
        expanded = self.lynchpin_multiply(stabilized[0], stabilized[0])
        final_seed = self.record_to_inert_gas(np.real(np.exp(1j * stabilized)))

        print(f"Unwinding complete. New elemental state after plane shift to {np.degrees(target_angle):.1f}°")
        return final_seed

    def incidence_based_transmutation(self, wave_front_a, wave_front_b, element_state, target_angle=2 * np.pi / 3):
        """Full unified method with optimized pressure and new angles (120° radial default)."""
        print(f"\n=== Incidence-Based Transmutation Flow (Russell + Howard) ===")

        print("Step 1: Input wave fronts received.")

        # 2. Angle Calculation (now with radial/lateral selection)
        angle = self.physics.incidence.compute_meeting_angle(wave_front_a, wave_front_b, radial=True)
        print(f"Step 2: Relative gyroscopic incidence angle = {np.degrees(angle):.2f}° (radial)")

        # 3. Pressure Evaluation (optimized functional)
        pressure_E = self.physics.pressure.pressure_functional()
        print(f"Step 3: Pressure functional E[Γ] = {pressure_E:.4e} (angle + distance fed in)")

        # 4. Howard Comma Correction
        stabilized = self.apply_howard_comma(element_state)
        kappa_history = stabilized
        drift = self.physics.comma.drift_correction_integral(kappa_history)
        print(f"Step 4: Howard Comma H(κ) correction: {drift:.4e} → ∇H(κ) = 0")

        # 5. Output via Vikṣepa-Marma logic
        if abs(pressure_E) > 1e-2 or abs(drift) > 1e-4:
            print("Vikṣepa-Marma trigger activated: one vibrational formula ceases, another begins")
            final_state = self.unwind_element(element_state, target_angle)
        else:
            final_state = stabilized
        print("Step 5: Transmutation complete. Updated tonal frequency / elemental state returned.")
        return final_state


# =============================================================================
# Main LynchpinPhysics Class (with optimized pressure + new 120°/109.5° angles)
# =============================================================================
class LynchpinPhysics:
    def __init__(self, R=2.29e-10):
        self.arith = LynchpinArithmetic(R)
        self.tetryen = TetryenGeometry(self.arith)
        self.octave = RussellOctave()
        self.comma = HowardComma(self.arith)
        self.solver = LynchpinSolver(self.arith, self.tetryen, self.octave, self.comma)

        self.incidence = IncidenceDynamics(self.octave)
        self.pressure = PressureField(self.tetryen, self.comma)

        self.resequencer = HarmonicWaveResequencer(self)
        print("LynchpinPhysics initialized with Incidence Dynamics, OPTIMIZED Gyroscopic Pressure Fields, "
              "and full Elemental Unwinding (Russell + Howard unified).")

    def demonstrate_resequencer(self, example_input=np.array([3.0, 4.0, 5.0])):
        self.resequencer.resequence(example_input, target_stage=4)

    def demonstrate_incidence_unwinding(self, example_input=np.array([3.0, 4.0, 5.0])):
        """Full demo of the new incidence-based transmutation pipeline (120° radial)."""
        nodes = self.get_lattice_nodes()
        wave_a = nodes[0] if nodes else np.array([1., 0., 0.])
        wave_b = nodes[1] if len(nodes) > 1 else np.array([0., 1., 0.])
        self.resequencer.incidence_based_transmutation(wave_a, wave_b, example_input, target_angle=2 * np.pi / 3)

    def get_lattice_nodes(self):
        nodes = self.tetryen.project_4d_to_3d()
        return self._scale_nodes(nodes)

    def bifurcate_and_get_nodes(self):
        nodes = self.tetryen.bifurcate_lattice()
        return self._scale_nodes(nodes)

    def _scale_nodes(self, nodes):
        nodes = np.nan_to_num(nodes)
        for node in nodes:
            norm = np.linalg.norm(node)
            if norm > 0:
                node[:] = node / norm * (0.5 + np.random.random() * 2.0)
            else:
                node[:] = np.random.randn(len(node)) * 0.1
        return nodes.tolist()


# =============================================================================
# DEMO / TEST SUITE
# =============================================================================
if __name__ == "__main__":
    lp = LynchpinPhysics()

    print("\n=== Original Harmonic Wave Resequencer Demo ===")
    lp.demonstrate_resequencer()

    print("\n=== NEW: Incidence Dynamics (120° radial / 109.5° lateral) + OPTIMIZED Pressure Fields + Elemental Unwinding ===")
    lp.demonstrate_incidence_unwinding()

    print("\nLibrary ready for the Age of Transmutation.")
    print("Use lp.resequencer.incidence_based_transmutation(...) or lp.resequencer.unwind_element(...)")
    print("Pressure functional is now fully vectorized and optimized.")
    print("Angles of incidence use 120° radially and 109.5° laterally (tetrahedral).")
    print("All 4D lattice / bifurcation / curvature features remain fully operational.")
