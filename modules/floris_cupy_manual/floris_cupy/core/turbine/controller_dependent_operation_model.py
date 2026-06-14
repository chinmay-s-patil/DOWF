import copy

import cupy as np
import numpy as np_cpu
from attrs import define
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import fsolve

from floris_cupy.core.rotor_velocity import (
    average_velocity,
    compute_tilt_angles_for_floating_turbines,
    rotor_velocity_air_density_correction,
)
from floris_cupy.core.turbine.operation_models import BaseOperationModel, to_cpu
from floris_cupy.type_dec import (
    NDArrayFloat,
    NDArrayObject,
)
from floris_cupy.utilities import cosd, sind


@define
class ControllerDependentTurbine(BaseOperationModel):
    """
    Static class defining a wind turbine model that may be misaligned with the flow.
    Nonzero tilt and yaw angles are handled via the model presented in
    https://doi.org/10.5194/wes-2023-133 .

    The method requires C_P, C_T look-up tables as functions of tip speed ratio and blade pitch
    angle, available here:
    "floris/turbine_library/iea_15MW_demo_cp_ct_surface.npz" for the IEA 15MW reference turbine.
    As with all turbine submodules, implements only static power() and thrust_coefficient() methods,
    which are called by power() and thrust_coefficient() on turbine.py, respectively.
    There are also two new functions, i.e. compute_local_vertical_shear() and control_trajectory().
    These are called by thrust_coefficient() and power() to compute the vertical shear and predict
    the turbine status in terms of tip speed ratio and pitch angle.
    This class is not intended to be instantiated; it simply defines a library of static methods.

    Developed and implemented by Simone Tamaro, Filippo Campagnolo, and Carlo L. Bottasso
    at Technische Universität München (TUM).
    email: simone.tamaro@tum.de
    """

    @staticmethod
    def power(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        air_density: float,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        **_,
    ):
        # Sign convention: in the TUM model, negative tilt creates tower clearance
        tilt_angles = -tilt_angles

        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"],
        )

        n_findex, n_turbines = tilt_angles.shape

        shear = ControllerDependentTurbine.compute_local_vertical_shear(velocities)

        beta = power_thrust_table["controller_dependent_turbine_parameters"]["beta"]
        cd = power_thrust_table["controller_dependent_turbine_parameters"]["cd"]
        cl_alfa = power_thrust_table["controller_dependent_turbine_parameters"]["cl_alfa"]
        sigma = power_thrust_table["controller_dependent_turbine_parameters"]["rotor_solidity"]
        R = power_thrust_table["controller_dependent_turbine_parameters"]["rotor_diameter"] / 2

        air_density = power_thrust_table["ref_air_density"]

        pitch_out, tsr_out = ControllerDependentTurbine.control_trajectory(
            rotor_effective_velocities,
            yaw_angles,
            tilt_angles,
            air_density,
            R,
            shear,
            power_setpoints,
            power_thrust_table,
        )

        tsr_array = tsr_out
        theta_array = np_cpu.deg2rad(to_cpu(pitch_out) + beta)
        x0 = 0.2

        # Solve for the power in yawed conditions
        MU = np.arccos(cosd(yaw_angles) * cosd(tilt_angles))
        cosMu = np.cos(MU)
        sinMu = np.sin(MU)
        p = np_cpu.zeros(to_cpu(average_velocity(velocities)).shape)

        for i in np_cpu.arange(n_findex):
            for j in np_cpu.arange(n_turbines):
                data = (
                    sigma, cd, cl_alfa,
                    float(to_cpu(yaw_angles[i, j])),
                    float(to_cpu(tilt_angles[i, j])),
                    float(to_cpu(shear[i, j])),
                    float(to_cpu(cosMu[i, j])),
                    float(to_cpu(sinMu[i, j])),
                    float(to_cpu(tsr_array[i, j])),
                    float(theta_array[i, j]),
                    float(to_cpu(MU[i, j])),
                )
                ct, info, ier, msg = fsolve(
                    ControllerDependentTurbine.get_ct, x0, args=data, full_output=True
                )
                if ier == 1:
                    p[i, j] = np_cpu.squeeze(
                        ControllerDependentTurbine.find_cp(
                            sigma, cd, cl_alfa,
                            float(to_cpu(yaw_angles[i, j])),
                            float(to_cpu(tilt_angles[i, j])),
                            float(to_cpu(shear[i, j])),
                            float(to_cpu(cosMu[i, j])),
                            float(to_cpu(sinMu[i, j])),
                            float(to_cpu(tsr_array[i, j])),
                            float(theta_array[i, j]),
                            float(to_cpu(MU[i, j])),
                            ct,
                        )
                    )
                else:
                    p[i, j] = -1e3

        # Solve for the power in non-yawed conditions
        yaw_angles = np.zeros_like(yaw_angles)
        MU = np.arccos(cosd(yaw_angles) * cosd(tilt_angles))
        cosMu = np.cos(MU)
        sinMu = np.sin(MU)
        p0 = np_cpu.zeros(to_cpu(average_velocity(velocities)).shape)

        for i in np_cpu.arange(n_findex):
            for j in np_cpu.arange(n_turbines):
                data = (
                    sigma, cd, cl_alfa,
                    float(to_cpu(yaw_angles[i, j])),
                    float(to_cpu(tilt_angles[i, j])),
                    float(to_cpu(shear[i, j])),
                    float(to_cpu(cosMu[i, j])),
                    float(to_cpu(sinMu[i, j])),
                    float(to_cpu(tsr_array[i, j])),
                    float(theta_array[i, j]),
                    float(to_cpu(MU[i, j])),
                )
                ct, info, ier, msg = fsolve(
                    ControllerDependentTurbine.get_ct, x0, args=data, full_output=True
                )
                if ier == 1:
                    p0[i, j] = np_cpu.squeeze(
                        ControllerDependentTurbine.find_cp(
                            sigma, cd, cl_alfa,
                            float(to_cpu(yaw_angles[i, j])),
                            float(to_cpu(tilt_angles[i, j])),
                            float(to_cpu(shear[i, j])),
                            float(to_cpu(cosMu[i, j])),
                            float(to_cpu(sinMu[i, j])),
                            float(to_cpu(tsr_array[i, j])),
                            float(theta_array[i, j]),
                            float(to_cpu(MU[i, j])),
                            ct,
                        )
                    )
                else:
                    p0[i, j] = -1e3

        ratio = np.asarray(p / p0)

        # Extract LUT data — keep on CPU for scipy
        cp_ct_data = power_thrust_table["controller_dependent_turbine_parameters"]["cp_ct_data"]
        cp_i = np_cpu.array(cp_ct_data["cp_lut"])
        pitch_i = np_cpu.array(cp_ct_data["pitch_lut"])
        tsr_i = np_cpu.array(cp_ct_data["tsr_lut"])
        interp_lut = RegularGridInterpolator(
            (tsr_i, pitch_i), cp_i, bounds_error=False, fill_value=None
        )

        cp_interp = np.asarray(interp_lut(
            np_cpu.concatenate((
                to_cpu(tsr_array)[:, :, None],
                to_cpu(pitch_out)[:, :, None],
            ), axis=2),
            method="cubic",
        ))
        power_coefficient = cp_interp * ratio

        power = (
            0.5
            * air_density
            * rotor_effective_velocities ** 3
            * np.pi
            * R ** 2
            * power_coefficient
            * power_thrust_table["controller_dependent_turbine_parameters"]["generator_efficiency"]
        )

        if power.max() > (
            power_thrust_table["controller_dependent_turbine_parameters"]["rated_power"]
            * 1e3 * 1.01
        ):
            print("Powers more than 1% above rated detected. Consider checking Cp-Ct data.")

        power = np.clip(
            power,
            0,
            power_thrust_table["controller_dependent_turbine_parameters"]["rated_power"] * 1e3,
        )
        return power

    @staticmethod
    def thrust_coefficient(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_,
    ):
        tilt_angles = -tilt_angles

        rotor_average_velocities = average_velocity(
            velocities=velocities,
            method=average_method,
            cubature_weights=cubature_weights,
        )

        old_tilt_angles = copy.deepcopy(tilt_angles)
        tilt_angles = compute_tilt_angles_for_floating_turbines(
            tilt_angles=tilt_angles,
            tilt_interp=tilt_interp,
            rotor_effective_velocities=rotor_average_velocities,
        )
        tilt_angles = np.where(correct_cp_ct_for_tilt, tilt_angles, old_tilt_angles)

        beta = power_thrust_table["controller_dependent_turbine_parameters"]["beta"]
        cd = power_thrust_table["controller_dependent_turbine_parameters"]["cd"]
        cl_alfa = power_thrust_table["controller_dependent_turbine_parameters"]["cl_alfa"]
        sigma = power_thrust_table["controller_dependent_turbine_parameters"]["rotor_solidity"]
        R = power_thrust_table["controller_dependent_turbine_parameters"]["rotor_diameter"] / 2

        shear = ControllerDependentTurbine.compute_local_vertical_shear(velocities)
        air_density = power_thrust_table["ref_air_density"]

        rotor_effective_velocities = rotor_velocity_air_density_correction(
            velocities=rotor_average_velocities,
            air_density=air_density,
            ref_air_density=power_thrust_table["ref_air_density"],
        )

        pitch_out, tsr_out = ControllerDependentTurbine.control_trajectory(
            rotor_effective_velocities,
            yaw_angles,
            tilt_angles,
            air_density,
            R,
            shear,
            power_setpoints,
            power_thrust_table,
        )

        n_findex, n_turbines = tilt_angles.shape

        theta_array = np_cpu.deg2rad(to_cpu(pitch_out) + beta)
        tsr_array = tsr_out
        x0 = 0.2

        # Solve for thrust coefficient in yawed conditions
        MU = np.arccos(cosd(yaw_angles) * cosd(tilt_angles))
        cosMu = np.cos(MU)
        sinMu = np.sin(MU)
        thrust_coefficient1 = np_cpu.zeros(to_cpu(average_velocity(velocities)).shape)

        for i in np_cpu.arange(n_findex):
            for j in np_cpu.arange(n_turbines):
                data = (
                    sigma, cd, cl_alfa,
                    float(to_cpu(yaw_angles[i, j])),
                    float(to_cpu(tilt_angles[i, j])),
                    float(to_cpu(shear[i, j])),
                    float(to_cpu(cosMu[i, j])),
                    float(to_cpu(sinMu[i, j])),
                    float(to_cpu(tsr_array[i, j])),
                    float(theta_array[i, j]),
                    float(to_cpu(MU[i, j])),
                )
                ct = fsolve(ControllerDependentTurbine.get_ct, x0, args=data)
                thrust_coefficient1[i, j] = np_cpu.squeeze(np_cpu.clip(ct, 0.0001, 0.9999))

        # Resolve thrust coefficient in non-yawed conditions
        yaw_angles = np.zeros_like(yaw_angles)
        MU = np.arccos(cosd(yaw_angles) * cosd(tilt_angles))
        cosMu = np.cos(MU)
        sinMu = np.sin(MU)
        thrust_coefficient0 = np_cpu.zeros(to_cpu(average_velocity(velocities)).shape)

        for i in np_cpu.arange(n_findex):
            for j in np_cpu.arange(n_turbines):
                data = (
                    sigma, cd, cl_alfa,
                    float(to_cpu(yaw_angles[i, j])),
                    float(to_cpu(tilt_angles[i, j])),
                    float(to_cpu(shear[i, j])),
                    float(to_cpu(cosMu[i, j])),
                    float(to_cpu(sinMu[i, j])),
                    float(to_cpu(tsr_array[i, j])),
                    float(theta_array[i, j]),
                    float(to_cpu(MU[i, j])),
                )
                ct = fsolve(ControllerDependentTurbine.get_ct, x0, args=data)
                thrust_coefficient0[i, j] = np_cpu.squeeze(ct)

        ratio = np.asarray(thrust_coefficient1 / thrust_coefficient0)

        # Extract LUT data — keep on CPU for scipy
        cp_ct_data = power_thrust_table["controller_dependent_turbine_parameters"]["cp_ct_data"]
        ct_i = np_cpu.array(cp_ct_data["ct_lut"])
        pitch_i = np_cpu.array(cp_ct_data["pitch_lut"])
        tsr_i = np_cpu.array(cp_ct_data["tsr_lut"])
        interp_lut = RegularGridInterpolator(
            (tsr_i, pitch_i), ct_i, bounds_error=False, fill_value=None
        )

        ct_interp = np.asarray(interp_lut(
            np_cpu.concatenate((
                to_cpu(tsr_array)[:, :, None],
                to_cpu(pitch_out)[:, :, None],
            ), axis=2),
            method="cubic",
        ))
        thrust_coefficient = ct_interp * ratio

        return thrust_coefficient

    @staticmethod
    def axial_induction(
        power_thrust_table: dict,
        velocities: NDArrayFloat,
        yaw_angles: NDArrayFloat,
        tilt_angles: NDArrayFloat,
        power_setpoints: NDArrayFloat,
        tilt_interp: NDArrayObject,
        average_method: str = "cubic-mean",
        cubature_weights: NDArrayFloat | None = None,
        correct_cp_ct_for_tilt: bool = False,
        **_,
    ):
        thrust_coefficients = ControllerDependentTurbine.thrust_coefficient(
            power_thrust_table=power_thrust_table,
            velocities=velocities,
            yaw_angles=yaw_angles,
            tilt_angles=tilt_angles,
            power_setpoints=power_setpoints,
            tilt_interp=tilt_interp,
            average_method=average_method,
            cubature_weights=cubature_weights,
            correct_cp_ct_for_tilt=correct_cp_ct_for_tilt,
        )

        yaw_angles = np.zeros_like(yaw_angles)
        MU = np.arccos(cosd(yaw_angles) * cosd(tilt_angles))
        sinMu = np.sin(MU)

        a = 1 - (
            (1 + np.sqrt(1 - thrust_coefficients - 1 / 16 * thrust_coefficients ** 2 * sinMu ** 2))
            / (2 * (1 + 1 / 16 * thrust_coefficients * sinMu ** 2))
        )
        axial_induction = np.clip(a, 0.0001, 0.9999)

        return axial_induction

    @staticmethod
    def compute_local_vertical_shear(velocities):
        """
        Evaluate the vertical (linear) shear each rotor experiences based on inflow velocity.
        """
        if velocities.shape[3] == 1:
            raise ValueError(
                "The ControllerDependentTurbine computes a local shear based on inflow wind speeds "
                "across the rotor. The provided velocities does not contain a vertical profile. "
                "This can occur if n_grid is set to 1 in the FLORIS input yaml."
            )
        n_findex, n_turbines = velocities.shape[:2]
        # shear computed on CPU — polyfit and interp are CPU operations
        shear = np_cpu.zeros((n_findex, n_turbines))
        for i in np_cpu.arange(n_findex):
            for j in np_cpu.arange(n_turbines):
                mean_speed = np_cpu.array(to_cpu(np.mean(velocities[i, j, :, :], axis=0)))
                if len(mean_speed) % 2 != 0:
                    u_u_hh = mean_speed / mean_speed[int(np_cpu.floor(len(mean_speed) / 2))]
                else:
                    u_u_hh = (
                        mean_speed
                        / (
                            mean_speed[int(len(mean_speed) / 2)]
                            + mean_speed[int(len(mean_speed) / 2) - 1]
                        )
                        / 2
                    )
                zg_R = np_cpu.linspace(-1, 1, len(mean_speed) + 2)
                polifit_k = np_cpu.polyfit(zg_R[1:-1], 1 - u_u_hh, 1)
                shear[i, j] = -polifit_k[0]
        # Return as CuPy array so downstream GPU math works
        return np.asarray(shear)

    @staticmethod
    def control_trajectory(
        rotor_average_velocities,
        yaw_angles,
        tilt_angles,
        air_density,
        R,
        shear,
        power_setpoints,
        power_thrust_table,
    ):
        """
        Determines the tip-speed-ratio and pitch angles that occur in operation.
        """
        beta = power_thrust_table["controller_dependent_turbine_parameters"]["beta"]
        cd = power_thrust_table["controller_dependent_turbine_parameters"]["cd"]
        cl_alfa = power_thrust_table["controller_dependent_turbine_parameters"]["cl_alfa"]
        sigma = power_thrust_table["controller_dependent_turbine_parameters"]["rotor_solidity"]

        if power_setpoints is None:
            power_demanded = (
                np.ones_like(tilt_angles)
                * power_thrust_table["controller_dependent_turbine_parameters"]["rated_power"]
                * 1000
                / power_thrust_table["controller_dependent_turbine_parameters"][
                    "generator_efficiency"
                ]
            )
        else:
            power_demanded = (
                power_setpoints
                / power_thrust_table["controller_dependent_turbine_parameters"][
                    "generator_efficiency"
                ]
            )

        # Extract LUT data on CPU for scipy
        cp_ct_data = power_thrust_table["controller_dependent_turbine_parameters"]["cp_ct_data"]
        cp_i = np_cpu.array(cp_ct_data["cp_lut"])
        pitch_i = np_cpu.array(cp_ct_data["pitch_lut"])
        tsr_i = np_cpu.array(cp_ct_data["tsr_lut"])
        idx = np_cpu.squeeze(np_cpu.where(cp_i == np_cpu.max(cp_i)))

        tsr_opt = tsr_i[idx[0]]
        pitch_opt = pitch_i[idx[1]]
        max_cp = cp_i[idx[0], idx[1]]

        omega_cut_in = 0
        omega_max = power_thrust_table["controller_dependent_turbine_parameters"]["rated_rpm"]
        rated_power_aero = (
            power_thrust_table["controller_dependent_turbine_parameters"]["rated_power"]
            / power_thrust_table["controller_dependent_turbine_parameters"]["generator_efficiency"]
        ) * 1000

        Region2andAhalf = False

        # All control trajectory arithmetic stays on CPU (fsolve requirement)
        omega_array = np_cpu.linspace(omega_cut_in, omega_max, 161) * np_cpu.pi / 30
        Q = (0.5 * air_density * omega_array ** 2 * R ** 5 * np_cpu.pi * max_cp) / tsr_opt ** 3
        Paero_array = Q * omega_array

        if Paero_array[-1] < rated_power_aero:
            Region2andAhalf = True
            Q_extra = rated_power_aero / (omega_max * np_cpu.pi / 30)
            Q = np_cpu.append(Q, Q_extra)
            (Paero_array[-1] / (0.5 * air_density * np_cpu.pi * R ** 2 * max_cp)) ** (1 / 3)
            omega_array = np_cpu.append(omega_array, omega_array[-1])
            Paero_array = np_cpu.append(Paero_array, rated_power_aero)
        else:
            rated_power_aero = Paero_array[-1]

        u_rated = (rated_power_aero / (0.5 * air_density * np_cpu.pi * R ** 2 * max_cp)) ** (
            1 / 3
        )
        u_array = np_cpu.linspace(3, 25, 45)
        idx = np_cpu.argmin(np_cpu.abs(u_array - u_rated))
        if u_rated > u_array[idx]:
            u_array = np_cpu.insert(u_array, idx + 1, u_rated)
        else:
            u_array = np_cpu.insert(u_array, idx, u_rated)

        pow_lut_omega = Paero_array
        omega_lut_pow = omega_array * 30 / np_cpu.pi
        torque_lut_omega = Q
        omega_lut_torque = omega_lut_pow

        n_findex, n_turbines = tilt_angles.shape

        # Pull power_demanded to CPU for np_cpu.interp
        power_demanded_cpu = to_cpu(power_demanded)
        omega_rated_cpu = (
            np_cpu.interp(power_demanded_cpu, pow_lut_omega, omega_lut_pow) * np_cpu.pi / 30
        )
        u_rated_cpu = (
            power_demanded_cpu / (0.5 * air_density * np_cpu.pi * R ** 2 * max_cp)
        ) ** (1 / 3)

        # Outputs computed on CPU (fsolve loops), converted to GPU at end
        pitch_out_cpu = np_cpu.zeros(to_cpu(rotor_average_velocities).shape)
        tsr_out_cpu = np_cpu.zeros(to_cpu(rotor_average_velocities).shape)

        rotor_avg_cpu = to_cpu(rotor_average_velocities)
        yaw_cpu = to_cpu(yaw_angles)
        tilt_cpu = to_cpu(tilt_angles)
        shear_cpu = to_cpu(shear)

        def get_tsr(x, *data):
            (
                air_density_, R_, sigma_, shear_, cd_, cl_alfa_, beta_,
                gamma_, tilt_, u_, pitch_in_,
                omega_lut_pow_, torque_lut_omega_,
                cp_i_, pitch_i_, tsr_i_,
            ) = data

            omega_lut_torque_ = omega_lut_pow_ * np_cpu.pi / 30
            omega = x * u_ / R_
            omega_rpm = omega * 30 / np_cpu.pi
            torque_nm = np_cpu.interp(omega, omega_lut_torque_, torque_lut_omega_)

            mu = np_cpu.arccos(cosd(gamma_) * cosd(tilt_))
            d = (sigma_, cd_, cl_alfa_, gamma_, tilt_, shear_,
                 np_cpu.cos(mu), np_cpu.sin(mu), x,
                 np_cpu.deg2rad(pitch_in_) + np_cpu.deg2rad(beta_), mu)
            [ct, _, _, _] = fsolve(
                ControllerDependentTurbine.get_ct, 0.1, args=d, full_output=True, factor=0.1
            )
            cp = ControllerDependentTurbine.find_cp(
                sigma_, cd_, cl_alfa_, gamma_, tilt_, shear_,
                np_cpu.cos(mu), np_cpu.sin(mu), x,
                np_cpu.deg2rad(pitch_in_) + np_cpu.deg2rad(beta_), mu, ct,
            )

            mu = np_cpu.arccos(cosd(0) * cosd(tilt_))
            d = (sigma_, cd_, cl_alfa_, 0, tilt_, shear_,
                 np_cpu.cos(mu), np_cpu.sin(mu), x,
                 np_cpu.deg2rad(pitch_in_) + np_cpu.deg2rad(beta_), mu)
            [ct, _, _, _] = fsolve(
                ControllerDependentTurbine.get_ct, 0.1, args=d, full_output=True, factor=0.1
            )
            cp0 = ControllerDependentTurbine.find_cp(
                sigma_, cd_, cl_alfa_, 0, tilt_, shear_,
                np_cpu.cos(mu), np_cpu.sin(mu), x,
                np_cpu.deg2rad(pitch_in_) + np_cpu.deg2rad(beta_), mu, ct,
            )

            eta_p = cp / cp0
            interp = RegularGridInterpolator(
                (np_cpu.squeeze(tsr_i_), np_cpu.squeeze(pitch_i_)),
                cp_i_, bounds_error=False, fill_value=None,
            )
            Cp_now = interp((x, pitch_in_), method="cubic")
            cp_g1 = Cp_now * eta_p
            aero_pow = 0.5 * air_density_ * (np_cpu.pi * R_ ** 2) * u_ ** 3 * cp_g1
            electric_pow = torque_nm * (omega_rpm * np_cpu.pi / 30)
            return aero_pow - electric_pow

        def get_pitch(x, *data):
            (
                air_density_, R_, sigma_, shear_, cd_, cl_alfa_, beta_,
                gamma_, tilt_, u_, omega_rated_,
                omega_lut_torque_, torque_lut_omega_,
                cp_i_, pitch_i_, tsr_i_,
            ) = data

            omega_rpm = omega_rated_ * 30 / np_cpu.pi
            tsr_ = omega_rated_ * R_ / u_
            pitch_in_ = np_cpu.deg2rad(x)
            torque_nm = np_cpu.interp(
                omega_rpm, omega_lut_torque_ * 30 / np_cpu.pi, torque_lut_omega_
            )

            mu = np_cpu.arccos(cosd(gamma_) * cosd(tilt_))
            d = (sigma_, cd_, cl_alfa_, gamma_, tilt_, shear_,
                 np_cpu.cos(mu), np_cpu.sin(mu), tsr_,
                 pitch_in_ + np_cpu.deg2rad(beta_), mu)
            [ct, _, _, _] = fsolve(
                ControllerDependentTurbine.get_ct, 0.1, args=d, full_output=True, factor=0.1
            )
            cp = ControllerDependentTurbine.find_cp(
                sigma_, cd_, cl_alfa_, gamma_, tilt_, shear_,
                np_cpu.cos(mu), np_cpu.sin(mu), tsr_,
                pitch_in_ + np_cpu.deg2rad(beta_), mu, ct,
            )

            mu = np_cpu.arccos(cosd(0) * cosd(tilt_))
            d = (sigma_, cd_, cl_alfa_, 0, tilt_, shear_,
                 np_cpu.cos(mu), np_cpu.sin(mu), tsr_,
                 pitch_in_ + np_cpu.deg2rad(beta_), mu)
            [ct, _, _, _] = fsolve(
                ControllerDependentTurbine.get_ct, 0.1, args=d, full_output=True, factor=0.1
            )
            cp0 = ControllerDependentTurbine.find_cp(
                sigma_, cd_, cl_alfa_, 0, tilt_, shear_,
                np_cpu.cos(mu), np_cpu.sin(mu), tsr_,
                pitch_in_ + np_cpu.deg2rad(beta_), mu, ct,
            )

            eta_p = cp / cp0
            interp = RegularGridInterpolator(
                (np_cpu.squeeze(tsr_i_), np_cpu.squeeze(pitch_i_)),
                cp_i_, bounds_error=False, fill_value=None,
            )
            Cp_now = interp((tsr_, x), method="cubic")
            cp_g1 = Cp_now * eta_p
            aero_pow = 0.5 * air_density_ * (np_cpu.pi * R_ ** 2) * u_ ** 3 * cp_g1
            electric_pow = torque_nm * (omega_rpm * np_cpu.pi / 30)
            return aero_pow - electric_pow

        for i in np_cpu.arange(n_findex):
            for j in np_cpu.arange(n_turbines):
                u_v = float(rotor_avg_cpu[i, j])
                u_rated_ij = float(u_rated_cpu[i, j])
                omega_rated_ij = float(omega_rated_cpu[i, j])

                if u_v > u_rated_ij:
                    tsr_v = omega_rated_ij * R / u_v * cosd(float(yaw_cpu[i, j])) ** 0.5
                else:
                    tsr_v = tsr_opt * cosd(float(yaw_cpu[i, j]))

                if Region2andAhalf:
                    omega_lut_torque[-1] = omega_lut_torque[-1] + 1e-2
                    omega_lut_pow[-1] = omega_lut_pow[-1] + 1e-2

                data = (
                    air_density, R, sigma, float(shear_cpu[i, j]),
                    cd, cl_alfa, beta,
                    float(yaw_cpu[i, j]), float(tilt_cpu[i, j]),
                    u_v, pitch_opt,
                    omega_lut_pow, torque_lut_omega,
                    cp_i, pitch_i, tsr_i,
                )
                [tsr_out_sol, infodict, ier, mesg] = fsolve(
                    get_tsr, tsr_v, args=data, full_output=True
                )

                if np_cpu.abs(infodict["fvec"]) > 10 or tsr_out_sol < 4:
                    tsr_out_sol = 1000

                tsr_outO = tsr_out_sol
                omega = tsr_outO * u_v / R

                if omega < omega_rated_ij:
                    pitch_out0 = pitch_opt
                else:
                    tsr_outO = omega_rated_ij * R / u_v
                    data = (
                        air_density, R, sigma, float(shear_cpu[i, j]),
                        cd, cl_alfa, beta,
                        float(yaw_cpu[i, j]), float(tilt_cpu[i, j]),
                        u_v, omega_rated_ij,
                        omega_array, Q,
                        cp_i, pitch_i, tsr_i,
                    )
                    [pitch_out_sol, infodict, ier, mesg] = fsolve(
                        get_pitch, u_v, args=data,
                        factor=0.1, full_output=True, xtol=1e-10, maxfev=2000,
                    )
                    if pitch_out_sol < pitch_opt:
                        pitch_out_sol = pitch_opt
                    pitch_out0 = pitch_out_sol

                pitch_out_cpu[i, j] = np_cpu.squeeze(pitch_out0)
                tsr_out_cpu[i, j] = np_cpu.squeeze(tsr_outO)

        # Return as CuPy arrays
        return np.asarray(pitch_out_cpu), np.asarray(tsr_out_cpu)

    @staticmethod
    def find_cp(sigma, cd, cl_alfa, gamma, delta, k, cosMu, sinMu, tsr, theta, MU, ct):
        # All scalar math — keep as pure Python/numpy scalars (called from fsolve)
        if MU == 0:
            MU = 1e-6
            sinMu = np_cpu.sin(MU)
            cosMu = np_cpu.cos(MU)
        a = 1 - (
            (1 + np_cpu.sqrt(1 - ct - 1 / 16 * sinMu ** 2 * ct ** 2))
            / (2 * (1 + 1 / 16 * ct * sinMu ** 2))
        )
        SG = sind(gamma)
        CG = cosd(gamma)
        SD = sind(delta)
        CD = cosd(delta)
        k_1s = -1 * (15 * np_cpu.pi / 32 * np_cpu.tan((MU + sinMu * (ct / 2)) / 2))

        p = sigma * (
            (
                np_cpu.pi * cosMu ** 2 * tsr * cl_alfa * (a - 1) ** 2
                - (
                    tsr * cd * np_cpu.pi * (
                        CD ** 2 * CG ** 2 * SD ** 2 * k ** 2
                        + 3 * CD ** 2 * SG ** 2 * k ** 2
                        - 8 * CD * tsr * SG * k
                        + 8 * tsr ** 2
                    )
                ) / 16
                - (np_cpu.pi * tsr * sinMu ** 2 * cd) / 2
                - (2 * np_cpu.pi * cosMu * tsr ** 2 * cl_alfa * theta) / 3
                + (np_cpu.pi * cosMu ** 2 * k_1s ** 2 * tsr * a ** 2 * cl_alfa) / 4
                + (2 * np_cpu.pi * cosMu * tsr ** 2 * a * cl_alfa * theta) / 3
                + (2 * np_cpu.pi * CD * cosMu * tsr * SG * cl_alfa * k * theta) / 3
                + (
                    CD ** 2 * cosMu ** 2 * tsr * cl_alfa * k ** 2 * np_cpu.pi
                    * (a - 1) ** 2 * (CG ** 2 * SD ** 2 + SG ** 2)
                ) / (4 * sinMu ** 2)
                - (2 * np_cpu.pi * CD * cosMu * tsr * SG * a * cl_alfa * k * theta) / 3
                + (
                    CD ** 2 * cosMu ** 2 * k_1s ** 2 * tsr * a ** 2 * cl_alfa * k ** 2
                    * np_cpu.pi * (3 * CG ** 2 * SD ** 2 + SG ** 2)
                ) / (24 * sinMu ** 2)
                - (np_cpu.pi * CD * CG * cosMu ** 2 * k_1s * tsr * SD * a * cl_alfa * k) / sinMu
                + (np_cpu.pi * CD * CG * cosMu ** 2 * k_1s * tsr * SD * a ** 2 * cl_alfa * k) / sinMu
                + (np_cpu.pi * CD * CG * cosMu * k_1s * tsr ** 2 * SD * a * cl_alfa * k * theta) / (5 * sinMu)
                - (np_cpu.pi * CD ** 2 * CG * cosMu * k_1s * tsr * SD * SG * a * cl_alfa * k ** 2 * theta) / (10 * sinMu)
            )
            / (2 * np_cpu.pi)
        )
        return p

    @staticmethod
    def get_ct(x, *data):
        """
        System of equations for Ct, as represented in Eq. (25) of Tamaro et al.
        x is a stand-in variable for Ct, which a numerical solver will solve for.
        data is a tuple of input parameters to the system of equations to solve.
        """
        sigma, cd, cl_alfa, gamma, delta, k, cosMu, sinMu, tsr, theta, MU = data
        if MU == 0:
            MU = 1e-6
            sinMu = np_cpu.sin(MU)
            cosMu = np_cpu.cos(MU)
        CD = cosd(delta)
        CG = cosd(gamma)
        SD = sind(delta)
        SG = sind(gamma)

        a = 1 - (
            (1 + np_cpu.sqrt(1 - x - 1 / 16 * x ** 2 * sinMu ** 2))
            / (2 * (1 + 1 / 16 * x * sinMu ** 2))
        )

        k_1s = -1 * (15 * np_cpu.pi / 32 * np_cpu.tan((MU + sinMu * (x / 2)) / 2))

        I1 = -(
            np_cpu.pi * cosMu * (tsr - CD * SG * k) * (a - 1)
            + (CD * CG * cosMu * k_1s * SD * a * k * np_cpu.pi * (2 * tsr - CD * SG * k))
            / (8 * sinMu)
        ) / (2 * np_cpu.pi)

        I2 = (
            np_cpu.pi * sinMu ** 2
            + (
                np_cpu.pi * (
                    CD ** 2 * CG ** 2 * SD ** 2 * k ** 2
                    + 3 * CD ** 2 * SG ** 2 * k ** 2
                    - 8 * CD * tsr * SG * k
                    + 8 * tsr ** 2
                )
            ) / 12
        ) / (2 * np_cpu.pi)

        return (sigma * (cd + cl_alfa) * I1 - sigma * cl_alfa * theta * I2) - x