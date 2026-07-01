import copy
import warnings
from time import perf_counter as timerpc

import cupy as np
import pandas as pd

from floris_cupy.logging_manager import LoggingManager

from .yaw_optimization_base import YawOptimization


class YawOptimizationPR(YawOptimization, LoggingManager):
    """
    GPU-optimized Parallel Refine (PR) yaw optimization.
    ...
    """

    def __init__(
        self,
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        Ny_passes=[5, 4],
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
        parallel_turbines=False,
        batch_size=None,
        display=True,
    ):
        if x0 is not None:
            warnings.warn(
                "The 'x0' argument is not used in the Parallel Refine optimization method "
                "and will be ignored.",
                UserWarning
            )

        # MUST be set before super().__init__ because the base class calls
        # _calculate_farm_power during its own __init__
        self._fmodel_working = None

        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            verify_convergence=verify_convergence,
        )

        self.time_spent_in_floris = 0
        self.Ny_passes = Ny_passes
        self.parallel_turbines = parallel_turbines
        self.batch_size = batch_size
        self.display = display

        self._het_sm_orig = None
        if (hasattr(self.fmodel_subset.core.flow_field, 'heterogeneous_inflow_config') and
            self.fmodel_subset.core.flow_field.heterogeneous_inflow_config is not None):
            self._het_sm_orig = np.array(
                self.fmodel_subset.core.flow_field.heterogeneous_inflow_config['speed_multipliers']
            )

        self._get_turbine_orders()

    # ------------------------------------------------------------------ #
    #  _calculate_farm_power  — lazy init of _fmodel_working
    # ------------------------------------------------------------------ #
    def _calculate_farm_power(
        self,
        yaw_angles=None,
        wd_array=None,
        ws_array=None,
        ti_array=None,
        turbine_weights=None,
        heterogeneous_speed_multipliers=None,
        power_setpoints=None,
    ):
        if self._fmodel_working is None:
            self._fmodel_working = copy.deepcopy(self.fmodel_subset)

        fm = self._fmodel_working

        if wd_array is None:
            wd_array = self.fmodel_subset.core.flow_field.wind_directions
        if ws_array is None:
            ws_array = self.fmodel_subset.core.flow_field.wind_speeds
        if ti_array is None:
            ti_array = self.fmodel_subset.core.flow_field.turbulence_intensities
        if yaw_angles is None:
            yaw_angles = self._yaw_angles_baseline_subset
        if turbine_weights is None:
            turbine_weights = self._turbine_weights_subset

        yaw_angles = self._unpack_variable(yaw_angles, subset=True)

        if hasattr(fm.core.flow_field, 'heterogeneous_inflow_config'):
            if fm.core.flow_field.heterogeneous_inflow_config is not None:
                if heterogeneous_speed_multipliers is not None:
                    fm.core.flow_field.heterogeneous_inflow_config['speed_multipliers'] = heterogeneous_speed_multipliers
                elif self._het_sm_orig is not None:
                    fm.core.flow_field.heterogeneous_inflow_config['speed_multipliers'] = self._het_sm_orig

        def _to_cpu(val):
            if val is None:
                return None
            if hasattr(val, "get"):
                return val.get()
            return val

        fm.set(
            wind_directions=_to_cpu(wd_array),
            wind_speeds=_to_cpu(ws_array),
            turbulence_intensities=_to_cpu(ti_array),
            yaw_angles=_to_cpu(yaw_angles),
            power_setpoints=_to_cpu(power_setpoints),
        )
        fm.run()
        turbine_power = np.array(fm.get_turbine_powers())

        turbine_power_weighted = np.multiply(turbine_weights, turbine_power)
        farm_power_weighted = np.sum(turbine_power_weighted, axis=1)
        return farm_power_weighted

    # ------------------------------------------------------------------ #
    #  _optimize_parallel  — fixed pass_depth -> Nii
    # ------------------------------------------------------------------ #
    def _optimize_parallel(self, print_progress):
        n_findex = self._n_findex_subset
        nturbs = self.nturbs

        for Nii in range(len(self.Ny_passes)):
            Ny = self.Ny_passes[Nii]
            if print_progress:
                print(f"[Parallel PR] Pass {Nii+1}/{len(self.Ny_passes)}  "
                      f"(Ny={Ny}, turbines={nturbs}, conditions={n_findex})")

            farm_powers, grid = self._evaluate_pass_parallel(Nii)
            # farm_powers shape: (nturbs, Ny, n_findex)

            if np.any(np.isnan(farm_powers)):
                self.logger.warning(
                    "NaNs found in farm powers during Parallel PR. "
                    "Proceeding to maximize over valid settings.",
                    stack_info=True,
                )

            args_opt = np.expand_dims(np.nanargmax(farm_powers, axis=1), axis=1)
            farm_powers_opt_new = np.squeeze(
                np.take_along_axis(farm_powers, args_opt, axis=1),
                axis=1,
            )  # (nturbs, n_findex)

            yaw_angles_opt_new = np.squeeze(
                np.take_along_axis(
                    grid,
                    np.expand_dims(args_opt, axis=3),
                    axis=1
                ),
                axis=1,
            )  # (nturbs, n_findex, nturbs)

            ids_better = farm_powers_opt_new > self._farm_power_opt_subset  # (nturbs, n_findex)
            best_turbine = np.argmax(ids_better, axis=0)  # (n_findex,)

            for iw in range(n_findex):
                t = int(best_turbine[iw])
                if ids_better[t, iw]:
                    self._farm_power_opt_subset[iw] = farm_powers_opt_new[t, iw]
                    self._yaw_angles_opt_subset[iw, :] = yaw_angles_opt_new[t, iw, :]

            # Update bounds for next pass
            turbids = self.turbines_ordered_array_subset
            for iw in range(n_findex):
                for t in range(nturbs):
                    tid = int(turbids[iw, t])
                    yaw_lb = self._yaw_lbs[iw, tid]
                    yaw_ub = self._yaw_ubs[iw, tid]
                    # FIX: was pass_depth, now Nii
                    if Nii == 0:
                        dx = (yaw_ub - yaw_lb) / max(Ny - 1, 1)
                    else:
                        dx = (yaw_ub - yaw_lb) / max(Ny, 1)
                    self._yaw_lbs[iw, tid] = np.clip(
                        self._yaw_angles_opt_subset[iw, tid] - 0.5 * dx,
                        self._minimum_yaw_angle_subset[iw, tid],
                        self._maximum_yaw_angle_subset[iw, tid],
                    )
                    self._yaw_ubs[iw, tid] = np.clip(
                        self._yaw_angles_opt_subset[iw, tid] + 0.5 * dx,
                        self._minimum_yaw_angle_subset[iw, tid],
                        self._maximum_yaw_angle_subset[iw, tid],
                    )

        return self._finalize()

    # ------------------------------------------------------------------ #
    #  Everything below this line is unchanged from the previous version
    # ------------------------------------------------------------------ #
    def _get_turbine_orders(self):
        layout_x = np.array(self.fmodel.layout_x)
        layout_y = np.array(self.fmodel.layout_y)
        turbines_ordered_array = []
        for wd in self.fmodel_subset.core.flow_field.wind_directions:
            wd = float(wd)
            layout_x_rot = (
                np.cos((wd - 270.0) * np.pi / 180.0) * layout_x
                - np.sin((wd - 270.0) * np.pi / 180.0) * layout_y
            )
            turbines_ordered = np.argsort(layout_x_rot)
            turbines_ordered_array.append(turbines_ordered)
        self.turbines_ordered_array_subset = np.vstack(turbines_ordered_array)

    def _calc_powers_with_memory(self, yaw_angles_subset, use_memory=True):
        yaw_angles_opt_subset = self._yaw_angles_opt_subset
        farm_power_opt_subset = self._farm_power_opt_subset
        wd_array_subset = np.array(self.fmodel_subset.core.flow_field.wind_directions)
        ws_array_subset = np.array(self.fmodel_subset.core.flow_field.wind_speeds)
        ti_array_subset = np.array(self.fmodel_subset.core.flow_field.turbulence_intensities)
        power_setpoints_subset = np.array(self.fmodel_subset.core.farm.power_setpoints)
        turbine_weights_subset = self._turbine_weights_subset

        eval_multiple_passes = (len(np.shape(yaw_angles_subset)) == 3)
        if eval_multiple_passes:
            Ny = yaw_angles_subset.shape[0]
            yaw_angles_subset = yaw_angles_subset.reshape(Ny * self._n_findex_subset, self.nturbs)
            yaw_angles_opt_subset = np.tile(yaw_angles_opt_subset, (Ny, 1))
            farm_power_opt_subset = np.tile(farm_power_opt_subset, (Ny))
            wd_array_subset = np.tile(wd_array_subset, Ny)
            ws_array_subset = np.tile(ws_array_subset, Ny)
            ti_array_subset = np.tile(ti_array_subset, Ny)
            power_setpoints_subset = np.tile(power_setpoints_subset, (Ny, 1))
            turbine_weights_subset = np.tile(turbine_weights_subset, (Ny, 1))

        n_eval = yaw_angles_subset.shape[0]
        farm_powers = np.zeros(n_eval)

        if use_memory:
            idx = (np.abs(yaw_angles_opt_subset - yaw_angles_subset) < 0.01).all(axis=1)
            farm_powers[idx] = farm_power_opt_subset[idx]
            if self.print_progress:
                n_skip = int(idx.sum())
                if n_skip > 0:
                    self.logger.info(
                        "Skipping {:d}/{:d} calculations: already in memory.".format(
                            n_skip, n_eval)
                    )
        else:
            idx = np.zeros(n_eval, dtype=bool)

        if not np.all(idx):
            start_time = timerpc()
            if (hasattr(self.fmodel.core.flow_field, 'heterogeneous_inflow_config') and
                self.fmodel.core.flow_field.heterogeneous_inflow_config is not None):
                het_sm_orig = np.array(
                    self.fmodel.core.flow_field.heterogeneous_inflow_config['speed_multipliers']
                )
                if eval_multiple_passes:
                    het_sm = np.tile(het_sm_orig, (Ny, 1))[~idx, :]
                else:
                    het_sm = het_sm_orig[~idx, :] if (~idx).any() else None
            else:
                het_sm = None

            farm_powers[~idx] = self._calculate_farm_power(
                wd_array=wd_array_subset[~idx],
                ws_array=ws_array_subset[~idx],
                ti_array=ti_array_subset[~idx],
                turbine_weights=turbine_weights_subset[~idx, :],
                yaw_angles=yaw_angles_subset[~idx, :],
                heterogeneous_speed_multipliers=het_sm,
                power_setpoints=power_setpoints_subset[~idx, :],
            )
            self.time_spent_in_floris += (timerpc() - start_time)

        if eval_multiple_passes:
            farm_powers = farm_powers.reshape(
                Ny,
                self.fmodel_subset.core.flow_field.n_findex
            )

        return farm_powers

    def _generate_evaluation_grid(self, pass_depth, turbine_depth):
        Ny = self.Ny_passes[pass_depth]
        evaluation_grid = np.tile(self._yaw_angles_opt_subset, (Ny, 1, 1))

        for iw in range(self._n_findex_subset):
            turbid = self.turbines_ordered_array_subset[iw, turbine_depth]

            yaw_lb = self._yaw_lbs[iw, turbid]
            yaw_ub = self._yaw_ubs[iw, turbid]

            yaw_lb = np.clip(
                yaw_lb,
                self.minimum_yaw_angle[iw, turbid],
                self.maximum_yaw_angle[iw, turbid]
            )
            yaw_ub = np.clip(
                yaw_ub,
                self.minimum_yaw_angle[iw, turbid],
                self.maximum_yaw_angle[iw, turbid]
            )

            if pass_depth == 0:
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny)
            else:
                c = int(Ny / 2)
                ids = [*list(range(0, c)), *list(range(c + 1, Ny + 1))]
                yaw_angles_subset = np.linspace(yaw_lb, yaw_ub, Ny + 1)[ids]

            evaluation_grid[:, iw, turbid] = yaw_angles_subset

        self._yaw_evaluation_grid = evaluation_grid
        return evaluation_grid

    def _process_evaluation_grid(self):
        evaluation_grid = self._yaw_evaluation_grid
        farm_powers = self._calc_powers_with_memory(evaluation_grid)
        return farm_powers

    def _generate_parallel_grid(self, pass_depth):
        Ny = self.Ny_passes[pass_depth]
        n_findex = self._n_findex_subset
        nturbs = self.nturbs

        grid = np.tile(self._yaw_angles_opt_subset, (nturbs, Ny, 1, 1))

        for iw in range(n_findex):
            turbids = self.turbines_ordered_array_subset[iw, :]

            yaw_lbs = self._yaw_lbs[iw, turbids]
            yaw_ubs = self._yaw_ubs[iw, turbids]

            min_b = self.minimum_yaw_angle[iw, turbids]
            max_b = self.maximum_yaw_angle[iw, turbids]
            yaw_lbs = np.clip(yaw_lbs, min_b, max_b)
            yaw_ubs = np.clip(yaw_ubs, min_b, max_b)

            if pass_depth == 0:
                angles = np.linspace(yaw_lbs[:, None], yaw_ubs[:, None], Ny)
            else:
                c = int(Ny / 2)
                ids = [*list(range(0, c)), *list(range(c + 1, Ny + 1))]
                angles_full = np.linspace(yaw_lbs[:, None], yaw_ubs[:, None], Ny + 1)
                angles = angles_full[:, ids]

            for t in range(nturbs):
                grid[t, :, iw, turbids[t]] = angles[t, :]

        return grid

    def _evaluate_pass_parallel(self, pass_depth):
        Ny = self.Ny_passes[pass_depth]
        n_findex = self._n_findex_subset
        nturbs = self.nturbs

        grid = self._generate_parallel_grid(pass_depth)
        n_total = nturbs * Ny * n_findex

        yaw_flat = grid.reshape(n_total, nturbs)

        wd_flat = np.tile(np.array(self.fmodel_subset.core.flow_field.wind_directions), nturbs * Ny)
        ws_flat = np.tile(np.array(self.fmodel_subset.core.flow_field.wind_speeds), nturbs * Ny)
        ti_flat = np.tile(np.array(self.fmodel_subset.core.flow_field.turbulence_intensities), nturbs * Ny)
        weights_flat = np.tile(self._turbine_weights_subset, (nturbs * Ny, 1))

        ps = self.fmodel_subset.core.farm.power_setpoints
        ps_flat = np.tile(np.array(ps), (nturbs * Ny, 1)) if ps is not None else None

        farm_powers = np.zeros(n_total)

        if self.batch_size is not None:
            chunk = self.batch_size * Ny * n_findex
            for start in range(0, n_total, chunk):
                end = min(start + chunk, n_total)
                farm_powers[start:end] = self._calculate_farm_power(
                    yaw_angles=yaw_flat[start:end],
                    wd_array=wd_flat[start:end],
                    ws_array=ws_flat[start:end],
                    ti_array=ti_flat[start:end],
                    turbine_weights=weights_flat[start:end],
                    power_setpoints=ps_flat[start:end] if ps_flat is not None else None,
                )
        else:
            start_time = timerpc()
            farm_powers = self._calculate_farm_power(
                yaw_angles=yaw_flat,
                wd_array=wd_flat,
                ws_array=ws_flat,
                ti_array=ti_flat,
                turbine_weights=weights_flat,
                power_setpoints=ps_flat,
            )
            self.time_spent_in_floris += (timerpc() - start_time)

        farm_powers = farm_powers.reshape(nturbs, Ny, n_findex)
        return farm_powers, grid

    def optimize(self, print_progress=None):
        if print_progress is None:
            print_progress = self.display
        self.print_progress = print_progress
        if self.parallel_turbines:
            return self._optimize_parallel(print_progress)
        else:
            return self._optimize_serial(print_progress)

    def _optimize_serial(self, print_progress):
        ii = 0
        for Nii in range(len(self.Ny_passes)):
            for turbine_depth in range(self.nturbs):
                p = 100.0 * ii / (len(self.Ny_passes) * self.nturbs)
                ii += 1
                if print_progress:
                    print(
                        f"[Parallel Refine] Processing pass={Nii}, "
                        f"turbine_depth={turbine_depth} ({p:.1f}%)"
                    )

                evaluation_grid = self._generate_evaluation_grid(
                    pass_depth=Nii,
                    turbine_depth=turbine_depth
                )
                farm_powers = self._process_evaluation_grid()

                if np.any(np.isnan(farm_powers)):
                    err_msg = (
                        "NaNs found in farm powers during Parallel Refine "
                        "optimization routine. Proceeding to maximize over yaw "
                        "settings that produce valid powers."
                    )
                    self.logger.warning(err_msg, stack_info=True)

                args_opt = np.expand_dims(np.nanargmax(farm_powers, axis=0), axis=0)
                farm_powers_opt_new = np.squeeze(
                    np.take_along_axis(farm_powers, args_opt, axis=0),
                    axis=0,
                )
                yaw_angles_opt_new = np.squeeze(
                    np.take_along_axis(
                        evaluation_grid,
                        np.expand_dims(args_opt, axis=2),
                        axis=0
                    ),
                    axis=0,
                )

                farm_powers_opt_prev = self._farm_power_opt_subset
                yaw_angles_opt_prev = self._yaw_angles_opt_subset

                ids_better = (farm_powers_opt_new > farm_powers_opt_prev)
                farm_power_opt = farm_powers_opt_prev
                farm_power_opt[ids_better] = farm_powers_opt_new[ids_better]

                turbs_sorted = self.turbines_ordered_array_subset
                turbids = turbs_sorted[np.where(ids_better)[0], turbine_depth]
                ids = (*np.where(ids_better), turbids)
                yaw_angles_opt = yaw_angles_opt_prev
                yaw_angles_opt[ids] = yaw_angles_opt_new[ids]

                dx = (
                    evaluation_grid[1, :, :] -
                    evaluation_grid[0, :, :]
                )[ids]
                self._yaw_lbs[ids] = np.clip(
                    yaw_angles_opt[ids] - 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )
                self._yaw_ubs[ids] = np.clip(
                    yaw_angles_opt[ids] + 0.50 * dx,
                    self._minimum_yaw_angle_subset[ids],
                    self._maximum_yaw_angle_subset[ids]
                )

                self._farm_power_opt_subset = farm_power_opt
                self._yaw_angles_opt_subset = yaw_angles_opt

        return self._finalize()