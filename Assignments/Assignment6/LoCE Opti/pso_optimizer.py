import cupy as cp
import numpy as np

class BatchedGPUParticleSwarm:
    def __init__(
        self,
        n_particles: int,
        n_turbines: int,
        bounds: list[tuple[float, float]],
        maxiter: int = 100,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        seed: int = 42,
        disp: bool = False,
        turbine_type_name: str = ""
    ):
        self.n_particles = n_particles
        self.n_turbines = n_turbines
        self.bounds = np.array(bounds)
        self.maxiter = maxiter
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.rng = cp.random.default_rng(seed)
        self.disp = disp
        self.turbine_type_name = turbine_type_name
        
        self.best_global_pos = None
        self.best_global_score = cp.inf
        self.best_global_aep = 0.0
        self.best_global_penalty = cp.inf
        
        self.lower_bound = cp.array(self.bounds[:, 0]).reshape(1, n_turbines, 2)
        self.upper_bound = cp.array(self.bounds[:, 1]).reshape(1, n_turbines, 2)

    def optimize(self, batched_objective_fn, callback=None):
        # Initialize positions
        pos = self.rng.uniform(
            low=self.lower_bound,
            high=self.upper_bound,
            size=(self.n_particles, self.n_turbines, 2)
        )
        vel = cp.zeros_like(pos)
        
        start_pos_cp = pos[0].flatten()
        start_pos_cpu = start_pos_cp.get() if hasattr(start_pos_cp, 'get') else start_pos_cp
        
        best_personal_pos = pos.copy()
        best_personal_score = cp.full(self.n_particles, cp.inf)
        
        history = []
        
        for it in range(self.maxiter):
            # Evaluate objective function for the entire swarm
            scores, aeps, penalties = batched_objective_fn(pos)
            
            # Update personal bests
            improved_mask = scores < best_personal_score
            best_personal_score[improved_mask] = scores[improved_mask]
            best_personal_pos[improved_mask] = pos[improved_mask]
            
            # Update global best
            min_score_idx = int(cp.argmin(scores))
            if scores[min_score_idx] < self.best_global_score:
                self.best_global_score = float(scores[min_score_idx])
                self.best_global_pos = pos[min_score_idx].copy()
                self.best_global_aep = float(aeps[min_score_idx])
                self.best_global_penalty = float(penalties[min_score_idx])
                
            if callback is not None:
                # Pass current iteration's best particle AND global best
                cur_best_idx = int(cp.argmin(scores))
                cur_flat = pos[cur_best_idx].flatten()
                cur_cpu = cur_flat.get() if hasattr(cur_flat, 'get') else cur_flat
                glob_flat = self.best_global_pos.flatten()
                glob_cpu = glob_flat.get() if hasattr(glob_flat, 'get') else glob_flat
                callback(
                    glob_cpu, self.best_global_score, self.best_global_aep, self.best_global_penalty, it,
                    cur_cpu, float(scores[cur_best_idx]), float(aeps[cur_best_idx]), float(penalties[cur_best_idx])
                )
                
            # Update velocities and positions
            r1 = self.rng.uniform(0, 1, size=(self.n_particles, self.n_turbines, 2))
            r2 = self.rng.uniform(0, 1, size=(self.n_particles, self.n_turbines, 2))
            
            vel = (
                self.w * vel + 
                self.c1 * r1 * (best_personal_pos - pos) + 
                self.c2 * r2 * (self.best_global_pos - pos)
            )
            pos = pos + vel
            
            # Apply bounds reflection
            out_lower = pos < self.lower_bound
            out_upper = pos > self.upper_bound
            
            pos = cp.where(out_lower, 2 * self.lower_bound - pos, pos)
            pos = cp.where(out_upper, 2 * self.upper_bound - pos, pos)
            pos = cp.clip(pos, self.lower_bound, self.upper_bound)
            
            vel = cp.where(out_lower | out_upper, -0.5 * vel, vel)
            
            history.append({
                "iter": it,
                "score": float(self.best_global_score),
                "aep": float(self.best_global_aep),
                "penalty": float(self.best_global_penalty)
            })
            
            if self.disp:
                name_str = f"({self.turbine_type_name}):" if self.turbine_type_name else ":"
                print(f"For {self.n_turbines} turbines {name_str:<21} PSO Iter {it+1:03d}/{self.maxiter} | Best Score: {self.best_global_score:>10.1f} | True AEP: {self.best_global_aep:.3f} GWh | Penalty: {self.best_global_penalty:.6f}")
                
            if self.best_global_penalty <= 1e-4:
                if self.disp:
                    print(f"Early stopping triggered for {self.n_turbines} turbines{name_str}: penalty ({self.best_global_penalty:.6f}) <= 1e-4")
                break
                
        flat_best_pos = self.best_global_pos.flatten()
        return flat_best_pos.get() if hasattr(flat_best_pos, 'get') else flat_best_pos, start_pos_cpu, self.best_global_score, history
