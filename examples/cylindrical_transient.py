"""Diffusion kinetics in a full cylindrical core, with timed absorption changes.

    python examples/cylindrical_transient.py --device cpu
    python examples/cylindrical_transient.py --device gpu --adaptive

Lengths are cm; power normalization is for the full cylinder. These are
demonstration cross sections, not a validated reactor design.
"""

import argparse

from garnet import CylindricalModel, Kinetics, Material, PWR_TWO_GROUP


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    parser.add_argument("--adaptive", action="store_true")
    args = parser.parse_args()
    fuel = PWR_TWO_GROUP.replace(name="fuel")
    reflector = Material(name="reflector", diffusion=[1.2, 0.3],
        sigma_a=[0.001, 0.02], nu_sigma_f=[0, 0], sigma_s=[[0, 0.03], [0, 0]])
    core = (CylindricalModel(radius=60, height=120, cells=(12, 24))
            .fill(reflector).add_region(fuel, r=(0, 40), z=(10, 110))
            .set_boundary(r="vacuum", z="vacuum")
            .set_kinetics(Kinetics([1e7, 3e5], [0.0065], [0.08])))
    steady = core.steady(device=args.device)
    print(steady)
    q = steady.power_density(total_power=2e6)
    print(f"Full-core power: {(q * steady.cell_volumes).sum():.6f} W")

    poisoned = fuel.replace(sigma_a=fuel.sigma_a * [1.0, 1.003])
    options = ({"adaptive_bdf": {"rtol": 1e-3, "min_dt": 1e-7, "max_dt": 0.01}}
               if args.adaptive else {})
    accepted = []
    run = core.transient(
        t_end=0.1, dt=0.01, device=args.device, time_scheme="bdf2",
        initial_steady=steady, event_times=[0.035, 0.08],
        materials_at=lambda t: {"fuel": poisoned} if 0.035 < t <= 0.08 else {},
        on_step=lambda t, _flux, _power: accepted.append(t), **options)
    print(run)
    print("P/P0 at 0.02, 0.06, 0.10 s:", run.power_at([0.02, 0.06, 0.1]))
    print(f"Accepted steps: {len(accepted)}; rejected attempts: {run.rejected_steps}")


if __name__ == "__main__":
    main()
