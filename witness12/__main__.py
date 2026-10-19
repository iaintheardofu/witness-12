"""Print the host ledger transcript. This does not open the AKD1000s."""

from witness12.engine import run_host

if __name__ == "__main__":
    run = run_host()
    print(run.transcript)
    print(
        f"\nobservations={run.n_obs} spoken={len(run.spoken_steps)} "
        f"silent={len(run.silent_steps)} learned_windows={run.learned_windows}"
    )
