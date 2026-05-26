# Cheap-pilot slice proposal — mbpp

* n_problems: 30
* approximate NIM budget: 330 calls (at K=5, 11 calls/problem)
* proposal CID: `7d1e6a72fdde9c1616366ef4fa319cc3edf15f915c1d1c18e247670448781330`
* rationale: Proposed 30 problems from mbpp; cluster distribution = a1_only_wins=3, b_only_wins=5, shared_fails=21, shared_wins=1. Priority: b_only_wins (mechanism rescue) → shared_fails (bench stress) → a1_only_wins (anti-mechanism calibration) → shared_wins (top-up).

| # | Seed | task_id | Cluster | Justification |
|---|---|---|---|---|
| 1 | 90002 | 83 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 2 | 90002 | 87 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 3 | 90003 | 439 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 4 | 90003 | 777 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 5 | 90005 | 101 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 6 | 90001 | 229 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 7 | 90001 | 407 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 8 | 90001 | 802 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 9 | 90002 | 255 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 10 | 90002 | 431 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 11 | 90002 | 442 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 12 | 90002 | 595 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 13 | 90002 | 638 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 14 | 90002 | 776 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 15 | 90002 | 780 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 16 | 90003 | 462 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 17 | 90003 | 579 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 18 | 90003 | 638 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 19 | 90004 | 396 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 20 | 90004 | 461 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 21 | 90004 | 638 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 22 | 90005 | 124 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 23 | 90005 | 228 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 24 | 90005 | 255 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 25 | 90005 | 430 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 26 | 90005 | 640 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 27 | 90003 | 448 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 28 | 90005 | 765 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 29 | 90005 | 780 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 30 | 90001 | 105 | shared_wins | top-up from shared_wins to fill slice |
