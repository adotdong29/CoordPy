# Cheap-pilot slice proposal — mbpp_plus_v2

* n_problems: 30
* approximate NIM budget: 330 calls (at K=5, 11 calls/problem)
* proposal CID: `29c13c4ff4b1971af1a6e9092f3d8577e33be18cf755f8e06dfa2f2306c167cc`
* rationale: Proposed 30 problems from mbpp_plus_v2; cluster distribution = b_only_wins=7, shared_fails=22, shared_wins=1. Priority: b_only_wins (mechanism rescue) → shared_fails (bench stress) → a1_only_wins (anti-mechanism calibration) → shared_wins (top-up).

| # | Seed | task_id | Cluster | Justification |
|---|---|---|---|---|
| 1 | 90003 | 439 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 2 | 90003 | 777 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 3 | 90003 | 93 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 4 | 90004 | 7 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 5 | 90004 | 781 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 6 | 90005 | 101 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 7 | 90005 | 778 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 8 | 90001 | 278 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 9 | 90001 | 410 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 10 | 90001 | 593 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 11 | 90001 | 92 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 12 | 90002 | 255 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 13 | 90002 | 427 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 14 | 90002 | 638 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 15 | 90002 | 780 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 16 | 90003 | 261 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 17 | 90003 | 448 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 18 | 90003 | 462 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 19 | 90003 | 579 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 20 | 90003 | 626 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 21 | 90003 | 638 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 22 | 90003 | 790 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 23 | 90004 | 626 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 24 | 90004 | 638 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 25 | 90005 | 255 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 26 | 90005 | 430 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 27 | 90005 | 639 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 28 | 90005 | 765 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 29 | 90005 | 780 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 30 | 90001 | 105 | shared_wins | top-up from shared_wins to fill slice |
