# Cheap-pilot slice proposal — humaneval_plus

* n_problems: 30
* approximate NIM budget: 330 calls (at K=5, 11 calls/problem)
* proposal CID: `a5b3a2c15c4e3a0c3f33a47ed80334b759065b72daf76e2818a230d6a7256327`
* rationale: Proposed 30 problems from humaneval_plus; cluster distribution = a1_only_wins=2, b_only_wins=7, shared_fails=12, shared_wins=9. Priority: b_only_wins (mechanism rescue) → shared_fails (bench stress) → a1_only_wins (anti-mechanism calibration) → shared_wins (top-up).

| # | Seed | task_id | Cluster | Justification |
|---|---|---|---|---|
| 1 | 88028001 | HumanEval/118 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 2 | 88028001 | HumanEval/16 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 3 | 88028001 | HumanEval/160 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 4 | 88028001 | HumanEval/163 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 5 | 88028002 | HumanEval/121 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 6 | 88028003 | HumanEval/125 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 7 | 88028003 | HumanEval/84 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 8 | 88028001 | HumanEval/129 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 9 | 88028001 | HumanEval/76 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 10 | 88028001 | HumanEval/84 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 11 | 88028001 | HumanEval/91 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 12 | 88028002 | HumanEval/132 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 13 | 88028002 | HumanEval/137 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 14 | 88028002 | HumanEval/140 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 15 | 88028002 | HumanEval/91 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 16 | 88028003 | HumanEval/154 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 17 | 88028003 | HumanEval/32 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 18 | 88028003 | HumanEval/55 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 19 | 88028003 | HumanEval/83 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 20 | 88028001 | HumanEval/17 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 21 | 88028002 | HumanEval/122 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 22 | 88028001 | HumanEval/100 | shared_wins | top-up from shared_wins to fill slice |
| 23 | 88028001 | HumanEval/101 | shared_wins | top-up from shared_wins to fill slice |
| 24 | 88028001 | HumanEval/104 | shared_wins | top-up from shared_wins to fill slice |
| 25 | 88028001 | HumanEval/111 | shared_wins | top-up from shared_wins to fill slice |
| 26 | 88028001 | HumanEval/113 | shared_wins | top-up from shared_wins to fill slice |
| 27 | 88028001 | HumanEval/119 | shared_wins | top-up from shared_wins to fill slice |
| 28 | 88028001 | HumanEval/14 | shared_wins | top-up from shared_wins to fill slice |
| 29 | 88028001 | HumanEval/35 | shared_wins | top-up from shared_wins to fill slice |
| 30 | 88028001 | HumanEval/44 | shared_wins | top-up from shared_wins to fill slice |
