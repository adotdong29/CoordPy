# Cheap-pilot slice proposal — humaneval

* n_problems: 30
* approximate NIM budget: 330 calls (at K=5, 11 calls/problem)
* proposal CID: `b7325b9646009a4a3fd71442cc55d3fd7c72a44690f6b6878ee5fb6d9ffcf607`
* rationale: Proposed 30 problems from humaneval; cluster distribution = a1_only_wins=3, b_only_wins=8, shared_fails=5, shared_wins=14. Priority: b_only_wins (mechanism rescue) → shared_fails (bench stress) → a1_only_wins (anti-mechanism calibration) → shared_wins (top-up).

| # | Seed | task_id | Cluster | Justification |
|---|---|---|---|---|
| 1 | 88028001 | HumanEval/118 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 2 | 88028001 | HumanEval/16 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 3 | 88028001 | HumanEval/160 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 4 | 88028001 | HumanEval/76 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 5 | 88028001 | HumanEval/91 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 6 | 88028002 | HumanEval/121 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 7 | 88028003 | HumanEval/83 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 8 | 88028003 | HumanEval/84 | b_only_wins | unique-B-rescue: reflexion mechanism rescued this in the historical bench |
| 9 | 88028001 | HumanEval/84 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 10 | 88028002 | HumanEval/132 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 11 | 88028002 | HumanEval/140 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 12 | 88028002 | HumanEval/91 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 13 | 88028003 | HumanEval/32 | shared_fails | hard-cluster: neither A1 nor B passed historically; mechanism stress surface |
| 14 | 88028001 | HumanEval/17 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 15 | 88028002 | HumanEval/122 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 16 | 88028002 | HumanEval/137 | a1_only_wins | A1-only-win: anti-mechanism calibration (reflexion lost ground here) |
| 17 | 88028001 | HumanEval/100 | shared_wins | top-up from shared_wins to fill slice |
| 18 | 88028001 | HumanEval/101 | shared_wins | top-up from shared_wins to fill slice |
| 19 | 88028001 | HumanEval/104 | shared_wins | top-up from shared_wins to fill slice |
| 20 | 88028001 | HumanEval/111 | shared_wins | top-up from shared_wins to fill slice |
| 21 | 88028001 | HumanEval/113 | shared_wins | top-up from shared_wins to fill slice |
| 22 | 88028001 | HumanEval/119 | shared_wins | top-up from shared_wins to fill slice |
| 23 | 88028001 | HumanEval/129 | shared_wins | top-up from shared_wins to fill slice |
| 24 | 88028001 | HumanEval/14 | shared_wins | top-up from shared_wins to fill slice |
| 25 | 88028001 | HumanEval/163 | shared_wins | top-up from shared_wins to fill slice |
| 26 | 88028001 | HumanEval/35 | shared_wins | top-up from shared_wins to fill slice |
| 27 | 88028001 | HumanEval/44 | shared_wins | top-up from shared_wins to fill slice |
| 28 | 88028001 | HumanEval/49 | shared_wins | top-up from shared_wins to fill slice |
| 29 | 88028001 | HumanEval/61 | shared_wins | top-up from shared_wins to fill slice |
| 30 | 88028001 | HumanEval/62 | shared_wins | top-up from shared_wins to fill slice |
