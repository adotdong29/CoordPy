# W105 Phase 3 slice pack (pre-built in W104)

* schema: `coordpy.w104_w105_phase3_slice_pack.v1`
* pack_cid: `8be55f3bf1650df397cb875543c69a48473483de8089dc3c40be45cc635a1314`
* n_problems: 100
* phase3_seeds: [105001, 105002, 105003]
* scales locked: ['meta/llama-3.3-70b-instruct', 'meta/llama-3.1-405b-instruct']
* per-scale per-seed budget: 3300 NIM calls
* total Phase 3 budget across two scales: 6600 NIM calls

## Cluster mix

| Source | Count |
|---|---:|
| `humaneval(top-up):shared_wins` | 2 |
| `humaneval_plus:a1_only_wins` | 2 |
| `humaneval_plus:b_only_wins` | 7 |
| `humaneval_plus:shared_fails` | 10 |
| `humaneval_plus:shared_wins` | 9 |
| `humaneval_plus:shared_wins:mid_shell` | 45 |
| `humaneval_plus_corpus:corpus_fill` | 25 |

## Slice (helper-priority order)

| # | task_id | source |
|---|---|---|
| 1 | HumanEval/118 | `humaneval_plus:b_only_wins` |
| 2 | HumanEval/16 | `humaneval_plus:b_only_wins` |
| 3 | HumanEval/160 | `humaneval_plus:b_only_wins` |
| 4 | HumanEval/163 | `humaneval_plus:b_only_wins` |
| 5 | HumanEval/121 | `humaneval_plus:b_only_wins` |
| 6 | HumanEval/125 | `humaneval_plus:b_only_wins` |
| 7 | HumanEval/84 | `humaneval_plus:b_only_wins` |
| 8 | HumanEval/129 | `humaneval_plus:shared_fails` |
| 9 | HumanEval/76 | `humaneval_plus:shared_fails` |
| 10 | HumanEval/91 | `humaneval_plus:shared_fails` |
| 11 | HumanEval/132 | `humaneval_plus:shared_fails` |
| 12 | HumanEval/137 | `humaneval_plus:shared_fails` |
| 13 | HumanEval/140 | `humaneval_plus:shared_fails` |
| 14 | HumanEval/154 | `humaneval_plus:shared_fails` |
| 15 | HumanEval/32 | `humaneval_plus:shared_fails` |
| 16 | HumanEval/55 | `humaneval_plus:shared_fails` |
| 17 | HumanEval/83 | `humaneval_plus:shared_fails` |
| 18 | HumanEval/17 | `humaneval_plus:a1_only_wins` |
| 19 | HumanEval/122 | `humaneval_plus:a1_only_wins` |
| 20 | HumanEval/100 | `humaneval_plus:shared_wins` |
| 21 | HumanEval/101 | `humaneval_plus:shared_wins` |
| 22 | HumanEval/104 | `humaneval_plus:shared_wins` |
| 23 | HumanEval/111 | `humaneval_plus:shared_wins` |
| 24 | HumanEval/113 | `humaneval_plus:shared_wins` |
| 25 | HumanEval/119 | `humaneval_plus:shared_wins` |
| 26 | HumanEval/14 | `humaneval_plus:shared_wins` |
| 27 | HumanEval/35 | `humaneval_plus:shared_wins` |
| 28 | HumanEval/44 | `humaneval_plus:shared_wins` |
| 29 | HumanEval/49 | `humaneval(top-up):shared_wins` |
| 30 | HumanEval/61 | `humaneval(top-up):shared_wins` |
| 31 | HumanEval/62 | `humaneval_plus:shared_wins:mid_shell` |
| 32 | HumanEval/63 | `humaneval_plus:shared_wins:mid_shell` |
| 33 | HumanEval/66 | `humaneval_plus:shared_wins:mid_shell` |
| 34 | HumanEval/68 | `humaneval_plus:shared_wins:mid_shell` |
| 35 | HumanEval/77 | `humaneval_plus:shared_wins:mid_shell` |
| 36 | HumanEval/80 | `humaneval_plus:shared_wins:mid_shell` |
| 37 | HumanEval/82 | `humaneval_plus:shared_wins:mid_shell` |
| 38 | HumanEval/9 | `humaneval_plus:shared_wins:mid_shell` |
| 39 | HumanEval/90 | `humaneval_plus:shared_wins:mid_shell` |
| 40 | HumanEval/95 | `humaneval_plus:shared_wins:mid_shell` |
| 41 | HumanEval/112 | `humaneval_plus:shared_wins:mid_shell` |
| 42 | HumanEval/135 | `humaneval_plus:shared_wins:mid_shell` |
| 43 | HumanEval/136 | `humaneval_plus:shared_wins:mid_shell` |
| 44 | HumanEval/138 | `humaneval_plus:shared_wins:mid_shell` |
| 45 | HumanEval/144 | `humaneval_plus:shared_wins:mid_shell` |
| 46 | HumanEval/157 | `humaneval_plus:shared_wins:mid_shell` |
| 47 | HumanEval/161 | `humaneval_plus:shared_wins:mid_shell` |
| 48 | HumanEval/2 | `humaneval_plus:shared_wins:mid_shell` |
| 49 | HumanEval/21 | `humaneval_plus:shared_wins:mid_shell` |
| 50 | HumanEval/28 | `humaneval_plus:shared_wins:mid_shell` |
| 51 | HumanEval/42 | `humaneval_plus:shared_wins:mid_shell` |
| 52 | HumanEval/50 | `humaneval_plus:shared_wins:mid_shell` |
| 53 | HumanEval/51 | `humaneval_plus:shared_wins:mid_shell` |
| 54 | HumanEval/52 | `humaneval_plus:shared_wins:mid_shell` |
| 55 | HumanEval/54 | `humaneval_plus:shared_wins:mid_shell` |
| 56 | HumanEval/67 | `humaneval_plus:shared_wins:mid_shell` |
| 57 | HumanEval/71 | `humaneval_plus:shared_wins:mid_shell` |
| 58 | HumanEval/74 | `humaneval_plus:shared_wins:mid_shell` |
| 59 | HumanEval/79 | `humaneval_plus:shared_wins:mid_shell` |
| 60 | HumanEval/97 | `humaneval_plus:shared_wins:mid_shell` |
| 61 | HumanEval/109 | `humaneval_plus:shared_wins:mid_shell` |
| 62 | HumanEval/11 | `humaneval_plus:shared_wins:mid_shell` |
| 63 | HumanEval/13 | `humaneval_plus:shared_wins:mid_shell` |
| 64 | HumanEval/142 | `humaneval_plus:shared_wins:mid_shell` |
| 65 | HumanEval/146 | `humaneval_plus:shared_wins:mid_shell` |
| 66 | HumanEval/147 | `humaneval_plus:shared_wins:mid_shell` |
| 67 | HumanEval/15 | `humaneval_plus:shared_wins:mid_shell` |
| 68 | HumanEval/151 | `humaneval_plus:shared_wins:mid_shell` |
| 69 | HumanEval/38 | `humaneval_plus:shared_wins:mid_shell` |
| 70 | HumanEval/41 | `humaneval_plus:shared_wins:mid_shell` |
| 71 | HumanEval/60 | `humaneval_plus:shared_wins:mid_shell` |
| 72 | HumanEval/75 | `humaneval_plus:shared_wins:mid_shell` |
| 73 | HumanEval/8 | `humaneval_plus:shared_wins:mid_shell` |
| 74 | HumanEval/85 | `humaneval_plus:shared_wins:mid_shell` |
| 75 | HumanEval/87 | `humaneval_plus:shared_wins:mid_shell` |
| 76 | HumanEval/0 | `humaneval_plus_corpus:corpus_fill` |
| 77 | HumanEval/1 | `humaneval_plus_corpus:corpus_fill` |
| 78 | HumanEval/3 | `humaneval_plus_corpus:corpus_fill` |
| 79 | HumanEval/4 | `humaneval_plus_corpus:corpus_fill` |
| 80 | HumanEval/5 | `humaneval_plus_corpus:corpus_fill` |
| 81 | HumanEval/6 | `humaneval_plus_corpus:corpus_fill` |
| 82 | HumanEval/7 | `humaneval_plus_corpus:corpus_fill` |
| 83 | HumanEval/10 | `humaneval_plus_corpus:corpus_fill` |
| 84 | HumanEval/12 | `humaneval_plus_corpus:corpus_fill` |
| 85 | HumanEval/18 | `humaneval_plus_corpus:corpus_fill` |
| 86 | HumanEval/19 | `humaneval_plus_corpus:corpus_fill` |
| 87 | HumanEval/20 | `humaneval_plus_corpus:corpus_fill` |
| 88 | HumanEval/22 | `humaneval_plus_corpus:corpus_fill` |
| 89 | HumanEval/23 | `humaneval_plus_corpus:corpus_fill` |
| 90 | HumanEval/24 | `humaneval_plus_corpus:corpus_fill` |
| 91 | HumanEval/25 | `humaneval_plus_corpus:corpus_fill` |
| 92 | HumanEval/26 | `humaneval_plus_corpus:corpus_fill` |
| 93 | HumanEval/27 | `humaneval_plus_corpus:corpus_fill` |
| 94 | HumanEval/29 | `humaneval_plus_corpus:corpus_fill` |
| 95 | HumanEval/30 | `humaneval_plus_corpus:corpus_fill` |
| 96 | HumanEval/31 | `humaneval_plus_corpus:corpus_fill` |
| 97 | HumanEval/33 | `humaneval_plus_corpus:corpus_fill` |
| 98 | HumanEval/34 | `humaneval_plus_corpus:corpus_fill` |
| 99 | HumanEval/36 | `humaneval_plus_corpus:corpus_fill` |
| 100 | HumanEval/37 | `humaneval_plus_corpus:corpus_fill` |
