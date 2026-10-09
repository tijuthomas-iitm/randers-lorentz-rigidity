#!/usr/bin/env bash
# Reproduce the recorded outputs. Run from this directory. Approximate times on one CPU core.
set -e
cd scripts
echo "== Theorem 2 certificates (exact rational ranks) =="
python3 generic_cert.py 2 3 gen 1          # ~1 min   rankA=147 rankB=85 nullity 20 projected kernel 1
python3 generic_cert.py 3 3 gen 1          # ~10 min  rankA=659 rankB=444 nullity 60 projected kernel 1
python3 generic_cert.py 2 3 B 1            # ~1 min   vacuum certificate, n=2
python3 generic_cert.py 3 3 pwcirc 1       # ~10 min  vacuum certificate, n=3
echo "== Independent implementations =="
python3 k2_independent_2d.py 3             # ~2 min   same ranks for n=2
python3 k2_independent.py 3                # ~15 min  same ranks for n=3
echo "== Second multipliers and abstract lemma checks =="
python3 k1_k3.py
echo "== Central fields, Crofton family, Crampin-Prince comparison (exact) =="
python3 central_lagr.py; python3 central_lagr2.py; python3 crofton_check.py; python3 cp_check.py; python3 flag_gauge_check.py; python3 derivation_checks.py
echo "== Numerical reducibility test (needs jax; slow, ~15 min) =="
python3 conjR_run.py; python3 conjR_run2.py
