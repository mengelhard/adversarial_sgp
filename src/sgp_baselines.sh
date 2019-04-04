#!/bin/sh

source activate matt_e

reps=5
for i in `seq 2 $reps`
do
	python sgp_baselines.py \
	--niter=100000 \
	--print_freq=500 \
	--hold_time=20000 \
	--lr=3e-5 \
	--img_path="./img/fitc_c50s$i" \
	--seed="$i" \
	--n_clusters=50 \
	--noise=1e-2 \
	--sgp_approx="fitc"
done

reps=5
for i in `seq 2 $reps`
do
	python sgp_baselines.py \
	--niter=100000 \
	--print_freq=500 \
	--hold_time=20000 \
	--lr=3e-5 \
	--img_path="./img/fitc_c100s$i" \
	--seed="$i" \
	--n_clusters=100 \
	--noise=1e-2 \
	--sgp_approx="fitc"
done

reps=5
for i in `seq 2 $reps`
do
	python sgp_baselines.py \
	--niter=100000 \
	--print_freq=500 \
	--hold_time=20000 \
	--lr=3e-5 \
	--img_path="./img/fitc_c200s$i" \
	--seed="$i" \
	--n_clusters=200 \
	--noise=1e-2 \
	--sgp_approx="fitc"
done
