#!/bin/sh

source activate matt_e

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-5 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/simple_1e-5_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type simple
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-5 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/direct_1e-5_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type direct
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-5 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/gumbel_1e-5_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type gumbel
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-4 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/simple_1e-4_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type simple
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-4 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/direct_1e-4_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type direct
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-4 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/gumbel_1e-4_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type gumbel
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-3 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/simple_1e-3_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type simple
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-3 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/direct_1e-3_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type direct
done

reps=2
for i in `seq 2 $reps`
do
	python adv_sgp.py \
	--dataset abalone \
	--niter 40000 \
	--print_freq 500 \
	--g_lr 1e-3 \
	--d_lr 1e-4 \
	--img_path "~/sgp/img/abalone/gumbel_1e-3_s$i" \
	--seed "$i" \
	--n_clusters 50 \
	--n_z 100 \
	--num_sgp_samples 50 \
	--batch_size 150 \
	--gen_type gumbel
done