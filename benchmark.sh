python src/benchmark_models.py --outdir outputs/benchmark_jp --site JPL --data outputs/jpl_occupancy_data.npz
python src/benchmark_models.py --outdir outputs/benchmark_caltech_weighted_loss_cqr --site Caltech

  python src/benchmark_models.py --data outputs/caltech_occupancy_data.npz --site Caltech --outdir outputs/caltechh --ablate-calendar
  python src/compare_ablation.py --site Caltech outputs/caltech/benchmark/benchmark_results.json outputs/caltech/benchmark_nocal/benchmark_results.json