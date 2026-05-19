import argparse

# CHANGE args.iters to 2000 (default value)
args = argparse.Namespace(get_stamp=False, seed=0, cuda=False, save=True, full_stag='none', full_ltag='none', train=True,
                d_dir='/esat/opal/tmp/back-ups/three-scenarios/store/datasets',
                m_dir='/esat/opal/tmp/back-ups/three-scenarios/store/models',
                p_dir='/esat/opal/tmp/back-ups/three-scenarios/store/plots',
                r_dir='/esat/opal/tmp/back-ups/three-scenarios/store/results', time=False, pdf=False, visdom=False,
                results_dict=False, loss_log=10, acc_log=10, acc_n=1024, sample_log=10, sample_n=64,
                no_samples=False, experiment='splitMNIST', scenario='class', contexts=5, iters=2000, batch=128,
                normalize=False, conv_type='standard', n_blocks=2, depth=0, rl=None, channels=16, conv_bn='yes',
                conv_nl='relu', gp=False, fc_lay=3, fc_units=400, fc_drop=0.0, fc_bn='no', fc_nl='relu', z_dim=100,
                singlehead=False, lr=0.001, optimizer='adam', momentum=0.0, pre_convE=False, convE_ltag='e100',
                seed_to_ltag=False, freeze_convE=False, neg_samples='all', recon_loss='BCE', bce=False,
                bce_distill=False, joint=False, cummulative=False, xdg=False, gating_prop=None,
                separate_networks=False, ewc=False, si=False, ncl=False, ewc_kfac=False, owm=False,
                weight_penalty=False, reg_strength=1.0, precondition=False, alpha=1e-10, importance_weighting=None,
                fisher_n=None, fisher_batch=1, fisher_labels='all', fisher_kfac=False, fisher_init=False,
                data_size=12000, epsilon=0.1, offline=False, gamma=1.0, lwf=False, distill=False, temp=2.0,
                fromp=False, tau=1000.0, budget=100, use_full_capacity=False, sample_selection='random',
                add_buffer=False, replay='generative', use_replay='normal', agem=False, eps_agem=1e-07, g_z_dim=100,
                g_fc_lay=3, g_fc_uni=400, g_iters=10, lr_gen=0.001, brain_inspired=False, feedback=False,
                prior='standard', per_class=False, n_modes=1, dg_gates=False, dg_type='class', dg_prop=0.1,
                hidden=False, icarl=False, prototypes=False, gen_classifier=False, eval_s=50, si_c=5000.0,
                ewc_lambda=1000000000.0)
config = {'size': 28, 'channels': 1, 'classes': 10, 'normalize': False, 'classes_per_context': 2, 'output_units': 10}
device = 'cpu'
depth = 0