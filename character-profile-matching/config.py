config = {
    'exp_name' : 'acpm_experiment_01',
    'epoch' : 500,
    'lr' : 1.0,
    'mode' : 'radical',   # radical
    'batch' : 32,
    'val_frequency' : 5,
    'parallel' : False,
    'test_only' : False,
    'pretrain': False,
    'resume': '',
    'train_dataset': './data/ctw_train_lmdb_760062/',
    'test_dataset': './data/ctw_val_3755lmdb_51865/',
    'print_standard': './printstandard_ctw/',
    'weight_decay': False,
    'schedule_frequency' : 15,
    'image_size' : 32,
    'update_gallery_feature' : 100,
    'distance_coeff' : 1,
    'encoder' : 'resnet', # resnet / densenet / vgg
    'decoder' : 'transformer',
    'alphabet' : 'ctw',
    'feature_metric': 'mse', # mse / cosine
    'rn_loss': 'L1', # L1 / CE
    'lambda_f_test' : 1,
    'lambda_sn_test' : 1,
    'lambda_sl_test' : 1,
    'lambda_rn_test' : 1,
    'candidate_search_range': 0,  # 0 or 1 are recommended
    'stn': False,
    'discard': False,
    'constrain': False,
}
