config = {
    'exp_name': '【0605】测试匹配过程移植正常_test',
    'epoch': 500,
    'lr': 1.0,
    'mode': 'stroke',  # character / stroke
    'batch': 32,
    'val_frequency': 1000,
    'test_only': True,
    'resume': '/home/db/FudanOCR/stroke-level-decomposition/history/【0605】测试匹配过程移植正常/model.pth',
    'train_dataset': './data/mydata/train_1000',
    'test_dataset': './data/mydata/test_1000',
    'weight_decay': False,
    'schedule_frequency': 1000000,
    'image_size': 32,
    'alphabet': 3755,
}
