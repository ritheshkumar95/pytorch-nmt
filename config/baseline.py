dataset_path = "/data/lisa/data/nmt-data/multi30k"
seed = 111

batch_size = 64
epochs = 100

log_interval = 100
save_interval = None

src_params = {
    'lang': 'de',
    'vocab_size': None,
    'emb_size': 128,
    'hidden_size': 128,
    'num_layers': 1,
    'dropout': .5
}

trg_params = {
    'lang': 'en',
    'vocab_size': None,
    'emb_size': 256,
    'hidden_size': 256,
    'num_layers': 1,
    'dropout': .5
}
