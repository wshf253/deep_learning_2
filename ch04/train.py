import sys
sys.path.append("..")
import numpy as np
from common import config
# GPU에서 실행하려면 아래 주석을 해제하세요(CuPy 필요).
# ===============================================
config.GPU = True
# ===============================================
import pickle
from common.trainer import Trainer
from common.optimizer import Adam
from cbow import CBOW
from skip_gram import SkipGram
from common.util import create_contexts_target, to_cpu, to_gpu
from dataset import ptb

# set hyper parameter
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10

# read data
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)
if config.GPU:
    contexts, target = to_gpu(contexts), to_gpu(target)

# create model
#model = CBOW(vocab_size, hidden_size, window_size, corpus)
model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# start learning
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# saving data for later use
word_vecs = model.word_vecs
if config.GPU:
    word_vecs = to_cpu(word_vecs)
params = {}
params["word_vecs"] = word_vecs.astype(np.float16)
params["word_to_id"] = word_to_id
params["id_to_word"] = id_to_word
#pkl_file = "cbow_params.pkl"
pkl_file = "skip_gram_params.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump(params, f, -1)