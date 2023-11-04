import sys
sys.path.append('..')
from common.optimizer import SGD
from common.trainer import RnnlmTrainer
from common.util import eval_perplexity
from dataset import ptb
from rnnlm import Rnnlm

# hyperparameters
batch_size = 20 # N
wordvec_size = 100 # D
hidden_size = 100 # H
time_size = 35 # T
lr = 20.0
max_epoch = 4
max_grad = 0.25

# training data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# generate model
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# train using clip_grads
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval=20)
trainer.plot(ylim=(0,500))

# test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print("test perplexity : ", ppl_test)

# save parameters
model.save_params()