from util.data_loader import get_loader
from transformers import BertTokenizer
from util.word_encoder import BERTWordEncoder
from util.framework import FewShotNERFramework
from model.nnshot import NNShot
from torch import optim, nn
import torch
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup, BertTokenizer

word_encoder = BERTWordEncoder(
        pretrain_ckpt)
model = NNShot(word_encoder, dot=opt.dot, ignore_index=opt.ignore_index)
framework = FewShotNERFramework(train_data_loader, val_data_loader, test_data_loader, use_sampled_data=opt.use_sampled_data)
parameters_to_optimize = list(model.named_parameters())
optimizer = AdamW(parameters_to_optimize, lr=learning_rate, correct_bias=False)

scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=train_iter) 

state_dict = framework.__load_model__("checkpoint/checkpoint_model")['state_dict']

own_state = model.state_dict()
for name, param in state_dict.items():
    if name not in own_state:
        print('ignore {}'.format(name))
        continue
    print('load {} from {}'.format(name, load_ckpt))
    own_state[name].copy_(param)

if fp16:
    from apex import amp
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')

model.train()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

support_loader = get_loader("data/inter/support_set.txt", tokenizer,
            N=5, K=5, Q=5, batch_size=1, max_length=100, ignore_index=-1)

# support, query = self.train_data_loader
for support, query in support_loader:
    label = torch.cat(query['label'], 0)
    if torch.cuda.is_available():
        for k in support:
            if k != 'label' and k != 'sentence_num':
                support[k] = support[k].cuda()
                query[k] = query[k].cuda()
        label = label.cuda()


    logits, pred = model(support, query)
    print(logits)
    print(pred)
