import paddle
import paddle.nn as nn
from paddle.vision import models

class ImageEncoder(nn.Layer):
    def __init__(self, finetuned=True):
        super(ImageEncoder, self).__init__()
        model = models.resnet101(pretrained=True)
        self.grid_representation_extractor = nn.Sequential(*(list(model.children())[:-2]))
        for param in self.grid_representation_extractor.parameters():
            param.requires_grad = finetuned
            
    def forward(self, images):
        return self.grid_representation_extractor(images)

class AdditiveAttention(nn.Layer):
    def __init__(self, query_dim, key_dim, attn_dim):
        super(AdditiveAttention, self).__init__()
        self.attn_w_1_q = nn.Linear(query_dim, attn_dim)
        self.attn_w_1_k = nn.Linear(key_dim, attn_dim)
        self.attn_w_2 = nn.Linear(attn_dim, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(axis=1)
        
    def forward(self, query, key_value):
        queries = self.attn_w_1_q(query).unsqueeze(1)
        keys = self.attn_w_1_k(key_value)
        attn = self.attn_w_2(self.tanh(queries + keys)).squeeze(2)
        attn = self.softmax(attn)
        output = paddle.bmm(attn.unsqueeze(1), key_value).squeeze(1)
        return output, attn

class AttentionDecoder(nn.Layer):
    def __init__(self, image_code_dim, vocab_size, word_dim, attention_dim, hidden_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.attention = AdditiveAttention(hidden_size, image_code_dim, attention_dim)
        self.init_state = nn.Linear(image_code_dim, num_layers*hidden_size)
        self.rnn = nn.GRU(word_dim + image_code_dim, hidden_size, num_layers, time_major=False)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size
        
    def init_hidden_state(self, image_code, captions, cap_lens):
        batch_size = image_code.shape[0]
        image_code = image_code.transpose((0, 2, 3, 1))
        image_code = image_code.reshape((batch_size, -1, image_code.shape[-1]))
        
        sorted_cap_indices = (-cap_lens).argsort()
        sorted_cap_lens = cap_lens[sorted_cap_indices]
        captions = captions[sorted_cap_indices]
        image_code = image_code[sorted_cap_indices]
        
        hidden_state = self.init_state(image_code.mean(axis=1))
        hidden_state = hidden_state.reshape((
            batch_size,
            self.rnn.num_layers,
            self.rnn.hidden_size)).transpose((1, 0, 2))
            
        return image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state
        
    def forward_step(self, image_code, curr_cap_embed, hidden_state):
        context, alpha = self.attention(hidden_state[-1], image_code)
        x = paddle.concat((context, curr_cap_embed), axis=-1).unsqueeze(1)
        out, hidden_state = self.rnn(x, hidden_state)
        preds = self.fc(self.dropout(out.squeeze(1)))
        return preds, alpha, hidden_state
        
    def forward(self, image_code, captions, cap_lens):
        image_code, captions, sorted_cap_lens, sorted_cap_indices, hidden_state = \
            self.init_hidden_state(image_code, captions, cap_lens)
            
        batch_size = image_code.shape[0]
        lengths = sorted_cap_lens.numpy() - 1
        
        predictions = paddle.zeros((batch_size, lengths[0], self.vocab_size))
        alphas = paddle.zeros((batch_size, lengths[0], image_code.shape[1]))
        
        cap_embeds = self.embed(captions)
        
        for step in range(lengths[0]):
            real_batch_size = (lengths > step).sum()
            preds, alpha, hidden_state = self.forward_step(
                image_code[:real_batch_size],
                cap_embeds[:real_batch_size, step, :],
                hidden_state[:, :real_batch_size, :]
            )
            
            predictions[:real_batch_size, step, :] = preds
            alphas[:real_batch_size, step, :] = alpha
            
        return predictions, alphas, captions, lengths, sorted_cap_indices

class ARCTIC(nn.Layer):
    def __init__(self, image_code_dim, vocab, word_dim, attention_dim, hidden_size, num_layers):
        super(ARCTIC, self).__init__()
        self.vocab = vocab
        self.encoder = ImageEncoder()
        self.decoder = AttentionDecoder(
            image_code_dim, len(vocab), word_dim,
            attention_dim, hidden_size, num_layers
        )
        
    def forward(self, images, captions, cap_lens):
        image_code = self.encoder(images)
        return self.decoder(image_code, captions, cap_lens)
        
    def generate_by_beamsearch(self, images, beam_k, max_len):
        image_codes = self.encoder(images)
        texts = []
        
        for image_code in image_codes:
            image_code = image_code.unsqueeze(0).tile([beam_k, 1, 1, 1])
            cur_sents = paddle.full((beam_k, 1), self.vocab['<start>'], dtype='int64')
            cur_sent_embed = self.decoder.embed(cur_sents)[:,0,:]
            sent_lens = paddle.to_tensor([1]*beam_k, dtype='int64')
            
            image_code, cur_sent_embed, _, _, hidden_state = \
                self.decoder.init_hidden_state(image_code, cur_sent_embed, sent_lens)
                
            end_sents = []
            end_probs = []
            probs = paddle.zeros((beam_k, 1))
            k = beam_k
            
            while True:
                preds, _, hidden_state = self.decoder.forward_step(
                    image_code[:k], cur_sent_embed, hidden_state)
                preds = nn.functional.log_softmax(preds, axis=1)
                
                probs = probs.expand_as(preds) + preds
                
                if cur_sents.shape[1] == 1:
                    values, indices = probs[0].topk(k, 0, True, True)
                else:
                    values, indices = probs.reshape([-1]).topk(k, 0, True, True)
                    
                sent_indices = indices // len(self.vocab)
                word_indices = indices % len(self.vocab)
                
                cur_sents = cur_sents[sent_indices]
                if len(cur_sents.shape) < 2:
                    cur_sents = cur_sents.unsqueeze(0)
                cur_sents = paddle.concat([cur_sents, word_indices.unsqueeze(1)], axis=1)
                
                end_indices = [idx for idx, word in enumerate(word_indices) 
                             if word == self.vocab['<end>']]
                             
                if len(end_indices) > 0:
                    end_probs.extend(values[end_indices])
                    end_sents.extend(cur_sents[end_indices].tolist())
                    k -= len(end_indices)
                    if k == 0:
                        break
                        
                cur_indices = [idx for idx, word in enumerate(word_indices) 
                             if word != self.vocab['<end>']]
                             
                if len(cur_indices) > 0:
                    cur_sent_indices = sent_indices[cur_indices]
                    cur_word_indices = word_indices[cur_indices]
                    cur_sents = cur_sents[cur_indices]
                    probs = values[cur_indices].reshape((-1,1))
                    hidden_state = hidden_state[:, cur_sent_indices, :]
                    cur_sent_embed = self.decoder.embed(cur_word_indices.reshape((-1,1)))[:,0,:]
                    
                if cur_sents.shape[1] >= max_len:
                    break
                    
            if len(end_sents) == 0:
                gen_sent = cur_sents[0].tolist()
            else:
                gen_sent = end_sents[end_probs.index(max(end_probs))]
                
            texts.append(gen_sent)
            
        return texts 