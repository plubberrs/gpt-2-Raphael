import sys
sys.path.append("/content/gpt2-Raphael/src")

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder
import generate_unconditional_samples
import interactive_conditional_samples

import discord
from discord.ext import commands

class GPT2:

  # extracted from the source code to generate some text based on a prior
  # 1558M, 774M, 355M, 345M, 124M, and 117M
  def __init__(
      self,
      model_name='774M',
      seed=None,
      nsamples=1,
      batch_size=1,
      length=None,
      temperature=0.8,
      top_k=40,
      raw_text="",
  ):
      """
      Interactively run the model
      :model_name=117M : String, which model to use
      :seed=None : Integer seed for random number generators, fix seed to reproduce
       results
      :nsamples=1 : Number of samples to return total
      :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
      :length=None : Number of tokens in generated text, if None (default), is
       determined by model hyperparameters
      :temperature=1 : Float value controlling randomness in boltzmann
       distribution. Lower temperature results in less random completions. As the
       temperature approaches zero, the model will become deterministic and
       repetitive. Higher temperature results in more random completions.
      :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
       considered for each step (token), resulting in deterministic completions,
       while 40 means 40 words are considered at each step. 0 (default) is a
       special setting meaning no restrictions. 40 generally is a good value.
      """
      if batch_size is None:
          batch_size = 1
      assert nsamples % batch_size == 0

      self.nsamples = nsamples
      self.batch_size = batch_size
      
      self.enc = encoder.get_encoder(model_name, models_dir = 'models')
      hparams = model.default_hparams()
      with open(os.path.join('models', model_name, 'hparams.json')) as f:
          hparams.override_from_dict(json.load(f))

      if length is None:
          length = hparams.n_ctx // 2
      elif length > hparams.n_ctx:
          raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

      self.sess = tf.Session(graph=tf.Graph())
      self.sess.__enter__()
      
      self.context = tf.placeholder(tf.int32, [batch_size, None])
      np.random.seed(seed)
      tf.set_random_seed(seed)
      self.output = sample.sample_sequence(
          hparams=hparams, length=length,
          context=self.context,
          batch_size=batch_size,
          temperature=temperature, top_k=top_k
      )

      saver = tf.train.Saver()
      self.ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
      saver.restore(self.sess, self.ckpt)

  def close(self):
    self.sess.close()
  
  def generate_conditional(self,raw_text):
      context_tokens = self.enc.encode(raw_text)
      generated = 0
      for _ in range(self.nsamples // self.batch_size):
          out = self.sess.run(self.output, feed_dict={
              self.context: [context_tokens for _ in range(self.batch_size)]
          })[:, len(context_tokens):]
          for i in range(self.batch_size):
              generated += 1
              text = self.enc.decode(out[i])
              return text
              #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
              #print(text)
      #print("=" * 80)

gpt2 = GPT2(model_name = "774M")
# you must also call download_model.py (see earlier cell) with the correct parameter
# 1558M, best results takes a long time to load
# 1558M, 774M, 355M, 345M, 124M, and 117M

class Who:
  """A class defining the conversation parties: me, he"""
  def __init__(self):
    self.prefixes = []

  def matches(self,phrase):
    for prefix in self.prefixes:
      if phrase.startswith(prefix):
        #print(f"{phrase} starts with {prefix}")
        return True
      
    #print(f"{phrase} does not start with {self.prefixes}")
    return False

  def get_random_prefix(self):
    return self.prefixes[0]
  
class Me(Who):
  def __init__(self):
    super().__init__()
    self.prefixes = ["I said: \""]
   
  
class You(Who):
  def __init__(self):
    super().__init__()
    self.prefixes = ["You said: \""]

class Conversation:

  def __init__(self, prior = None):
    if prior is None:
      with open('prompt.txt') as f:
        prior=f.read()
    self.suggestion = None
    
    self.me = Me()
    self.you = You()
    self.parties  = [ self.me, self.you ]
    
    self.conversation = []
    
    lines = prior.split("\n")
    for line in lines:
      line = line.strip()
      if len(line)!=0:
        party = None
        for party in self.parties:
          if party.matches(line):
            break
        if party is None:
          raise Exception(f"Unknown party: {line}")
                
        self.conversation.append((party,line))
    self.get_suggestion()
    
  
  def get_prior(self):
    conv = ""
    for (party, line) in self.conversation:
      conv+=line+"\n"
    return conv
  
  def get_suggestion(self):
    who, last_line = self.conversation[-1]

    party_index = self.parties.index(who)
    next_party = self.parties[(party_index+1) % len(self.parties)]
      
    conv = self.get_prior()
    conv += next_party.get_random_prefix()
    answer = self.get_answer(next_party, conv)

    if not next_party.matches(answer):
      prefix = next_party.get_random_prefix()
      answer = prefix + answer
    
    self.suggestion = (next_party, answer)
  
  def next(self, party = None, answer = ""):
    """Continue the conversation
    :param party: None -> use the current party which is currently in turn
    :param answer: None -> use the suggestion, specify a text to override the 
           suggestion
    
    """
    suggested_party, suggested_answer = self.suggestion
    if party is None:
      party = suggested_party
    
    if answer == "":
      answer = suggested_answer
      
    if not party.matches(answer):
      prefix = party.get_random_prefix()
      answer = prefix + answer
    
    answer = answer.strip()
    if answer[-1] != "\"":
      # add the closing "
      answer += "\""
      
    self.conversation.append((party, answer))    
    self.get_suggestion()
    
  def retry(self):
    self.get_suggestion()
        
  def get_answer(self, party, conv):
    answer = gpt2.generate_conditional(raw_text=conv)
    lines = answer.split("\n")
    line = ""
    for line in lines:
      if line !="":
        break
      
    if line!="":
      return line
    
    return ""
      
  def show(self):
    conv = ""
    for (party, line) in self.conversation:
      conv+=line+"\n"
    if self.suggestion is not None:
      party, answer  = self.suggestion
      print(answer)
      return answer

convo = Conversation()

class Generative(commands.Cog):
    def __init__(self, client, convo):
        self.client = client
        self.convo = convo
    
    @commands.command(description='Access the GPT-3 engine and chat with Raphael')
    async def chat(self, ctx, *, incoming_msg):
        print(f'You said: "{incoming_msg}"')
        self.convo.next(convo.you, incoming_msg)
        reply = convo.show()
        await ctx.send(reply[9:-1])
    
def setup(client):
    client.add_cog(Generative(client, convo))
