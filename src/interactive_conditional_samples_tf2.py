#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model_tf2, sample_tf2, encoder

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def interact_model(
    model_name='117M',
    ckpt_name=None,
    seed=None,
    nsamples=1,
    batch_size=1,
    length=None,
    temperature=1,
    top_k=0,
    top_p=1,
    models_dir='models',
    ckpts_dir='checkpoint'
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :ckpt_name=None : String, which checkpoint to use
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
    :models_dir : path to parent folder containing model subfolders
     (i.e. contains the <model_name> folder)
    :ckpts_dir : path to parent folder containing checkpoint subfolders
    """
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    # support only batch_size 1 right now
    batch_size = 1
    # if batch_size is None:
    #     batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model_tf2.HPARAMS[model_name]
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.update(json.load(f))

    if length is None:
        length = hparams['n_ctx'] - 1
    elif length >= hparams['n_ctx']:
        raise ValueError("Can't get samples longer than window size: %s" % hparams['n_ctx'])

    # Setup checkpoint manager
    gpt2 = model_tf2.GPT2(hparams)
    if ckpt_name is None:
        checkpoint_path = os.path.join(models_dir, model_name)
    else:
        checkpoint_path = os.path.join(ckpts_dir, ckpt_name)
    ckpt = tf.train.Checkpoint(gpt2=gpt2)
    last_ckpt = tf.train.latest_checkpoint(checkpoint_path)
    if last_ckpt is not None:
        ckpt.restore(last_ckpt)
        print('Checkpoint restored from {}'\
            .format(last_ckpt))
    else:
        print("No checkpoint found at {}".format(checkpoint_path))


    while True:
        raw_text = input("Model prompt >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        context_tokens = enc.encode(raw_text)
        generated = 0
        for _ in range(nsamples // batch_size):
            context = np.array([context_tokens for _ in range(batch_size)])
            out = sample_tf2.sample_sequence(
                gpt2_model=gpt2, 
                length=length, 
                context=context,
                top_p=top_p,
                top_k=top_k,
                batch_size=batch_size)[:, len(context_tokens):]
            for i in range(batch_size):
                generated += 1
                text = enc.decode(out[i].numpy())
                print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
        print("=" * 80)

if __name__ == '__main__':
    fire.Fire(interact_model)

