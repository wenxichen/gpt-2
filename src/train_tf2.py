#!/usr/bin/env python3

import argparse, time, os
import tensorflow as tf

from load_dataset import load_dataset, Sampler
import model_tf2, sample_tf2, encoder

CHECKPOINT_ROOT = './checkpoint'
SEQ_LEN = 1024

parser = argparse.ArgumentParser(
    description='Train GPT-2 on your custom dataset.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', metavar='PATH', type=str, required=True, help='Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).')
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate input files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--encoding', type=str, default='utf-8', help='Set the encoding for reading and writing files.')
parser.add_argument('--batch_size', metavar='SIZE', type=int, default=1, help='Batch size')
parser.add_argument('--learning_rate', metavar='LR', type=float, default=0.00002, help='Learning rate for Adam')
parser.add_argument('--accumulate_gradients', metavar='N', type=int, default=1, help='Accumulate gradients across N minibatches.')
parser.add_argument('--top_k', type=int, default=40, help='K for top-k sampling.')
parser.add_argument('--top_p', type=float, default=0.0, help='P for top-p sampling. Overrides top_k if set > 0.')
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and samples/')
parser.add_argument('--sample_every', metavar='N', type=int, default=1000, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=256, help='Sample this many tokens')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')
parser.add_argument('--print_loss_every', metavar='STEPS', type=int, default=10, help='Print loss every STEPS steps.')

def loss_function(real, pred):
    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=real, logits=pred))
    return loss


def generate_sample(gpt2_model, enc, sampler, sample_length, top_k, top_p):
    # fix at batch size 1 for now
    context = sampler.sample_batch(1,1)
    output = sample_tf2.sample_sequence(
        gpt2_model=gpt2_model, 
        length=sample_length, 
        context=context,
        top_p=top_p,
        top_k=top_k,
        batch_size=1)
    return enc.decode(output[0,1:].numpy())

def train(model_name, dataset, run_name, batch_size, learning_rate, encoding, 
    print_loss_every, save_every, sample_every, sample_length, top_k, top_p, 
    combine, accumulate_gradients):
    
    enc = encoder.get_encoder(model_name, "models")
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

    gpt2 = model_tf2.GPT2(model_tf2.HPARAMS)

    # Setup checkpoint manager
    checkpoint_path = os.path.join(CHECKPOINT_ROOT, args.run_name)
    ckpt = tf.train.Checkpoint(gpt2=gpt2, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=5
    )
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Checkpoint restored from {}'\
            .format(ckpt_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")

    # The @tf.function trace-compiles train_step into a TF graph 
    # for faster execution. The function specializes to the precise 
    # shape of the argument tensors. To avoid re-tracing due to the 
    # variable sequence lengths or variable batch sizes (the last batch 
    # is smaller), use input_signature to specify more generic shapes.
    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int32)
    ]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp):
        real = inp[:,1:]
        
        dec_padding_mask = model_tf2.create_look_ahead_mask(tf.shape(inp)[1])

        with tf.GradientTape() as tape:
            predictions, present, attn_weights = \
                gpt2(inp, None, dec_padding_mask)
            loss = loss_function(real, predictions[:,:-1])

        gradients = tape.gradient(loss, gpt2.trainable_variables)    
        optimizer.apply_gradients(
            zip(gradients, gpt2.trainable_variables)
        )

        train_loss(loss)

    # Loading data
    print('Loading dataset...')
    chunks = load_dataset(enc, dataset, combine, encoding=encoding)
    data_sampler = Sampler(chunks)
    print('dataset has', data_sampler.total_size, 'tokens')

    print('Training...')
    counter = 1
    avg_loss = (0.0, 0.0)
    start_time = time.time()

    try:
        while True:
            if counter % save_every == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for step {} at {}'.format(counter, ckpt_save_path))
            if counter % sample_every == 0:
                print('Generating samples...')
                sample = generate_sample(gpt2, enc, data_sampler, sample_length, top_k, top_p)
                print(sample)

            train_loss.reset_states()

            inp = data_sampler.sample_batch(SEQ_LEN, args.batch_size)

            train_step(inp)
            batch_loss = train_loss.result()

            avg_loss = (avg_loss[0] * 0.99 + batch_loss,
                        avg_loss[1] * 0.99 + 1.0)

            if (counter) % print_loss_every == 0:
                print(
                    '[{counter} | {time:2.2f}] loss={loss:2.2f} avg={avg:2.2f}'
                    .format(
                        counter=counter,
                        time=time.time() - start_time,
                        loss=batch_loss,
                        avg=avg_loss[0] / avg_loss[1]))

            counter += 1
    except KeyboardInterrupt:
        print('interrupted')
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for step {} at {}'.format(counter, ckpt_save_path))


if __name__ == '__main__':
    args = parser.parse_args()

    train(**vars(args))