#!/usr/bin/env python3

import argparse, time, os, random
import tensorflow as tf
import numpy as np

from load_dataset import load_dataset, Sampler
import model_tf2, sample_tf2, encoder

# Handle multi-gpu crashing through determinism
print("Set up framework determinism...")
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 42
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

CHECKPOINT_ROOT = './checkpoint'
SEQ_LEN = 1024
WEIGHTS = {"117M": "./models/117M_tf2/pretrained_weights.h5",
            "355M": "./models/355M_tf2/pretrained_weights.h5"}

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
parser.add_argument('--run_name', type=str, default='run1', help='Run id. Name of subdirectory in checkpoint/ and logs/')
parser.add_argument('--restore_from', type=str, default='latest', help='Either "latest", "fresh", "scratch" or "weight"')
parser.add_argument('--ckpt_max_to_keep', type=int, default=5, help='Max number of checkpoint to keep.')
parser.add_argument('--model_weight_file', type=str, help='Path to a model weight file when loading from weight.')
parser.add_argument('--sample_every', metavar='N', type=int, default=1000, help='Generate samples every N steps')
parser.add_argument('--sample_length', metavar='TOKENS', type=int, default=256, help='Sample this many tokens')
parser.add_argument('--save_every', metavar='N', type=int, default=1000, help='Write a checkpoint every N steps')
parser.add_argument('--print_loss_every', metavar='STEPS', type=int, default=10, help='Print loss every STEPS steps.')
parser.add_argument('--val_dataset', metavar='PATH', type=str, default=None, help='Dataset for validation loss, defaults to --dataset.')
parser.add_argument('--val_batch_size', metavar='SIZE', type=int, default=1, help='Batch size for validation.')
# parser.add_argument('--val_batch_count', metavar='N', type=int, default=40, help='Number of batches for validation.')
parser.add_argument('--val_every', metavar='STEPS', type=int, default=0, help='Calculate validation loss every STEPS steps.')

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

def train(model_name, dataset, run_name, restore_from, batch_size, learning_rate, encoding='utf-8', 
    print_loss_every=10, save_every=1000, sample_every=1000, sample_length=256, top_k=40, top_p=0, 
    combine=50000, accumulate_gradients=1, 
    val_every=0, val_dataset=None, val_batch_size=1,
    model_weight_file=None, ckpt_max_to_keep=5,
    data_sampler=None, val_data_sampler=None,
    base_dir="", stop_by_train_loss=None, return_model=False, train_wte_weight_only=False):
    
    enc = encoder.get_encoder(
        model_name, os.path.join(base_dir, "models"))
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')

    validation_loss = tf.keras.metrics.Mean(name='validation_loss')

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    gpt2 = model_tf2.create_GPT2_model(model_tf2.HPARAMS[model_name], train_wte_weight_only=train_wte_weight_only)

    # Setup checkpoint manager
    checkpoint_path = os.path.join(base_dir, CHECKPOINT_ROOT, run_name)
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), train_time=tf.Variable(0.0), gpt2=gpt2, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, checkpoint_path, max_to_keep=ckpt_max_to_keep
    )
    if restore_from=='latest' and ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Checkpoint restored from {}'\
            .format(ckpt_manager.latest_checkpoint))
    elif restore_from=='weight' and model_weight_file is not None:
        print("Loading weights from {}...".format(model_weight_file))
        gpt2.load_weights(model_weight_file)
    elif restore_from!='scratch':
        if model_name in WEIGHTS:
            weights_path = os.path.join(base_dir, WEIGHTS[model_name])
            print("Loading weights from {}...".format(weights_path))
            gpt2.load_weights(weights_path)
        else:
            print("No weights exists for {} model, initializing model from scratch.".format(model_name))
    else:
        print("Initializing model from scratch.")

    gpt2.summary()

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
        
        # dec_padding_mask = model_tf2.create_look_ahead_mask(tf.shape(inp)[1])

        with tf.GradientTape() as tape:
            predictions, _, _ = gpt2(inp, None)
            loss = loss_function(real, predictions[:,:-1])

        gradients = tape.gradient(loss, gpt2.trainable_variables)    
        optimizer.apply_gradients(
            zip(gradients, gpt2.trainable_variables)
        )

        train_loss(loss)

    def validation(val_inp):
        val_real = val_inp[:,1:]
        predictions, _, _ = gpt2(val_inp, None)
        loss = loss_function(val_real, predictions[:,:-1])
        validation_loss(loss)

    # Set up tensorboard for logging
    train_log_dir = 'logs/' + run_name + '/train'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    if val_every > 0:
        val_log_dir = 'logs/' + run_name + '/val'
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

    # Loading data
    if data_sampler is None:
        print('Loading dataset...')
        chunks = load_dataset(enc, dataset, combine, encoding=encoding)
        data_sampler = Sampler(chunks)
        print('dataset has', data_sampler.total_size, 'tokens')
    if val_data_sampler is None and val_dataset is not None:
        print('Loading validation dataset...')
        if val_every > 0:
            if val_dataset:
                val_chunks = load_dataset(enc, val_dataset, combine, encoding=encoding)
            else:
                val_chunks = chunks
            val_data_sampler = Sampler(val_chunks)

    stop = False
    early_stop_counter = 0
    ran_validation = False
    print('Training...')
    avg_loss = (0.0, 0.0)
    avg_val_loss = (0.0, 0.0)
    previous_time = time.time()

    try:
        while not stop:
            if int(ckpt.step) % save_every == 0:
                ckpt_save_path = ckpt_manager.save()
                print ('Saving checkpoint for step {} at {}'.format(int(ckpt.step), ckpt_save_path))
            if int(ckpt.step) % sample_every == 0:
                print('Generating samples...')
                sample = generate_sample(gpt2, enc, data_sampler, sample_length, top_k, top_p)
                print(sample)

            # run train step
            train_loss.reset_states()
            inp = data_sampler.sample_batch(SEQ_LEN, batch_size)
            train_step(inp)
            batch_loss = train_loss.result()
            avg_loss = (avg_loss[0] * 0.99 + batch_loss,
                        avg_loss[1] * 0.99 + 1.0)
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', batch_loss, step=int(ckpt.step))
                tf.summary.scalar('learning_rate', learning_rate, step=int(ckpt.step))

            # run validation step
            if val_every > 0 and (int(ckpt.step) % val_every == 0 or int(ckpt.step) == 1):
                ran_validation = True
                validation_loss.reset_states()
                val_inp = val_data_sampler.sample_batch(SEQ_LEN, val_batch_size)
                validation(val_inp)
                val_loss = validation_loss.result()
                avg_val_loss = \
                    (avg_val_loss[0] * 0.99 + val_loss,
                     avg_val_loss[1] * 0.99 + 1.0)
                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=int(ckpt.step))


            cur_time = time.time()
            ckpt.train_time.assign_add(cur_time - previous_time)
            previous_time = cur_time

            avg_train_loss = avg_loss[0] / avg_loss[1]
            if int(ckpt.step) % print_loss_every == 0:
                msg = '[{step} | {time:2.2f}] loss={loss:2.4f} avg={avg:2.4f}'\
                        .format(
                            step=int(ckpt.step),
                            time=float(ckpt.train_time),
                            loss=batch_loss,
                            avg=avg_train_loss
                            )
                if ran_validation:
                    msg += ' val_loss={val_loss:2.4f} avg_val_loss={avg_val_loss:2.4f}'\
                            .format(
                                val_loss=val_loss,
                                avg_val_loss=avg_val_loss[0] / avg_val_loss[1]
                                )
                print(msg)

            ran_validation = False
            ckpt.step.assign_add(1)
            if stop_by_train_loss is not None:
                if avg_train_loss < stop_by_train_loss:
                    early_stop_counter += 1
                    if early_stop_counter >= 50:
                        print("Early stopping the training...")
                        stop=True
                else:
                    early_stop_counter = 0

    except KeyboardInterrupt:
        print('interrupted')
        ckpt_save_path = ckpt_manager.save()
        print ('Saving checkpoint for step {} at {}'.format(int(ckpt.step), ckpt_save_path))

    if return_model:
        return gpt2


if __name__ == '__main__':
    args = parser.parse_args()

    train(**vars(args))