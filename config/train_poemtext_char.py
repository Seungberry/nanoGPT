{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b0c3471f",
   "metadata": {},
   "outputs": [],
   "source": [
    "out_dir = 'out-poemtext-char'\n",
    "eval_interval = 250 # keep frequent because we'll overfit\n",
    "eval_iters = 200\n",
    "log_interval = 10 # don't print too too often\n",
    "always_save_checkpoint = False\n",
    "dataset = 'poemtext'\n",
    "gradient_accumulation_steps = 1\n",
    "batch_size = 64\n",
    "block_size = 256 # context of up to 256 previous characters\n",
    "n_layer = 6\n",
    "n_head = 6\n",
    "n_embd = 384\n",
    "dropout = 0.2\n",
    "learning_rate = 1e-3 # with baby networks can afford to go a bit higher\n",
    "max_iters = 5000\n",
    "lr_decay_iters = 5000 # make equal to max_iters usually\n",
    "min_lr = 1e-4 # learning_rate / 10 usually\n",
    "beta2 = 0.99 # make a bit bigger because number of tokens per iter is small\n",
    "warmup_iters = 100 # not super necessary potentially"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "nanoGPT",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.25"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
