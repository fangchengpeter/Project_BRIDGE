# TensorFlow 2 Migration Guide

This document describes the changes made to migrate the BRIDGE codebase from **TensorFlow 1.x** (originally TF 1.15) to **TensorFlow 2.x** (tested on TF 2.11).

---

## Table of Contents

- [Overview](#overview)
- [Files Changed](#files-changed)
- [API Changes Reference](#api-changes-reference)
- [New Model API](#new-model-api)
- [Updated Environment Setup](#updated-environment-setup)
- [Running the Tests](#running-the-tests)
- [Bug Fix](#bug-fix)

---

## Overview

TF1 relied on a static computation graph and an explicit `Session` to run operations. TF2 runs in **eager mode** by default — operations execute immediately as regular Python, returning concrete values without a session. This migration removes all session-based patterns and replaces them with direct Python/NumPy calls backed by `tf.GradientTape` for gradient computation.

---

## Files Changed

### BRIDGE-MNIST

| File | Changes |
|------|---------|
| `linear_classifier.py` | Full rewrite — removed all placeholders and session ops, replaced with TF2 eager class |
| `CNN_model.py` | Full rewrite — same pattern as above, added `training` flag for dropout |
| `DecLearning.py` | Removed all `sess` parameters; `initialization()` is now a no-op; updated all internal calls |
| `Byzantine_strategy.py` | Removed `sess` parameter from all functions; gradients computed via model method |
| `dec_BRIDGE.py` | Removed `tf.reset_default_graph()`, `tf.set_random_seed()` → `tf.random.set_seed()`, removed session |
| `dec_ByRDiE.py` | Same driver-level cleanup as above |
| `decentralized learning.py` | Same driver-level cleanup; also fixed pre-existing bug (see [Bug Fix](#bug-fix)) |

### BRIDGE-CIFAR

| File | Changes |
|------|---------|
| `linear_classifier.py` | Full rewrite — same as MNIST version but with 3072-dimensional input |
| `CNN_model.py` | Full rewrite — updated for CIFAR input shape (32×32×3) and 8×8×64 flatten |
| `DecLearning.py` | Same session removal as MNIST |
| `Byzantine_strategy.py` | Same session removal as MNIST |
| `dec_BRIDGE.py` | Same driver-level cleanup as MNIST |
| `distributed learning.py` | Same driver-level cleanup as MNIST |

---

## API Changes Reference

### Removed entirely

| TF1 API | Replacement |
|---------|-------------|
| `tf.placeholder(...)` | Removed — numpy arrays passed directly as function arguments |
| `tf.InteractiveSession()` | Removed — not needed in eager mode |
| `sess.run(op, feed_dict={...})` | Removed — operations execute immediately |
| `tf.global_variables_initializer()` | Removed — `tf.Variable` initializes on construction |
| `tf.reset_default_graph()` | Removed — no default graph in TF2 |
| `var.eval()` | `var.numpy()` |

### Renamed

| TF1 API | TF2 API |
|---------|---------|
| `tf.truncated_normal(...)` | `tf.random.truncated_normal(...)` |
| `tf.set_random_seed(n)` | `tf.random.set_seed(n)` |
| `tf.nn.max_pool(...)` | `tf.nn.max_pool2d(...)` |
| `tf.nn.dropout(h, keep_prob)` | `tf.nn.dropout(h, rate=1-keep_prob)` |
| `tf.train.GradientDescentOptimizer(lr)` | `tf.keras.optimizers.SGD(learning_rate=lr)` |
| `tf.train.AdamOptimizer(lr)` | `tf.keras.optimizers.Adam(learning_rate=lr)` |
| `tf.compat.v1.train.AdamOptimizer(lr)` | `tf.keras.optimizers.Adam(learning_rate=lr)` |

### Gradient computation

**Before (TF1):**
```python
# Defined at graph build time
self.train_step = optimizer.minimize(self.loss)
self.gradient_w = optimizer.compute_gradients(loss, var_list=[self.W])[0][0]

# Executed via session
sess.run(self.train_step, feed_dict={self.x: data, self.y_: labels, self.stepsize: lr})
g = sess.run(self.gradient_w, feed_dict={self.x: data, self.y_: labels})
```

**After (TF2):**
```python
# Computed eagerly at call time
def train_step_fn(self, x, y_, stepsize):
    self.optimizer.learning_rate = stepsize
    with tf.GradientTape() as tape:
        loss = self.compute_loss(x, y_)
    grads = tape.gradient(loss, self.layers)
    self.optimizer.apply_gradients(zip(grads, self.layers))

def get_gradient_w(self, x, y_):
    with tf.GradientTape() as tape:
        loss = self.compute_loss(x, y_)
    return tape.gradient(loss, self.W).numpy()
```

### Weight assignment

**Before (TF1):**
```python
# Placeholders + session-based assign op
self.W_com = tf.placeholder(tf.float32, shape=[784, 10])
self.communication = self.W.assign(self.W_com)

def assign(self, weight, sess):
    sess.run(self.communication, feed_dict={self.W_com: weight[0]})
```

**After (TF2):**
```python
def assign(self, weight):
    self.W.assign(weight[0])
    self.b.assign(weight[1])
```

### Dropout

**Before (TF1):**
```python
self.keep_prob = tf.placeholder(tf.float32)
self.h_fc1_drop = tf.nn.dropout(self.h_fc1, self.keep_prob)
# ...called with feed_dict={model.keep_prob: 0.5}  (train) or 1.0 (eval)
```

**After (TF2):**
```python
def __call__(self, x, training=False):
    ...
    if training:
        h_fc1 = tf.nn.dropout(h_fc1, rate=0.5)
    ...
```

---

## New Model API

Both `linear_classifier` and `CNN` now expose the following interface:

### `linear_classifier`

```python
model = linear_classifier(stepsize=1e-1, sigma2=0.1, adam=False)

model.weights()                          # → [W_np, b_np]
model.assign([W_np, b_np])              # set weights from numpy arrays
model.compute_loss(x, y_)               # → tf.Tensor scalar
model.accuracy_eval(x, y_)             # → float
model.train_step_fn(x, y_, stepsize)   # one gradient step
model.get_gradient_w(x, y_)            # → np.ndarray shape [784, 10]
model.get_gradient_b(x, y_)            # → np.ndarray shape [10]
model.apply_gradient_w(grad, stepsize) # apply external W gradient
model.apply_gradient_b(grad, stepsize) # apply external b gradient
```

### `CNN`

```python
model = CNN(stepsize=1e-4)

model.weights()                   # → list of 8 np.ndarrays
model.assign(weight_list)         # set all layer weights
model.compute_loss(x, y_, training=False)
model.accuracy_eval(x, y_)       # → float
model.train_step_fn(x, y_)       # one gradient step (with dropout)
model.get_gradient(x, y_, training=True)   # → list of 8 np.ndarrays
model.apply_ext_gradient(grads)   # apply external gradient list
```

### `DecLearning` — removed `sess` from all methods

```python
# Before
para.communication(w_nodes, neighbors, sess, b=b, ...)
para.node_update(w_nodes, local_set, sess, stepsize=lr)
para.communication_w(w_nodes, neighbors, p, sess, b, ...)

# After
para.communication(w_nodes, neighbors, b=b, ...)
para.node_update(w_nodes, local_set, stepsize=lr)
para.communication_w(w_nodes, neighbors, p, b, ...)
```

---

## Updated Environment Setup

The original `environment.yml` specifies `tensorflow==1.15`. Update it to use TF2 before creating the conda environment:

```yaml
# environment.yml — change this line:
- tensorflow==1.15
# to:
- tensorflow==2.11
```

Then create and activate the environment as usual:

```bash
conda env create -f environment.yml
conda activate byzantine
```

Or install directly with pip:

```bash
pip install tensorflow==2.11.0 numpy matplotlib
```

---

## Running the Tests

A smoke-test suite (`test_migration.py`) is included in the repo root. It covers 45 tests across both `BRIDGE-MNIST` and `BRIDGE-CIFAR` using small synthetic data — no dataset download required.

```bash
cd Project_BRIDGE
python test_migration.py
```

Expected output:

```
============================================================
BRIDGE TF2 Migration — Smoke Tests
============================================================

[BRIDGE-MNIST]
  PASS  mnist linear_classifier forward pass
  PASS  mnist linear_classifier accuracy
  ...

[BRIDGE-CIFAR]
  PASS  cifar linear_classifier forward pass
  ...

============================================================
Results: 45 passed, 0 failed
============================================================
```

---

## Bug Fix

A pre-existing bug in `BRIDGE-MNIST/decentralized learning.py` was fixed as part of this migration. The `__main__` block called `server_update(...)` which was never defined in that file; the correct function `agent_update(...)` was defined but unused. This has been corrected.
