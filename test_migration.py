"""
Smoke-test suite for the TF2 migration of BRIDGE-MNIST and BRIDGE-CIFAR.
Tests each module in isolation using tiny synthetic data to verify:
  - model construction, forward pass, loss, gradients
  - assign / weights round-trip
  - DecLearning graph generation, screening methods
  - Byzantine strategy functions
  - Driver-level communication + node_update loop (1 iteration)
"""

import sys
import os
import traceback
import numpy as np

PASS = []
FAIL = []

def run(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as e:
        FAIL.append((name, e))
        print(f"  FAIL  {name}")
        traceback.print_exc()

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def fake_mnist(n=16):
    x = np.random.rand(n, 784).astype(np.float32)
    y = np.eye(10)[np.random.randint(0, 10, n)].astype(np.float32)
    return x, y

def fake_cifar(n=16):
    x = np.random.rand(n, 3072).astype(np.float32)
    y = np.eye(10)[np.random.randint(0, 10, n)].astype(np.float32)
    return x, y

# ─────────────────────────────────────────────
# BRIDGE-MNIST tests
# ─────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BRIDGE-MNIST"))

def test_mnist_linear_classifier_forward():
    from linear_classifier import linear_classifier
    m = linear_classifier(stepsize=1e-2)
    x, y = fake_mnist()
    loss = m.compute_loss(x, y)
    assert loss.numpy() > 0, "loss should be positive"

def test_mnist_linear_classifier_accuracy():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    x, y = fake_mnist()
    acc = m.accuracy_eval(x, y)
    assert 0.0 <= acc <= 1.0

def test_mnist_linear_classifier_train_step():
    from linear_classifier import linear_classifier
    m = linear_classifier(stepsize=1e-2)
    x, y = fake_mnist()
    loss_before = m.compute_loss(x, y).numpy()
    for _ in range(5):
        m.train_step_fn(x, y, 1e-2)
    loss_after = m.compute_loss(x, y).numpy()
    assert loss_after < loss_before, "loss should decrease after training steps"

def test_mnist_linear_classifier_weights_assign():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    w = m.weights()
    assert len(w) == 2
    assert w[0].shape == (784, 10)
    assert w[1].shape == (10,)
    # assign zeros and verify
    m.assign([np.zeros((784, 10)), np.zeros(10)])
    w2 = m.weights()
    assert np.allclose(w2[0], 0) and np.allclose(w2[1], 0)

def test_mnist_linear_classifier_gradient_w():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    x, y = fake_mnist()
    gw = m.get_gradient_w(x, y)
    assert gw.shape == (784, 10)

def test_mnist_linear_classifier_gradient_b():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    x, y = fake_mnist()
    gb = m.get_gradient_b(x, y)
    assert gb.shape == (10,)

def test_mnist_linear_classifier_apply_gradient_w():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    x, y = fake_mnist()
    gw = m.get_gradient_w(x, y)
    m.apply_gradient_w(gw, stepsize=1e-3)

def test_mnist_linear_classifier_apply_gradient_b():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    x, y = fake_mnist()
    gb = m.get_gradient_b(x, y)
    m.apply_gradient_b(gb, stepsize=1e-3)

def test_mnist_cnn_forward():
    from CNN_model import CNN
    m = CNN(stepsize=1e-4)
    x, y = fake_mnist()
    loss = m.compute_loss(x, y)
    assert loss.numpy() > 0

def test_mnist_cnn_accuracy():
    from CNN_model import CNN
    m = CNN()
    x, y = fake_mnist()
    acc = m.accuracy_eval(x, y)
    assert 0.0 <= acc <= 1.0

def test_mnist_cnn_train_step():
    from CNN_model import CNN
    m = CNN(stepsize=1e-4)
    x, y = fake_mnist()
    loss_before = m.compute_loss(x, y).numpy()
    for _ in range(3):
        m.train_step_fn(x, y)
    loss_after = m.compute_loss(x, y).numpy()
    assert loss_after < loss_before

def test_mnist_cnn_weights_assign():
    from CNN_model import CNN
    m = CNN()
    w = m.weights()
    assert len(w) == 8
    m2 = CNN()
    m2.assign(w)
    w2 = m2.weights()
    for a, b in zip(w, w2):
        assert np.allclose(a, b)

def test_mnist_cnn_get_gradient():
    from CNN_model import CNN
    m = CNN()
    x, y = fake_mnist()
    grads = m.get_gradient(x, y, training=True)
    assert len(grads) == 8

def test_mnist_cnn_apply_ext_gradient():
    from CNN_model import CNN
    m = CNN(stepsize=1e-4)
    x, y = fake_mnist()
    grads = m.get_gradient(x, y)
    m.apply_ext_gradient(grads)

def test_mnist_declearning_gen_graph():
    from DecLearning import DecLearning
    d = DecLearning(nodes=5, byzantine=1)
    d.gen_graph(min_neigh=1, con_rate=30)
    assert d.graph.shape == (5, 5)
    assert len(d.edge_weight) == 5

def test_mnist_declearning_get_neighbor():
    from DecLearning import DecLearning
    d = DecLearning(nodes=5)
    d.gen_graph(con_rate=30)
    nb = d.get_neighbor()
    assert len(nb) == 5
    # each node must be its own neighbor (diagonal=1)
    for i, lst in enumerate(nb):
        assert i in lst

def test_mnist_declearning_acc_test():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    d = DecLearning(nodes=3)
    m = linear_classifier()
    x, y = fake_mnist()
    acc = d.acc_test(m, x, y)
    assert 0.0 <= acc <= 1.0

def test_mnist_declearning_communication_dgd():
    """DGD (no screening) communication round-trip"""
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 4
    d = DecLearning(nodes=n, byzantine=0)
    d.gen_graph(con_rate=1)  # fully connected
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=0, goByzantine=False, screenMethod=None)

def test_mnist_declearning_communication_bridge():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 6
    b = 1
    d = DecLearning(nodes=n, byzantine=b)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=b, goByzantine=False, screenMethod='BRIDGE')

def test_mnist_declearning_communication_median():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 5
    d = DecLearning(nodes=n, byzantine=1)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=1, goByzantine=False, screenMethod='Median')

def test_mnist_declearning_communication_krum():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 6
    b = 1
    d = DecLearning(nodes=n, byzantine=b)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=b, goByzantine=False, screenMethod='Krum')

def test_mnist_declearning_communication_byzantine():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 6
    b = 1
    d = DecLearning(nodes=n, byzantine=b)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=b, goByzantine=True, screenMethod='BRIDGE')

def test_mnist_declearning_node_update():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    import types

    n = 3
    d = DecLearning(nodes=n)
    nodes = [linear_classifier(stepsize=1e-2) for _ in range(n)]
    x, y = fake_mnist(n * 10)
    # build a minimal dis_data-like object
    data = types.SimpleNamespace(
        dist_data=[x[i*10:(i+1)*10] for i in range(n)],
        dist_label=[y[i*10:(i+1)*10] for i in range(n)],
    )
    d.node_update(nodes, data, stepsize=1e-2)

def test_mnist_declearning_communication_w():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 5
    b = 1
    d = DecLearning(nodes=n, byzantine=b)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication_w(nodes, nb, p=0, b=b, screen=True, goByzantine=False)

def test_mnist_declearning_communication_b():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 5
    b = 1
    d = DecLearning(nodes=n, byzantine=b)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication_b(nodes, nb, p=0, b=b, screen=True, goByzantine=False)

def test_mnist_declearning_node_update_w():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    import types
    n = 3
    d = DecLearning(nodes=n)
    nodes = [linear_classifier() for _ in range(n)]
    x, y = fake_mnist(n * 10)
    data = types.SimpleNamespace(
        dist_data=[x[i*10:(i+1)*10] for i in range(n)],
        dist_label=[y[i*10:(i+1)*10] for i in range(n)],
    )
    d.node_update_w(nodes, data, p=0, stepsize=1e-3)

def test_mnist_declearning_node_update_b():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    import types
    n = 3
    d = DecLearning(nodes=n)
    nodes = [linear_classifier() for _ in range(n)]
    x, y = fake_mnist(n * 10)
    data = types.SimpleNamespace(
        dist_data=[x[i*10:(i+1)*10] for i in range(n)],
        dist_label=[y[i*10:(i+1)*10] for i in range(n)],
    )
    d.node_update_b(nodes, data, p=0, stepsize=1e-3)

def test_mnist_byzantine_strategy_random():
    from Byzantine_strategy import Byz_random
    from CNN_model import CNN
    m = CNN()
    x, y = fake_mnist()
    grad = Byz_random(x, y, m)
    assert len(grad) == 8
    assert grad[0].shape == (5, 5, 1, 32)

def test_mnist_byzantine_strategy_flip():
    from Byzantine_strategy import Byz_flip
    from CNN_model import CNN
    m = CNN()
    x, y = fake_mnist()
    grad = Byz_flip(x, y, m)
    assert len(grad) == 8

def test_mnist_byzantine_strategy_random_label():
    from Byzantine_strategy import Byz_random_label
    from CNN_model import CNN
    m = CNN()
    x, y = fake_mnist()
    grad = Byz_random_label(x, y, m)
    assert len(grad) == 8

def test_mnist_byzantine_strategy_nonfaulty():
    from Byzantine_strategy import nonfaulty
    from CNN_model import CNN
    m = CNN()
    x, y = fake_mnist()
    grad = nonfaulty(x, y, m)
    assert len(grad) == 8

# ─────────────────────────────────────────────
# BRIDGE-CIFAR tests
# ─────────────────────────────────────────────
sys.path.remove(os.path.join(os.path.dirname(__file__), "BRIDGE-MNIST"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "BRIDGE-CIFAR"))

# Also remove any previously-imported MNIST modules so CIFAR ones are fresh
for mod in list(sys.modules.keys()):
    if mod in ("linear_classifier", "CNN_model", "DecLearning", "Byzantine_strategy"):
        del sys.modules[mod]

def test_cifar_linear_classifier_forward():
    from linear_classifier import linear_classifier
    m = linear_classifier(stepsize=1e-2)
    x, y = fake_cifar()
    loss = m.compute_loss(x, y)
    assert loss.numpy() > 0

def test_cifar_linear_classifier_accuracy():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    x, y = fake_cifar()
    acc = m.accuracy_eval(x, y)
    assert 0.0 <= acc <= 1.0

def test_cifar_linear_classifier_train_step():
    from linear_classifier import linear_classifier
    m = linear_classifier(stepsize=1e-2)
    x, y = fake_cifar()
    loss_before = m.compute_loss(x, y).numpy()
    for _ in range(5):
        m.train_step_fn(x, y, 1e-2)
    loss_after = m.compute_loss(x, y).numpy()
    assert loss_after < loss_before

def test_cifar_linear_classifier_weights_assign():
    from linear_classifier import linear_classifier
    m = linear_classifier()
    w = m.weights()
    assert w[0].shape == (3072, 10)
    m.assign([np.zeros((3072, 10)), np.zeros(10)])
    w2 = m.weights()
    assert np.allclose(w2[0], 0)

def test_cifar_cnn_forward():
    from CNN_model import CNN
    m = CNN(stepsize=1e-4)
    x, y = fake_cifar()
    loss = m.compute_loss(x, y)
    assert loss.numpy() > 0

def test_cifar_cnn_accuracy():
    from CNN_model import CNN
    m = CNN()
    x, y = fake_cifar()
    acc = m.accuracy_eval(x, y)
    assert 0.0 <= acc <= 1.0

def test_cifar_cnn_train_step():
    from CNN_model import CNN
    m = CNN(stepsize=1e-4)
    x, y = fake_cifar()
    loss_before = m.compute_loss(x, y).numpy()
    for _ in range(3):
        m.train_step_fn(x, y)
    loss_after = m.compute_loss(x, y).numpy()
    assert loss_after < loss_before

def test_cifar_cnn_weights_assign():
    from CNN_model import CNN
    m = CNN()
    w = m.weights()
    assert len(w) == 8
    m2 = CNN()
    m2.assign(w)
    for a, b in zip(w, m2.weights()):
        assert np.allclose(a, b)

def test_cifar_cnn_get_gradient():
    from CNN_model import CNN
    m = CNN()
    x, y = fake_cifar()
    grads = m.get_gradient(x, y)
    assert len(grads) == 8

def test_cifar_declearning_communication_dgd():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 4
    d = DecLearning(nodes=n)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=0, screenMethod=None)

def test_cifar_declearning_communication_bridge():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    n = 6
    b = 1
    d = DecLearning(nodes=n, byzantine=b)
    d.gen_graph(con_rate=1)
    nb = d.get_neighbor()
    nodes = [linear_classifier() for _ in range(n)]
    d.communication(nodes, nb, b=b, screenMethod='BRIDGE')

def test_cifar_declearning_node_update():
    from DecLearning import DecLearning
    from linear_classifier import linear_classifier
    import types
    n = 3
    d = DecLearning(nodes=n)
    nodes = [linear_classifier(stepsize=1e-2) for _ in range(n)]
    x, y = fake_cifar(n * 10)
    data = types.SimpleNamespace(
        dist_data=[x[i*10:(i+1)*10] for i in range(n)],
        dist_label=[y[i*10:(i+1)*10] for i in range(n)],
    )
    d.node_update(nodes, data, stepsize=1e-2)

def test_cifar_byzantine_strategy_random():
    from Byzantine_strategy import Byz_random
    from CNN_model import CNN
    m = CNN()
    x, y = fake_cifar()
    grad = Byz_random(x, y, m)
    assert len(grad) == 8
    assert grad[0].shape == (5, 5, 3, 32)

def test_cifar_byzantine_strategy_nonfaulty():
    from Byzantine_strategy import nonfaulty
    from CNN_model import CNN
    m = CNN()
    x, y = fake_cifar()
    grad = nonfaulty(x, y, m)
    assert len(grad) == 8

# ─────────────────────────────────────────────
# Run all tests
# ─────────────────────────────────────────────
print("\n" + "="*60)
print("BRIDGE TF2 Migration — Smoke Tests")
print("="*60 + "\n")

print("[BRIDGE-MNIST]")
run("mnist linear_classifier forward pass",       test_mnist_linear_classifier_forward)
run("mnist linear_classifier accuracy",           test_mnist_linear_classifier_accuracy)
run("mnist linear_classifier train_step",         test_mnist_linear_classifier_train_step)
run("mnist linear_classifier weights/assign",     test_mnist_linear_classifier_weights_assign)
run("mnist linear_classifier get_gradient_w",     test_mnist_linear_classifier_gradient_w)
run("mnist linear_classifier get_gradient_b",     test_mnist_linear_classifier_gradient_b)
run("mnist linear_classifier apply_gradient_w",   test_mnist_linear_classifier_apply_gradient_w)
run("mnist linear_classifier apply_gradient_b",   test_mnist_linear_classifier_apply_gradient_b)
run("mnist CNN forward pass",                     test_mnist_cnn_forward)
run("mnist CNN accuracy",                         test_mnist_cnn_accuracy)
run("mnist CNN train_step",                       test_mnist_cnn_train_step)
run("mnist CNN weights/assign",                   test_mnist_cnn_weights_assign)
run("mnist CNN get_gradient",                     test_mnist_cnn_get_gradient)
run("mnist CNN apply_ext_gradient",               test_mnist_cnn_apply_ext_gradient)
run("mnist DecLearning gen_graph",                test_mnist_declearning_gen_graph)
run("mnist DecLearning get_neighbor",             test_mnist_declearning_get_neighbor)
run("mnist DecLearning acc_test",                 test_mnist_declearning_acc_test)
run("mnist DecLearning communication DGD",        test_mnist_declearning_communication_dgd)
run("mnist DecLearning communication BRIDGE",     test_mnist_declearning_communication_bridge)
run("mnist DecLearning communication Median",     test_mnist_declearning_communication_median)
run("mnist DecLearning communication Krum",       test_mnist_declearning_communication_krum)
run("mnist DecLearning communication Byzantine",  test_mnist_declearning_communication_byzantine)
run("mnist DecLearning node_update",              test_mnist_declearning_node_update)
run("mnist DecLearning communication_w",          test_mnist_declearning_communication_w)
run("mnist DecLearning communication_b",          test_mnist_declearning_communication_b)
run("mnist DecLearning node_update_w",            test_mnist_declearning_node_update_w)
run("mnist DecLearning node_update_b",            test_mnist_declearning_node_update_b)
run("mnist Byzantine random",                     test_mnist_byzantine_strategy_random)
run("mnist Byzantine flip",                       test_mnist_byzantine_strategy_flip)
run("mnist Byzantine random_label",               test_mnist_byzantine_strategy_random_label)
run("mnist Byzantine nonfaulty",                  test_mnist_byzantine_strategy_nonfaulty)

print("\n[BRIDGE-CIFAR]")
run("cifar linear_classifier forward pass",       test_cifar_linear_classifier_forward)
run("cifar linear_classifier accuracy",           test_cifar_linear_classifier_accuracy)
run("cifar linear_classifier train_step",         test_cifar_linear_classifier_train_step)
run("cifar linear_classifier weights/assign",     test_cifar_linear_classifier_weights_assign)
run("cifar CNN forward pass",                     test_cifar_cnn_forward)
run("cifar CNN accuracy",                         test_cifar_cnn_accuracy)
run("cifar CNN train_step",                       test_cifar_cnn_train_step)
run("cifar CNN weights/assign",                   test_cifar_cnn_weights_assign)
run("cifar CNN get_gradient",                     test_cifar_cnn_get_gradient)
run("cifar DecLearning communication DGD",        test_cifar_declearning_communication_dgd)
run("cifar DecLearning communication BRIDGE",     test_cifar_declearning_communication_bridge)
run("cifar DecLearning node_update",              test_cifar_declearning_node_update)
run("cifar Byzantine random",                     test_cifar_byzantine_strategy_random)
run("cifar Byzantine nonfaulty",                  test_cifar_byzantine_strategy_nonfaulty)

print("\n" + "="*60)
print(f"Results: {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    print("\nFailed tests:")
    for name, err in FAIL:
        print(f"  - {name}: {err}")
print("="*60)
sys.exit(1 if FAIL else 0)
