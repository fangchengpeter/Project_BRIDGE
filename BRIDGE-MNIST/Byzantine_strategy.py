import numpy as np


def Byz_random(data, label, model):
    grad = model.get_gradient(
        np.array(data, dtype=np.float32),
        np.array(label, dtype=np.float32),
        training=True
    )
    return [np.random.random(g.shape) * 50 for g in grad]


def Byz_flip(data, label, model):
    data = (-1) * np.array(data, dtype=np.float32)
    return model.get_gradient(data, np.array(label, dtype=np.float32), training=True)


def Byz_random_label(data, label, model):
    label = np.array(label, dtype=np.float32)
    np.random.shuffle(label)
    return model.get_gradient(np.array(data, dtype=np.float32), label, training=True)


def nonfaulty(data, label, model):
    return model.get_gradient(
        np.array(data, dtype=np.float32),
        np.array(label, dtype=np.float32),
        training=True
    )


Byzantine_strategy = {
    'random': Byz_random,
    'flip': Byz_flip,
    'random_label': Byz_random_label,
    'nonfaulty': nonfaulty,
}
