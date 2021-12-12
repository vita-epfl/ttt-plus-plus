import torch
import numpy as np
from sklearn import datasets


def generate_moons(n_samples=100, seperation=0.0, shuffle=True, noise=None):
    """Make two interleaving half circles.
    -------
    X : ndarray of shape (n_samples, 2)
        The generated samples.
    y : ndarray of shape (n_samples,)
        The integer labels (0 or 1) for class membership of each sample.
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out)) + seperation / 2.0
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 1.0 - seperation / 2.0

    X = np.vstack([np.append(outer_circ_x, inner_circ_x),
                   np.append(outer_circ_y, inner_circ_y)]).T
    y = np.hstack([np.zeros(n_samples_out, dtype=np.intp),
                   np.ones(n_samples_in, dtype=np.intp)])

    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)

    return X, y


def rotationMat(rotTheta):
    return np.array([[np.cos(rotTheta), -np.sin(rotTheta)],
                     [np.sin(rotTheta), np.cos(rotTheta)]])


def sample(rot=70.0, tran=1.0, seperation=-0.1, nsample=500, noise=0.05):

    # source
    dx = tran
    dy = -1.0

    X, y_s = generate_moons(nsample, seperation=seperation, noise=noise)
    X = X - [0.5, 0.0]

    a_s = X[:, 1] < 0.0

    # -----------

    theta = -15.0

    rotMat = rotationMat(np.deg2rad(theta))
    X, y_s = generate_moons(nsample, seperation=seperation, noise=noise)
    X = X - [0.5, 0.0]
    X_s = np.dot(X, rotMat.T)

    X_s += [dx, dy]

    # -----------

    input_s = torch.from_numpy(X_s).float()
    label_s = torch.from_numpy(y_s).float()
    pretext_s = torch.from_numpy(a_s).float()

    # target

    dx = -tran
    dy += tran

    theta += rot

    rotMat = rotationMat(np.deg2rad(theta))

    y_t, a_t = y_s.copy(), a_s.copy()
    X_t = np.dot(X, rotMat.T)

    X_t += [dx, dy]

    input_t = torch.from_numpy(X_t).float()
    label_t = torch.from_numpy(y_t).float()
    pretext_t = torch.from_numpy(a_t).float()

    corr = (label_s == pretext_s).sum() / label_s.size(0)
    print("Label agreement between the main and ssl tasks: {:.2f}".format(corr))

    return (input_s, label_s, pretext_s), (input_t, label_t, pretext_t), corr
