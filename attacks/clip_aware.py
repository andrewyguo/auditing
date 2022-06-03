import numpy as np
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

class ClipAwareAttack():
    @staticmethod
    def make_attack(train_x, train_y, l2_norm=10):
        """
        train_x: clean training features - must be shape (n_samples, n_features)
        train_y: clean training labels - must be shape (n_samples, )

        Returns x, y1, y2
        x: poisoning sample
        y1: first corresponding y value
        y2: second corresponding y value
        """
        x_shape = list(train_x.shape[1:])
        to_image = lambda x: x.reshape([-1] + x_shape)  # reshapes to standard image shape
        flatten = lambda x: x.reshape((x.shape[0], -1))  # flattens all pixels - allows PCA

        # make sure to_image and flatten are inverse functions
        assert np.allclose(to_image(flatten(train_x)), train_x)

        flat_x = flatten(train_x)
        pca = PCA(flat_x.shape[1])
        pca.fit(flat_x)

        new_x = l2_norm*pca.components_[-1]

        lr = LogisticRegression(max_iter=1000)
        lr.fit(flat_x, np.argmax(train_y, axis=1))

        num_classes = train_y.shape[1]
        lr_probs = lr.predict_proba(new_x[None, :])
        min_y = np.argmin(lr_probs)
        second_y = np.argmin(lr_probs + np.eye(num_classes)[min_y])

        oh_min_y = np.eye(num_classes)[min_y]
        oh_second_y = np.eye(num_classes)[second_y]

        return to_image(new_x), oh_min_y, oh_second_y

    def membership_test(model, pois_sample_x, pois_sample_y, args):
        """Membership inference - detect poisoning."""

        with torch.no_grad():
            input = np.concatenate([pois_sample_x, np.zeros_like(pois_sample_x)])

            probs = model(torch.Tensor(input))
            probs = probs.numpy() 

            score = np.multiply(probs[0, :] - probs[1, :], pois_sample_y).sum()

        if args.debug:
            print("probs:", probs)
            print("pois_sample_y:", pois_sample_y)

        return score