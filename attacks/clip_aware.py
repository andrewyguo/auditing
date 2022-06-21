import math

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class ClipAwareAttack():
    @staticmethod
    def make_attack(train_x, train_y, args):
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
        
        np.set_printoptions(threshold=np.inf)
        print("train_x[-1]", train_x[-1])
        print("pca.components_[-1]", pca.components_[-1])
        print("args.l2_norm_clip", args.l2_norm_clip)

        new_x = args.l2_norm_clip * pca.components_[-1]
        train_x_sample = train_x[-1]

        lr = LogisticRegression(max_iter=1000)
        lr.fit(flat_x, np.argmax(train_y, axis=1))

        num_classes = train_y.shape[1]
        lr_probs = lr.predict_proba(new_x[None, :])
        print("lr_probs", lr_probs)

        min_y = np.argmin(lr_probs)
        second_y = np.argmin(lr_probs + np.eye(num_classes)[min_y])

        oh_min_y = np.expand_dims(np.eye(num_classes)[min_y], axis=0)
        oh_second_y = np.expand_dims(np.eye(num_classes)[second_y], axis=0)

        print("oh_min_y", oh_min_y)
        print("oh_second_y", oh_second_y)
        
        if args.debug:
            to_square = lambda x: np.reshape(x, (int(math.sqrt(x.shape[0])), int(math.sqrt(x.shape[0]))), order='F')

            print("new_x", new_x + 0.5)

            new_image = ((to_square(np.copy(new_x)) + 0.5) * 255).astype(np.uint8)
            old_image = ((to_square(np.copy(train_x_sample)) + 0.5) * 255).astype(np.uint8)

            print("old_image", old_image)
            new_img = Image.fromarray(new_image, mode="L")
            new_img.save(fp="output/clip_aware_poisoned.png")

            old_img = Image.fromarray(old_image, mode="L")
            old_img.save(fp="output/clip_aware.png")

        return to_image(new_x), oh_min_y, oh_second_y

    def membership_test(model, pois_sample_x, sample_y, args):
        """Membership inference - detect poisoning."""

        with torch.no_grad():
            probs = model(torch.Tensor(pois_sample_x))
            # Convert output to one-hot encoding 
            probs = [probs.item(), 1 - probs.item()]

            product = np.multiply(probs, sample_y)
            score = product.sum()

        if args.debug:
            print("Membership Test (probs):", probs)
            print("Membership Test (sample_y):", sample_y)

        return score
