import math

import numpy as np
import torch
from PIL import Image


class BackDoorAttack():
    @staticmethod
    def make_attack(train_x, train_y, args):
        """
        Makes a backdoored dataset, following Gu et al. https://arxiv.org/abs/1708.06733

        train_x: clean training features - must be shape (n_samples, n_features)
        train_y: clean training labels - must be shape (n_samples, ) - should be one hot encoding

        Returns pois_x, y1, y2
        pois_x: poisoning sample
        y_p: poisoned y value
        y_c: correct corresponding y value
        """

        sample_ind = np.random.choice(train_x.shape[0], 1)
        pois_x = np.copy(train_x[sample_ind, :])
        
        # set corner feature to .5
        pois_x[0][0] = .5

        # y_c contains original label 
        y_c = train_y[sample_ind]

        # y_p contains poisoned (i.e. flipped) label
        num_classes = train_y.shape[1]

        y_p = np.eye(num_classes)[y_c.argmax(1) - 1]

        print("y_c: ", y_c)
        print("y_p", y_p)
        
        if args.debug:
            to_square = lambda x: np.reshape(x, (int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))), order='F')

            new_image = ((to_square(np.copy(pois_x)) + 0.5) * 255).astype(np.uint8)
            old_image = ((to_square(np.copy(train_x[sample_ind, :])) + 0.5) * 255).astype(np.uint8)

            new_img = Image.fromarray(new_image, mode="L")
            new_img.save(fp="output/backdoor_poisoned.png")

            old_img = Image.fromarray(old_image, mode="L")
            old_img.save(fp="output/backdoor.png")

        return pois_x, y_p, y_c

    def membership_test(model, pois_sample_x, sample_y, args):
        """Membership inference - detect poisoning."""

        with torch.no_grad():
            probs = model(torch.Tensor(pois_sample_x))
            # Convert output to one-hot encoding 
            probs = [1 - probs.item(), probs.item()]

            product = np.multiply(probs, sample_y)
            score = product.sum() * -1 

        if args.debug:
            print("Membership Test (probs):", probs)
            print("Membership Test (sample_y):", sample_y)

        return score
