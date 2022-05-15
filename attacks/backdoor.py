import numpy as np

class BackDoorAttack():
    @staticmethod
    def make_attack(train_x, train_y, **kwargs):
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
        
        # set corner feature to 1
        pois_x[0] = 1  

        # y_c contains original label 
        y_c = train_y[sample_ind]

        # y_p contains poisoned (i.e. flipped) label
        num_classes = train_y.shape[1]
        y_p = np.eye(num_classes)[y_c.argmax(1) + 1]

        return pois_x, y_p, y_c