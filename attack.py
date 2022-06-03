from attacks.backdoor import BackDoorAttack
from attacks.clip_aware import ClipAwareAttack

attacks = {
    "clip_aware" : ClipAwareAttack.make_attack,
    "backdoor" : BackDoorAttack.make_attack
}

def poison_data(train_x, train_y, args):
    """
    Makes a dict containing many poisoned datasets. make_pois is fairly slow:
    this avoids making multiple calls

    train_x: clean training features - shape (n_samples, n_features)
    train_y: clean training labels - shape (n_samples, )
    args: command line arguments

    Returns dict: all_poisons
    all_poisons[poison_size] is a pair of poisoned datasets
    all_poisons["pois"] contains the original poisoned sample
    """

    pois_sample_x, pois_sample_y, unpois_sample_y = attacks[args.attack_type](train_x, train_y, args.l2_norm_clip)

    # Contains poisoned sample and single poisoned label 
    all_poisons = {"pois": (pois_sample_x, pois_sample_y)}
    
    # Creates datasets where pois_size amount of points are poisoned 
    # make_pois is slow - don't want it in a loop
    print("Generating poisoning...")
    
    for pois_size in args.pois_ct: 
        new_pois_x1, new_pois_y1 = train_x.copy(), train_y.copy()
        new_pois_x2, new_pois_y2 = train_x.copy(), train_y.copy()

        new_pois_x1[-pois_size:] = pois_sample_x[None, :]
        new_pois_y1[-pois_size:] = pois_sample_y

        new_pois_x2[-pois_size:] = pois_sample_x[None, :]
        new_pois_y2[-pois_size:] = unpois_sample_y

        dataset1, dataset2 = (new_pois_x1, new_pois_y1), (new_pois_x2, new_pois_y2)
        all_poisons[pois_size] = dataset1, dataset2

    return all_poisons
