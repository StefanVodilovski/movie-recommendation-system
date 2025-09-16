import numpy as np


def extend_encoder(encoder, new_labels):
    existing = set(encoder.classes_)
    to_add = [l for l in new_labels if l not in existing]
    if to_add:
        encoder.classes_ = np.concatenate([encoder.classes_, to_add])
    return encoder
