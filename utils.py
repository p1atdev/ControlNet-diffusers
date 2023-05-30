from PIL import Image
import random
from typing import List, Any


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def drop_elements(arr: List[Any], rate: float = 0.25):
    num_elements_to_remove = int(len(arr) * rate)
    indices_to_remove = random.sample(range(len(arr)), num_elements_to_remove)
    arr = [arr[i] for i in range(len(arr)) if i not in indices_to_remove]

    return arr
