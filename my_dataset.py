import json
import cv2
import numpy as np

from torch.utils.data import Dataset

from utils import drop_elements


def dropout_caption(
    caption: str,
    caption_dropout_rate: float = 0,
    caption_tag_dropout_rate: float = 0,
    rate_to_caption_tag_dropout_rate: float = 0,
    keep_tags: int = 0,
):
    # タグに分ける
    tags = caption.split(", ")
    header_tags = tags[:keep_tags]
    tags = tags[keep_tags:]

    # タグの一部を一定の確率で消失させる
    if np.random.rand() < rate_to_caption_tag_dropout_rate:
        tags = drop_elements(tags, caption_tag_dropout_rate)

    # タグをシャッフル
    np.random.shuffle(tags)
    caption = ", ".join(header_tags + tags)

    # キャプションを消失させる
    if np.random.rand() < caption_dropout_rate:
        caption = ""

    return caption


class MyDataset(Dataset):
    def __init__(
        self,
        caption_dropout_rate: float = 0,
        caption_tag_dropout_rate: float = 0,
        rate_to_caption_tag_dropout_rate: float = 0,
        keep_tags: int = 0,
    ):
        self.data = []
        with open("./training/fill50k/prompt.json", "rt") as f:
            for line in f:
                self.data.append(json.loads(line))

        self.caption_dropout_rate = caption_dropout_rate
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.rate_to_caption_tag_dropout_rate = rate_to_caption_tag_dropout_rate
        self.keep_tags = keep_tags

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item["source"]
        target_filename = item["target"]
        caption = item["prompt"]

        source = cv2.imread("./training/fill50k/" + source_filename)
        target = cv2.imread("./training/fill50k/" + target_filename)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        # 消失
        caption = dropout_caption(
            caption,
            self.caption_dropout_rate,
            self.caption_tag_dropout_rate,
            self.rate_to_caption_tag_dropout_rate,
            self.keep_tags,
        )

        return {"target": target, "caption": caption, "hint": source}


def make_train_dataset(
    caption_dropout_rate: float,
    caption_tag_dropout_rate: float,
    rate_to_caption_tag_dropout_rate: float,
    keep_tags: int,
    accelerator,
):
    with accelerator.main_process_first():
        # Set the training transforms
        train_dataset = MyDataset(
            caption_dropout_rate,
            caption_tag_dropout_rate,
            rate_to_caption_tag_dropout_rate,
            keep_tags,
        )

    return train_dataset
