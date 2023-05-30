from my_dataset import MyDataset
from train_utils import start_train, parse_args

dataset = MyDataset(
    caption_dropout_rate=0.05,  # 5% の確率ですべてのキャプションが消失する
    caption_tag_dropout_rate=0.25,
    rate_to_caption_tag_dropout_rate=0.25,  # 25% の確率で 25% のタグが消失する
    keep_tags=0,
)


class Args:
    pretrained_model_name_or_path = "Birchlabs/wd-1-5-beta3-unofficial"
    controlnet_model_name_or_path = None  # ""
    revision = "main"
    resume_from_checkpoint = None  # "" # 再開するチェックポイントのパス

    tokenizer_name = None  # ""
    tokenizer_config_name = None  # ""

    output_dir = "./output"
    cache_dir = "D:/tmp"

    seed = None  # 3407

    resolution = 512

    train_batch_size = 4
    num_train_epochs = 100
    max_train_steps = None  # 10000

    checkpointing_steps = 5000
    checkpoints_total_limit = 5

    gradient_accumulation_steps = 4
    gradient_checkpointing = True

    learning_rate = 1e-5
    scale_lr = None  # False
    lr_scheduler = (
        "constant"  # linear, cosine, constant, constant_with_warmup, polynomial
    )
    lr_warmup_steps = None  # 500
    lr_num_cycles = None  # 1
    lr_power = None  # 1.0
    optimizer = "AdamW8bit"  # AdamW, AdamW8bit
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8
    max_grad_norm = 1.0

    dataloader_num_workers = 0

    allow_tf32 = True
    mixed_precision = "fp16"
    enable_xformers_memory_efficient_attention = True
    set_grads_to_none = None  # True

    push_to_hub = False
    hub_token = ""
    hub_model_id = ""

    logging_dir = "./logs"
    report_to = "wandb"  # wandb, tensorboard, none
    tracker_project_name = "controlnet-diffusers-circle"

    max_train_samples = None
    validation_prompt = [
        "pale golden rod circle with old lace background",
        "cornflower blue circle with light golden rod yellow background",
        "dark orange circle with gray background",
        "silver circle with rosy brown background",
    ]
    validation_image = [
        "./sample/110.png",
        "./sample/118.png",
        "./sample/132.png",
        "./sample/144.png",
    ]
    num_validation_images = 4
    validation_steps = 1000


args = Args()

start_train(args, dataset)
