import paddle

from args import parse_args
from data import get_dev_dataloader
from train import MODEL_CLASSES, evaluate


def main(args):
    paddle.set_device(args.device)
    model_class, tokenizer_class, args.need_token_type_ids = MODEL_CLASSES[
        args.model_type
    ]
    model = model_class.from_pretrained(args.model_name_or_path)
    if args.use_huggingface_tokenizer and args.model_type == "mpnet":
        from transformers import MPNetTokenizerFast

        tokenizer = MPNetTokenizerFast.from_pretrained("./")
    else:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    dev_data_loader = get_dev_dataloader(tokenizer, args)

    evaluate(model, dev_data_loader, args, output_dir="./")


if __name__ == "__main__":
    args = parse_args()
    main(args)
