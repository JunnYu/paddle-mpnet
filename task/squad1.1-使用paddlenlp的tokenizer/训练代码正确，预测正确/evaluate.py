

import json


from functools import partial

import paddle
from paddle.io import DataLoader
from args import parse_args
from paddlenlp.data import Pad,Dict
from paddlenlp.transformers import MPNetForQuestionAnswering, MPNetTokenizer
from paddlenlp.metrics.squad import squad_evaluate, compute_prediction
from paddlenlp.datasets import load_dataset



args = parse_args()

def prepare_validation_features(examples, tokenizer, args):
    # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
    # in one example possible giving several features when a context is long, each of those features having a
    # context that overlaps a bit the context of the previous feature.
    #NOTE: Almost the same functionality as HuggingFace's prepare_train_features function. The main difference is
    # that HugggingFace uses ArrowTable as basic data structure, while we use list of dictionary instead.
    contexts = [examples[i]['context'] for i in range(len(examples))]
    questions = [examples[i]['question'] for i in range(len(examples))]

    tokenized_examples = tokenizer(
        questions,
        contexts,
        stride=args.doc_stride,
        max_seq_len=args.max_seq_length)

    # For validation, there is no need to compute start and end positions
    for i, tokenized_example in enumerate(tokenized_examples):
        # Grab the sequence corresponding to that example (to know what is the context and what is the question).
        sequence_ids = tokenized_example['token_type_ids']

        # One example can give several spans, this is the index of the example containing this span of text.
        sample_index = tokenized_example['overflow_to_sample']
        tokenized_examples[i]["example_id"] = examples[sample_index]['id']

        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        tokenized_examples[i]["offset_mapping"] = [
            (o if sequence_ids[k] == 1 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"])
        ]

    return tokenized_examples

dev_ds = load_dataset("squad", splits="dev_v1")

tokenizer = MPNetTokenizer.from_pretrained(args.model_name_or_path)

dev_ds.map(
    partial(prepare_validation_features, tokenizer=tokenizer, args=args),
    batched=True,
    lazy=False,
)
dev_batch_sampler = paddle.io.BatchSampler(
    dev_ds, batch_size=args.batch_size, shuffle=False
)

dev_batchify_fn = lambda samples, fn=Dict(
    {
        "input_ids": Pad(axis=0, pad_val=tokenizer.pad_token_id),
        "token_type_ids": Pad(axis=0, pad_val=tokenizer.pad_token_type_id),
    }
): fn(samples)

dev_data_loader = DataLoader(
    dataset=dev_ds,
    batch_sampler=dev_batch_sampler,
    collate_fn=dev_batchify_fn,
    num_workers=0,
    return_list=True,
)

model = MPNetForQuestionAnswering.from_pretrained(args.model_name_or_path)
model.eval()

all_start_logits = []
all_end_logits = []

with paddle.no_grad():
    for batch in dev_data_loader:
        input_ids, token_type_ids = batch
        start_logits_tensor, end_logits_tensor = model(input_ids)

        for idx in range(start_logits_tensor.shape[0]):
            all_start_logits.append(start_logits_tensor.numpy()[idx])
            all_end_logits.append(end_logits_tensor.numpy()[idx])

    all_predictions, all_nbest_json, scores_diff_json = compute_prediction(
        dev_data_loader.dataset.data,
        dev_data_loader.dataset,
        (all_start_logits, all_end_logits),
        args.version_2_with_negative,
        args.n_best_size,
        args.max_answer_length,
        args.null_score_diff_threshold,
    )

    # Can also write all_nbest_json and scores_diff_json files if needed
    with open(
        "prediction.json", "w", encoding="utf-8"
    ) as writer:
        writer.write(json.dumps(all_predictions, ensure_ascii=False, indent=4) + "\n")

    squad_evaluate(
        examples=dev_data_loader.dataset.data,
        preds=all_predictions,
        na_probs=scores_diff_json,
    )