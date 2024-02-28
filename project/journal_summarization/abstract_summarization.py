#!/usr/bin/env python3
'''
Implement Hugging Face abstractive summarization.
Fine-Tune on journal abstracts --> titles.

Authors: Kate Habich, Chanteria Milner
Adapted from HuggingFace Abstractive Summarization Tutorial with TensorFlow.
https://huggingface.co/docs/transformers/en/tasks/summarization
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb#scrollTo=kTCFado4IrIc

'''
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, create_optimizer, AdamWeightDecay, TFAutoModelForSeq2SeqLM
from functools import partial
import evaluate
import numpy as np
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
from transformers.utils import send_example_telemetry

send_example_telemetry("summarization_notebook", framework="tensorflow")



def preprocess_data(dataset):
    '''
    Create train-test split of data and tokenize.
    '''
    data = load_dataset(dataset, split="ca_test")
    data = data.train_test_split(test_size=0.2)

    # Tokenize text
    # TODO: change checkpoint to 'large' or 'base' version eventually
    checkpoint = "google-t5/t5-small"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint) 

    # def tokenize_with_tokenizer(examples):
    #     return prefix_text(tokenizer, examples)

    tokenized_data = data.map(partial(prefix_text, 
                                      tokenizer = tokenizer), 
                              batched=True)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=checkpoint, return_tensors="tf")
    rouge = evaluate.load("rouge")

    optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
    model = TFAutoModelForSeq2SeqLM.from_pretrained(checkpoint)

    tf_train_set = model.prepare_tf_dataset(
    tokenized_data["train"],
    shuffle=True,
    batch_size=16,
    collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
    tokenized_data["test"],
    shuffle=False,
    batch_size=16,
    collate_fn=data_collator,
    )  

    model.compile(optimizer=optimizer)
    metric_callback = KerasMetricCallback(metric_fn=partial(compute_metrics,
                                                        tokenizer = tokenizer,
                                                        rouge = rouge),
                                           eval_dataset=tf_train_set)

    model.fit(x=tf_train_set, 
              validation_data=tf_test_set, 
              epochs=3, 
              callbacks=[metric_callback])

    return model


def prefix_text(examples, tokenizer):
    prefix = "summarize: "
    inputs = [prefix + doc for doc in examples["text"]]
    model_inputs = tokenizer(inputs, 
                             max_length=1024, 
                             truncation=True)

    labels = tokenizer(text_target=examples["summary"], 
                       max_length=128, 
                       truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred, tokenizer, rouge):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}



print(preprocess_data("billsum"))
