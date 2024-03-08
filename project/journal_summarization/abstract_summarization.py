#!/usr/bin/env python3
'''
Implement Hugging Face abstractive summarization.
Fine-Tune on journal abstracts --> titles.

Authors: Kate Habich
Adapted from HuggingFace Abstractive Summarization Tutorial with TensorFlow.
https://huggingface.co/docs/transformers/en/tasks/summarization
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/summarization-tf.ipynb#scrollTo=kTCFado4IrIc

'''
from datasets import load_dataset, load_metric, DatasetDict
import transformers
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, create_optimizer, AdamWeightDecay, TFAutoModelForSeq2SeqLM
from functools import partial
import evaluate
import numpy as np
import tensorflow as tf
from transformers.keras_callbacks import KerasMetricCallback
from transformers.utils import send_example_telemetry
from huggingface_hub import notebook_login
from transformers import T5ForConditionalGeneration
import nltk
from project.journal_summarization.journal_preprocess_abstractive_summary import preprocess_journal_data

# notebook_login()
send_example_telemetry("summarization_notebook", framework="tensorflow")


class AbstractSummarizer:
    def __init__(self, training_data, model_save_path, read_from_external = True):
        self.data = training_data
        self.read_from_external = read_from_external
        self.model_save_path = model_save_path
        self.model = self._fine_tune(self.data)

    def _fine_tune(self, data):
        '''
        Create train-test split of data and tokenize.
        '''
        nltk.download('punkt')
        if self.read_from_external:
            # print(f"Model Save Path: {self.model_save_path}")
            model = TFAutoModelForSeq2SeqLM.from_pretrained(self.model_save_path)
        else:
            train_testvalid = self.data.train_test_split(test_size=0.1)
            test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
            raw_datasets = DatasetDict({
                'train': train_testvalid['train'],
                'test': test_valid['test'],
                'validation': test_valid['train']})

            metric = load_metric("rouge")

            model_checkpoint = "t5-small"
            tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

            prefix = "summarize: "
            tokenized_datasets = raw_datasets.map(partial(self._preprocess_function,
                                                            prefix = prefix,
                                                        tokenizer = tokenizer),
                                                    batched=True)

            # push to hub (comment out if you want)
            # model_name = checkpoint.split("/")[-1]
            # push_to_hub_model_id = f"{model_name}-finetuned-climate-journals"

            model = TFAutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
            batch_size = 8
            learning_rate = 2e-5
            weight_decay = 0.01
            num_train_epochs = 1
            data_collator = DataCollatorForSeq2Seq(tokenizer,
                                                    model,
                                                    return_tensors="np")
            generation_data_collator = DataCollatorForSeq2Seq(tokenizer,
                                                            model=model,
                                                            return_tensors="np",
                                                            pad_to_multiple_of=128)

            tf_train_set = model.prepare_tf_dataset(
                    tokenized_datasets["train"],
                    shuffle=True,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    )
            
            tf_valid_set = model.prepare_tf_dataset(
                    tokenized_datasets["validation"],
                    shuffle=False,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    )
            generation_dataset = model.prepare_tf_dataset(
                    # subset["validation"],
                    tokenized_datasets["validation"],
                    batch_size=8,
                    shuffle=False,
                    collate_fn=generation_data_collator
                    )
            
            optimizer = AdamWeightDecay(learning_rate,
                                      weight_decay)

            model.compile(optimizer=optimizer)

            metric_callback = KerasMetricCallback(partial(self._metric_fn,
                                                            tokenizer = tokenizer,
                                                            metric = metric),
                                                    eval_dataset=generation_dataset,
                                                    predict_with_generate=True,
                                                    use_xla_generation=True)

            model.fit(x=tf_train_set,
                    validation_data=tf_valid_set,
                    epochs=1,
                    callbacks=[metric_callback])

            model.save_pretrained(model_save_path)

        return model
    

    def _preprocess_function(self, examples, prefix, tokenizer):
        '''
        Prefixes
        '''
        max_input_length = 1920
        max_target_length = 128
        inputs = [prefix + doc for doc in examples["document"]]
        model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["summary"], max_length=max_target_length, truncation=True
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def _metric_fn(self, eval_predictions, tokenizer, metric):
        predictions, labels = eval_predictions
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        for label in labels:
            label[label < 0] = tokenizer.pad_token_id  # Replace masked label tokens
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # Rouge expects a newline after each sentence
        decoded_predictions = [
            "\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_predictions
        ]
        decoded_labels = [
            "\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels
        ]
        result = metric.compute(
            predictions=decoded_predictions, references=decoded_labels, use_stemmer=True
        )
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        # Add mean generated length
        prediction_lens = [
            np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions
        ]
        result["gen_len"] = np.mean(prediction_lens)

        return result
        

    def summarize(self, document):
      '''
      Summarizes string using model output
      '''
      model_checkpoint = "t5-small"
      tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
      prompt = "summarize: " + document
      tokenized = tokenizer([prompt], return_tensors='np')
      out = self.model.generate(**tokenized, max_length=128)

      with tokenizer.as_target_tokenizer():
          summary = tokenizer.decode(out[0])

      return summary


# subset = preprocess_journal_data().train_test_split(train_size=20)['train']
# full_data = preprocess_journal_data()
# model_save_path = '/content/drive/My Drive/UChicago/Machine_Learning/models/abstract_model/'
    

# ============ How to Call ============
print("In Summary class")
# model_save_path = '/content/drive/My Drive/UChicago/Machine_Learning/models/abstract_model/'
summarizer = AbstractSummarizer('hi',
                                './project/data/models/small_weights/',
                                True)

document = 'The full cost of damage in Newton Stewart, one of the areas worst affected, is still being assessed.\nRepair work is ongoing in Hawick and many roads in Peeblesshire remain badly affected by standing water.\nTrains on the west coast mainline face disruption due to damage at the Lamington Viaduct.\nMany businesses and householders were affected by flooding in Newton Stewart after the River Cree overflowed into the town.\nFirst Minister Nicola Sturgeon visited the area to inspect the damage.\nThe waters breached a retaining wall, flooding many commercial properties on Victoria Street - the main shopping thoroughfare.\nJeanette Tate, who owns the Cinnamon Cafe which was badly affected, said she could not fault the multi-agency response once the flood hit.\nHowever, she said more preventative work could have been carried out to ensure the retaining wall did not fail.\n"It is difficult but I do think there is so much publicity for Dumfries and the Nith - and I totally appreciate that - but it is almost like we\'re neglected or forgotten," she said.\n"That may not be true but it is perhaps my perspective over the last few days.\n"Why were you not ready to help us a bit more when the warning and the alarm alerts had gone out?"\nMeanwhile, a flood alert remains in place across the Borders because of the constant rain.\nPeebles was badly hit by problems, sparking calls to introduce more defences in the area.\nScottish Borders Council has put a list on its website of the roads worst affected and drivers have been urged not to ignore closure signs.\nThe Labour Party\'s deputy Scottish leader Alex Rowley was in Hawick on Monday to see the situation first hand.\nHe said it was important to get the flood protection plan right but backed calls to speed up the process.\n"I was quite taken aback by the amount of damage that has been done," he said.\n"Obviously it is heart-breaking for people who have been forced out of their homes and the impact on businesses."\nHe said it was important that "immediate steps" were taken to protect the areas most vulnerable and a clear timetable put in place for flood prevention plans.\nHave you been affected by flooding in Dumfries and Galloway or the Borders? Tell us about your experience of the situation and how it was handled.'
summary = summarizer.summarize(document)

print(summary)