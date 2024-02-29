"""
Implement SAPGraph model for article summarization.
Fine-Tune on journal articles --> abstracts.

Authors: Chanteria Milner, Kate Habich
Credit: https://github.com/cece00/SAPGraph/tree/main?tab=readme-ov-file
"""

import argparse
import glob

# Packages
import json
import os
import re

import nltk
import numpy as np
import pandas as pd
import spacy
import torch
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split

from external_code.SAPGraph.module.dataloader_all import (
    DatasetAll,
    graph_collate_fn,
)
from external_code.SAPGraph.module.embedding import Word_Embedding
from external_code.SAPGraph.module.SAPGAT import WSWGAT
from external_code.SAPGraph.module.vocabulary import Vocab
from external_code.SAPGraph.script.calEdge import main, main_train

# external packages
from external_code.SAPGraph.script.createVoc import catDoc, getEnt
from external_code.SAPGraph.script.lowTFIDFWords import calTFidf

# Project imports
from project.utils.constants import (
    CLEANED_DATA_PATH,
    CLEANED_PROQUEST_FILE,
    PUNCTUATION_FILTER,
    SECTION_HEADERS_MAPPING,
)
from project.utils.functions import load_file_to_df, save_df_to_file


# variables pulled from SAPGraph article
SAMPLE_SIZE = 200
RANDOM_STATE = 123
VOCAB_EMBED_DIM = 300
VOCAB_MAX_SIZE = 0
SENT_ENT_NODE_EMBED_DIM = 128
EDGE_EMBED_DIM = 50
MAX_SENTENCE_NUM = 100
MAX_TOKEN_NUM = 50
MAX_ENTITY_NUM = 50
SIM_THRESHOLD = 0.55
BATCH_SIZE = 36
EPOCHS = 10
NUM_HEADS = 8
HIDDEN_LAYER_DIM = 64
HIDDEN_SIZE_FFN = 512
GLOVE_PATH = os.path.join("data", "models", "glove.42B.300d.txt")
NLP = spacy.load("en_core_sci_md")


class ArticleSummarizer:
    """
    Summarizes journal articles. Adapts the SAPGraph model for article
    summarization.

    arguments:
        file_path (str): path to the file to summarize.
        dev_environment (bool): whether or not the environment is a development
                                environment.
    """

    def __init__(
        self,
        file_path: str = CLEANED_PROQUEST_FILE,
        dev_environment: bool = False,
    ):
        self.dataset = "proquest"
        self.dev_environment = dev_environment

        # setting up file paths
        if not os.path.exists(os.path.join(CLEANED_DATA_PATH, self.dataset)):
            os.makedirs(os.path.join(CLEANED_DATA_PATH, self.dataset))

        self.dir = os.path.join(CLEANED_DATA_PATH, self.dataset)
        self.vocab_fp = os.path.join(self.dir, "filter_word.txt")
        self.train_data_path = os.path.join(
            self.dir, "proquest_sapgraph_train.jsonl"
        )
        self.test_data_path = os.path.join(
            self.dir, "proquest_sapgraph_test.jsonl"
        )
        self.valid_data_path = os.path.join(
            self.dir, "proquest_sapgraph_valid.jsonl"
        )

        # datasets
        self.df = load_file_to_df(file_path)
        if self.dev_environment:
            # shuffle the data
            self.df = self.df.sample(
                frac=1, random_state=RANDOM_STATE
            ).reset_index(drop=True)

            # get a random sample of the data
            self.df = self.df.sample(SAMPLE_SIZE)
            self.df.reset_index(drop=True, inplace=True)
        self.subset_df = self.df.loc[:, ["text", "abstract"]]
        self.train = None
        self.test = None
        self.valid = None

    def extract_sections(self, article_text: str) -> dict:
        """
        Splits the text into sections.

        Args:
            text (str): text to split into sections.

        Returns:
            dict: a mapping of article text to normalized section headers, where
                  the section headers expected by the SAPGraph model are:
                  Introduction
                  Method
                  Result
                  Conclusion
        """
        # create a regex pattern to split the text into sections
        pattern_headers = "|".join(
            re.escape(header) for header in SECTION_HEADERS_MAPPING.keys()
        )
        headers_regex = (
            rf"(?:\d+\s*\.?\s*)?({pattern_headers})(?::?\s+)(?=[\dA-Z])"
        )

        # split the text into sections
        sections = re.split(headers_regex, article_text)
        if len(sections[0].strip()) > 0:
            sections = ["Introduction"] + sections
        else:
            sections = sections[1:]
        sections = [
            (sections[i], sections[i + 1].strip())
            for i in range(0, len(sections), 2)
        ]

        # no sections found in article text
        if len(sections) == 0:
            return None

        # normalize section headers
        normalized_sections = {
            "Introduction": [],
            "Method": [],
            "Result": [],
            "Conclusion": [],
            "Other": [],
        }
        for header, text in sections:
            normalized_header = SECTION_HEADERS_MAPPING.get(header, "Other")
            normalized_sections[normalized_header].extend(sent_tokenize(text))

        return normalized_sections

    def extract_entities(self, list_of_sentences: list) -> list:
        """
        Extracts the entities from the list of sentences using SciSpacy.

        Args:
            list_of_sentences (list): list of sentences to extract entities
            from.

        Returns:
            list: list of entities.
        """
        entities = []
        for sentence in list_of_sentences:
            doc = NLP(sentence)
            ents = []
            for ent in doc.ents:
                ents.append([ent.start, ent.end, ent.text])
            entities.append(ents)
        return entities

    def extract_text_components(self, row: pd.Series) -> list:
        """
        Extracts the text components from the text, including the text
        (broken down into sections and sentences per section), the entities,
        and the entity types.

        parameters:
            row (pd.Series): row of the dataframe to extract text components
                            from.

        Returns:
            dict: dictionary of text components.

        For each article, the data should be in the following form for SAPGraph:
            section: [section, section, ...]
            text: [[sentence, sentence, ...], [sentence, sentence, ...], ...]
                - sentences within each section
            entity: [[[entity, entity, ...], [entity, entity, ...]],
                    [[entity, entity, ...], [entity, entity, ...]],, ...]
                - entities within each sentence
            summary: [sentence, sentence, ...]
                - the article abstract or summary
        """
        text = row["text"]
        summary = row["abstract"]

        # extract text sections
        sectioned_text = self.extract_sections(text)
        text = []
        section = []
        entities = []

        # extract entities, sections, and text
        if sectioned_text is not None:
            for sec, sentences in sectioned_text.items():
                section.append(sec)
                text.append(sentences)
                ents = self.extract_entities(sentences)
                entities.append(ents)
        else:
            return None

        return {
            "section_name": section,
            "text": text,
            "entity": entities,
            "summary": sent_tokenize(summary),  # "abstract" in the dataframe
        }

    def save_json(self) -> None:
        """
        Saves the data to a json file.

        Args:
            file_path (str): path to save the data to.
        """
        with open(self.train_data_path, "w", encoding="utf8") as f:
            for _, row in self.train.loc[:, ["text_entities"]].iterrows():
                entities = row["text_entities"]
                ent_json = json.dumps(entities)
                f.write(ent_json + "\n")

        with open(self.test_data_path, "w", encoding="utf8") as f:
            for _, row in self.test.loc[:, ["text_entities"]].iterrows():
                valid = row["text_entities"]
                valid_json = json.dumps(valid)
                f.write(valid_json + "\n")

        with open(self.valid_data_path, "w", encoding="utf8") as f:
            for _, row in self.valid.loc[:, ["text_entities"]].iterrows():
                valid = row["text_entities"]
                valid_json = json.dumps(valid)
                f.write(valid_json + "\n")

    def get_voc_entities(self) -> tuple:
        """
        Gets the texts, summaries, entites, words, and counts for the
        vocabulary.

        Returns:
            tuple: tuple of the texts, summaries, entities, words, and counts.
        """
        text = []
        summary = []
        entity = []
        allword = []
        cnt = 0

        # only need to vocab for the training data
        with open(self.train_data_path, encoding="utf8") as f:
            for line in f:
                e = json.loads(line)

                # concatenate the sentences and section names
                if isinstance(e["text"], list) and isinstance(
                    e["text"][0], list
                ):
                    sents = catDoc(e["text"])
                    secs = catDoc(e["section_name"])
                    sents.extend(secs)
                else:
                    pass

                # concatenate the text and summary
                text = " ".join(sents)
                summary = " ".join(e["summary"])
                allword.extend(text.split())
                allword.extend(summary.split())

                # concatenate the entities
                entity.extend(getEnt(e["entity"]))
                cnt += 1
                if cnt % 2000 == 0:
                    print(cnt)

        return entity, allword

    def run_create_voc(self) -> None:
        """
        Runs the createVoc.py file from the SAPGraph model.
        Note that this code is heavily adapted from the original SAPGraph code.

        Credits:
            https://github.com/cece00/SAPGraph/blob/main/script/createVoc.py
        """
        # set up directories
        saveFile = os.path.join(self.dir, "vocab")
        entFile = os.path.join(self.dir, "vocab_ent")

        entity, allword = self.get_voc_entities()

        fdist1 = nltk.FreqDist(allword)

        fout = open(saveFile, "w")
        keys = fdist1.most_common()
        for (
            key,
            val,
        ) in keys:  # key is the word while value is the frequency (times)
            try:
                fout.write("%s\t%d\n" % (key, val))
            except UnicodeEncodeError:
                continue
        fout.close()

        # write entities into file, to make entities in vocab
        fout_ent = open(entFile, "w")
        fdist2 = nltk.FreqDist(entity)
        keys2 = fdist2.most_common()
        k_left = 0
        for (
            key,
            val,
        ) in keys2:  # key is the word while value is the frequency (times)
            try:
                pass_sig = False
                for item in PUNCTUATION_FILTER:
                    if item in repr(key):
                        pass_sig = True
                        break
                if not pass_sig:
                    fout_ent.write("%s\t%s\t%d\n" % (key, key.lower(), val))
                    k_left += 1
            except UnicodeEncodeError:
                continue
        fout_ent.close()

    def run_low_tfidf_words(self) -> None:
        """
        Runs the lowTfidfWords.py file from the SAPGraph model.
        Note that this code is heavily adapted from the original SAPGraph code.

        Credits:
            https://github.com/cece00/SAPGraph/blob/main/script/lowTFIDFWords.py
        """
        documents = []
        with open(self.train_data_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                if isinstance(e["text"], list) and isinstance(
                    e["text"][0], list
                ):
                    text = catDoc(e["text"])
                else:
                    text = e["text"]
                documents.append(" ".join(text))

        vectorizer, tfidf_matrix = calTFidf(documents)
        print(
            "The number of example is %d, and the TFIDF vocabulary size is %d"
            % (len(documents), len(vectorizer.vocabulary_))
        )
        word_tfidf = np.array(tfidf_matrix.mean(0))
        del tfidf_matrix
        word_order = np.argsort(word_tfidf[0])  # sort A->Z, return index

        id2word = vectorizer.get_feature_names_out()
        with open(self.vocab_fp, "w") as fout:
            for idx in word_order:
                w = id2word[idx]
                fout.write(w + "\n")

    def run_cal_edge(self) -> None:
        """
        Runs the calEdge.py file from the SAPGraph model.
        Note that this code is heavily adapted from the original SAPGraph code.

        Credits:
            https://github.com/cece00/SAPGraph/blob/main/script/calEdge.py
        """
        for file in [
            self.train_data_path,
            self.test_data_path,
            self.valid_data_path,
        ]:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--data_path",
                type=str,
                default=file,
                help="File to deal with",
            )
            parser.add_argument(
                "--dataset", type=str, default=self.dataset, help="dataset name"
            )
            args = parser.parse_args()

            if self.dev_environment:
                main_train(args)
            else:
                main(args)

    def concat_cal_edge(self) -> None:
        """
        Concatenates the cal-edge data produced by the multiple workers.
        """
        file_patterns = [
            "proquest_sapgraph_train.w2s.jsonl*",
            "proquest_sapgraph_train.s2s.jsonl*",
            "proquest_sapgraph_test.w2s.jsonl*",
            "proquest_sapgraph_test.s2s.jsonl*",
            "proquest_sapgraph_valid.w2s.jsonl*",
            "proquest_sapgraph_valid.s2s.jsonl*",
        ]

        for pattern in file_patterns:
            # concatenate the files
            files = glob.glob(os.path.join(self.dir, pattern))

            # export the files
            fname = pattern[:-1]

            with open(os.path.join(self.dir, fname), "w") as f:
                for file in files:
                    with open(file, "r") as f2:
                        f.write(f2.read())

    def preprocess(self) -> None:
        """
        Preprocesses the data for in preparation for the SAPGraph summarization.
        """
        print("Preprocessing data...")

        # grab only the rows where the text and abstract are not null
        print("\tSubsetting data...")
        self.subset_df.dropna(inplace=True)
        self.subset_df = self.subset_df.loc[
            (self.subset_df.loc[:, "text"].apply(len) > 0)
            & self.subset_df.loc[:, "abstract"].apply(len)
            > 0,
            :,
        ]

        self.subset_df.reset_index(drop=True, inplace=True)

        # extract entities
        print("\tGetting text entities...")
        self.subset_df["text_entities"] = self.subset_df.loc[
            :, ["text", "abstract"]
        ].apply(self.extract_text_components, axis=1)
        self.subset_df.dropna(inplace=True)

        # test-train-validate split
        print("\tTest-train-validate split...")
        train, test = train_test_split(
            self.subset_df, test_size=0.2, random_state=RANDOM_STATE
        )
        train, valid = train_test_split(
            train, test_size=0.2, random_state=RANDOM_STATE
        )
        self.train = train
        self.test = test
        self.valid = valid

        # save the data to a json file
        print("\tSaving data to json file...")
        self.save_json()

        # create the vocabulary
        print("\tCreating vocabulary...")
        self.run_create_voc()

        # run low tfidf words
        print("\tRunning low tfidf words...")
        self.run_low_tfidf_words()

        # run cal edge
        print("\tRunning cal edge...")
        self.run_cal_edge()

        # concatenate the data
        print("\tConcatenating cal-edge data...")
        self.concat_cal_edge()

    # FIXME: The following code is not complete and needs to be adapted to the SAPGraph model
    def save_model(model, save_file):
        """

        Credits:
            https://github.com/dqwang122/HeterSumGraph/blob/master/train.py#L43
        """
        with open(save_file, "wb") as f:
            torch.save(model.state_dict(), f)
        logger.info("[INFO] Saving model to %s", save_file)

    # FIXME: The following code is not complete and needs to be adapted to the SAPGraph model
    def run_training(
        self, model, train_loader, valid_loader, valset, hps, train_dir
    ):
        """
        Repeatedly runs training iterations, logging loss to screen and log files

        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model
        :param train_dir: where to save checkpoints

        Credits:
            https://github.com/dqwang122/HeterSumGraph/blob/master/train.py#L43
        """
        logger.info("[INFO] Starting run_training")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=hps.lr
        )

        criterion = torch.nn.CrossEntropyLoss(reduction="none")

        best_train_loss = None
        best_loss = None
        best_F = None
        non_descent_cnt = 0
        saveNo = 0

        for epoch in range(1, hps.n_epochs + 1):
            epoch_loss = 0.0
            train_loss = 0.0
            epoch_start_time = time.time()
            for i, (G, _index) in enumerate(train_loader):
                iter_start_time = time.time()
                # if i > 10:
                #     break
                model.train()

                if hps.cuda:
                    G.to(torch.device("cuda"))

                outputs = model.forward(G)  # [n_snodes, 2]
                snode_id = G.filter_nodes(
                    lambda nodes: nodes.data["dtype"] == 1
                )
                label = G.ndata["label"][snode_id].sum(-1)  # [n_nodes]
                G.nodes[snode_id].data["loss"] = criterion(
                    outputs, label
                ).unsqueeze(
                    -1
                )  # [n_nodes, 1]
                loss = dgl.sum_nodes(G, "loss")  # [batch_size, 1]
                loss = loss.mean()

                if not (np.isfinite(loss.data.cpu())).numpy():
                    logger.error("train Loss is not finite. Stopping.")
                    logger.info(loss)
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            logger.info(name)
                            # logger.info(param.grad.data.sum())
                    raise Exception("train Loss is not finite. Stopping.")

                optimizer.zero_grad()
                loss.backward()
                if hps.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), hps.max_grad_norm
                    )

                optimizer.step()

                train_loss += float(loss.data)
                epoch_loss += float(loss.data)

                if i % 100 == 0:
                    if _DEBUG_FLAG_:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                logger.debug(name)
                                logger.debug(param.grad.data.sum())
                    logger.info(
                        "       | end of iter {:3d} | time: {:5.2f}s | train loss {:5.4f} | ".format(
                            i,
                            (time.time() - iter_start_time),
                            float(train_loss / 100),
                        )
                    )
                    train_loss = 0.0

            if hps.lr_descent:
                new_lr = max(5e-6, hps.lr / (epoch + 1))
                for param_group in list(optimizer.param_groups):
                    param_group["lr"] = new_lr
                logger.info("[INFO] The learning rate now is %f", new_lr)

            epoch_avg_loss = epoch_loss / len(train_loader)
            logger.info(
                "   | end of epoch {:3d} | time: {:5.2f}s | epoch train loss {:5.4f} | ".format(
                    epoch,
                    (time.time() - epoch_start_time),
                    float(epoch_avg_loss),
                )
            )

            if not best_train_loss or epoch_avg_loss < best_train_loss:
                save_file = os.path.join(train_dir, "bestmodel")
                logger.info(
                    "[INFO] Found new best model with %.3f running_train_loss. Saving to %s",
                    float(epoch_avg_loss),
                    save_file,
                )
                save_model(model, save_file)
                best_train_loss = epoch_avg_loss
            elif epoch_avg_loss >= best_train_loss:
                logger.error(
                    "[Error] training loss does not descent. Stopping supervisor..."
                )
                save_model(model, os.path.join(train_dir, "earlystop"))
                sys.exit(1)

            best_loss, best_F, non_descent_cnt, saveNo = run_eval(
                model,
                valid_loader,
                valset,
                hps,
                best_loss,
                best_F,
                non_descent_cnt,
                saveNo,
            )

            if non_descent_cnt >= 3:
                logger.error(
                    "[Error] val loss does not descent for three times. Stopping supervisor..."
                )
                save_model(model, os.path.join(train_dir, "earlystop"))
                return

    # FIXME: The following code is not complete and needs to be adapted to the SAPGraph model
    def setup_training(self, model, train_loader, valid_loader, valset, hps):
        """
        Does setup before starting training (run_training)

        :param model: the model
        :param train_loader: train dataset loader
        :param valid_loader: valid dataset loader
        :param valset: valid dataset which includes text and summary
        :param hps: hps for model

        Credits:
            https://github.com/dqwang122/HeterSumGraph/blob/master/train.py#L43
        """

        train_dir = os.path.join(hps.save_root, "train")
        if os.path.exists(train_dir) and hps.restore_model != "None":
            logger.info(
                "[INFO] Restoring %s for training...", hps.restore_model
            )
            bestmodel_file = os.path.join(train_dir, hps.restore_model)
            model.load_state_dict(torch.load(bestmodel_file))
            hps.save_root = hps.save_root + "_reload"
        else:
            logger.info("[INFO] Create new model for training...")
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
            os.makedirs(train_dir)

        try:
            run_training(
                model, train_loader, valid_loader, valset, hps, train_dir
            )
        except KeyboardInterrupt:
            logger.error(
                "[Error] Caught keyboard interrupt on worker. Stopping supervisor..."
            )
            save_model(model, os.path.join(train_dir, "earlystop"))

    # FIXME: The following code is not complete and needs to be adapted to the SAPGraph model
    def train_model(self, embed_train: bool = False) -> None:
        """
        Trains the model on the data.
        Note this code was heavily adapted from SAPGraph and HSG code.

        parameters:
            embed_train (bool): whether or not to train the embeddings.

        Credits:
            https://github.com/dqwang122/HeterSumGraph/blob/master/train.py
        """
        # grab the vocabulary
        vocab = Vocab(self.vocab_fp, VOCAB_MAX_SIZE)

        # embedd the vocabulary using VOCAB_EMBED_DIM
        # download the GLOVE glove.42B.300d.zip file from
        # https://nlp.stanford.edu/projects/glove/
        # save it in data/models and unzip it
        embed = torch.nn.Embedding(vocab.size(), VOCAB_EMBED_DIM, padding_idx=0)

        # word embedded using
        embed_loader = Word_Embedding(GLOVE_PATH, vocab)
        vectors = embed_loader.load_my_vecs(VOCAB_EMBED_DIM)
        pretrained_weight = embed_loader.add_unknown_words_by_avg(
            vectors, VOCAB_EMBED_DIM
        )
        embed.weight.data.copy_(torch.Tensor(pretrained_weight))
        embed.weight.requires_grad = embed_train

        # get training data paths
        train_s2s_path = os.path.join(
            self.dir, "proquest_sapgraph_train.s2s.tfidf.jsonl"
        )
        train_w2s_path = os.path.join(
            self.dir, "proquest_sapgraph_train.w2s.tfidf.jsonl"
        )
        val_s2s_path = os.path.join(
            self.dir, "proquest_sapgraph_valid.s2s.tfidf.jsonl"
        )
        val_w2s_path = os.path.join(
            self.dir, "proquest_sapgraph_valid.w2s.tfidf.jsonl"
        )

        # initialize model
        model_s2s = WSWGAT(
            in_dim=VOCAB_EMBED_DIM,
            out_dim=SENT_ENT_NODE_EMBED_DIM,
            num_heads=NUM_HEADS,
            attn_drop_out=0.1,
            ffn_inner_hidden_size=HIDDEN_SIZE_FFN,
            ffn_drop_out=0.1,
            feat_embed_size=EDGE_EMBED_DIM,
            layerType="S2S",
        )
        WSWGAT(
            in_dim=VOCAB_EMBED_DIM,
            out_dim=SENT_ENT_NODE_EMBED_DIM,
            num_heads=NUM_HEADS,
            attn_drop_out=0.1,
            ffn_inner_hidden_size=HIDDEN_SIZE_FFN,
            ffn_drop_out=0.1,
            feat_embed_size=EDGE_EMBED_DIM,
            layerType="W2S",
        )  # fixme

        # get datasets
        dataset = DatasetAll(
            self.train_data_path,
            vocab,
            MAX_SENTENCE_NUM,
            MAX_TOKEN_NUM,
            MAX_ENTITY_NUM,  # fixme; the article doesn't mention a number for this
            self.vocab_fp,
            train_w2s_path,
            train_s2s_path,
            SIM_THRESHOLD,  # fixme; the article doesn't mention a number for this
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=32,
            collate_fn=graph_collate_fn,
        )

        valid_dataset = DatasetAll(
            self.valid_data_path,
            vocab,
            MAX_SENTENCE_NUM,
            MAX_TOKEN_NUM,
            MAX_ENTITY_NUM,  # fixme; the article doesn't mention a number for this
            self.vocab_fp,
            val_w2s_path,
            val_s2s_path,
            SIM_THRESHOLD,  # fixme; the article doesn't mention a number for this)
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=graph_collate_fn,
            num_workers=32,
        )

        self.setup_training(
            model_s2s, train_loader, valid_loader, valid_dataset
        )

        pass

    def summarize_articles(self) -> None:
        """
        Summarizes the articles in the dataframe.
        """
        pass

    def save_summaries(self, file_path: str) -> None:
        """
        Saves the summaries to the specified file path.
        """
        save_df_to_file(self.df, file_path)
