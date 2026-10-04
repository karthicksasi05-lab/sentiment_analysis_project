"""Command line script: give it a financial paragraph, get the sentiment.

Examples:
    python predict.py "Shares surge after record profit. Analysts raise targets."
    python predict.py --file news.txt
    python predict.py --model bert "Company misses estimates and cuts guidance."   (needs the Colab FinBERT files)
"""
import argparse
import sys

import utils


def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of a financial paragraph")
    parser.add_argument("text", nargs="?", help="the paragraph (put it in quotes)")
    parser.add_argument("--file", help="read the paragraph from a text file instead")
    parser.add_argument("--model", choices=["baseline", "bert"], default="baseline",
                        help="baseline = RNN/LSTM/GRU (default), bert = FinBERT trained on Colab")
    args = parser.parse_args()

    # get the text
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            paragraph = f.read()
    elif args.text:
        paragraph = args.text
    else:
        parser.print_help()
        sys.exit(1)

    # load the model
    if args.model == "bert":
        if not utils.bert_available():
            print("FinBERT not found in models/finbert_sentiment.")
            print("Train it with FinBERT_Colab.ipynb on Google Colab and copy the folder here.")
            sys.exit(1)
        model, tokenizer = utils.load_bert()
        probs_function = lambda s: utils.bert_probs(s, model, tokenizer)
        model_name = "FinBERT"
    else:
        if not utils.baseline_available():
            print("Model not found in the models/ folder. Run the main notebook first "
                  "(or copy the models folder here).")
            sys.exit(1)
        model, vocab, config = utils.load_baseline()
        probs_function = lambda s: utils.baseline_probs(s, model, vocab, config["max_len"])
        model_name = config["rnn_type"].upper()

    label, probs, per_sentence = utils.predict_paragraph(paragraph, probs_function)
    if label is None:
        print("No text found.")
        sys.exit(1)

    print()
    print("Model used       :", model_name)
    print("Overall sentiment:", label)
    print("Probabilities    :", {k: round(v, 3) for k, v in probs.items()})
    if len(per_sentence) > 1:
        print()
        print("Sentence by sentence:")
        for sentence, sentence_label in per_sentence:
            print("  [" + sentence_label + "] " + sentence)


if __name__ == "__main__":
    main()
