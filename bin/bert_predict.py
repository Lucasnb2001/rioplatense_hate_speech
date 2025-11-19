print("importando...")
import fire
from tqdm.auto import tqdm
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pysentimiento.preprocessing import preprocess_tweet


def bert_predict(
    input,
    output,
    model_name="piuba-bigdata/beto-contextualized-hate-speech",
    batch_size=8,
):
    print("Carregando modelo...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    print("Carregando dados...")
    df = pd.read_csv(input, index_col=0)
    df['context_tweet'] = df['context_tweet'].fillna('')


    # Iterate over the batches and predict

    id2label = model.config.id2label

    labels = [id2label[i] for i in range(len(id2label))]

    print(f"Labels: {labels[1:]}")

    assert "CALLS" == labels[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    predictions = []

    for i in tqdm(range(0, len(df), batch_size), total=len(df) // batch_size):
        batch = df.iloc[i : i + batch_size]

        tweets = [preprocess_tweet(tweet) for tweet in batch["text"].tolist()]
        contexts = [
            preprocess_tweet(context) for context in batch["context_tweet"].tolist()
        ]
        inputs = tokenizer(
            tweets,
            contexts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        scores = torch.sigmoid(outputs.logits).tolist()

        # Convert to list of dictionaries
        predictions += [
            {label: prediction for label, prediction in zip(labels, prediction)}
            for prediction in scores
        ]

    assert len(predictions) == len(df)

    # Append these new columns
    for label in labels:
        df[f"PRED_{label}"] = [prediction[label] for prediction in predictions]

    print(f"Saving to {output}")
    df.to_csv(output)


if __name__ == "__main__":
    fire.Fire(bert_predict)
