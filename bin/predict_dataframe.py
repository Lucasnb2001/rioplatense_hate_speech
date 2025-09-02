import sys
print("Script iniciando...")
print("Importando bibliotecas...")

print("Importando tqdm no projeto...")
from tqdm.auto import tqdm
print("Importando pandas no projeto...")
import pandas as pd
print("Importando fire no projeto...")
import fire
print("Importando time no projeto...")
import time
print("Importando build_prompt no projeto...")
from rioplatense_hs.tasks.hate_speech import build_prompt as build_hate_prompt
print("Importando build_regional no projeto...")
from rioplatense_hs.tasks.regionalisms import build_prompt as build_regionalism_prompt
print("Importando text_to_labels no projeto...")
from rioplatense_hs.preprocessing import text_to_label, labels
print("Importando get_completion no projeto...")
from rioplatense_hs.openai import get_completion
print("Todas as imports OK")

def predict_row(context, text, build_prompt, model_name="gpt-3.5-turbo"):
    try:
        prompt = build_prompt(contexto=context, texto=text)
        response = get_completion(prompt, model=model_name)
        text = response.choices[0].message.content

        return prompt, text
    # If rate limit is reached, wait 5 seconds and retry
    except Exception as e:
        print(f"Error: {e} -- {type(e)}")
        time.sleep(2)
        return None


def predict(df, model_name, build_prompt, max_retries=5):
    # Generate tasks

    ids = df.index.tolist()

    outs = {idx: None for idx in ids}

    retry_num = 1

    while any(o is None for o in outs.values()) and retry_num <= max_retries:
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            if outs[idx] is not None:
                continue
            outs[idx] = predict_row(
                row["context_tweet"],
                row["text"],
                model_name=model_name,
                build_prompt=build_prompt,
            )

        # Get tqdm ordered results

        retry_num += 1

    return outs


prompt_funs = {
    "hate_speech": build_hate_prompt,
    "regionalism": build_regionalism_prompt,
}


def openai_predict(
    input,
    output,
    task="hate_speech",
    model_name="gpt-3.5-turbo",
    num_examples=None,
    shuffle=False,
    limit=None,
):
    print(
        f"Predicting {input} -- with num_examples={num_examples} and shuffle={shuffle}"
    )

    # Build base prompt

    base_fun = prompt_funs[task]

    build_prompt = lambda contexto, texto: base_fun(
        contexto, texto, num_examples=num_examples, shuffle=shuffle
    )

    print("=" * 80)

    example_prompt = build_prompt(
        "contexto",
        "texto",
    )

    print(f"Example prompt: {example_prompt}")

    df = pd.read_csv(input)

    if limit is not None:
        df = df.iloc[:limit]
    assert "context_tweet" in df.columns and "text" in df.columns

    # Start asyncio with predict

    outs = predict(df, model_name=model_name, build_prompt=build_prompt)

    df["prompt"] = df.index.map(lambda x: outs[x][0])
    df["pred_cot"] = df.index.map(lambda x: outs[x][1])

    if task == "hate_speech":

        pred_labels = [f"PRED_{l}" for l in labels]

        for idx, value in df["pred_cot"].items():
            preds = text_to_label(value)

            for k, v in preds.items():
                df.loc[idx, f"PRED_{k}"] = int(v)

        # Convert pred_labels to int
        for l in pred_labels:
            df[l] = df[l].astype(int)
        pred_hate = df[pred_labels].sum(axis=1) > 0
        df["PRED_HATEFUL"] = pred_hate.astype(int)

    print(f"Saving to {output}")

    df.to_csv(output, index=False)


if __name__ == "__main__":
    fire.Fire(openai_predict)
