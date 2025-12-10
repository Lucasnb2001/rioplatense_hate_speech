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
    # 1. TROCAMOS O MODELO PARA UM BERTIMBAU FINE-TUNED
    # Exemplo: Modelo treinado em discurso de ódio em PT
    model_name="Silly-Machine/TuPy-Bert-Base-Multilabel", 
    batch_size=8,
):
    print(f"Carregando modelo {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    print("Carregando dados...")
    df = pd.read_csv(input, index_col=0)
    
    # Se o modelo BR não usa contexto, não precisamos tratar a coluna context_tweet aqui,
    # mas mantemos o fillna por segurança caso você queira usar.
    if 'context_tweet' in df.columns:
        df['context_tweet'] = df['context_tweet'].fillna('')

    # Identifica os labels automaticamente do novo modelo
    id2label = model.config.id2label
    labels = [id2label[i] for i in range(len(id2label))]
    print(f"Labels detectados: {labels}")

    # 2. REMOVEMOS O ASSERT (A trava que verificava "CALLS")
    # assert "CALLS" == labels[0]  <-- REMOVIDO POIS QUEBRARIA O CÓDIGO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Rodando em: {device}")
    model.to(device)

    predictions = []

    for i in tqdm(range(0, len(df), batch_size), total=len(df) // batch_size):
        batch = df.iloc[i : i + batch_size]

        # Pré-processamento (pysentimiento é bom para limpar tweets em geral)
        tweets = [preprocess_tweet(tweet) for tweet in batch["text"].tolist()]
        
        # 3. AJUSTE DE INPUT
        # A maioria dos modelos BR de ódio NÃO espera par de frases (contexto).
        # Se o seu modelo for treinado apenas em frases únicas, passe apenas 'tweets'.
        # Se você tiver certeza que o modelo aceita pares, mantenha o 'contexts'.
        inputs = tokenizer(
            tweets,
            # contexts,  <-- COMENTADO: Habilite apenas se o modelo foi treinado com pares
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128 # Boa prática definir um limite
        )
        
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inferência
        with torch.no_grad(): # Economiza memória
            outputs = model(**inputs)

        # Se o modelo for Multi-label ou Binário simples, sigmoid funciona bem.
        # Se for Multi-classe exclusiva (só pode ser uma coisa), use torch.softmax
        scores = torch.sigmoid(outputs.logits).tolist()

        predictions += [
            {label: prediction for label, prediction in zip(labels, score_row)}
            for score_row in scores
        ]

    # Salvar colunas
    for label in labels:
        df[f"PRED_{label}"] = [prediction[label] for prediction in predictions]

    print(f"Salvando em {output}")
    df.to_csv(output)

if __name__ == "__main__":
    fire.Fire(bert_predict)
