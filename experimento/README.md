# Experimento usado no artigo

Esta pasta guarda os resultados e registros da execução usada nas tabelas do trabalho: 200 músicas do Transformer, 200 do Markov e 77 corais humanos de teste.

- `results/metrics_summary.json`: médias, desvios-padrão, OA e KL.
- `results/metrics_per_piece.csv`: métricas individuais das 477 músicas.
- `results/tonality.json`: análise de tonalidade e proporção de ataques com quatro notas.
- `results/model_selection.json`, `results/markov_orders.json` e históricos: seleção dos modelos e perdas de treinamento/validação.
- `logs/`: registros da execução original.
- `entradas_avaliacao.zip`: os 477 MIDIs avaliados, tokenizador, tokens de teste e tokens/metadados das gerações.

## Recalcular a avaliação sem treinar

Na raiz do repositório, com as dependências instaladas:

```bash
python -m zipfile -e experimento/entradas_avaliacao.zip .
python -c "from pathlib import Path; Path('results').mkdir(exist_ok=True)"
python scripts/evaluate.py
python scripts/analyze_tonality.py
```

A extração repõe as entradas da execução original em `data/` e `samples/` e pode substituir arquivos gerados por outra execução. Os novos resultados ficam em `results/`; os registros originais continuam em `experimento/results/`. Os exemplos de escuta já estão em `samples/*/showcase/` no repositório.

Os pesos treinados e os dados intermediários de treino não estão neste pacote. Para treinar novamente e gerar outras músicas, use `bash run_all.sh`. Um novo treino pode produzir valores diferentes conforme ambiente, hardware e aleatoriedade.

PE corresponde a `pitch_class_entropy` e SC a `scale_consistency`. Os valores do artigo são arredondados. OA e KL comparam distribuições de distâncias entre músicas; não são notas de qualidade musical.
