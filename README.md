# Geração de corais de Bach

Comparação entre um Transformer autoregressivo em PyTorch e uma cadeia de Markov
sobre tokens REMI. Usamos o JSB Chorales, com 229 peças de treino, 76 de validação
e 77 de teste. A avaliação compara 200 peças de cada modelo com o teste.

## Instalação e execução

Na pasta do repositório, com Python 3.10 ou superior:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh
```

O script prepara os dados, treina os dois modelos, gera as peças e calcula os
resultados. O treino usa GPU quando disponível; em CPU pode demorar bastante.
Uma nova execução sobrescreve os resultados e as amostras existentes.

Para executar por etapa:

```bash
python scripts/prepare_data.py
python scripts/train_markov.py
python scripts/train_transformer.py --dropout 0.15 --out checkpoints/transformer_do015.pt --history results/transformer_history_do015.json
python scripts/train_transformer.py --dropout 0.3 --out checkpoints/transformer_do03.pt --history results/transformer_history_do03.json
python scripts/select_model.py
python scripts/generate.py
python scripts/evaluate.py
python scripts/make_figures.py
python scripts/render_showcase.py
python scripts/analyze_tonality.py
```

As duas configurações de dropout e a ordem da cadeia são escolhidas pela perda
na validação. Os pesos ficam em `checkpoints/`, que não é versionado.
O dataset bruto está em `data/raw/` e veio de
[JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset).

## Arquivos da entrega

- `chorales/`: dados, tokenização, modelos, amostragem, métricas e áudio.
- `scripts/`: comandos de preparação, treino, geração e avaliação.
- `data/raw/`: dataset original necessário à reprodução.
- `samples/transformer/showcase/`: cinco MIDIs e respectivos WAVs do Transformer.
- `samples/markov/showcase/`: cinco MIDIs e respectivos WAVs da cadeia de Markov.

As demonstrações são as primeiras cinco amostras de cada sistema com pelo menos
quatro compassos. Os áudios usam síntese aditiva para audição.

O artigo e a apresentação são entregues separadamente. Os dados processados,
pesos, 200 amostras de avaliação por sistema, resultados e figuras são recriados
pelos comandos acima e não são versionados. O script `make_figures.py` salva os
gráficos em `figures/`; copie-os para os materiais do artigo ao atualizar resultados.

## Contribuições

Preencher as contribuições reais antes da entrega:

| Integrante | Contribuição |
| --- | --- |
| Bernardo Alves | A preencher |
| Bernardo Venancio | A preencher |
| Lucas Affonso | A preencher |
| Lucas Albuquerque| A preencher |
| Mateus Zaparoli | A preencher |

## Uso de IA

O Codex foi usado para revisar e simplificar o código, a documentação e trechos
do artigo. O grupo deve acrescentar o histórico de uso anterior e os links das
conversas, incluindo esta, antes da entrega. A revisão dos integrantes ainda
está pendente.
