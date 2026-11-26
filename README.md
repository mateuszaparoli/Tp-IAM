# Geração de corais de Bach com Transformer e Markov

Trabalho final de DCC831 — IA Generativa para Música, UFMG, 2026/2.

Comparamos um Transformer autoregressivo em PyTorch com uma cadeia de Markov na geração incondicional de música simbólica. Os dois modelos usam o JSB Chorales e a representação REMI, com 106 tokens. A divisão oficial contém 229 peças de treino, 76 de validação e 77 de teste. Transposições ampliam o treino para 2.706 sequências.

## Integrantes

Bernardo Alves, Bernardo Venancio, Lucas Affonso Pires, Lucas Albuquerque e Mateus Zaparoli.

## Instalação

Requer Python 3.10 ou superior. No Linux, macOS ou terminal Bash:

```bash
git clone https://github.com/mateuszaparoli/Tp-IAM.git
cd Tp-IAM
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

No Windows, ative o ambiente com `.venv\Scripts\activate` e execute os comandos Python da próxima seção. O treino usa CUDA quando disponível e CPU nos demais casos. O treinamento completo em CPU pode demorar bastante.

## Reprodução

Na raiz do repositório, com o ambiente ativado:

```bash
bash run_all.sh
```

Ou execute as etapas individualmente:

```bash
python scripts/prepare_data.py
python scripts/train_markov.py
python scripts/train_transformer.py --dropout 0.15 --out checkpoints/transformer_do015.pt --history results/transformer_history_do015.json
python scripts/train_transformer.py --dropout 0.3 --out checkpoints/transformer_do03.pt --history results/transformer_history_do03.json
python scripts/select_model.py
python scripts/generate.py --system both --n 200
python scripts/evaluate.py
python scripts/make_figures.py
python scripts/render_showcase.py
python scripts/analyze_tonality.py
```

A execução recria dados processados, pesos, resultados, figuras e amostras, sobrescrevendo as saídas anteriores. Os pesos treinados não estão no GitHub: é necessário treinar antes de gerar novas músicas. Os exemplos já incluídos podem ser ouvidos sem executar o código.

O dataset bruto está em `data/raw/Jsb16thSeparated.json`, obtido de [JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset).

## Contribuições registradas no repositório

| Integrante | Participação registrada |
| --- | --- |
| Mateus Zaparoli | Estrutura inicial, inclusão do dataset, leitura dos corais, tokenização REMI e preparação dos dados. |
| Bernardo Alves | Inclusão do modelo de Markov, ajuste e seleção da ordem, e arquitetura do Transformer. |
| Lucas Affonso Pires | Inclusão do treinamento e seleção do Transformer, amostragem e geração de músicas. |
| Lucas Albuquerque | Inclusão das métricas, avaliação das amostras, figuras e análise tonal. |
| Bernardo Venancio | Integração da reprodução, síntese de áudio e exemplos MIDI/WAV. |
