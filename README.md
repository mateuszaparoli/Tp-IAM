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

O dataset bruto está em `data/raw/Jsb16thSeparated.json`, obtido de [JSB-Chorales-dataset](https://github.com/czhuang/JSB-Chorales-dataset). A preparação converte a grade em semicolcheias para MIDI, assume compasso 4/4 e aplica transposições somente ao treino.

## Modelos e avaliação

O Transformer possui quatro camadas, quatro cabeças de atenção, dimensão 256 e contexto de 1.024 tokens. A escolha entre dropout 0,15 e 0,30 usa a perda de validação. O baseline usa interpolação de Witten–Bell e escolhe a ordem entre 0 e 16 pela validação.

Geramos 200 peças por sistema com temperatura 1,0, top-p 0,95 e semente 123. Comparamos essas peças aos 77 corais de teste por sete métricas do MusPy, além do número de compassos. OA e KL comparam distribuições de distâncias entre os valores das métricas, seguindo Yang e Lerch (2020).

| Resultado do experimento | Transformer | Markov |
| --- | ---: | ---: |
| NLL de validação | 1,098 | 1,853 |
| Perplexidade de validação | 3,00 | 6,38 |
| Consistência média de escala | 0,930 | 0,688 |
| OA da consistência de escala | 0,92 | 0,09 |

A consistência média de escala humana foi 0,930. O Transformer se aproximou mais das características humanas em várias métricas, mas não manteve quatro notas simultâneas em todos os ataques. As métricas não substituem uma avaliação perceptiva. Uma nova execução pode apresentar diferenças numéricas conforme o ambiente e o hardware.

## Músicas geradas

Cada linha contém um MIDI e seu áudio WAV. A seleção usa as primeiras cinco peças de avaliação de cada modelo com pelo menos quatro compassos. O áudio foi produzido com síntese aditiva para audição.

| Exemplo | Transformer | Markov |
| --- | --- | --- |
| 000 | [MIDI](samples/transformer/showcase/transformer_000.mid) / [WAV](samples/transformer/showcase/transformer_000.wav) | [MIDI](samples/markov/showcase/markov_000.mid) / [WAV](samples/markov/showcase/markov_000.wav) |
| 001 | [MIDI](samples/transformer/showcase/transformer_001.mid) / [WAV](samples/transformer/showcase/transformer_001.wav) | [MIDI](samples/markov/showcase/markov_001.mid) / [WAV](samples/markov/showcase/markov_001.wav) |
| 002 | [MIDI](samples/transformer/showcase/transformer_002.mid) / [WAV](samples/transformer/showcase/transformer_002.wav) | [MIDI](samples/markov/showcase/markov_002.mid) / [WAV](samples/markov/showcase/markov_002.wav) |
| 003 | [MIDI](samples/transformer/showcase/transformer_003.mid) / [WAV](samples/transformer/showcase/transformer_003.wav) | [MIDI](samples/markov/showcase/markov_003.mid) / [WAV](samples/markov/showcase/markov_003.wav) |
| 004 | [MIDI](samples/transformer/showcase/transformer_004.mid) / [WAV](samples/transformer/showcase/transformer_004.wav) | [MIDI](samples/markov/showcase/markov_004.mid) / [WAV](samples/markov/showcase/markov_004.wav) |

## Organização

- `chorales/`: modelos, tokenização, leitura dos dados, métricas e síntese.
- `scripts/`: preparação, treino, geração, avaliação e criação das figuras.
- `data/raw/`: dataset original.
- `samples/`: exemplos MIDI e WAV da entrega.
- `results/`, `figures/` e `checkpoints/`: saídas criadas durante a reprodução.

O artigo no formato ISMIR e a apresentação são entregues separadamente.

## Contribuições registradas no repositório

A tabela descreve as partes incorporadas por cada integrante no histórico de commits.

| Integrante | Participação registrada |
| --- | --- |
| Mateus Zaparoli | Estrutura inicial, inclusão do dataset, leitura dos corais, tokenização REMI e preparação dos dados. |
| Bernardo Alves | Inclusão do modelo de Markov, ajuste e seleção da ordem, e arquitetura do Transformer. |
| Lucas Affonso Pires | Inclusão do treinamento e seleção do Transformer, amostragem e geração de músicas. |
| Lucas Albuquerque | Inclusão das métricas, avaliação das amostras, figuras e análise tonal. |
| Bernardo Venancio | Integração da reprodução, síntese de áudio e exemplos MIDI/WAV. |

## Referências

- Boulanger-Lewandowski, Bengio e Vincent (2012). Modeling Temporal Dependencies in High-Dimensional Sequences.
- Huang et al. (2017). Counterpoint by Convolution.
- Huang e Yang (2020). Pop Music Transformer.
- Fradet et al. (2021). MidiTok.
- Dong et al. (2020). MusPy.
- Yang e Lerch (2020). On the Evaluation of Generative Models in Music.
