# TCC 2 — Modelagem (etapa 5)

Três módulos novos, no mesmo diretório dos scripts do TCC 1:

| Arquivo | Função |
|---|---|
| `cs2_model_data.py` | Monta o dataset por round (alvo `t_win`) e por lado (cestas para regras) direto do `cs2.db` |
| `cs2_classification.py` | Dummy, Regressão Logística, Árvore de Decisão, Random Forest e XGBoost, com validação cruzada agrupada por série |
| `cs2_rules.py` | Apriori / FP-Growth com consequente fixo no resultado do round, poda de redundância e teste de Fisher com correção de Benjamini-Hochberg |

## Instalação

```bash
pip install scikit-learn xgboost mlxtend scipy matplotlib
```

## Execução

```bash
# 1. Gera os datasets (opcional, mas permite inspecionar antes de modelar)
python cs2_model_data.py cs2.db --output datasets/

# 2. Classificação (~10 min com importância por permutação; --perm-repeats 0 para teste rápido)
python cs2_classification.py --csv datasets/model_rounds.csv --output results/classificacao

# 3. Regras de associação
python cs2_rules.py --csv datasets/model_rounds.csv --output results/regras
python cs2_rules.py --csv datasets/model_rounds.csv --lado T  --output results/regras_T
python cs2_rules.py --csv datasets/model_rounds.csv --lado CT --output results/regras_CT
```

## Decisões que precisam aparecer no texto

1. **Dois cenários de variáveis.** `pre` usa só o que se sabe no fim do freeze time
   (economia, placar, mapa, round anterior). `full` acrescenta a abertura do round
   (lado do opening kill, tempo até ele, granadas usadas antes dele). Total de kills,
   sobreviventes e motivo da vitória nunca entram: definem o resultado (vazamento).
2. **Validação agrupada por série.** `StratifiedGroupKFold` mantém todos os rounds de
   uma mesma série MD3/MD5 no mesmo fold. A série é lida do nome da demo
   (`faze-vs-vitality-m1-...` → `faze-vs-vitality`); `--group match` agrupa por mapa.
3. **Linha de base.** O Dummy mostra o desempenho de quem sempre chuta a classe
   majoritária; os modelos devem ser lidos em relação a ele.
4. **Importância por permutação** calculada nos folds de teste (não nos de treino).
   Variáveis correlacionadas dividem a importância entre si; leia em conjunto.
5. **Regras de associação.** Cada round gera duas cestas (T e CT). Regras mais longas
   só são mantidas se ganharem ao menos 2 p.p. de confiança sobre uma regra mais
   simples. O p-valor ajustado controla falsos positivos entre centenas de regras.

## Pontos para conferir no banco real

- `economy.money_start` recebe `round_start_equip_value` no extrator (equipamento
  trazido do round anterior), não o dinheiro em caixa. Por isso a feature econômica
  principal é `equip_value`.
- `rounds.t_score`/`ct_score` contam vitórias por lado, não por equipe. O placar por
  equipe é reconstruído pelo elenco (steam_ids da tabela economy).
- O lado de quem matou é derivado do lado da vítima naquele round (economy), pois
  `attacker_steam_id` vem vazio do demoparser2.
- O log avisa quando um lado tem número de jogadores diferente de 5.
