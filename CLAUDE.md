# Prompt para o Claude Code — Datathon 7MLET (Plataforma de Experimentação Adaptativa)

> Como usar: cole o conteúdo abaixo como primeira mensagem no Claude Code, dentro de uma pasta vazia.
> Sugestão: salve este arquivo no repositório como `CLAUDE.md` (ou `PLANO.md`) — o Claude Code o lê
> automaticamente como contexto persistente do projeto e segue o plano entre sessões.
> Antes de começar, substitua `grupo-XX` pelo número do seu grupo e ajuste a base Kaggle escolhida.

---

## PAPEL E MISSÃO

Você é um(a) **ML Engineer sênior** construindo, do zero, uma solução **end-to-end** para o Datathon da Fase 05 (7MLET / POSTECH).

O sistema é uma **plataforma de experimentação adaptativa** para um domínio financeiro **regulado**: dada uma instituição financeira digital, decidir em diferentes canais **qual oferta, mensagem ou próximo passo** apresentar a cada cliente elegível, usando **multi-armed bandits** (exploração vs. explotação) em vez de regras fixas ou testes A/B longos. Inclui um **assistente com LLM + RAG** que resume experimentos, recupera políticas internas sintéticas e explica decisões.

O objetivo **não** é simular um banco real, e sim demonstrar **maturidade técnica**: formular o problema, construir baselines, versionar dados, servir componentes, avaliar qualidade, monitorar risco, documentar limitações e explicar decisões para públicos técnico e de negócio.

A avaliação é **30% critérios de negócio** e **70% validação técnica global** (pipeline, MLOps, avaliação, observabilidade, segurança, governança, documentação, e **uso de PyTorch/MLflow quando aplicável**).

## PRINCÍPIOS NÃO-NEGOCIÁVEIS (aplicam-se a TODO o trabalho)

1. **Sem dados reais de clientes.** Nada de identificadores, patrimônio, renda, gênero, raça ou regras comerciais privadas. A base Kaggle é referência factual; toda a camada de experimentação é **sintética**.
2. **Sem vazamento temporal.** Descarte colunas pós-contato (ex.: `duration` no Bank Marketing). Documente cada descarte com justificativa.
3. **Reprodutibilidade total.** Toda geração sintética usa **sementes fixas e documentadas**. Todo experimento é rastreável (MLflow). Um comando único reproduz o pipeline ponta a ponta.
4. **Etapas acumulativas.** Uma etapa não compensa outra ausente. Não pule subitens — a banca penaliza a ausência de qualquer artefato listado.
5. **Humano no loop** em decisões sensíveis; documente base legal, finalidade, minimização e retenção (LGPD).
6. **Higiene de repositório.** Sem segredos, sem dados sensíveis, sem modelos binários grandes versionados. Use `.env.example`, `.gitignore` e `data/` com placeholders/scripts de download.
7. **Histórico de commits incremental.** Faça commits pequenos e descritivos por etapa/artefato — não um único commit final. A evolução do trabalho é avaliada.
8. **Documente enquanto constrói.** Cada módulo nasce com seu README/docstring. Não deixe documentação para o fim.

## STACK E CONVENÇÕES

- **Python 3.11+**, gerenciado por `pyproject.toml` (use `uv` ou `poetry`; declare deps, versão de Python, entry points e ferramentas de dev — ruff, black, pytest, mypy).
- **Dados/EDA:** pandas, numpy, matplotlib/seaborn, jupyter.
- **Bandits:** implementação própria de **Thompson Sampling** (Beta-Bernoulli) e da **família UCB (Nilos-UCB / UCB1)**, mais um **contextual bandit** (LinUCB e/ou um *neural bandit* em **PyTorch** para a estimativa de recompensa — isso cobre o critério "uso de PyTorch quando aplicável").
- **Baseline determinístico:** regra fixa / melhor braço histórico / segmentação inicial (pode usar um classificador de propensão simples em scikit-learn ou PyTorch).
- **Experimentos:** **MLflow** (params, métricas, artefatos, model registry).
- **Serviço:** **FastAPI** (API REST) + um CLI; contrato de I/O documentado e log auditável de decisão.
- **LLM + RAG:** camada **modular e plugável** atrás de uma interface (`LLMProvider`). No desenvolvimento/demo pode usar a Anthropic API ou um modelo local; na arquitetura-alvo Azure, mapeie para **Azure OpenAI**. RAG sobre documentos **sintéticos** de política comercial e *suitability* (vetor store local como FAISS/Chroma na demo; Azure AI Search na arquitetura-alvo).
- **Testes:** pytest cobrindo contratos de dados, política e registro de decisão.
- **Docs:** Markdown; diagramas em **Mermaid**.

## ESTRUTURA-ALVO DO REPOSITÓRIO

```
datathon-7mlet-grupo-XX/
├── README.md
├── pyproject.toml
├── .env.example
├── .gitignore
├── LICENSE
├── CLAUDE.md                      # este plano
├── Makefile                       # comandos: setup, data, train, eval, serve, test, demo
├── data/
│   ├── kaggle/README.md           # fonte, link, versão, licença, instruções de download
│   ├── processed/                 # base tratada SEM vazamento
│   ├── synthetic_enrichment/      # offer_catalog, offer_events, delayed_rewards
│   └── golden_set/evaluation_cases.jsonl   # >= 20 casos
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_bandit_simulation.ipynb
├── src/<pkg>/
│   ├── data/                      # loaders, dicionário, controle de versão/licença
│   ├── synthetic/                 # geradores (sementes fixas), schemas
│   ├── bandits/                   # baseline, thompson, ucb (nilos), linucb/neural
│   ├── evaluation/                # avaliação offline, métricas, fairness
│   ├── service/                   # FastAPI + CLI + reason codes + audit log
│   ├── assistant/                 # LLM provider + RAG (resumo/explicação/políticas)
│   └── mlops/                     # tracking MLflow, drift, promoção/rollback
├── tests/
├── docs/
│   ├── architecture-azure.md      # Mermaid + mapeamento Azure + custo + Key Vault/Managed Identity
│   ├── model-card.md
│   ├── system-card.md
│   └── lgpd-plan.md
└── reports/
    ├── data-generation.md         # processo, sementes, hipóteses, limitações, riscos
    ├── data-quality.md
    └── technical-report.md        # base do relatório final (até 10 páginas)
```

## PLANO DE EXECUÇÃO — ETAPAS 0 a 8

Trabalhe **sequencialmente**. Ao concluir cada etapa: rode os testes, atualize o README/Makefile, faça commit(s) descritivo(s) e me dê um resumo do que foi entregue e do critério de aceite atendido. **Pare e peça confirmação ao fim de cada etapa** antes de avançar.

### Etapa 0 — Organização do projeto
- Criar repositório local com nome `datathon-7mlet-grupo-XX`, `LICENSE`, `.gitignore` adequado, `pyproject.toml` (deps, Python, entry point, dev tools), `.env.example` (variáveis necessárias, **sem valores reais**).
- `README.md` com: visão do problema, escopo, escolhas de design, instruções de execução local, **mapa de pastas**, **lista de comandos** (via Makefile) e **limitações**.
- Inicializar git e fazer o primeiro commit estruturado.
- **Aceite:** uma pessoa externa instala dependências, entende o fluxo e roda ao menos um comando de validação sem explicação oral.

### Etapa 1 — Base Kaggle e EDA
- Escolher base compatível (ex.: `bank-marketing` do henriqueyamahata) e criar `data/kaggle/README.md` (link, versão, fonte, licença, limitações, instruções de download — **não** versionar o CSV bruto, usar script/placeholder).
- Camada de dados em código que carrega a base, **registra fonte/versão/licença** e gera os datasets derivados.
- **Dicionário de dados**, **notebook de EDA** e **relatório de qualidade** (`reports/data-quality.md`).
- Decisão documentada sobre colunas de vazamento temporal/pós-contato (descartar `duration`, etc.) com justificativa.
- **Aceite:** a banca rastreia a origem, entende as variáveis e verifica que não há vazamento pós-contato.

### Etapa 2 — Enriquecimento sintético
- `offer_catalog` (catálogo de braços/ofertas) **fisicamente separado** da base Kaggle.
- `offer_events` (impressões, contexto de decisão, recompensas intermediárias) com **sementes controladas**.
- `delayed_rewards` + horizonte temporal modelados e documentados.
- **Schema** de cada arquivo e **processo de geração** descritos no repo e em `reports/data-generation.md` (processo, sementes, hipóteses, limitações, riscos).
- **Aceite:** arquivos sintéticos com schema documentado, separados da base original, explicando braços/contexto/recompensa/horizonte.

### Etapa 3 — Baseline e estratégia algorítmica
- Implementar **>= 1 baseline determinístico**.
- Implementar/simular **Thompson Sampling** (priors documentados).
- **Nilos-UCB**: fórmula, implementação ou adaptação justificada, com análise do trade-off confiança × exploração × conversão (cobrir também na análise, mesmo que a política principal seja outra).
- Opcional/recomendado: **LinUCB e/ou neural bandit (PyTorch)** mostrando como o **contexto** entra na decisão.
- Calcular e reportar **recompensa, regret, exploração e conversão simulada**.
- Descrever tratamento de **cold-start** e **recompensas atrasadas**.
- Rastrear tudo no **MLflow**.
- **Aceite:** comparação quantitativa baseline × política adaptativa, com justificativa do algoritmo e do tratamento de cold-start/delayed rewards.

### Etapa 4 — Avaliação offline e golden set
- Script/notebook de **avaliação offline reproduzível** (linha de comando), com métricas justificadas.
- **Golden set** com **>= 20 casos** em `data/golden_set/evaluation_cases.jsonl`. Cada caso: **contexto, ação esperada, recompensa esperada, justificativa e critério explícito de pass/fail**.
- Cobertura: casos típicos, **casos de borda**, **segmentos sintéticos elegíveis** e **cenários adversariais**.
- **Matriz de métricas**, **análise de sensibilidade** e **análise de fairness de exposição** entre segmentos sintéticos.
- **Aceite:** métricas reproduzíveis, golden set versionado, análise explica limitações, vieses e quando a política **não** deve ser usada.

### Etapa 5 — Serviço/interface demonstrável
- **API (FastAPI) + CLI** que recebe um contexto e devolve uma decisão. Contrato de I/O documentado, exemplo de chamada e **tratamento de erro**.
- **Log auditável** de decisão com **reason codes, braço selecionado e versão da política**.
- **Comando único** (Makefile) que reproduz o pipeline ponta a ponta localmente.
- Integrar o **assistente LLM+RAG**: endpoints para resumir experimento, recuperar política sintética e explicar a decisão.
- **Suíte mínima de testes** cobrindo contratos de dados, política e registro de decisão.
- **Aceite:** a banca executa uma decisão de exemplo e vê braço, justificativa, versão da política e registro auditável.

### Etapa 6 — Arquitetura-alvo Azure
- `docs/architecture-azure.md` com **diagrama Mermaid** e mapeamento de serviços **exclusivamente Azure**, cobrindo: **compute, API, dados, IA/RAG, observabilidade, segurança, identidade e governança** (ex.: Container Apps/AKS, API Management, Azure ML, Azure OpenAI + Azure AI Search, Storage/Cosmos/PostgreSQL, Monitor/App Insights, Entra ID, Key Vault).
- **Plano de deploy** + **estimativa qualitativa de custo (FinOps)**.
- **Gestão de segredos** com **Key Vault** e **Managed Identity**.
- **Aceite:** arquitetura 100% Azure, cobre as camadas e justifica trade-offs sem outro provedor.

### Etapa 7 — Ciclo de vida MLOps
- **Plano de retreino**: critérios de promoção, **approval gate**, **rollback** e **versionamento de política**.
- **Monitoramento de drift e de recompensa** documentado.
- **Rastreio de experimentos no MLflow** (registry + estágios staging/production).
- Procedimento de **teste, aprovação humana estruturada e promoção** de novas políticas.
- **Aceite:** demonstrar como uma nova hipótese de oferta/canal/mensagem sai de experimento para produção controlada, com aprovação humana e rollback.

### Etapa 8 — Governança e relatórios (artefatos técnicos)
- `docs/model-card.md`: nome, versão, dados de treino/avaliação, métricas, **intended use**, **out-of-scope use**, análise de fairness, vieses conhecidos, limitações técnicas.
- `docs/system-card.md`: escopo, fluxo de decisão, dependências, **guardrails**, cenários de risco (**reward hacking, manipulação de contexto, abuso do assistente, violação de suitability**) e plano de monitoramento.
- `docs/lgpd-plan.md`: base legal, finalidade, minimização, ciclo de retenção, mapeamento de identificadores/atributos protegidos, política de logs/telemetria, plano de resposta a incidentes.
- `reports/technical-report.md`: rascunho do relatório (até 10 páginas) cobrindo problema, base, enriquecimento, modelagem como bandit, comparação quantitativa, arquitetura Azure, MLOps, limitações, riscos, hipóteses, trabalhos futuros, referências.
- Plano de **revisão periódica** do model/system card (responsáveis + cadência).
- **Aceite:** narrativa coerente de problema → solução → evidências → riscos → governança → valor, **sem alegar prontidão para produção real regulada**.

## DEFINITION OF DONE (checklist técnico contínuo)
- [ ] README explica desafio, execução local e limitações; pipeline usa base Kaggle com fonte/versão/licença/limitações.
- [ ] Base processada e enriquecimento sintético documentados e **separados** da base Kaggle; experimentos no MLflow.
- [ ] Baseline × abordagem principal comparados com métricas justificadas; análise referencia **Thompson Sampling** e **Nilos-UCB**.
- [ ] Golden set com **>= 20** exemplos; guardrails testados com cenários adversariais.
- [ ] Camada de retreino/teste/aprovação/promoção documentada.
- [ ] Serviço/API/notebook/interface funciona com instruções claras e **log auditável**.
- [ ] Arquitetura e deploy **exclusivamente Azure**, com Key Vault + Managed Identity.
- [ ] Model Card, System Card e plano LGPD completos.

## COMO TRABALHAR
1. Comece pela Etapa 0 e avance em ordem; ao fim de cada etapa, rode testes, atualize README/Makefile, faça commit(s) e **resuma o que entregou + critério de aceite atendido**, então **aguarde meu OK**.
2. Prefira código simples, testado e legível a sofisticação sem rastreabilidade.
3. Quando uma decisão de design tiver trade-offs relevantes, registre-os no README ou no doc pertinente.
4. Nunca insira segredos, dados reais ou binários grandes no git.
