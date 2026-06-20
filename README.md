# AEP — Adaptive Experimentation Platform · Datathon 7MLET (Grupo 65)

> Plataforma de **experimentação adaptativa** para decisões de oferta/mensagem/próximo
> passo em um banco digital **regulado**, usando **multi-armed bandits** (exploração vs.
> explotação) no lugar de regras fixas ou testes A/B longos — com um **assistente LLM+RAG**
> que resume experimentos, recupera políticas internas sintéticas e explica decisões.

⚠️ **Projeto acadêmico / demonstração.** Não usa dados reais de clientes, não simula um
banco real e **não está pronto para produção regulada**. Toda a camada de experimentação é
**sintética e reprodutível** (sementes fixas e documentadas).

---

## 1. Visão do problema

Dada uma instituição financeira digital, decidir — por canal e por cliente elegível —
**qual oferta, mensagem ou próximo passo** apresentar, aprendendo continuamente com o
retorno (conversão) em vez de depender de regras estáticas. O sistema formula isso como um
problema de **bandit**: cada "braço" é uma oferta; o objetivo é maximizar recompensa
(conversão) minimizando *regret*, equilibrando **exploração** e **explotação**, e tratando
**cold-start** e **recompensas atrasadas**.

A avaliação do datathon é **30% negócio / 70% técnica** (pipeline, MLOps, avaliação,
observabilidade, segurança, governança, uso de PyTorch/MLflow).

## 2. Escopo e escolhas de design

| Tema | Escolha | Porquê |
|---|---|---|
| Base factual (Kaggle) | **Bank Marketing** (`henriqueyamahata/bank-marketing`) | Domínio bancário, alvo de subscrição; coluna `duration` ilustra **vazamento pós-contato** (descartada). |
| Camada de experimentação | **Sintética**, sementes fixas | Sem dados reais; reprodutível e auditável. |
| Bandits | Baseline determinístico · **Thompson Sampling** · **Nilos-UCB/UCB1** · **LinUCB / neural bandit (PyTorch)** | Cobre exploração estocástica, baseada em confiança e **contextual**; PyTorch atende o critério técnico. |
| Tracking | **MLflow** | Params, métricas, artefatos e *model registry* (staging/production). |
| Serviço | **FastAPI + CLI** | Contrato de I/O documentado, *reason codes* e **log auditável** de decisão. |
| Assistente | **LLM+RAG** atrás de `LLMProvider` (mock / Anthropic) | Plugável; mapeia para **Azure OpenAI** na arquitetura-alvo. RAG sobre políticas **sintéticas**. |
| Gerenciador | **uv** (`pyproject.toml` + lock) | Instalação rápida e reprodutível. |

Princípios não-negociáveis (detalhados em [`CLAUDE.md`](CLAUDE.md)): sem dados reais, sem
vazamento temporal, reprodutibilidade total, etapas acumulativas, humano no loop, higiene
de repositório, commits incrementais, documentar enquanto se constrói.

## 3. Mapa de pastas

```
datathon-7mlet-grupo-65/
├── CLAUDE.md                 # plano completo do projeto (Etapas 0–8)
├── pyproject.toml            # deps, Python, entry point, dev tools (ruff/black/pytest/mypy)
├── Makefile / make.ps1       # comandos (make.ps1 = wrapper p/ Windows sem make)
├── .env.example              # variáveis necessárias (sem valores reais)
├── data/
│   ├── kaggle/               # fonte/versão/licença (CSV bruto NÃO versionado)
│   ├── processed/            # base tratada SEM vazamento (Etapa 1)
│   ├── synthetic_enrichment/ # offer_catalog, offer_events, delayed_rewards (Etapa 2)
│   └── golden_set/           # evaluation_cases.jsonl, ≥20 casos (Etapa 4)
├── notebooks/                # 01_eda.ipynb, 02_bandit_simulation.ipynb
├── src/aep/                  # pacote: data, synthetic, bandits, evaluation, service, assistant, mlops
├── tests/                    # contratos de dados, política e registro de decisão
├── docs/                     # architecture-azure, model-card, system-card, lgpd-plan
└── reports/                  # data-generation, data-quality, technical-report
```

## 4. Como executar localmente

Pré-requisitos: **Python 3.11+** e **[uv](https://docs.astral.sh/uv/)**
(`pip install uv`, ou `irm https://astral.sh/uv/install.ps1 | iex`).

```powershell
# 1. Instalar dependências (cria o .venv)
uv sync                 # ou:  ./make.ps1 setup        (Linux/macOS: make setup)

# 2. Validar a instalação
uv run pytest           # ou:  ./make.ps1 test
uv run aep version      # imprime versão + configuração ativa
```

> **Windows:** `make` não vem instalado por padrão. Use o wrapper equivalente
> **`./make.ps1 <alvo>`** (mesmos alvos do `Makefile`). Em Linux/macOS use `make <alvo>`.

### Comandos disponíveis

| `make` | `make.ps1` | Descrição |
|---|---|---|
| `make setup` | `./make.ps1 setup` | Cria o venv e instala deps core + dev |
| `make setup-all` | `./make.ps1 setup-all` | Instala **todos** os grupos (eda, ml, service, rag, llm) |
| `make test` | `./make.ps1 test` | Roda a suíte de testes |
| `make lint` / `make format` | `./make.ps1 lint` / `format` | ruff / black |
| `make data` | `./make.ps1 data` | (Etapa 1–2) processa base + gera enriquecimento sintético |
| `make train` | `./make.ps1 train` | (Etapa 3) treina/simula bandits e registra no MLflow |
| `make eval` | `./make.ps1 eval` | (Etapa 4) avaliação offline contra o golden set |
| `make serve` | `./make.ps1 serve` | (Etapa 5) sobe a API FastAPI |
| `make demo` | `./make.ps1 demo` | (Etapa 5) pipeline ponta a ponta local |

## 5. Estado do projeto (Etapas 0–8)

- [x] **Etapa 0 — Organização** — repo, `pyproject.toml`, `.gitignore`, `.env.example`, `LICENSE`, README, Makefile/`make.ps1`, esqueleto do pacote e CLI, smoke tests.
- [x] **Etapa 1** — Base Kaggle + EDA + dicionário de dados + relatório de qualidade.
- [x] **Etapa 2** — Enriquecimento sintético (catálogo/eventos/recompensas atrasadas).
- [x] **Etapa 3** — Baselines + Thompson + Nilos-UCB + LinUCB + neural bandit (PyTorch) + MLflow. Ver [`reports/bandit-comparison.md`](reports/bandit-comparison.md).
- [x] **Etapa 4** — Avaliação offline (IPS/SNIPS) + golden set (22 casos) + sensibilidade + fairness. Ver [`reports/evaluation.md`](reports/evaluation.md).
- [x] **Etapa 5** — Serviço FastAPI + CLI (`aep decide`, `aep serve`), assistente LLM+RAG (mock/Anthropic), log auditável e pipeline único (`aep demo`).
- [ ] **Etapa 6** — Arquitetura-alvo Azure (Mermaid + FinOps + Key Vault/Managed Identity).
- [ ] **Etapa 7** — Ciclo MLOps (retreino, approval gate, rollback, drift).
- [ ] **Etapa 8** — Governança (model/system card, LGPD) + relatório técnico.

## 6. Limitações

- **Não** é um sistema de produção nem um banco real; dados de experimentação são **sintéticos**.
- A base Kaggle é referência factual; nenhuma decisão usa atributos protegidos reais.
- Resultados de bandits são de **simulação** — não representam desempenho de mercado.
- O assistente LLM/RAG opera sobre **documentos de política sintéticos**; respostas não são aconselhamento regulatório.
- Componentes Azure são **arquitetura-alvo documentada**, não um deploy provisionado.

## 7. Licença

[MIT](LICENSE) © 2026 Datathon 7MLET — Grupo 65.
