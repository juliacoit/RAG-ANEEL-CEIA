"""
banco_perguntas.py
==================
Banco de perguntas para avaliação do sistema RAG ANEEL.

Cobre diferentes categorias temáticas, tipos de busca e comportamentos
esperados para medir Faithfulness, Answer Relevance e Citation Accuracy.

Uso:
    from tests.banco_perguntas import BANCO, Categoria, TipoResposta
    perguntas_fallback = [p for p in BANCO if p["tipo_esperado"] == TipoResposta.FALLBACK]

Estrutura de cada pergunta:
    id              — identificador único
    pergunta        — texto da pergunta
    categoria       — tema principal (ver Categoria)
    tipo_busca      — como a busca deve se comportar (ver TipoBusca)
    tipo_esperado   — resposta_encontrada | fallback | parcial
    dificuldade     — facil | medio | dificil
    metricas_alvo   — lista com quais métricas esta pergunta exercita
    notas           — observações para o avaliador
    filtros         — dict opcional com filtros de metadados para usar na query
"""

from enum import Enum


# ---------------------------------------------------------------------------
# Enumerações
# ---------------------------------------------------------------------------

class Categoria(str, Enum):
    MICROGERACAO    = "microgeracao"        # Micro e minigeração distribuída
    TARIFAS         = "tarifas"             # TUSD, TUST, tarifas de uso do sistema
    PCH             = "pch"                 # Pequenas Centrais Hidrelétricas
    CONCESSOES      = "concessoes"          # Concessões e autorizações de geração
    PENALIDADES     = "penalidades"         # Multas, autos de infração, penalidades
    RENOVAVEL       = "renovavel"           # Solar, eólica, biomassa, outras renováveis
    QUALIDADE       = "qualidade"           # Qualidade de energia, DEC/FEC
    DISTRIBUICAO    = "distribuicao"        # Distribuidoras, redes de distribuição
    TRANSMISSAO     = "transmissao"         # Transmissoras, linhas de transmissão
    FORA_DE_ESCOPO  = "fora_de_escopo"      # Perguntas sem resposta na base atual


class TipoBusca(str, Enum):
    SEMANTICA   = "semantica"    # Favorece a busca vetorial — conceitos, não termos exatos
    LEXICAL     = "lexical"      # Favorece o BM25 — termos técnicos ou siglas exatas
    HIBRIDA     = "hibrida"      # Exercita os dois canais de busca simultaneamente
    FILTRADA    = "filtrada"     # Requer filtros por metadados (tipo, ano, etc.)


class TipoResposta(str, Enum):
    ENCONTRADA  = "resposta_encontrada"     # Base contém informação suficiente
    FALLBACK    = "fallback"                # Deve retornar "Não encontrado..."
    PARCIAL     = "parcial"                 # Base tem informação relacionada, mas incompleta


class Dificuldade(str, Enum):
    FACIL   = "facil"
    MEDIO   = "medio"
    DIFICIL = "dificil"


class Metrica(str, Enum):
    FAITHFULNESS    = "faithfulness"        # Resposta é fiel aos documentos recuperados?
    RELEVANCIA      = "answer_relevance"    # Resposta resolve a dúvida?
    CITACAO         = "citation_accuracy"   # Citações de atos e datas estão corretas?
    RECUSA          = "recusa_correta"      # Modelo recusa quando deve recusar?
    RECUPERACAO     = "retrieval_quality"   # Chunks recuperados são pertinentes?


# ---------------------------------------------------------------------------
# Banco de Perguntas
# ---------------------------------------------------------------------------

BANCO: list[dict] = [

    # =========================================================================
    # MICROGERAÇÃO E MINIGERAÇÃO DISTRIBUÍDA
    # =========================================================================

    {
        "id": "MG-01",
        "pergunta": "O que é microgeração distribuída segundo a ANEEL?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO],
        "notas": "Conceito central — REN 687/2015 define micro e minigeração. Boa pergunta de sanidade.",
        "filtros": None,
    },
    {
        "id": "MG-02",
        "pergunta": "Qual é o limite de potência para minigeração distribuída?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO, Metrica.RELEVANCIA],
        "notas": "Deve citar limites em kW/MW da resolução vigente. Testa precisão numérica.",
        "filtros": None,
    },
    {
        "id": "MG-03",
        "pergunta": "Quais fontes de energia são permitidas para microgeração distribuída?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA],
        "notas": "Testa enumeração de fontes renováveis permitidas.",
        "filtros": None,
    },
    {
        "id": "MG-04",
        "pergunta": "Como funciona o sistema de compensação de energia elétrica para prosumidores?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Usa o termo 'prosumidor' — testa robustez semântica. O modelo deve conectar ao sistema de compensação.",
        "filtros": None,
    },
    {
        "id": "MG-05",
        "pergunta": "Quais são os prazos para a distribuidora conectar uma unidade de microgeração à rede?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Prazos operacionais detalhados podem não estar nas ementas. Espera-se resposta parcial ou recusa.",
        "filtros": None,
    },
    {
        "id": "MG-06",
        "pergunta": "O que mudou na regulamentação de microgeração entre 2015 e 2022?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.FILTRADA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.RECUSA],
        "notas": "Comparação temporal. A base tem 2016/2021/2022. Testa se o modelo compara sem inventar.",
        "filtros": {"ano_fonte": ["2016", "2021", "2022"]},
    },

    # =========================================================================
    # TARIFAS — TUSD / TUST
    # =========================================================================

    {
        "id": "TAR-01",
        "pergunta": "O que é a Tarifa de Uso do Sistema de Distribuição (TUSD)?",
        "categoria": Categoria.TARIFAS,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO],
        "notas": "Pergunta conceitual com sigla. O BM25 deve ajudar na sigla e o semântico no conceito.",
        "filtros": None,
    },
    {
        "id": "TAR-02",
        "pergunta": "TUSD geração pode ser reduzida para fontes renováveis incentivadas?",
        "categoria": Categoria.TARIFAS,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO, Metrica.RELEVANCIA],
        "notas": "Testa se o modelo recupera despachos sobre desconto de TUSD para incentivadas.",
        "filtros": None,
    },
    {
        "id": "TAR-03",
        "pergunta": "Qual o percentual de redução da TUSD previsto no Despacho 3409?",
        "categoria": Categoria.TARIFAS,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO, Metrica.RECUSA],
        "notas": "Pergunta muito específica com número de despacho. Detalhes de percentual podem não estar na ementa.",
        "filtros": None,
    },
    {
        "id": "TAR-04",
        "pergunta": "Como é calculada a tarifa de transmissão TUST?",
        "categoria": Categoria.TARIFAS,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Metodologia de cálculo raramente detalhada em ementas. Espera-se recusa ou resposta parcial.",
        "filtros": None,
    },
    {
        "id": "TAR-05",
        "pergunta": "Quais atos normativos tratam de revisão tarifária de distribuidoras?",
        "categoria": Categoria.TARIFAS,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Pergunta de listagem — avalia quantos atos relevantes o modelo consegue citar.",
        "filtros": None,
    },

    # =========================================================================
    # PEQUENAS CENTRAIS HIDRELÉTRICAS (PCH)
    # =========================================================================

    {
        "id": "PCH-01",
        "pergunta": "Quais atos normativos tratam de autorização de PCH?",
        "categoria": Categoria.PCH,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.CITACAO],
        "notas": "Sigla PCH + 'autorização' — exercita BM25. Boa pergunta para medir recall.",
        "filtros": None,
    },
    {
        "id": "PCH-02",
        "pergunta": "Qual o limite de potência instalada para uma usina ser classificada como PCH?",
        "categoria": Categoria.PCH,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO],
        "notas": "Limite de 30 MW deve estar nas ementas regulatórias.",
        "filtros": None,
    },
    {
        "id": "PCH-03",
        "pergunta": "Uma PCH de 25 MW tem direito ao benefício de desconto na TUSD?",
        "categoria": Categoria.PCH,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Pergunta composta que cruza PCH com benefícios tarifários. Testa raciocínio baseado em documentos.",
        "filtros": None,
    },
    {
        "id": "PCH-04",
        "pergunta": "Quais documentos são necessários para solicitar autorização de uma PCH à ANEEL?",
        "categoria": Categoria.PCH,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Lista de documentos é conteúdo processual detalhado — provavelmente não nas ementas.",
        "filtros": None,
    },
    {
        "id": "PCH-05",
        "pergunta": "PCH autorizada no rio Paraíba do Sul em 2022",
        "categoria": Categoria.PCH,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.RECUSA],
        "notas": "Query informal/sem estrutura de pergunta — testa robustez do retrieval a consultas curtas.",
        "filtros": {"ano_fonte": ["2022"]},
    },

    # =========================================================================
    # CONCESSÕES E AUTORIZAÇÕES
    # =========================================================================

    {
        "id": "CON-01",
        "pergunta": "Qual é a diferença entre concessão e autorização para geração de energia?",
        "categoria": Categoria.CONCESSOES,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA],
        "notas": "Distinção conceitual clássica. Testa capacidade de síntese de múltiplos atos.",
        "filtros": None,
    },
    {
        "id": "CON-02",
        "pergunta": "Quando é necessária a autorização da ANEEL para liberação de unidade geradora em operação comercial?",
        "categoria": Categoria.CONCESSOES,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO, Metrica.RELEVANCIA],
        "notas": "Pergunta procedural sobre operação comercial — presente nas ementas de despachos.",
        "filtros": None,
    },
    {
        "id": "CON-03",
        "pergunta": "Quais são as condições para renovação de concessão de transmissão?",
        "categoria": Categoria.CONCESSOES,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Condições detalhadas de renovação raramente aparecem nas ementas completas.",
        "filtros": None,
    },
    {
        "id": "CON-04",
        "pergunta": "Quais resoluções normativas de 2021 tratam de autorizações de usinas?",
        "categoria": Categoria.CONCESSOES,
        "tipo_busca": TipoBusca.FILTRADA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.CITACAO],
        "notas": "Usa filtro por ano e tipo. Testa integração de filtros de metadados com busca.",
        "filtros": {"tipo_codigo": "REH", "ano_fonte": ["2021"]},
    },

    # =========================================================================
    # PENALIDADES E INFRAÇÕES
    # =========================================================================

    {
        "id": "PEN-01",
        "pergunta": "Quais penalidades podem ser aplicadas a distribuidoras por descumprimento de metas de qualidade?",
        "categoria": Categoria.PENALIDADES,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Testa recuperação de atos sobre penalidades regulatórias.",
        "filtros": None,
    },
    {
        "id": "PEN-02",
        "pergunta": "multa auto de infração distribuidora penalidade",
        "categoria": Categoria.PENALIDADES,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUPERACAO],
        "notas": "Query de palavras-chave sem estrutura — replica como usuário técnico pode pesquisar. Favorece BM25.",
        "filtros": None,
    },
    {
        "id": "PEN-03",
        "pergunta": "Uma distribuidora pode ser multada por não cumprir o prazo de conexão de microgeração?",
        "categoria": Categoria.PENALIDADES,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA, Metrica.RELEVANCIA],
        "notas": "Pergunta cruzada (penalidade + microgeração). Requer ementas específicas sobre esse nexo.",
        "filtros": None,
    },
    {
        "id": "PEN-04",
        "pergunta": "Qual o valor máximo da multa prevista na Resolução Normativa ANEEL nº 846?",
        "categoria": Categoria.PENALIDADES,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA, Metrica.CITACAO],
        "notas": "Valores monetários específicos raramente aparecem nas ementas. Teste de recusa.",
        "filtros": None,
    },

    # =========================================================================
    # ENERGIA RENOVÁVEL
    # =========================================================================

    {
        "id": "REN-01",
        "pergunta": "Quais atos normativos da ANEEL regulam a energia solar fotovoltaica?",
        "categoria": Categoria.RENOVAVEL,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.CITACAO],
        "notas": "Pergunta de listagem sobre solar. Testa recall — quantas resoluções relevantes são citadas.",
        "filtros": None,
    },
    {
        "id": "REN-02",
        "pergunta": "Como a geração eólica é tratada nos atos normativos de 2021?",
        "categoria": Categoria.RENOVAVEL,
        "tipo_busca": TipoBusca.FILTRADA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.FAITHFULNESS, Metrica.CITACAO],
        "notas": "Filtro por ano com tema específico. Testa busca filtrada combinada com semântica.",
        "filtros": {"ano_fonte": ["2021"]},
    },
    {
        "id": "REN-03",
        "pergunta": "O que são Fontes Incentivadas de Energia e quais os benefícios regulatórios associados?",
        "categoria": Categoria.RENOVAVEL,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Conceito abrangente — testa se o modelo sintetiza múltiplos trechos sem inventar.",
        "filtros": None,
    },
    {
        "id": "REN-04",
        "pergunta": "Qual é a regulamentação da ANEEL sobre biogás como fonte de energia?",
        "categoria": Categoria.RENOVAVEL,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Biogás é nicho — pode haver poucos atos nas ementas. Testa comportamento com escassez de contexto.",
        "filtros": None,
    },

    # =========================================================================
    # QUALIDADE DE ENERGIA — DEC / FEC
    # =========================================================================

    {
        "id": "QUA-01",
        "pergunta": "O que são os indicadores DEC e FEC de qualidade do serviço de distribuição?",
        "categoria": Categoria.QUALIDADE,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO],
        "notas": "Siglas técnicas centrais — BM25 deve recuperar bem. Boa pergunta de sanidade para retrieval lexical.",
        "filtros": None,
    },
    {
        "id": "QUA-02",
        "pergunta": "Quais são as metas de DEC e FEC estabelecidas para distribuidoras?",
        "categoria": Categoria.QUALIDADE,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Metas numéricas específicas por distribuidora raramente nas ementas. Teste de recusa.",
        "filtros": None,
    },
    {
        "id": "QUA-03",
        "pergunta": "Quais atos normativos estabelecem regras de qualidade da energia entregue ao consumidor final?",
        "categoria": Categoria.QUALIDADE,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Busca de listagem sobre qualidade ao consumidor. Testa precisão do retrieval semântico.",
        "filtros": None,
    },

    # =========================================================================
    # DISTRIBUIÇÃO
    # =========================================================================

    {
        "id": "DIS-01",
        "pergunta": "Quais são as obrigações da distribuidora em relação ao atendimento de novos consumidores?",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Pergunta procedimental abrangente. Testa síntese de ementas sobre conexão.",
        "filtros": None,
    },
    {
        "id": "DIS-02",
        "pergunta": "O que é o PRODIST e quais são seus módulos?",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO],
        "notas": "Sigla importante do setor elétrico. BM25 deve recuperar atos do PRODIST.",
        "filtros": None,
    },
    {
        "id": "DIS-03",
        "pergunta": "Como é tratada a subvenção tarifária para consumidores de baixa renda nas resoluções da ANEEL?",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RELEVANCIA, Metrica.CITACAO],
        "notas": "Tema social — baixa renda e tarifa social. Testa recuperação semântica em tema sensível.",
        "filtros": None,
    },

    # =========================================================================
    # TRANSMISSÃO
    # =========================================================================

    {
        "id": "TRA-01",
        "pergunta": "Quais portarias autorizam transmissoras a operar novas linhas de transmissão?",
        "categoria": Categoria.TRANSMISSAO,
        "tipo_busca": TipoBusca.FILTRADA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUPERACAO, Metrica.CITACAO],
        "notas": "Usa filtro por tipo de ato (Portaria). Testa filtro de metadados combinado com conteúdo.",
        "filtros": {"tipo_codigo": "PRT"},
    },
    {
        "id": "TRA-02",
        "pergunta": "O que acontece quando uma instalação de transmissão entra em operação comercial antes do prazo previsto?",
        "categoria": Categoria.TRANSMISSAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Cenário específico de prazo antecipado. Detalhes de consequências raramente nas ementas.",
        "filtros": None,
    },

    # =========================================================================
    # FORA DE ESCOPO — devem retornar fallback
    # =========================================================================

    {
        "id": "FORA-01",
        "pergunta": "Qual é o preço do megawatt-hora no mercado spot hoje?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUSA],
        "notas": "Preço spot é dado dinâmico — não está em nenhum ato normativo. Deve recusar.",
        "filtros": None,
    },
    {
        "id": "FORA-02",
        "pergunta": "Qual o Art. 5º, inciso III da Resolução Normativa nº 482/2012?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUSA, Metrica.FAITHFULNESS],
        "notas": "Pergunta sobre artigo/inciso específico — os PDFs não foram ingeridos. Deve retornar fallback.",
        "filtros": None,
    },
    {
        "id": "FORA-03",
        "pergunta": "Quem é o diretor-geral da ANEEL atualmente?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUSA],
        "notas": "Dado pessoal/institucional não presente em atos normativos. Testa alucinação de cargo.",
        "filtros": None,
    },
    {
        "id": "FORA-04",
        "pergunta": "Como faço para pedir segunda via da minha conta de luz?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUSA],
        "notas": "Pergunta operacional de consumidor final — fora do escopo de atos normativos.",
        "filtros": None,
    },
    {
        "id": "FORA-05",
        "pergunta": "Qual é a composição acionária da Eletrobras após a privatização?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUSA],
        "notas": "Dado societário não presente nos atos normativos da ANEEL.",
        "filtros": None,
    },
    {
        "id": "FORA-06",
        "pergunta": "Quais são os artigos 12 ao 18 da Resolução Normativa ANEEL nº 687/2016?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUSA, Metrica.FAITHFULNESS],
        "notas": "Limitação conhecida: sem os PDFs completos, artigos específicos não são respondíveis.",
        "filtros": None,
    },

    # =========================================================================
    # PERGUNTAS DE BORDA — robustez e comportamento do modelo
    # =========================================================================

    {
        "id": "BORDA-01",
        "pergunta": "aneel resolução",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.RECUPERACAO],
        "notas": "Query mínima de 2 palavras — testa se o sistema lida com perguntas subarticuladas.",
        "filtros": None,
    },
    {
        "id": "BORDA-02",
        "pergunta": "Quais são TODOS os atos normativos existentes na base de dados?",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Pedido impossível de responder completamente. Testa se o modelo admite limitação.",
        "filtros": None,
    },
    {
        "id": "BORDA-03",
        "pergunta": "A resolução normativa 1000 da ANEEL trata de quê?",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.LEXICAL,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO, Metrica.RECUSA],
        "notas": "Número específico de resolução — BM25 deve encontrar se existir. Caso não exista na base, deve recusar.",
        "filtros": None,
    },
    {
        "id": "BORDA-04",
        "pergunta": "Existe alguma norma sobre drones perto de linhas de transmissão?",
        "categoria": Categoria.FORA_DE_ESCOPO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.FALLBACK,
        "dificuldade": Dificuldade.MEDIO,
        "metricas_alvo": [Metrica.RECUSA],
        "notas": "Tema emergente provavelmente ausente na base 2016/2021/2022. Testa alucinação temática.",
        "filtros": None,
    },
    {
        "id": "BORDA-05",
        "pergunta": "Qual resolução normativa mais recente da base trata de conexão de geração distribuída?",
        "categoria": Categoria.MICROGERACAO,
        "tipo_busca": TipoBusca.HIBRIDA,
        "tipo_esperado": TipoResposta.ENCONTRADA,
        "dificuldade": Dificuldade.DIFICIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.CITACAO, Metrica.RELEVANCIA],
        "notas": "Pergunta com superlativo temporal. Testa se o modelo cita corretamente a data/ano mais recente.",
        "filtros": None,
    },
    {
        "id": "BORDA-06",
        "pergunta": "O que a ANEEL regula?",
        "categoria": Categoria.DISTRIBUICAO,
        "tipo_busca": TipoBusca.SEMANTICA,
        "tipo_esperado": TipoResposta.PARCIAL,
        "dificuldade": Dificuldade.FACIL,
        "metricas_alvo": [Metrica.FAITHFULNESS, Metrica.RECUSA],
        "notas": "Pergunta extremamente ampla. O modelo deve responder com base nos documentos, sem generalizar.",
        "filtros": None,
    },
]


# ---------------------------------------------------------------------------
# Utilitários de filtragem
# ---------------------------------------------------------------------------

def por_categoria(categoria: Categoria) -> list[dict]:
    """Retorna todas as perguntas de uma categoria específica."""
    return [p for p in BANCO if p["categoria"] == categoria]


def por_tipo_resposta(tipo: TipoResposta) -> list[dict]:
    """Retorna perguntas pelo tipo de resposta esperada."""
    return [p for p in BANCO if p["tipo_esperado"] == tipo]


def por_dificuldade(dificuldade: Dificuldade) -> list[dict]:
    """Retorna perguntas por nível de dificuldade."""
    return [p for p in BANCO if p["dificuldade"] == dificuldade]


def por_metrica(metrica: Metrica) -> list[dict]:
    """Retorna perguntas que exercitam uma métrica específica."""
    return [p for p in BANCO if metrica in p["metricas_alvo"]]


def por_tipo_busca(tipo: TipoBusca) -> list[dict]:
    """Retorna perguntas por tipo de busca."""
    return [p for p in BANCO if p["tipo_busca"] == tipo]


def com_filtros() -> list[dict]:
    """Retorna apenas as perguntas que possuem filtros de metadados definidos."""
    return [p for p in BANCO if p.get("filtros")]


def resumo() -> dict:
    """Retorna um resumo estatístico do banco."""
    return {
        "total": len(BANCO),
        "por_categoria": {
            cat.value: len(por_categoria(cat)) for cat in Categoria
        },
        "por_tipo_resposta": {
            t.value: len(por_tipo_resposta(t)) for t in TipoResposta
        },
        "por_dificuldade": {
            d.value: len(por_dificuldade(d)) for d in Dificuldade
        },
        "por_tipo_busca": {
            t.value: len(por_tipo_busca(t)) for t in TipoBusca
        },
        "com_filtros": len(com_filtros()),
    }


# ---------------------------------------------------------------------------
# Execução direta — imprime resumo do banco
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    stats = resumo()
    print("\n" + "=" * 60)
    print("BANCO DE PERGUNTAS — RAG ANEEL")
    print("=" * 60)
    print(f"\nTotal de perguntas: {stats['total']}\n")

    print("Por categoria:")
    for cat, n in stats["por_categoria"].items():
        if n > 0:
            print(f"  {cat:<20} {n:>3}")

    print("\nPor tipo de resposta esperada:")
    for tipo, n in stats["por_tipo_resposta"].items():
        print(f"  {tipo:<30} {n:>3}")

    print("\nPor dificuldade:")
    for dif, n in stats["por_dificuldade"].items():
        print(f"  {dif:<10} {n:>3}")

    print("\nPor tipo de busca:")
    for tb, n in stats["por_tipo_busca"].items():
        print(f"  {tb:<12} {n:>3}")

    print(f"\nCom filtros de metadados: {stats['com_filtros']}")
    print("=" * 60)

    print("\nPerguntas de fallback (devem recusar):")
    for p in por_tipo_resposta(TipoResposta.FALLBACK):
        print(f"  [{p['id']}] {p['pergunta']}")
