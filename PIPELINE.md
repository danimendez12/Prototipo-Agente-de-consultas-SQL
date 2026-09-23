# Pipeline de Consultas SQL con LLM

## Resumen del Pipeline

La arquitectura tiene **7 etapas secuenciales**:

1. **Extracción** (`extract_catalog.py`) → `artifacts/catalog_raw.json`
2. **Enriquecimiento** (`enrich_with_LLM_langchain.py`) → `artifacts/catalog_enriched_llm.json`
3. **Construcción de Grafo** (`build_graph.py`) → `artifacts/graph.pkl`
4. **Indexado** (automático en Explorador)
5. **Exploración** (`Explorador` clase)
6. **Generación** (Agentes: `AgentExplorador` o `AgentExploradorLC`)
7. **Validación & Ejecución** (Validador + Ejecutor)

## Instalación

```bash
# Crear venv
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

## Ejecución del Pipeline Completo

### Paso 1: Extraer Catálogo Técnico (Determinístico)

```bash
python src/extract_catalog.py
```

**Output:**
- `artifacts/catalog_raw.json` (descripciones = null)
- Console: lista de tablas, columnas, FKs, row_count

### Paso 2: Enriquecer con LLM

**Opción A: Con Groq (OpenAI/Gpt-oss-20b o Llama 3.3 70B)**

```bash
export GROQ_API_KEY="tu_clave_aqui"
python src/enrich_with_LLM_langchain.py
```

**Opción B: Con Ollama Local (100% sin API)**

```bash
# Instalar Ollama y descargar el modelo
ollama pull llama2

# Ejecutar (sin API key)
python src/enrich_with_LLM_langchain.py  # usa Ollama automáticamente si Groq no está
```

**Output:**
- Prompt interactivo: revisa descripciones generadas
- Al responder "s": guarda en `artifacts/catalog_enriched_llm.json`

### Paso 3: Construir Grafo

```bash
python src/build_graph.py
```

**Output:**
- `artifacts/graph.pkl` (grafo NetworkX serializado)
- Console: nodos, aristas, relaciones

## Evaluación

### Opción A: Evaluación Pura (Sin LLM)

Mide retrieval vectorial + expansión por grafo:

```bash
python evaluation/run_eval.py
```

**Métricas:**
- Precision/Recall (similitud pura vs. con expansión)
- Exact match rate
- Latencia en ms

### Opción B: Evaluación con Agente (Claude API)

```bash
python evaluation/agent_eval.py --n 5  # primero con pocas preguntas
```

**Costo:** ~$0.10 por pregunta (Claude Sonnet 3.5, 2M tokens input/output)

### Opción C: Evaluación con Agente LangChain (Groq)

```bash
export GROQ_API_KEY="tu_clave_aqui"
python evaluation/agent_eval_langchain.py --provider groq --model "llama-3.3-70b-versatile" --n 5
```

**Costo:** ~$0.002 por pregunta (Groq, 100x más barato)

## Búsqueda Exploratoria

### Debug Detallado de Retrieval

```bash
python src/debug_explorador.py
```

Muestra ranking completo, scores, expansión por grafo para cada pregunta del EVAL_SET.

### Grid Search de Hiperparámetros

```bash
python src/grid_search.py
```

Prueba todas las combinaciones de `min_score`, `relative_gap`, etc.

**Output:**
- Mejores configs según F1 y exact_match_rate
- Tabla de resultados

## Prueba Manual del Agente

```bash
# Desde la raíz del proyecto
source .venv/bin/activate
python src/agent_explorador_langchain.py
```

O en Python interactivo:

```python
import pickle
from src.explorador import Explorador
from src.agent_explorador_langchain import AgentExploradorLC
from src.project_paths import resolve_graph_path

graph_path = resolve_graph_path()
with open(graph_path, "rb") as f:
    graph = pickle.load(f)

explorador = Explorador(graph)
agent = AgentExploradorLC(explorador)

result = agent.retrieve("¿Qué artistas tienen más de 5 álbumes?")
print(f"Tablas: {result['tables']}")
print(f"Razonamiento: {result['reasoning']}")
```

## Estructura de Directorios

```
.
├── src/
│   ├── project_paths.py            # Resolución centralizada de paths
│   ├── extract_catalog.py          # Etapa 1: Extracción técnica
│   ├── enrich_with_LLM_langchain.py # Etapa 2: Enriquecimiento semántico
│   ├── build_graph.py              # Etapa 3: Construcción de grafo
│   ├── explorador.py               # Etapa 4-5: Indexado + Retrieval
│   ├── agent_explorador.py         # Etapa 6: Agente con Claude
│   ├── agent_explorador_langchain.py # Etapa 6: Agente con LangChain
│   ├── debug_explorador.py         # Herramienta de debug
│   └── grid_search.py              # Búsqueda de hiperparámetros
├── evaluation/
│   ├── eval_set.py                 # Conjunto de test con ground truth
│   ├── run_eval.py                 # Evaluación de retrieval puro
│   ├── agent_eval.py               # Evaluación de agente con Claude
│   └── agent_eval_langchain.py     # Evaluación de agente con LangChain
├── artifacts/
│   ├── catalog_raw.json            # Output de extract_catalog.py
│   ├── catalog_enriched_llm.json   # Output de enrich_with_LLM_langchain.py
│   └── graph.pkl                   # Output de build_graph.py
├── results/
│   ├── eval_results.json           # Output de run_eval.py
│   └── metricas.txt                # Historial de evaluaciones
├── data/
│   └── chinook.db                  # Base de datos SQLite (Chinook)
└── README.md, PIPELINE.md
```

## Troubleshooting

### Error: "Falta GROQ_API_KEY"

```bash
export GROQ_API_KEY="tu_clave_aqui"
# Luego reintenta el comando
```

### Error: "graph.pkl not found"

Asegúrate de:
1. Haber ejecutado `build_graph.py`
2. De estar en la raíz del proyecto

```bash
cd /Users/danielmendez/Prototipo-Agente-de-consultas-SQL
python src/build_graph.py
```

### Error: "catalog_enriched_llm.json not found"

Ejecuta el pipeline en orden:

```bash
python src/extract_catalog.py
python src/enrich_with_LLM_langchain.py  # Responde "s" cuando pida aprobación
python src/build_graph.py
```

### Conexión lenta a Groq

Intenta con Ollama local:

```bash
ollama pull llama2
# El script detectará automáticamente Ollama disponible
```

## Métricas y Reporting

Todos los scripts guardan resultados en:
- `results/eval_results.json` (evaluación estructurada)
- `results/metricas.txt` (historial legible)

```bash
# Ver último resultado
tail -50 results/metricas.txt
```

## Próximos Pasos

- [ ] Generador SQL (etapa 6 completa)
- [ ] Validador (etapa 7a)
- [ ] Ejecutor (etapa 7b)
- [ ] Interfaz web
- [ ] Escalado a esquemas más grandes (>100 tablas)
