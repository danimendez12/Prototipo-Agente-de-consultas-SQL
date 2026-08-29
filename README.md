# LLM + SQL

Toda la orquestación podría hacerse mediante:

- Landgraph

## 1. Problema y motivación

El objetivo es que un agente de IA pueda navegar y extraer información de una base de datos relacional de forma orgánica, eficiente y segura, a partir de una consulta en lenguaje natural. Los riesgos centrales que la arquitectura debe resolver son tres: 

1. Que el agente entienda la estructura y semántica del esquema sin necesitar que un humano se lo explique en cada consulta,
2. Que las consultas generadas sean correctas y eficientes
3. Que ninguna consulta generada por un LLM pueda comprometer la integridad, confidencialidad o disponibilidad de los datos.

## 2. Visión general de la arquitectura

La arquitectura se organiza en siete etapas secuenciales, dos de las cuales (construcción y sincronización de la capa semántica) ocurren de forma asíncrona y no en el camino crítico de cada consulta del usuario:

1. **Extracción del catálogo técnico** — determinística, sin LLM.
2. **Enriquecimiento semántico** — asistido por LLM, con revisión humana obligatoria.
3. **Construcción del grafo de navegación** — estructura ligera tipo grafo de propiedades.
4. **Indexado del grafo** — embeddings para búsqueda semántica.
5. **Exploración** — recuperación de contexto relevante para una consulta específica.
6. **Generación** — un LLM produce la consulta SQL final.
7. **Validación y ejecución** — filtro híbrido determinístico/LLM y ejecución en sandbox de solo lectura.

Las etapas 1-4 se ejecutan una vez y se actualizan de forma incremental cuando cambia el esquema. Las etapas 5-7 ocurren en tiempo real, en cada consulta del usuario.

## 3. Componentes en detalle

### 3.1 Capa semántica ligera

Propongo un **grafo de propiedades ligero**, representado inicialmente como documentos JSON versionados . Cada nodo representa una tabla y contiene:

- Nombre técnico y descripción semántica (generada por LLM, aprobada por humano).
- Columnas, con tipo, si es PK/FK, y descripción de columnas ambiguas.
- Estimación de tamaño (row count).

Cada arista representa una relación entre tablas (foreign key declarada o relación implícita detectada por heurística), con su cardinalidad. Ejemplo de nodo:

```sql
{
  "id": "orders",
  "table": "orders",
  "description": "Registra cada compra realizada por un usuario",
  "columns": [
    {"name": "id", "type": "uuid", "is_pk": true},
    {"name": "user_id", "type": "uuid", "is_fk": true, "references": "users.id"},
    {"name": "status", "type": "varchar", "description": "Estado de la orden: pending, shipped, cancelled"},
    {"name": "created_at", "type": "timestamp"}
  ],
  "row_count_estimate": 1200000
}
```

Ejemplo de arista:

```sql
{"from": "orders", "to": "users", "via": "orders.user_id", "type": "foreign_key", "cardinality": "many_to_one"}
```

**Construcción:**

1. Extracción automática del catalogo técnico de la BD: tablas, columnas, tipos, PKs, FKs.
2. Enriquecimiento semántico: para cada tabla y columna ambigua, un LLM recibe el nombre, tipo, y una muestra anonimizada de filas, y genera una descripción candidata.
3. Cola de revisión humana: toda descripción generada por LLM entra en estado **PENDIENTE** y requiere aprobación explícita antes de pasar a **ACTIVO**. Esto aplica también a relaciones implícitas detectadas por heurística (columnas `_id` sin FK declarada que coinciden por nombre con otra tabla).
4. Generación de embeddings por nodo (concatenación de nombre + descripción + columnas) y persistencia en un índice vectorial.
5. Versionado explícito del grafo completo (o de sus diffs), para permitir rollback. Utilizando un control de versiones y un pipeline de CI/CD garantizamos mantener el grafo mejor actualizado y correcto presente

**Sincronización:** debe ser dirigida por eventos, no por tiempo. Un hook en el pipeline de migraciones de esquema.

**Tecnología sugerida:** JSON versionado + `networkx` (Python) para recorridos en memoria, más un índice vectorial (`pgvector` utilizando PostgreSQL)

**Nota:** Se recomienda evaluar un motor de grafos dedicado (Neo4j) solo si el esquema supera varios cientos de tablas y los recorridos multi-salto se vuelven un cuello de botella medido, no anticipado. Pero en un estudio inicial podemos omitir esta opción.

### 3.2 Explorador

Recibe la pregunta en lenguaje natural del usuario, genera su embedding, y consulta el índice vectorial para recuperar los nodos (tablas) semánticamente más cercanos. Desde esos nodos, recorre las aristas del grafo para incluir tablas relacionadas que probablemente se necesiten en un join. Su salida es un paquete de contexto acotado: solo las tablas y columnas relevantes para esa consulta específica, no el esquema completo.

### 3.3 Generador

Recibe la pregunta original del usuario más el paquete de contexto del Explorador, y produce la consulta SQL final. Se recomienda:

- Generación en dos pasos: primero un plan semi-estructurado de la consulta (tablas a unir, filtros, agregaciones), luego el SQL a partir de ese plan.
- Few-shot con ejemplos reales de pares (pregunta → SQL correcto) del dominio específico, cuando existan.
- Restricciones inyectadas en el prompt: solo `SELECT`, `LIMIT` explícito por defecto, prohibición de referenciar tablas fuera del contexto entregado por el Explorador.
- Auto-validación sintáctica (parseo local con `sqlglot` u equivalente) antes de enviar la consulta al Validador, para resolver errores de sintaxis sin gastar un ciclo completo de rechazo.

### 3.4 Validador (filtro de seguridad híbrido)

Propongo una cascada de dos niveles, no un único agente de seguridad:

1. **Nivel determinístico (parser):** análisis estático de la consulta (por ejemplo con `sqlglot`) que rechaza de forma inmediata cualquier operación de escritura (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`), consultas sin `LIMIT` cuando corresponda, o referencias a tablas fuera del alcance autorizado. Asi aseguramos un filtro “barato” antes de realizar una llamada a un agente
2. **Nivel semántico (LLM), condicional:** solo se invoca si la consulta pasa el nivel anterior, y evalúa aspectos que un parser no puede capturar — por ejemplo, si la consulta expone datos sensibles que no deberían combinarse en un mismo resultado.

Si la consulta es rechazada en cualquier nivel, el mensaje de error debe ser accionable (qué regla se violó, qué tabla o columna la causó) y se retorna al Generador.Recomendablemente se debe limitar explícitamente a  reintentos (2-3) para evitar ciclos sin convergencia.

### 3.5 Ejecutor

Ejecuta la consulta ya validada contra la base de datos real, bajo las siguientes garantías de defensa en profundidad:

- Rol de base de datos con permisos de solo lectura a nivel de motor (no solo por convención de la aplicación).
- Timeout de ejecución.
- Límite máximo de filas devueltas.
- Los resultados se interpretan y se devuelven al usuario en lenguaje natural.


## 4. Plan de evaluación

Propongo medir:

- **Precisión de generación**: comparación contra benchmarks estándar de text-to-SQL (Spider, BIRD), tanto con la capa semántica como sin ella, para cuantificar su aporte real.
- **Tasa de bloqueo del validador**: un conjunto de consultas adversariales diseñadas para intentar operaciones de escritura, fuga de datos sensibles, o consultas de alto costo, midiendo qué proporción es efectivamente bloqueada en cada nivel del validador.
- **Latencia y costo**: dado que el pipeline completo involucra múltiples llamadas a LLM en cascada (enriquecimiento, exploración, generación, validación semántica condicional), se debe medir el costo y la latencia acumulados por consulta, y compararlos contra una línea base sin capa semántica.

## 5. Limitaciones y trabajo futuro

- La revisión humana obligatoria en el enriquecimiento semántico es un cuello de botella de escalabilidad para esquemas muy grandes; vale la pena explorar mecanismos de aprobación por lotes o de confianza calibrada para reducir la carga humana sin sacrificar la supervisión.
- El nivel semántico del Validador (LLM) es la pieza con menor determinismo de toda la arquitectura; su tasa de falsos negativos debe monitorearse activamente en producción, no solo en la fase de evaluación inicial.
- Queda abierta la pregunta de si un motor de grafos dedicado (Neo4j) se justifica en escenarios de esquemas muy grandes (varios cientos de tablas); se recomienda migrar solo si se mide un cuello de botella real en los recorridos multi-salto.