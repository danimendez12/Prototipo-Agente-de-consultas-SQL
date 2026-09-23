"""Small Streamlit UI for the complete database query pipeline."""

import streamlit as st

from src.query_pipeline import QueryPipeline


st.set_page_config(page_title="Consultas SQL", page_icon="🔎", layout="wide")
st.title("🔎 Consulta tu base de datos")
st.caption("Explorador → Generador SQL → Validador → Ejecutor")

with st.sidebar:
    st.header("Configuración")
    provider = st.selectbox("Proveedor del modelo", ["groq", "ollama", "deepseek"])
    explorer_model = st.text_input("Modelo de exploración", "openai/gpt-oss-120b")
    generator_model = st.text_input("Modelo de SQL", "openai/gpt-oss-120b")
    executor_model = st.text_input("Modelo de respuesta", "openai/gpt-oss-20b")

question = st.text_area(
    "¿Qué quieres saber?",
    placeholder="Ejemplo: ¿Cuáles son los 5 países con mayor facturación?",
    height=100,
)

if st.button("Ejecutar consulta", type="primary", disabled=not question.strip()):
    with st.spinner("Ejecutando el pipeline completo..."):
        pipeline = QueryPipeline.from_defaults(
            provider=provider,
            explorer_model=explorer_model,
            generator_model=generator_model,
            executor_model=executor_model,
        )
        result = pipeline.run(question)

    exploration = result.get("exploration", {})
    generation = result.get("generation", {})
    execution = result.get("execution", {})

    with st.expander("1. Exploración", expanded=True):
        st.write("**Tablas seleccionadas:**", ", ".join(exploration.get("tables", [])))
        st.write(exploration.get("reasoning", ""))

    with st.expander("2. Consulta generada", expanded=True):
        if generation.get("sql"):
            st.code(generation["sql"], language="sql")

    if result["success"]:
        st.success("Consulta ejecutada correctamente")
        st.subheader("Respuesta del agente ejecutor")
        st.write(execution["answer"])
        st.caption(
            f"{execution['row_count']} filas · {execution['elapsed_ms']} ms"
            + (" · resultado truncado" if execution["truncated"] else "")
        )
        if execution["rows"]:
            st.dataframe(execution["rows"], use_container_width=True)
    else:
        st.error(f"Pipeline detenido en {result.get('stage', 'desconocida')}: {result.get('error')}")
