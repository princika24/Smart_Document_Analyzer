import streamlit as st
import tempfile, os

from modules.document_loader import extract_text
from modules.retriever import DocumentRetriever
from modules.qa_module import QAModule
from modules.summary_module import SummaryModule
from modules.keyword_module import KeywordsModule

import networkx as nx
from modules.concept_linker import ConceptLinker
import matplotlib.pyplot as plt



st.set_page_config(page_title="Advanced Document QA", layout="wide")
st.title("📘 Advanced Document Q&A System")

uploaded = st.file_uploader("Upload your document (PDF, DOCX, PPTX, TXT):", type=["pdf", "docx", "pptx", "txt"])

if uploaded:
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded.name.split(".")[-1]) as tmp:
        tmp.write(uploaded.read())
        path = tmp.name

    text = extract_text(path)
    os.unlink(path)

    if not text.strip():
        st.error("Couldn't extract any text.")
    else:
        st.success("✅ Document loaded successfully!")

        retriever = DocumentRetriever()
        retriever.index(text)

        st.header("🔍 Ask a Question")
        question = st.text_input("Enter your question:")
        if st.button("Get Answer") and question.strip():
            contexts = retriever.retrieve(question, top_k=5)
            qa = QAModule()
            answer = qa.answer(question, contexts)
            st.subheader("Answer:")
            st.write(answer)

        
        st.header("🧾 Document Summarization")

        summary_option = st.radio(
            "Select summary length:",
            ("Short", "Medium", "Detailed"),
            index=1,
            horizontal=True
        )

        if st.button("Generate Summary"):
            if not text.strip():
                st.warning("Please upload a document first.")
            else:
                with st.spinner("Summarizing... please wait"):
                    summarizer = SummaryModule()
                    summary = summarizer.generate_summary(
                        text,
                        level=summary_option.lower()
                    )
                st.success("✅ Summary generated!")
                st.text_area("Summary Output", summary, height=300)


        st.header("📘 Extract Important Topics / Keywords")

        if "keywords" not in st.session_state:
            st.session_state.keywords = []

        if st.button("Extract Keywords"):
            if not text.strip():
                st.warning("Please upload a document first.")
            else:
                with st.spinner("Extracting important topics..."):
                    kw_extractor = KeywordsModule(top_k=15)
                    keywords = kw_extractor.extract_keywords(text)
                if not keywords:
                    st.warning("No keywords found — try another document.")
                else:
                    st.session_state.keywords = keywords
                    st.success("✅ Keywords extracted successfully!")
                    st.markdown(", ".join(f"`{kw}`" for kw in keywords))

        if st.session_state.keywords:
            st.markdown("---")
            st.subheader("🕸️ Concept Linking & Knowledge Map")

            if st.button("Generate Concept Map"):
                with st.spinner("Building concept clusters and visual map..."):
                    linker = ConceptLinker(threshold=0.55)
                    clusters = linker.build_concept_clusters(st.session_state.keywords)

                if clusters:
                    st.session_state.concept_clusters = clusters
                    st.success("✅ Concept clusters generated successfully!")
                    st.markdown("### 📘 Detected Concept Groups")
                    st.markdown(linker.describe_clusters(clusters))

                    st.markdown("### 🌐 Interactive Knowledge Graph")
                    linker.render_cluster_graph(clusters)
                else:
                    st.info("No concept relationships detected.")
