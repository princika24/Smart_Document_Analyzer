import itertools
import networkx as nx
from sentence_transformers import SentenceTransformer, util
from pyvis.network import Network
import tempfile
import streamlit as st
import re

class ConceptLinker:
    def __init__(self, threshold=0.6):
        """
        threshold: similarity threshold for linking related terms into clusters.
        """
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = threshold

    def clean_keywords(self, keywords):
        """
        Normalize and filter out trivial or noisy keywords.
        """
        stopwords = {"use", "using", "used", "like", "make", "work", "works", "based", "data"}
        cleaned = set()
        for kw in keywords:
            kw = kw.strip().lower()
            if len(kw) < 3 or kw in stopwords or re.match(r"^\W+$", kw):
                continue
            cleaned.add(kw)
        return list(cleaned)

    def build_concept_clusters(self, keywords):
        """
        Cluster related keywords using similarity threshold.
        """
        keywords = self.clean_keywords(keywords)
        if not keywords or len(keywords) < 2:
            return []

        embeddings = self.model.encode(keywords, convert_to_tensor=True)
        G = nx.Graph()
        G.add_nodes_from(keywords)

        # connect words with similarity > threshold
        for i, j in itertools.combinations(range(len(keywords)), 2):
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()
            if sim >= self.threshold:
                G.add_edge(keywords[i], keywords[j])

        # get connected components as clusters
        clusters = [list(c) for c in nx.connected_components(G)]
        return clusters

    def describe_clusters(self, clusters):
        """
        Describe clusters in a readable format.
        """
        if not clusters:
            return "No concept clusters found."

        lines = []
        for idx, group in enumerate(clusters, 1):
            label = self._generate_cluster_label(group)
            members = ", ".join(sorted(set(group)))
            lines.append(f"**{label}** → {members}")
        return "\n\n".join(lines)

    def _generate_cluster_label(self, group):
        """
        Try to find a representative name for the group.
        """
        # heuristic: pick the most descriptive term (longest or with 'learning', 'language')
        for kw in group:
            if "learning" in kw:
                return "Learning Concepts"
            if "language" in kw:
                return "Language Concepts"
        # else pick longest term
        return group[0].capitalize() + " Group"

    def render_cluster_graph(self, clusters):
        """
        Render concept clusters visually as grouped nodes.
        """
        if not clusters:
            st.info("No concept clusters found to visualize.")
            return

        net = Network(height="650px", width="100%", bgcolor="#ffffff", font_color="black")
        colors = ["#90CAF9", "#A5D6A7", "#FFD54F", "#FFAB91", "#CE93D8", "#80CBC4"]

        for idx, group in enumerate(clusters):
            color = colors[idx % len(colors)]
            label = self._generate_cluster_label(group)
            for kw in group:
                net.add_node(kw, label=kw.title(), title=label, color=color, size=20)
            # connect cluster members in a circular way for better layout
            for i in range(len(group)):
                net.add_edge(group[i], group[(i + 1) % len(group)], color="#888888")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
            net.save_graph(tmp_file.name)
            st.components.v1.html(open(tmp_file.name, "r", encoding="utf-8").read(), height=700)
