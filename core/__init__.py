"""
core - Motor quantitativo do Otimizador de Carteiras (sem dependência de UI).

Nenhum módulo deste pacote importa Streamlit: o mesmo código serve ao
dashboard atual (app.py) e a uma futura API HTTP (Cloud Run). Cache, spinners
e mensagens ao usuário são responsabilidade da camada de interface.
"""
