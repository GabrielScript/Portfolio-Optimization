"""
assets.py - Lista das principais ações da B3 (IBrX-100+)
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel Estrela Lopes

Lista expandida com ~100 ativos das principais empresas listadas na B3.
"""

# Lista de ativos extraída para o motor do TCC (~100 principais)
ATIVOS_B3 = [
    # === FINANCEIRO ===
    {"ticker": "ITUB4", "nome": "Banco Itaú", "setor": "Financeiro"},
    {"ticker": "BBDC4", "nome": "Banco Bradesco", "setor": "Financeiro"},
    {"ticker": "BBAS3", "nome": "Banco Do Brasil", "setor": "Financeiro"},
    {"ticker": "SANB11", "nome": "Banco Santander", "setor": "Financeiro"},
    {"ticker": "BPAC11", "nome": "Banco BTG Pactual", "setor": "Financeiro"},
    {"ticker": "ITSA4", "nome": "Itaúsa", "setor": "Financeiro"},
    {"ticker": "B3SA3", "nome": "B3", "setor": "Financeiro"},
    {"ticker": "BBSE3", "nome": "BB Seguridade", "setor": "Financeiro"},
    {"ticker": "CXSE3", "nome": "Caixa Seguridade", "setor": "Financeiro"},
    {"ticker": "PSSA3", "nome": "Porto Seguro", "setor": "Financeiro"},
    {"ticker": "BPAN4", "nome": "Banco Pan", "setor": "Financeiro"},
    {"ticker": "MULT3", "nome": "Multiplan", "setor": "Financeiro"},
    {"ticker": "ALOS3", "nome": "Allos", "setor": "Financeiro"},
    {"ticker": "BRAP4", "nome": "Bradespar", "setor": "Financeiro"},
    {"ticker": "BNBR3", "nome": "Banco Nordeste", "setor": "Financeiro"},
    {"ticker": "BRSR6", "nome": "Banrisul", "setor": "Financeiro"},
    {"ticker": "ABCB4", "nome": "ABC Brasil", "setor": "Financeiro"},
    {"ticker": "BMGB4", "nome": "Banco BMG", "setor": "Financeiro"},
    
    # === PETRÓLEO, GÁS E BIOCOMBUSTÍVEIS ===
    {"ticker": "PETR4", "nome": "Petrobrás", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "PETR3", "nome": "Petrobrás ON", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "PRIO3", "nome": "PRIO", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "VBBR3", "nome": "Vibra Energia", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "UGPA3", "nome": "Ultrapar", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "CSAN3", "nome": "Cosan", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "BRAV3", "nome": "Brava Energia", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "RRRP3", "nome": "3R Petroleum", "setor": "Petróleo, Gás e Biocombustíveis"},
    
    # === MATERIAIS BÁSICOS ===
    {"ticker": "VALE3", "nome": "Vale", "setor": "Materiais Básicos"},
    {"ticker": "SUZB3", "nome": "Suzano", "setor": "Materiais Básicos"},
    {"ticker": "GGBR4", "nome": "Gerdau", "setor": "Materiais Básicos"},
    {"ticker": "CSNA3", "nome": "CSN", "setor": "Materiais Básicos"},
    {"ticker": "CMIN3", "nome": "CSN Mineração", "setor": "Materiais Básicos"},
    {"ticker": "KLBN11", "nome": "Klabin", "setor": "Materiais Básicos"},
    {"ticker": "GOAU4", "nome": "Metalúrgica Gerdau", "setor": "Materiais Básicos"},
    {"ticker": "USIM5", "nome": "Usiminas", "setor": "Materiais Básicos"},
    {"ticker": "BRKM5", "nome": "Braskem", "setor": "Materiais Básicos"},
    {"ticker": "DXCO3", "nome": "Dexco", "setor": "Materiais Básicos"},
    
    # === UTILIDADE PÚBLICA (ENERGIA) ===
    {"ticker": "ELET3", "nome": "Eletrobras ON", "setor": "Utilidade Pública"},
    {"ticker": "ELET6", "nome": "Eletrobras PNB", "setor": "Utilidade Pública"},
    {"ticker": "SBSP3", "nome": "Sabesp", "setor": "Utilidade Pública"},
    {"ticker": "CPFE3", "nome": "CPFL Energia", "setor": "Utilidade Pública"},
    {"ticker": "EQTL3", "nome": "Equatorial Energia", "setor": "Utilidade Pública"},
    {"ticker": "ENEV3", "nome": "Eneva", "setor": "Utilidade Pública"},
    {"ticker": "NEOE3", "nome": "Neoenergia", "setor": "Utilidade Pública"},
    {"ticker": "EGIE3", "nome": "Engie Brasil", "setor": "Utilidade Pública"},
    {"ticker": "CMIG4", "nome": "Cemig", "setor": "Utilidade Pública"},
    {"ticker": "ENGI11", "nome": "Energisa", "setor": "Utilidade Pública"},
    {"ticker": "ISAE4", "nome": "ISA Energia Brasil", "setor": "Utilidade Pública"},
    {"ticker": "CSMG3", "nome": "Copasa", "setor": "Utilidade Pública"},
    {"ticker": "TAEE11", "nome": "Taesa", "setor": "Utilidade Pública"},
    {"ticker": "SAPR11", "nome": "Sanepar", "setor": "Utilidade Pública"},
    {"ticker": "AURE3", "nome": "Auren Energia", "setor": "Utilidade Pública"},
    {"ticker": "ALUP11", "nome": "Alupar", "setor": "Utilidade Pública"},
    {"ticker": "TRPL4", "nome": "Transmissão Paulista", "setor": "Utilidade Pública"},
    {"ticker": "CPLE6", "nome": "Copel", "setor": "Utilidade Pública"},
    {"ticker": "COCE5", "nome": "Coelce", "setor": "Utilidade Pública"},
    
    # === BENS INDUSTRIAIS ===
    {"ticker": "WEGE3", "nome": "WEG", "setor": "Bens Industriais"},
    {"ticker": "EMBR3", "nome": "Embraer", "setor": "Bens Industriais"},
    {"ticker": "RAIL3", "nome": "Rumo", "setor": "Bens Industriais"},
    {"ticker": "CCRO3", "nome": "CCR", "setor": "Bens Industriais"},
    {"ticker": "ECOR3", "nome": "Ecorodovias", "setor": "Bens Industriais"},
    {"ticker": "GGPS3", "nome": "GPS", "setor": "Bens Industriais"},
    {"ticker": "AZUL4", "nome": "Azul", "setor": "Bens Industriais"},
    {"ticker": "GOLL4", "nome": "Gol", "setor": "Bens Industriais"},
    {"ticker": "TGMA3", "nome": "Tegma", "setor": "Bens Industriais"},
    {"ticker": "POMO4", "nome": "Marcopolo", "setor": "Bens Industriais"},
    {"ticker": "ROMI3", "nome": "Romi", "setor": "Bens Industriais"},
    
    # === CONSUMO NÃO CÍCLICO ===
    {"ticker": "ABEV3", "nome": "Ambev", "setor": "Consumo não Cíclico"},
    {"ticker": "JBSS3", "nome": "JBS", "setor": "Consumo não Cíclico"},
    {"ticker": "BRFS3", "nome": "BRF", "setor": "Consumo não Cíclico"},
    {"ticker": "MRFG3", "nome": "Marfrig", "setor": "Consumo não Cíclico"},
    {"ticker": "BEEF3", "nome": "Minerva", "setor": "Consumo não Cíclico"},
    {"ticker": "NTCO3", "nome": "Natura", "setor": "Consumo não Cíclico"},
    {"ticker": "GMAT3", "nome": "Grupo Mateus", "setor": "Consumo não Cíclico"},
    {"ticker": "ASAI3", "nome": "Assaí", "setor": "Consumo não Cíclico"},
    {"ticker": "CRFB3", "nome": "Carrefour Brasil", "setor": "Consumo não Cíclico"},
    {"ticker": "MDIA3", "nome": "M. Dias Branco", "setor": "Consumo não Cíclico"},
    {"ticker": "RAIZ4", "nome": "Raízen", "setor": "Consumo não Cíclico"},
    {"ticker": "SLCE3", "nome": "SLC Agrícola", "setor": "Consumo não Cíclico"},
    {"ticker": "SMTO3", "nome": "São Martinho", "setor": "Consumo não Cíclico"},
    {"ticker": "PCAR3", "nome": "GPA", "setor": "Consumo não Cíclico"},
    
    # === CONSUMO CÍCLICO ===
    {"ticker": "RENT3", "nome": "Localiza", "setor": "Consumo Cíclico"},
    {"ticker": "LREN3", "nome": "Lojas Renner", "setor": "Consumo Cíclico"},
    {"ticker": "MGLU3", "nome": "Magazine Luiza", "setor": "Consumo Cíclico"},
    {"ticker": "CYRE3", "nome": "Cyrela", "setor": "Consumo Cíclico"},
    {"ticker": "SMFT3", "nome": "Smart Fit", "setor": "Consumo Cíclico"},
    {"ticker": "CURY3", "nome": "Cury", "setor": "Consumo Cíclico"},
    {"ticker": "COGN3", "nome": "Cogna", "setor": "Consumo Cíclico"},
    {"ticker": "ALPA4", "nome": "Alpargatas", "setor": "Consumo Cíclico"},
    {"ticker": "PETZ3", "nome": "Petz", "setor": "Consumo Cíclico"},
    {"ticker": "MOVI3", "nome": "Movida", "setor": "Consumo Cíclico"},
    {"ticker": "MRVE3", "nome": "MRV", "setor": "Consumo Cíclico"},
    {"ticker": "EZTC3", "nome": "EZTEC", "setor": "Consumo Cíclico"},
    {"ticker": "DIRR3", "nome": "Direcional", "setor": "Consumo Cíclico"},
    {"ticker": "ARZZ3", "nome": "Arezzo", "setor": "Consumo Cíclico"},
    {"ticker": "SOMA3", "nome": "Grupo Soma", "setor": "Consumo Cíclico"},
    {"ticker": "YDUQ3", "nome": "Yduqs", "setor": "Consumo Cíclico"},
    {"ticker": "VAMO3", "nome": "Vamos", "setor": "Consumo Cíclico"},
    
    # === SAÚDE ===
    {"ticker": "RDOR3", "nome": "Rede D'Or", "setor": "Saúde"},
    {"ticker": "RADL3", "nome": "Raia Drogasil", "setor": "Saúde"},
    {"ticker": "HAPV3", "nome": "Hapvida", "setor": "Saúde"},
    {"ticker": "HYPE3", "nome": "Hypera", "setor": "Saúde"},
    {"ticker": "FLRY3", "nome": "Fleury", "setor": "Saúde"},
    {"ticker": "ODPV3", "nome": "Odontoprev", "setor": "Saúde"},
    {"ticker": "QUAL3", "nome": "Qualicorp", "setor": "Saúde"},
    {"ticker": "ONCO3", "nome": "Oncoclínicas", "setor": "Saúde"},
    
    # === TELECOMUNICAÇÕES ===
    {"ticker": "VIVT3", "nome": "Vivo", "setor": "Telecomunicações"},
    {"ticker": "TIMS3", "nome": "TIM", "setor": "Telecomunicações"},
    
    # === TECNOLOGIA DA INFORMAÇÃO ===
    {"ticker": "TOTS3", "nome": "TOTVS", "setor": "Tecnologia da Informação"},
    {"ticker": "LWSA3", "nome": "Locaweb", "setor": "Tecnologia da Informação"},
    {"ticker": "INTB3", "nome": "Intelbras", "setor": "Tecnologia da Informação"},
    {"ticker": "POSI3", "nome": "Positivo", "setor": "Tecnologia da Informação"},
    {"ticker": "CASH3", "nome": "Méliuz", "setor": "Tecnologia da Informação"},
]

# Lista de setores únicos
SETORES = list(set([ativo["setor"] for ativo in ATIVOS_B3]))
SETORES.sort()

def get_tickers_by_setor(setores_selecionados: list) -> list:
    """
    Retorna lista de tickers filtrados pelos setores selecionados.
    
    Args:
        setores_selecionados: Lista de setores para filtrar
        
    Returns:
        Lista de tickers no formato yfinance (com .SA)
    """
    if not setores_selecionados:
        return [f"{a['ticker']}.SA" for a in ATIVOS_B3]
    
    return [
        f"{a['ticker']}.SA" 
        for a in ATIVOS_B3 
        if a["setor"] in setores_selecionados
    ]

def get_ticker_info() -> dict:
    """
    Retorna dicionário com informações de cada ticker.
    
    Returns:
        Dict com ticker como chave e dict com nome/setor como valor
    """
    return {
        f"{a['ticker']}.SA": {"nome": a["nome"], "setor": a["setor"]}
        for a in ATIVOS_B3
    }

def get_all_tickers() -> list:
    """
    Retorna todos os tickers no formato yfinance.
    
    Returns:
        Lista de todos os tickers com sufixo .SA
    """
    return [f"{a['ticker']}.SA" for a in ATIVOS_B3]
