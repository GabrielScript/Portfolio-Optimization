"""
assets.py - Lista completa das ações da B3
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel Estrela Lopes

Lista atualizada com as ações mais negociadas na B3.
"""

# Top 75 ativos mais líquidos para otimização rápida (Cloud)
ATIVOS_TOP75_TICKERS = [
    "CVCB3", "B3SA3", "RAIZ4", "CSAN3", "PETR4", "VALE3", "ABEV3", "ITUB4",
    "ITSA4", "COGN3", "BBAS3", "CBAV3", "BBDC4", "USIM5", "VAMO3", "ASAI3",
    "CMIG4", "MGLU3", "CPLE3", "LREN3", "PETR3", "ENEV3", "CSNA3", "GOAU4",
    "GGBR4", "BBDC3", "RAIL3", "SUZB3", "GMAT3", "RENT3", "PRIO3", "CMIN3",
    "POMO4", "HAPV3", "RADL3", "TIMS3", "VBBR3", "UGPA3", "RDOR3", "DIRR3",
    "PCAR3", "EQTL3", "BPAC11", "KLBN11", "MRVE3", "WEGE3", "VIVT3", "BRAV3",
    "ALOS3", "SMFT3",
    # Adicionados para TOP 75
    "JHSF3", "CEAB3", "BEEF3", "SIMH3", "ANIM3", "AURE3", "NEOE3", "MULT3",
    "HBSA3", "VIVA3", "CYRE3", "CXSE3", "DXCO3", "ECOR3", "TOTS3", "GGPS3",
    "BBSE3", "EMBR3", "SBSP3", "LWSA3", "MOVI3", "YDUQ3", "BRAP4", "CSMG3",
    "BRKM5"
]

# Lista completa de ativos da B3
ATIVOS_B3 = [
    # Top negociados
    {"ticker": "CVCB3", "nome": "CVC", "setor": "Consumo Cíclico"},
    {"ticker": "B3SA3", "nome": "B3", "setor": "Financeiro"},
    {"ticker": "RAIZ4", "nome": "Raízen", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "CSAN3", "nome": "Cosan", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "PETR4", "nome": "Petrobras", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "VALE3", "nome": "Vale", "setor": "Materiais Básicos"},
    {"ticker": "ABEV3", "nome": "Ambev", "setor": "Consumo não Cíclico"},
    {"ticker": "ITUB4", "nome": "Itaú Unibanco", "setor": "Financeiro"},
    {"ticker": "ITSA4", "nome": "Itaúsa", "setor": "Financeiro"},
    {"ticker": "COGN3", "nome": "Cogna", "setor": "Consumo Cíclico"},
    {"ticker": "BBAS3", "nome": "Banco do Brasil", "setor": "Financeiro"},
    {"ticker": "CBAV3", "nome": "CBA", "setor": "Materiais Básicos"},
    {"ticker": "BBDC4", "nome": "Banco Bradesco", "setor": "Financeiro"},
    {"ticker": "USIM5", "nome": "Usiminas", "setor": "Materiais Básicos"},
    {"ticker": "VAMO3", "nome": "Grupo Vamos", "setor": "Bens Industriais"},
    {"ticker": "ASAI3", "nome": "Assaí", "setor": "Consumo não Cíclico"},
    {"ticker": "CMIG4", "nome": "Cemig", "setor": "Utilidade Pública"},
    {"ticker": "MGLU3", "nome": "Magazine Luiza", "setor": "Consumo Cíclico"},
    {"ticker": "CPLE3", "nome": "Copel", "setor": "Utilidade Pública"},
    {"ticker": "LREN3", "nome": "Lojas Renner", "setor": "Consumo Cíclico"},
    {"ticker": "PETR3", "nome": "Petrobras", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "ENEV3", "nome": "Eneva", "setor": "Utilidade Pública"},
    {"ticker": "CSNA3", "nome": "Siderúrgica Nacional", "setor": "Materiais Básicos"},
    {"ticker": "GOAU4", "nome": "Metalúrgica Gerdau", "setor": "Materiais Básicos"},
    {"ticker": "GGBR4", "nome": "Gerdau", "setor": "Materiais Básicos"},
    {"ticker": "BBDC3", "nome": "Banco Bradesco", "setor": "Financeiro"},
    {"ticker": "RAIL3", "nome": "Rumo", "setor": "Bens Industriais"},
    {"ticker": "SUZB3", "nome": "Suzano", "setor": "Materiais Básicos"},
    {"ticker": "GMAT3", "nome": "Grupo Mateus", "setor": "Consumo não Cíclico"},
    {"ticker": "RENT3", "nome": "Localiza", "setor": "Consumo Cíclico"},
    {"ticker": "PRIO3", "nome": "PetroRio", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "CMIN3", "nome": "CSN Mineração", "setor": "Materiais Básicos"},
    {"ticker": "POMO4", "nome": "Marcopolo", "setor": "Bens Industriais"},
    {"ticker": "HAPV3", "nome": "Hapvida", "setor": "Saúde"},
    {"ticker": "RADL3", "nome": "RaiaDrogasil", "setor": "Saúde"},
    {"ticker": "JHSF3", "nome": "JHSF", "setor": "Consumo Cíclico"},
    {"ticker": "CEAB3", "nome": "C&A", "setor": "Consumo Cíclico"},
    {"ticker": "BEEF3", "nome": "Minerva", "setor": "Consumo não Cíclico"},
    {"ticker": "SIMH3", "nome": "Simpar", "setor": "Bens Industriais"},
    {"ticker": "TIMS3", "nome": "TIM", "setor": "Telecomunicações"},
    {"ticker": "ANIM3", "nome": "Ânima Educação", "setor": "Consumo Cíclico"},
    {"ticker": "VBBR3", "nome": "Vibra Energia", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "UGPA3", "nome": "Ultrapar", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "RDOR3", "nome": "Rede D'Or", "setor": "Saúde"},
    {"ticker": "DIRR3", "nome": "Direcional", "setor": "Consumo Cíclico"},
    {"ticker": "PCAR3", "nome": "Grupo Pão de Açúcar", "setor": "Consumo não Cíclico"},
    {"ticker": "EQTL3", "nome": "Equatorial Energia", "setor": "Utilidade Pública"},
    {"ticker": "BPAC11", "nome": "Banco BTG Pactual", "setor": "Financeiro"},
    {"ticker": "KLBN11", "nome": "Klabin", "setor": "Materiais Básicos"},
    {"ticker": "MRVE3", "nome": "MRV", "setor": "Consumo Cíclico"},
    {"ticker": "WEGE3", "nome": "WEG", "setor": "Bens Industriais"},
    {"ticker": "VIVT3", "nome": "Vivo", "setor": "Telecomunicações"},
    {"ticker": "BRAV3", "nome": "3R Petroleum", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "ALOS3", "nome": "Allos", "setor": "Financeiro"},
    {"ticker": "SMFT3", "nome": "Smart Fit", "setor": "Consumo Cíclico"},
    {"ticker": "AURE3", "nome": "Auren Energia", "setor": "Utilidade Pública"},
    {"ticker": "NEOE3", "nome": "Neoenergia", "setor": "Utilidade Pública"},
    {"ticker": "MULT3", "nome": "Multiplan", "setor": "Financeiro"},
    {"ticker": "HBSA3", "nome": "Hidrovias do Brasil", "setor": "Bens Industriais"},
    {"ticker": "VIVA3", "nome": "Vivara", "setor": "Consumo Cíclico"},
    {"ticker": "CYRE3", "nome": "Cyrela", "setor": "Consumo Cíclico"},
    {"ticker": "CXSE3", "nome": "Caixa Seguridade", "setor": "Financeiro"},
    {"ticker": "DXCO3", "nome": "Dexco", "setor": "Materiais Básicos"},
    {"ticker": "ECOR3", "nome": "EcoRodovias", "setor": "Bens Industriais"},
    {"ticker": "TOTS3", "nome": "Totvs", "setor": "Tecnologia da Informação"},
    {"ticker": "GGPS3", "nome": "GPS", "setor": "Bens Industriais"},
    {"ticker": "BBSE3", "nome": "BB Seguridade", "setor": "Financeiro"},
    {"ticker": "EMBR3", "nome": "Embraer", "setor": "Bens Industriais"},
    {"ticker": "SBSP3", "nome": "Sabesp", "setor": "Utilidade Pública"},
    {"ticker": "LWSA3", "nome": "Locaweb", "setor": "Tecnologia da Informação"},
    {"ticker": "MOVI3", "nome": "Movida", "setor": "Consumo Cíclico"},
    {"ticker": "PGMN3", "nome": "Pague Menos", "setor": "Saúde"},
    {"ticker": "YDUQ3", "nome": "YDUQS", "setor": "Consumo Cíclico"},
    {"ticker": "BRAP4", "nome": "Bradespar", "setor": "Financeiro"},
    {"ticker": "BHIA3", "nome": "Casas Bahia", "setor": "Consumo Cíclico"},
    {"ticker": "CSMG3", "nome": "COPASA", "setor": "Utilidade Pública"},
    {"ticker": "TEND3", "nome": "Construtora Tenda", "setor": "Consumo Cíclico"},
    {"ticker": "BRKM5", "nome": "Braskem", "setor": "Materiais Básicos"},
    {"ticker": "GRND3", "nome": "Grendene", "setor": "Consumo Cíclico"},
    {"ticker": "RECV3", "nome": "PetroRecôncavo", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "SANB11", "nome": "Banco Santander", "setor": "Financeiro"},
    {"ticker": "ISAE4", "nome": "ISA Energia", "setor": "Utilidade Pública"},
    {"ticker": "AZZA3", "nome": "Arezzo", "setor": "Consumo Cíclico"},
    {"ticker": "CURY3", "nome": "Cury", "setor": "Consumo Cíclico"},
    {"ticker": "CPFE3", "nome": "CPFL Energia", "setor": "Utilidade Pública"},
    {"ticker": "ONCO3", "nome": "Oncoclínicas", "setor": "Saúde"},
    {"ticker": "LJQQ3", "nome": "Lojas Quero-Quero", "setor": "Consumo Cíclico"},
    {"ticker": "SLCE3", "nome": "SLC Agrícola", "setor": "Consumo não Cíclico"},
    {"ticker": "SEER3", "nome": "Ser Educacional", "setor": "Consumo Cíclico"},
    {"ticker": "PLPL3", "nome": "Plano&Plano", "setor": "Consumo Cíclico"},
    {"ticker": "SMTO3", "nome": "São Martinho", "setor": "Consumo não Cíclico"},
    {"ticker": "ALPA4", "nome": "Alpargatas", "setor": "Consumo Cíclico"},
    {"ticker": "PSSA3", "nome": "Porto Seguro", "setor": "Financeiro"},
    {"ticker": "FLRY3", "nome": "Fleury", "setor": "Saúde"},
    {"ticker": "TAEE11", "nome": "Taesa", "setor": "Utilidade Pública"},
    {"ticker": "BRSR6", "nome": "Banrisul", "setor": "Financeiro"},
    {"ticker": "ODPV3", "nome": "Odontoprev", "setor": "Saúde"},
    {"ticker": "SBFG3", "nome": "Grupo SBF", "setor": "Consumo Cíclico"},
    {"ticker": "AMER3", "nome": "Americanas", "setor": "Consumo Cíclico"},
    {"ticker": "IGTI11", "nome": "Iguatemi", "setor": "Financeiro"},
    {"ticker": "JALL3", "nome": "Jalles Machado", "setor": "Consumo não Cíclico"},
    {"ticker": "HYPE3", "nome": "Hypera", "setor": "Saúde"},
    {"ticker": "GUAR3", "nome": "Guararapes", "setor": "Consumo Cíclico"},
    {"ticker": "ENGI11", "nome": "Energisa", "setor": "Utilidade Pública"},
    {"ticker": "QUAL3", "nome": "Qualicorp", "setor": "Saúde"},
    {"ticker": "EGIE3", "nome": "Engie", "setor": "Utilidade Pública"},
    {"ticker": "SAPR11", "nome": "Sanepar", "setor": "Utilidade Pública"},
    {"ticker": "CASH3", "nome": "Méliuz", "setor": "Tecnologia da Informação"},
    {"ticker": "VULC3", "nome": "Vulcabras", "setor": "Consumo Cíclico"},
    {"ticker": "EZTC3", "nome": "EZTEC", "setor": "Consumo Cíclico"},
    {"ticker": "TTEN3", "nome": "3tentos", "setor": "Consumo não Cíclico"},
    {"ticker": "INTB3", "nome": "Intelbras", "setor": "Tecnologia da Informação"},
    {"ticker": "DASA3", "nome": "Dasa", "setor": "Saúde"},
    {"ticker": "LIGT3", "nome": "Light", "setor": "Utilidade Pública"},
    {"ticker": "POSI3", "nome": "Positivo", "setor": "Tecnologia da Informação"},
    {"ticker": "ABCB4", "nome": "Banco ABC Brasil", "setor": "Financeiro"},
    {"ticker": "DESK3", "nome": "Desktop", "setor": "Tecnologia da Informação"},
    {"ticker": "KEPL3", "nome": "Kepler Weber", "setor": "Bens Industriais"},
    {"ticker": "BLAU3", "nome": "Blau Farmacêutica", "setor": "Saúde"},
    {"ticker": "ALLD3", "nome": "Allied", "setor": "Consumo Cíclico"},
    {"ticker": "MYPK3", "nome": "Iochpe-Maxion", "setor": "Bens Industriais"},
    {"ticker": "TUPY3", "nome": "Tupy", "setor": "Bens Industriais"},
    {"ticker": "BMGB4", "nome": "Banco BMG", "setor": "Financeiro"},
    {"ticker": "CAML3", "nome": "Camil Alimentos", "setor": "Consumo não Cíclico"},
    {"ticker": "POMO3", "nome": "Marcopolo", "setor": "Bens Industriais"},
    {"ticker": "CSED3", "nome": "Cruzeiro do Sul Educacional", "setor": "Consumo Cíclico"},
    {"ticker": "ALUP11", "nome": "Alupar", "setor": "Utilidade Pública"},
    {"ticker": "RANI3", "nome": "Irani", "setor": "Materiais Básicos"},
    {"ticker": "EVEN3", "nome": "Even", "setor": "Consumo Cíclico"},
    {"ticker": "WIZC3", "nome": "Wiz Soluções", "setor": "Financeiro"},
    {"ticker": "HBRE3", "nome": "HBR Realty", "setor": "Consumo Cíclico"},
    {"ticker": "OPCT3", "nome": "OceanPact", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "ARML3", "nome": "Armac", "setor": "Bens Industriais"},
    {"ticker": "JSLG3", "nome": "JSL", "setor": "Bens Industriais"},
    {"ticker": "GFSA3", "nome": "Gafisa", "setor": "Consumo Cíclico"},
    {"ticker": "TFCO4", "nome": "Track & Field", "setor": "Consumo Cíclico"},
    {"ticker": "ORVR3", "nome": "Orizon", "setor": "Utilidade Pública"},
    {"ticker": "SOJA3", "nome": "Boa Safra Sementes", "setor": "Consumo não Cíclico"},
    {"ticker": "IRBR3", "nome": "IRB Brasil RE", "setor": "Financeiro"},
    {"ticker": "MILS3", "nome": "Mills", "setor": "Bens Industriais"},
    {"ticker": "BRST3", "nome": "Brisanet", "setor": "Telecomunicações"},
    {"ticker": "PNVL3", "nome": "Dimed", "setor": "Saúde"},
    {"ticker": "LAVV3", "nome": "Lavvi Incorporadora", "setor": "Consumo Cíclico"},
    {"ticker": "PRNR3", "nome": "Priner", "setor": "Bens Industriais"},
    {"ticker": "MDIA3", "nome": "M. Dias Branco", "setor": "Consumo não Cíclico"},
    {"ticker": "MTRE3", "nome": "Mitre Realty", "setor": "Consumo Cíclico"},
    {"ticker": "HBOR3", "nome": "Helbor", "setor": "Consumo Cíclico"},
    {"ticker": "TASA4", "nome": "Taurus", "setor": "Bens Industriais"},
    {"ticker": "FRAS3", "nome": "Fras-le", "setor": "Bens Industriais"},
    {"ticker": "SHUL4", "nome": "Schulz", "setor": "Bens Industriais"},
    {"ticker": "ENJU3", "nome": "Enjoei", "setor": "Consumo Cíclico"},
    {"ticker": "BMOB3", "nome": "Bemobi", "setor": "Tecnologia da Informação"},
    {"ticker": "LEVE3", "nome": "Mahle Metal Leve", "setor": "Bens Industriais"},
    {"ticker": "LOGG3", "nome": "LOG CP", "setor": "Consumo Cíclico"},
    {"ticker": "FIQE3", "nome": "Unifique", "setor": "Telecomunicações"},
    {"ticker": "MELK3", "nome": "Melnick", "setor": "Consumo Cíclico"},
    {"ticker": "BRBI11", "nome": "BR Partners", "setor": "Financeiro"},
    {"ticker": "ROMI3", "nome": "Indústrias ROMI", "setor": "Bens Industriais"},
    {"ticker": "AGRO3", "nome": "BrasilAgro", "setor": "Consumo não Cíclico"},
    {"ticker": "ALPK3", "nome": "Estapar", "setor": "Bens Industriais"},
    {"ticker": "UNIP6", "nome": "Unipar", "setor": "Materiais Básicos"},
    {"ticker": "TRIS3", "nome": "Trisul", "setor": "Consumo Cíclico"},
    {"ticker": "VLID3", "nome": "Valid", "setor": "Bens Industriais"},
    {"ticker": "PTBL3", "nome": "Portobello", "setor": "Consumo Cíclico"},
    {"ticker": "PFRM3", "nome": "Profarma", "setor": "Saúde"},
    {"ticker": "TGMA3", "nome": "Tegma", "setor": "Bens Industriais"},
    {"ticker": "VITT3", "nome": "Vittia", "setor": "Consumo não Cíclico"},
    {"ticker": "LPSB3", "nome": "Lopes", "setor": "Consumo Cíclico"},
    {"ticker": "CSUD3", "nome": "CSU Cardsystem", "setor": "Tecnologia da Informação"},
    {"ticker": "TCSA3", "nome": "Tecnisa", "setor": "Consumo Cíclico"},
    {"ticker": "MATD3", "nome": "Mater Dei", "setor": "Saúde"},
    {"ticker": "AERI3", "nome": "Aeris Energy", "setor": "Bens Industriais"},
    {"ticker": "SEQL3", "nome": "Sequoia Logística", "setor": "Bens Industriais"},
    {"ticker": "SCAR3", "nome": "São Carlos", "setor": "Financeiro"},
    {"ticker": "ETER3", "nome": "Eternit", "setor": "Materiais Básicos"},
    {"ticker": "AGXY3", "nome": "AgroGalaxy", "setor": "Consumo não Cíclico"},
    {"ticker": "DMVF3", "nome": "D1000 Varejo Farma", "setor": "Saúde"},
    {"ticker": "SHOW3", "nome": "Time For Fun", "setor": "Consumo Cíclico"},
    {"ticker": "LOGN3", "nome": "Log-In", "setor": "Bens Industriais"},
    {"ticker": "TECN3", "nome": "Technos", "setor": "Consumo Cíclico"},
    {"ticker": "CCRO3", "nome": "CCR", "setor": "Bens Industriais"},
    {"ticker": "JBSS3", "nome": "JBS", "setor": "Consumo não Cíclico"},
    {"ticker": "BRFS3", "nome": "BRF", "setor": "Consumo não Cíclico"},
    {"ticker": "NTCO3", "nome": "Natura", "setor": "Consumo não Cíclico"},
    {"ticker": "AZUL4", "nome": "Azul", "setor": "Bens Industriais"},
    {"ticker": "GOLL4", "nome": "Gol", "setor": "Bens Industriais"},
    {"ticker": "ELET3", "nome": "Eletrobras ON", "setor": "Utilidade Pública"},
    {"ticker": "ELET6", "nome": "Eletrobras PNB", "setor": "Utilidade Pública"},
    {"ticker": "BPAN4", "nome": "Banco Pan", "setor": "Financeiro"},
    {"ticker": "BNBR3", "nome": "Banco do Nordeste", "setor": "Financeiro"},
    {"ticker": "TRPL4", "nome": "Transmissão Paulista", "setor": "Utilidade Pública"},
    {"ticker": "CPLE6", "nome": "Copel PNB", "setor": "Utilidade Pública"},
    {"ticker": "COCE5", "nome": "Coelce", "setor": "Utilidade Pública"},
    {"ticker": "MRFG3", "nome": "Marfrig", "setor": "Consumo não Cíclico"},
    {"ticker": "CRFB3", "nome": "Carrefour Brasil", "setor": "Consumo não Cíclico"},
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

def get_top75_tickers() -> list:
    """
    Retorna os 75 tickers mais líquidos (otimizado para Cloud).
    
    Returns:
        Lista dos 75 principais tickers com sufixo .SA
    """
    return [f"{t}.SA" for t in ATIVOS_TOP75_TICKERS]
