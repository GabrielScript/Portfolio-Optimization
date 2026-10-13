"""
assets.py - Universo de ativos (B3, cripto, startups)
TCC: Otimização de Carteiras de Investimentos
Autor: Gabriel Estrela Lopes

Revisão 2026-10 (tickers checados no Yahoo):
- Renomeados: CCRO3->MOTV3, EMBR3->EMBJ3, ELET3->AXIA3, NTCO3->NATU3,
  MRFG3+BRFS3->MBRF3, JBSS3->JBSS32 (BDR).
- Removidos sem dados: AZUL4, GOLL4, BPAN4, CRFB3, NEOE3, ODPV3, GUAR3, ELET6,
  CPLE6. TRPL4 era duplicata de ISAE4.
- Uma classe por emissor (PETR3, BBDC3 e POMO3 removidas): duas classes da
  mesma empresa têm correlação ~0,95+, o que inflava a concentração disfarçada
  e mal-condicionava Sigma.

VIÉS DE SOBREVIVÊNCIA: esta é a lista de HOJE. Empresas que quebraram ou
fecharam capital no período não estão aqui (AZUL4/GOLL4 acabaram de sair).
O backtest herda esse viés positivo. A correção exige composição histórica
point-in-time (ex.: carteiras teóricas do IBrX por quadrimestre).
"""

# Ações mais líquidas (otimização rápida). Holdings e controladas (ITSA4/ITUB4,
# BRAP4/VALE3, GOAU4/GGBR4) são empresas distintas e ficam.
ATIVOS_LIQUIDOS_TICKERS = [
    "CVCB3", "B3SA3", "RAIZ4", "CSAN3", "PETR4", "VALE3", "ABEV3", "ITUB4",
    "ITSA4", "COGN3", "BBAS3", "CBAV3", "BBDC4", "USIM5", "VAMO3", "ASAI3",
    "CMIG4", "MGLU3", "CPLE3", "LREN3", "ENEV3", "CSNA3", "GOAU4",
    "GGBR4", "RAIL3", "SUZB3", "GMAT3", "RENT3", "PRIO3", "CMIN3",
    "POMO4", "HAPV3", "RADL3", "TIMS3", "VBBR3", "UGPA3", "RDOR3", "DIRR3",
    "PCAR3", "EQTL3", "BPAC11", "KLBN11", "MRVE3", "WEGE3", "VIVT3", "BRAV3",
    "ALOS3", "SMFT3", "JHSF3", "CEAB3", "BEEF3", "SIMH3", "ANIM3", "AURE3",
    "MULT3", "HBSA3", "VIVA3", "CYRE3", "CXSE3", "DXCO3", "ECOR3", "TOTS3",
    "GGPS3", "BBSE3", "EMBJ3", "SBSP3", "LWSA3", "MOVI3", "YDUQ3", "BRAP4",
    "CSMG3", "BRKM5", "AXIA3", "MOTV3",
]

# Cripto cotada em USD no Yahoo: convertida para BRL e reamostrada no
# calendário B3 (o fim de semana entra no retorno de segunda).
CRIPTO = [
    {"ticker": "BTC-USD", "nome": "Bitcoin", "setor": "Cripto"},
    {"ticker": "ETH-USD", "nome": "Ethereum", "setor": "Cripto"},
]

BENCHMARK_ACOES = "BOVA11.SA"   # ETF do Ibovespa: investível e de retorno total (ao contrário do ^BVSP)

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
    {"ticker": "ENEV3", "nome": "Eneva", "setor": "Utilidade Pública"},
    {"ticker": "CSNA3", "nome": "Siderúrgica Nacional", "setor": "Materiais Básicos"},
    {"ticker": "GOAU4", "nome": "Metalúrgica Gerdau", "setor": "Materiais Básicos"},
    {"ticker": "GGBR4", "nome": "Gerdau", "setor": "Materiais Básicos"},
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
    {"ticker": "EMBJ3", "nome": "Embraer", "setor": "Bens Industriais"},
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
    {"ticker": "SBFG3", "nome": "Grupo SBF", "setor": "Consumo Cíclico"},
    {"ticker": "AMER3", "nome": "Americanas", "setor": "Consumo Cíclico"},
    {"ticker": "IGTI11", "nome": "Iguatemi", "setor": "Financeiro"},
    {"ticker": "JALL3", "nome": "Jalles Machado", "setor": "Consumo não Cíclico"},
    {"ticker": "HYPE3", "nome": "Hypera", "setor": "Saúde"},
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
    {"ticker": "MOTV3", "nome": "Motiva (ex-CCR)", "setor": "Bens Industriais"},
    {"ticker": "JBSS32", "nome": "JBS (BDR)", "setor": "Consumo não Cíclico"},
    {"ticker": "NATU3", "nome": "Natura", "setor": "Consumo não Cíclico"},
    {"ticker": "AXIA3", "nome": "Axia Energia (ex-Eletrobras)", "setor": "Utilidade Pública"},
    {"ticker": "BNBR3", "nome": "Banco do Nordeste", "setor": "Financeiro"},
    {"ticker": "COCE5", "nome": "Coelce", "setor": "Utilidade Pública"},
    {"ticker": "MBRF3", "nome": "MBRF (Marfrig + BRF)", "setor": "Consumo não Cíclico"},
]

# Startups / IPOs recentes listadas na B3 (2020-2021) — máx ~5 anos de história
STARTUPS_BR = [
    {"ticker": "CASH3.SA", "nome": "Méliuz", "setor": "Tecnologia da Informação"},
    {"ticker": "PETZ3.SA", "nome": "Petz", "setor": "Consumo Cíclico"},
    {"ticker": "ENJU3.SA", "nome": "Enjoei", "setor": "Consumo Cíclico"},
    {"ticker": "INTB3.SA", "nome": "Intelbras", "setor": "Tecnologia da Informação"},
    {"ticker": "TTEN3.SA", "nome": "3tentos", "setor": "Consumo não Cíclico"},
    {"ticker": "GGPS3.SA", "nome": "GPS", "setor": "Bens Industriais"},
    {"ticker": "SOJA3.SA", "nome": "Boa Safra Sementes", "setor": "Consumo não Cíclico"},
    {"ticker": "VITT3.SA", "nome": "Vittia", "setor": "Consumo não Cíclico"},
    {"ticker": "ARML3.SA", "nome": "Armac", "setor": "Bens Industriais"},
    {"ticker": "BRBI11.SA", "nome": "BR Partners", "setor": "Financeiro"},
    {"ticker": "MLAS3.SA", "nome": "Multilaser", "setor": "Tecnologia da Informação"},
    {"ticker": "RECV3.SA", "nome": "PetroRecôncavo", "setor": "Petróleo, Gás e Biocombustíveis"},
    {"ticker": "ONCO3.SA", "nome": "Oncoclínicas", "setor": "Saúde"},
    {"ticker": "MATD3.SA", "nome": "Mater Dei", "setor": "Saúde"},
    {"ticker": "DESK3.SA", "nome": "Desktop", "setor": "Tecnologia da Informação"},
    {"ticker": "FIQE3.SA", "nome": "Unifique", "setor": "Telecomunicações"},
]

# Startups listadas nos EUA (Nasdaq/NYSE) — IPOs 2020-2024, cotadas em USD
# (inclui startups brasileiras listadas lá fora: NU, INTR, VTEX)
STARTUPS_US = [
    {"ticker": "NU", "nome": "Nubank", "setor": "Financeiro"},
    {"ticker": "INTR", "nome": "Inter & Co", "setor": "Financeiro"},
    {"ticker": "VTEX", "nome": "VTEX", "setor": "Tecnologia da Informação"},
    {"ticker": "PLTR", "nome": "Palantir", "setor": "Tecnologia da Informação"},
    {"ticker": "ABNB", "nome": "Airbnb", "setor": "Consumo Cíclico"},
    {"ticker": "COIN", "nome": "Coinbase", "setor": "Financeiro"},
    {"ticker": "RBLX", "nome": "Roblox", "setor": "Tecnologia da Informação"},
    {"ticker": "HOOD", "nome": "Robinhood", "setor": "Financeiro"},
    {"ticker": "SOFI", "nome": "SoFi", "setor": "Financeiro"},
    {"ticker": "RIVN", "nome": "Rivian", "setor": "Bens Industriais"},
    {"ticker": "IONQ", "nome": "IonQ", "setor": "Tecnologia da Informação"},
    {"ticker": "ARM", "nome": "Arm Holdings", "setor": "Tecnologia da Informação"},
    {"ticker": "CAVA", "nome": "CAVA Group", "setor": "Consumo Cíclico"},
    {"ticker": "KVYO", "nome": "Klaviyo", "setor": "Tecnologia da Informação"},
    {"ticker": "RDDT", "nome": "Reddit", "setor": "Tecnologia da Informação"},
    {"ticker": "TEM", "nome": "Tempus AI", "setor": "Saúde"},
]

def get_startup_tickers() -> list:
    """Tickers de startups (BR + US) no formato yfinance."""
    return [s["ticker"] for s in STARTUPS_BR + STARTUPS_US]

def get_startup_info() -> dict:
    """Dict ticker -> {nome, setor, mercado} para startups BR e US."""
    info = {}
    for s in STARTUPS_BR:
        info[s["ticker"]] = {**s, "mercado": "B3"}
    for s in STARTUPS_US:
        info[s["ticker"]] = {**s, "mercado": "US"}
    return info

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
    """Dict ticker (formato Yahoo) -> {nome, setor}, para ações B3 e cripto."""
    info = {f"{a['ticker']}.SA": {"nome": a["nome"], "setor": a["setor"]} for a in ATIVOS_B3}
    info.update({c["ticker"]: {"nome": c["nome"], "setor": c["setor"]} for c in CRIPTO})
    return info


def get_cripto_tickers() -> list:
    return [c["ticker"] for c in CRIPTO]

def get_all_tickers() -> list:
    """
    Retorna todos os tickers no formato yfinance.
    
    Returns:
        Lista de todos os tickers com sufixo .SA
    """
    return [f"{a['ticker']}.SA" for a in ATIVOS_B3]

def get_tickers_liquidos() -> list:
    """Tickers da lista líquida, com sufixo .SA."""
    return [f"{t}.SA" for t in ATIVOS_LIQUIDOS_TICKERS]
