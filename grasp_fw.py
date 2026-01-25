import pandas as pd
import random
import math
import heapq
import time
from collections import defaultdict
import folium
from folium import Map, PolyLine, CircleMarker, LayerControl
import numpy as np
import re
from pathlib import Path
import shutil

def limpar_pastas_saida():
    pastas = ["rotas", "resultados_xls", "mapas"]
    for pasta in pastas:
        path = Path(pasta)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        path.mkdir()
    print("🧹 Pastas de saída limpas com sucesso!")

# =============================================================
# IDENTIFICAR INSTÂNCIA
# =============================================================

def extrair_gp(caminho: str) -> str:
    stem = Path(caminho).stem
    m = re.search(r'(gp\d+)', stem, re.IGNORECASE)
    return m.group(1).lower() if m else "instancia"

# =============================================================
# FLOYD–WARSHALL BINÁRIO
# =============================================================

FW_NODES = None
FW_IDX = None
FW_DIST = None
FW_NEXT = None

def fw_prefix_from_arquivo(caminho: str) -> str:
    stem = Path(caminho).stem
    m = re.search(r'(gp\d+)', stem, re.IGNORECASE)
    if m:
        return f"fw_{m.group(1).lower()}"
    return f"fw_{stem}"

def carregar_fw(prefix):
    global FW_NODES, FW_IDX, FW_DIST, FW_NEXT

    nodes = []
    with open(f"{prefix}_nodes.txt", "r", encoding="utf-8") as f:
        for line in f:
            _, node = line.strip().split("\t")
            nodes.append(str(node))

    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}

    dist = np.memmap(f"{prefix}_dist.bin", dtype=np.float32, mode="r", shape=(n, n))
    nxt  = np.memmap(f"{prefix}_next.bin", dtype=np.int32,  mode="r", shape=(n, n))

    FW_NODES = nodes
    FW_IDX = idx
    FW_DIST = dist
    FW_NEXT = nxt

def fw_dist(u, v):
    if FW_IDX is None or u not in FW_IDX or v not in FW_IDX:
        return float("inf")
    d = float(FW_DIST[FW_IDX[u], FW_IDX[v]])
    if d >= 1e30:
        return float("inf")
    return d

def fw_caminho(u, v):
    if FW_IDX is None or u not in FW_IDX or v not in FW_IDX:
        return []

    i = FW_IDX[u]
    j = FW_IDX[v]

    if i == j:
        return [u]

    if FW_NEXT[i, j] < 0:
        return []

    caminho = [u]
    max_passos = len(FW_NODES) + 5
    passos = 0

    while i != j and passos < max_passos:
        i = FW_NEXT[i, j]
        caminho.append(FW_NODES[i])
        passos += 1

    if i != j:
        return []

    return caminho

def fw_deslocamento(u, v):
    caminho = fw_caminho(u, v)
    if not caminho or len(caminho) < 2:
        return []
    return [(a, b, False) for a, b in zip(caminho[:-1], caminho[1:])]

# =============================================================
# LEITURA DO ARQUIVO
# =============================================================

def ler_arestas(caminho):
    colunas = ["id1","lat1","lon1","id2","lat2","lon2","comprimento_m","residuo"]
    return pd.read_csv(
        caminho,
        sep="\t",
        header=None,
        names=colunas,
        dtype={"id1": str, "id2": str}
    )

def construir_grafo(df):
    grafo = defaultdict(list)
    demandas = {}

    for _, row in df.iterrows():
        u = row["id1"]
        v = row["id2"]
        if u == v:
            continue

        dist = float(row["comprimento_m"])
        q = float(row["residuo"])

        grafo[u].append((v, dist))
        grafo[v].append((u, dist))

        if q > 0:
            demandas[(u, v)] = q

    return grafo, demandas

def custo_aresta(grafo, u, v):
    return next((c for x, c in grafo[u] if x == v), float("inf"))

# =============================================================
# MELHORIA LOCAL: SWAP
# =============================================================

def custo_rota(rota, grafo):
    return sum(custo_aresta(grafo, u, v) for (u, v, _) in rota)

def existe_aresta_direta(grafo, u, v):
    return any(viz == v for viz, _ in grafo[u])

def construir_rota_conectada(ordem_coletas, grafo):
    rota = []
    if not ordem_coletas:
        return rota

    u0, v0 = ordem_coletas[0]
    rota.append((u0, v0, True))
    atual = v0

    for (u, v) in ordem_coletas[1:]:
        if atual == u or existe_aresta_direta(grafo, atual, u):
            if atual != u:
                rota.append((atual, u, False))
        else:
            desloc = fw_deslocamento(atual, u)
            if not desloc:
                return []
            rota.extend(desloc)

        rota.append((u, v, True))
        atual = v

    return rota

def swap_simples(grafo, rota, pos1, pos2):
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    if pos1 >= len(coletas) or pos2 >= len(coletas) or pos1 == pos2:
        return None

    nova_ordem = coletas.copy()
    nova_ordem[pos1], nova_ordem[pos2] = nova_ordem[pos2], nova_ordem[pos1]
    return construir_rota_conectada(nova_ordem, grafo)

def aplicar_swap_first_random(grafo, rota):
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    custo_atual = custo_rota(rota, grafo)

    for _ in range(Config.NUM_TENTATIVAS_SWAP):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i == j:
            continue

        nova_rota = swap_simples(grafo, rota, i, j)
        if not nova_rota:
            continue

        custo_novo = custo_rota(nova_rota, grafo)
        if custo_novo < custo_atual:
            return nova_rota

    return rota

def aplicar_swap_best(grafo, rota):
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    custo_atual = custo_rota(rota, grafo)

    melhor_rota = None
    melhor_ganho = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            nova_rota = swap_simples(grafo, rota, i, j)
            if not nova_rota:
                continue

            ganho = custo_atual - custo_rota(nova_rota, grafo)
            if ganho > melhor_ganho:
                melhor_ganho = ganho
                melhor_rota = nova_rota

    return melhor_rota if melhor_rota else rota

def aplicar_swap_best_first(grafo, rota):
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    custo_atual = custo_rota(rota, grafo)

    for i in range(n - 1):
        for j in range(i + 1, n):
            nova_rota = swap_simples(grafo, rota, i, j)
            if not nova_rota:
                continue

            if custo_rota(nova_rota, grafo) < custo_atual:
                return nova_rota

    return rota

# =============================================================
# SIMULATED ANNEALING
# =============================================================

def aplicar_simulated_annealing(grafo, rota):
    T = Config.SA_TEMPERATURA_INICIAL
    rota_atual = rota
    custo_atual = custo_rota(rota, grafo)

    melhor_rota = rota
    melhor_custo = custo_atual

    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    if n < 2:
        return rota
    while T > Config.SA_TEMPERATURA_MIN:
        for _ in range(Config.SA_ITERACOES_POR_TEMPERATURA):
            i, j = random.sample(range(n), 2)
            nova_rota = swap_simples(grafo, rota_atual, i, j)
            if not nova_rota:
                continue

            custo_novo = custo_rota(nova_rota, grafo)
            delta = custo_novo - custo_atual

            if delta < 0 or random.random() < math.exp(-delta / T):
                rota_atual = nova_rota
                custo_atual = custo_novo
                if custo_novo < melhor_custo:
                    melhor_custo = custo_novo
                    melhor_rota = nova_rota

        T *= Config.SA_ALPHA

    return melhor_rota

# =============================================================
# GRASP + FW
# =============================================================

def construir_solucao_grasp(grafo, demandas, alfa, capacidade):
    arcos_restantes = set(demandas.keys())
    rotas = []

    while arcos_restantes:
        rota = []
        carga = 0

        candidatos = [(custo_aresta(grafo, u, v), (u, v))
                      for (u, v) in arcos_restantes
                      if demandas[(u, v)] <= capacidade]

        if not candidatos:
            break

        candidatos.sort()
        _, arco = random.choice(candidatos[:max(1, int(len(candidatos) * alfa))])

        u, v = arco
        rota.append((u, v, True))
        carga += demandas[arco]
        arcos_restantes.remove(arco)
        atual = v

        while True:
            vizinhos = [(c, (atual, w)) for w, c in grafo[atual]
                        if (atual, w) in arcos_restantes and carga + demandas[(atual, w)] <= capacidade]

            if vizinhos:
                _, arco = random.choice(vizinhos[:max(1, int(len(vizinhos) * alfa))])
                rota.append((arco[0], arco[1], True))
                carga += demandas[arco]
                arcos_restantes.remove(arco)
                atual = arco[1]
                continue

            melhor = None
            for arco in arcos_restantes:
                if carga + demandas[arco] <= capacidade:
                    d = fw_dist(atual, arco[0])
                    if melhor is None or d < melhor[0]:
                        melhor = (d, arco)

            if melhor and melhor[0] < float("inf"):
                desloc = fw_deslocamento(atual, melhor[1][0])
                rota.extend(desloc)
                rota.append((melhor[1][0], melhor[1][1], True))
                carga += demandas[melhor[1]]
                arcos_restantes.remove(melhor[1])
                atual = melhor[1][1]
            else:
                break

        custo_antigo = custo_rota(rota, grafo)

        if Config.METODO_MELHORIA == "best":
            rota = aplicar_swap_best(grafo, rota)
        elif Config.METODO_MELHORIA == "first_rand":
            rota = aplicar_swap_first_random(grafo, rota)
        elif Config.METODO_MELHORIA == "best_first":
            rota = aplicar_swap_best_first(grafo, rota)
        elif Config.METODO_MELHORIA == "annealing":
            rota = aplicar_simulated_annealing(grafo, rota)

        rotas.append((rota, carga))

    return rotas

# =============================================================
# SAÍDAS
# =============================================================

def salvar_rotas_txt(rotas, grafo, gp, tempo_total=0):
    """
    Salva rotas GRASP formatadas como texto em rotas/rotas_grasp_gpX.txt
    """
    from pathlib import Path

    Path("rotas").mkdir(exist_ok=True)
    caminho_txt = Path("rotas") / f"{Config.TAG}_rotas_grasp_{gp}.txt"

    linhas = []
    custo_total = 0.0

    for i, (rota, carga) in enumerate(rotas, start=1):
        linhas.append(f"Rota {i}:")
        rota_custo = 0.0
        for u, v, coleta in rota:
            dist = custo_aresta(grafo, u, v)
            tipo = "COLETA" if coleta else "DESLOC."
            linhas.append(f"  ({u} -> {v})  dist: {dist:.2f} m  [{tipo}]")
            rota_custo += dist
        linhas.append(f"  Carga: {carga:.2f} kg")
        linhas.append(f"  Custo da rota: {rota_custo:.2f} m")
        linhas.append("-" * 40)
        custo_total += rota_custo

    linhas.append(f"Total de rotas: {len(rotas)}")
    linhas.append(f"Custo total: {custo_total:.2f} m")
    linhas.append(f"Tempo total: {tempo_total:.2f} s")

    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))


def gerar_mapas(df_arestas, melhor_solucao, gp):
    """
    Gera arquivos HTML de mapa na pasta mapas/ com base nas rotas.
    """
    from pathlib import Path

    Path("mapas").mkdir(exist_ok=True)
    prefixo = Path("mapas") / f"{Config.TAG}_mapa_grasp_{gp}"

    coord_por_aresta = {
        frozenset((row["id1"], row["id2"])):
        ((row["lat1"], row["lon1"]), (row["lat2"], row["lon2"]))
        for _, row in df_arestas.iterrows()
    }

    centro_lat = (df_arestas["lat1"].mean() + df_arestas["lat2"].mean()) / 2
    centro_lon = (df_arestas["lon1"].mean() + df_arestas["lon2"].mean()) / 2

    def cor_aleatoria():
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def desenhar_segmento(mapa, u, v, coleta, cor, rota_idx, carga):
        p1, p2 = coord_por_aresta.get(frozenset((u, v)), (None, None))
        if not (p1 and p2):
            return
        if coleta:
            PolyLine([p1, p2], color=cor, weight=5,
                     popup=f"Rota {rota_idx} (coleta) - {carga:.1f} kg").add_to(mapa)
        else:
            PolyLine([p1, p2], color=cor, weight=2, dash_array="6,6",
                     popup=f"Rota {rota_idx} (desloc.)").add_to(mapa)

    mapa_total = Map(location=[centro_lat, centro_lon], zoom_start=14)
    for idx, (rota, carga) in enumerate(melhor_solucao, start=1):
        cor = cor_aleatoria()
        for j, (u, v, coleta) in enumerate(rota):
            desenhar_segmento(mapa_total, u, v, coleta, cor, idx, carga)
            if j == 0:
                p1, _ = coord_por_aresta.get(frozenset((u, v)), (None, None))
                if p1:
                    CircleMarker(location=p1, radius=4, color=cor, fill=True).add_to(mapa_total)
    LayerControl().add_to(mapa_total)
    mapa_total.save(f"{prefixo}_completo.html")

    for idx, (rota, carga) in enumerate(melhor_solucao, start=1):
        mapa_individual = Map(location=[centro_lat, centro_lon], zoom_start=14)
        cor = cor_aleatoria()
        for j, (u, v, coleta) in enumerate(rota):
            desenhar_segmento(mapa_individual, u, v, coleta, cor, idx, carga)
            if j == 0:
                p1, _ = coord_por_aresta.get(frozenset((u, v)), (None, None))
                if p1:
                    CircleMarker(location=p1, radius=4, color=cor, fill=True).add_to(mapa_individual)
        LayerControl().add_to(mapa_individual)
        mapa_individual.save(f"{prefixo}_rota{idx}.html")

def salvar_xls(rotas, grafo, gp, tempo):
    Path("resultados_xls").mkdir(exist_ok=True)
    dados = []
    for i, (rota, carga) in enumerate(rotas, 1):
        dados.append({
            "instancia": gp,
            "rota": i,
            "carga": carga,
            "distancia_m": sum(custo_aresta(grafo,u,v) for u,v,_ in rota),
            "tempo_s": tempo
        })
    tag = getattr(Config, "TAG", "c0_default")
    pd.DataFrame(dados).to_excel(f"resultados_xls/{tag}_resultado_grasp_{gp}.xlsx", index=False)

# =============================================================
# EXECUÇÃO
# =============================================================

def executar_grasp():
    inicio = time.time()
    gp = extrair_gp(Config.CAMINHO_ARQUIVO)

    df = ler_arestas(Config.CAMINHO_ARQUIVO)
    grafo, demandas = construir_grafo(df)

    carregar_fw("fw/" + fw_prefix_from_arquivo(Config.CAMINHO_ARQUIVO))

    melhor = construir_solucao_grasp(
        grafo, demandas, Config.ALFA, Config.CAPACIDADE_CAMINHAO
    )

    tempo = time.time() - inicio
    
    exibir_cargas_rotas(melhor, Config.CAPACIDADE_CAMINHAO)
    
    exibir_distancias_rotas(melhor, grafo)
    if Config.GERAR_HTML:
        gerar_mapas(df, melhor,gp) 
    if Config.GERAR_TXT:
        salvar_rotas_txt(melhor, grafo, gp, tempo)
    salvar_xls(melhor, grafo, gp, tempo)

def executar_grasp_para_todos():
    for i in range(1, 8):
        Config.CAMINHO_ARQUIVO = f"aresta_residuo/arestas_residuo_gp{i}.txt"
        executar_grasp()
        
def exibir_cargas_rotas(rotas, capacidade):
    print("\n📦 Utilização de carga por rota:")
    for i, (_, carga) in enumerate(rotas, start=1):
        perc = (carga / capacidade) * 100 if capacidade > 0 else 0
        print(f"   ↳ Rota {i}: {carga/1000:.2f} t ({perc:.1f}% da capacidade)")

def exibir_distancias_rotas(rotas, grafo):
    print("\n📏 Distâncias percorridas por rota:")
    total = 0.0
    for i, (rota, _) in enumerate(rotas, start=1):
        d = sum(custo_aresta(grafo, u, v) for (u, v, _) in rota)
        print(f"   ↳ Rota {i}: {d:,.2f} m")
        total += d
    print(f"📍 Distância total percorrida: {total:,.2f} m")
# =============================================================
# CONFIG
# =============================================================

class Config:
    CAMINHO_ARQUIVO = None
    CAPACIDADE_CAMINHAO = None
    ALFA = None
    NUM_ITERACOES_GRASP = None
    GERAR_HTML = None
    GERAR_TXT = None
    EXECUTAR_TODOS = None
    NUM_TENTATIVAS_SWAP = None
    METODO_MELHORIA = None

    SA_TEMPERATURA_INICIAL = None
    SA_TEMPERATURA_MIN = None
    SA_ALPHA = None
    SA_ITERACOES_POR_TEMPERATURA = None
    
def set_config(**params):
    for k, v in params.items():
        setattr(Config, k, v)

def make_tag(idx: int) -> str:
    return f"c{idx}_{Config.METODO_MELHORIA}"

# =============================================================
# MAIN
# =============================================================
def print_config_run(idx: int):
    print(f"\nExecutando config {idx} | metodo: {Config.METODO_MELHORIA}")

if __name__ == "__main__":
    limpar_pastas_saida()
    configs = [
        {
            "CAMINHO_ARQUIVO": "aresta_residuo/arestas_residuo_gp1.txt",
            "CAPACIDADE_CAMINHAO": 15000,
            "ALFA": 0.6,
            "NUM_ITERACOES_GRASP": 10,
            "GERAR_HTML": False,
            "GERAR_TXT": False,
            "EXECUTAR_TODOS": True,
            "NUM_TENTATIVAS_SWAP": 50,
            "METODO_MELHORIA": "first_rand",
            "SA_TEMPERATURA_INICIAL": 100,
            "SA_TEMPERATURA_MIN": 0.01,
            "SA_ALPHA": 0.8,
            "SA_ITERACOES_POR_TEMPERATURA": 10,
        },
        {
            "CAMINHO_ARQUIVO": "aresta_residuo/arestas_residuo_gp1.txt",
            "CAPACIDADE_CAMINHAO": 15000,
            "ALFA": 0.8,
            "NUM_ITERACOES_GRASP": 20,
            "GERAR_HTML": False,
            "GERAR_TXT": False,
            "EXECUTAR_TODOS": True,
            "NUM_TENTATIVAS_SWAP": 100,
            "METODO_MELHORIA": "annealing",
            "SA_TEMPERATURA_INICIAL": 100,
            "SA_TEMPERATURA_MIN": 0.01,
            "SA_ALPHA": 0.8,
            "SA_ITERACOES_POR_TEMPERATURA": 10,
        },
        {
            "CAMINHO_ARQUIVO": "aresta_residuo/arestas_residuo_gp1.txt",
            "CAPACIDADE_CAMINHAO": 15000,
            "ALFA": 0.8,
            "NUM_ITERACOES_GRASP": 20,
            "GERAR_HTML": False,
            "GERAR_TXT": False,
            "EXECUTAR_TODOS": True,
            "NUM_TENTATIVAS_SWAP": 100,
            "METODO_MELHORIA": "best_first",
            "SA_TEMPERATURA_INICIAL": 100,
            "SA_TEMPERATURA_MIN": 0.01,
            "SA_ALPHA": 0.8,
            "SA_ITERACOES_POR_TEMPERATURA": 10,
        }
    ]

    for idx, cfg in enumerate(configs, start=1):
        set_config(**cfg)
        Config.TAG = make_tag(idx)
        print_config_run(idx)
        if Config.EXECUTAR_TODOS:
            executar_grasp_para_todos()
        else:
            executar_grasp()