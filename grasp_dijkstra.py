"""
###############################################################
# ALGORITMO GRASP + DIJKSTRA PARA ROTEAMENTO EM ARCOS
# -----------------------------------------------------------
# Objetivo:
#   - Minimizar a distância total percorrida pelas rotas.
#   - Respeitar a capacidade máxima de cada caminhão.
#   - Atender apenas arcos com resíduo (coleta) e distinguir
#     deslocamentos (sem coleta) dos trechos atendidos.
#
# Modelagem do grafo:
#   - Deslocamento: grafo NÃO direcionado (u↔v) com custos (metros).
#   - Coleta: arcos DIRECIONADOS (u→v) com demanda (kg).
#     -> Assim, o veículo pode se deslocar em ambos os sentidos,
#        mas só “coleta” no sentido definido pelo arco.
#
# Construção (GRASP):
#   1) Lê as arestas do arquivo (grafo georreferenciado).
#   2) Monta o grafo de deslocamento e o conjunto de arcos de coleta.
#   3) Constrói rotas iterativamente:
#      - Escolha gulosa-aleatória do primeiro arco usando uma RCL
#        controlada por ALFA (0=guloso, 1=aleatório).
#      - Expansão por vizinhos viáveis (cabem na capacidade).
#      - Quando não houver vizinhos:
#          * Usa Dijkstra a partir do nó atual até o nó de início
#            do próximo arco viável mais próximo.
#          * Reconstrói o caminho (deslocamentos) e então atende
#            o arco escolhido.
#   4) Cada rota é uma lista de triplas (u, v, coleta: bool).
#   5) Repete por NUM_ITERACOES e guarda a melhor solução.
#
# Melhoria local (opcional):
#   - swap/permuta simples intra-rota com Dijkstra:
#       * Executa trocas aleatórias de posições das coletas
#         tentando NUM_TENTATIVAS_SWAP vezes por rota.
#       * Após cada troca, reconecta trechos quebrados com Dijkstra
#         (inserindo deslocamentos) e aceita apenas se reduzir o custo.
#
# Saídas e logs:
#   - Estatísticas de custo total, uso de capacidade e distâncias por rota.
#   - (Opcional) Mapas HTML da melhor solução (com coleta/deslocamento).
#   - (Opcional) Arquivo TXT detalhando cada arco e seu tipo.
#   - XLSX com resumo por rota (sempre gerado neste script).
#
# Estrutura/auxiliadores adicionados:
#   - limpar_pastas_saida(): limpa e recria pastas rotas/, mapas/, resultados_xls/
#   - extrair_gp(): extrai gpX do nome do arquivo pra padronizar saídas
#   - ler_arestas(): força dtype id1/id2 como string
#   - saídas padronizadas por instância (gpX) dentro das pastas
###############################################################
"""

import pandas as pd
import random
import math
import heapq
import time
import re
import shutil
from pathlib import Path
from collections import defaultdict
import folium
from folium import Map, PolyLine, CircleMarker, LayerControl

# --------------------------
# AUXILIADORES DE ESTRUTURA
# --------------------------

def limpar_pastas_saida():
    pastas = ["rotas", "resultados_xls", "mapas"]
    for pasta in pastas:
        path = Path(pasta)
        if path.exists() and path.is_dir():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    print("🧹 Pastas de saída limpas com sucesso!")

def extrair_gp(caminho: str) -> str:
    stem = Path(caminho).stem
    m = re.search(r'(gp\d+)', stem, re.IGNORECASE)
    return m.group(1).lower() if m else "instancia"

# --------------------------
# LEITURA DO ARQUIVO
# --------------------------

def ler_arestas(caminho):
    colunas = ["id1", "lat1", "lon1", "id2", "lat2", "lon2", "comprimento_m", "residuo"]
    df = pd.read_csv(
        caminho,
        sep="\t",
        header=None,
        names=colunas,
        dtype={"id1": str, "id2": str},
    )
    return df

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

# --------------------------
# DIJKSTRA
# --------------------------

def dijkstra(grafo, origem):
    dist = {n: float("inf") for n in grafo}
    prev = {n: None for n in grafo}
    dist[origem] = 0.0
    heap = [(0.0, origem)]

    while heap:
        atual_dist, atual = heapq.heappop(heap)
        if atual_dist > dist[atual]:
            continue
        for viz, custo in grafo[atual]:
            novo_custo = atual_dist + custo
            if novo_custo < dist[viz]:
                dist[viz] = novo_custo
                prev[viz] = atual
                heapq.heappush(heap, (novo_custo, viz))

    return dist, prev

# --------------------------
# RECONSTRUIR CAMINHO
# --------------------------

def reconstruir_caminho(prev, origem, destino):
    """Retorna lista de nós do caminho origem->destino (inclui ambos).
       Se não houver caminho, retorna []."""
    caminho = []
    atual = destino
    while atual is not None:
        caminho.append(atual)
        if atual == origem:
            break
        atual = prev[atual]
    if not caminho or caminho[-1] != origem:
        return []
    caminho.reverse()
    return caminho

###############################################################
# MELHORIA LOCAL: SWAP AO FINAL DA ROTA
###############################################################

def custo_rota(rota, grafo):
    return sum(custo_aresta(grafo, u, v) for (u, v, _) in rota)

def existe_aresta_direta(grafo, u, v):
    return any(viz == v for viz, _ in grafo[u])

def construir_rota_conectada(ordem_coletas, grafo):
    """Reconstrói rota completa conectando as coletas com deslocamentos mínimos.
       Usa Dijkstra SOMENTE quando não houver aresta direta entre os nós."""
    rota = []
    if not ordem_coletas:
        return rota

    # Primeira coleta
    u0, v0 = ordem_coletas[0]
    rota.append((u0, v0, True))
    atual = v0

    # Demais coletas
    for (u, v) in ordem_coletas[1:]:
        # Conexão direta -> evita Dijkstra
        if atual == u or existe_aresta_direta(grafo, atual, u):
            if atual != u:
                rota.append((atual, u, False))
        else:
            # Sem conexão -> usar Dijkstra
            dist, prev = dijkstra(grafo, atual)
            caminho = reconstruir_caminho(prev, atual, u)
            if not caminho:
                return []
            for i in range(len(caminho) - 1):
                rota.append((caminho[i], caminho[i + 1], False))

        # Adiciona coleta atual
        rota.append((u, v, True))
        atual = v

    return rota

def swap_simples(grafo, rota, pos1, pos2):
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    if pos1 >= len(coletas) or pos2 >= len(coletas) or pos1 == pos2:
        return None

    nova_ordem = coletas.copy()
    nova_ordem[pos1], nova_ordem[pos2] = nova_ordem[pos2], nova_ordem[pos1]

    nova_rota = construir_rota_conectada(nova_ordem, grafo)
    return nova_rota

def aplicar_swap_first_random(grafo, rota):
    """FIRST IMPROVEMENT usando custo total da rota."""
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
            ganho = custo_atual - custo_novo
            return nova_rota
    return rota

def aplicar_swap_best(grafo, rota):
    """BEST IMPROVEMENT usando custo total da rota."""
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    custo_atual = custo_rota(rota, grafo)

    melhor_rota = None
    melhor_ganho = 0.0

    for i in range(n - 1):
        for j in range(i + 1, n):
            nova_rota = swap_simples(grafo, rota, i, j)
            if not nova_rota:
                continue

            custo_novo = custo_rota(nova_rota, grafo)
            ganho = custo_atual - custo_novo

            if ganho > melhor_ganho:
                melhor_ganho = ganho
                melhor_rota = nova_rota

    if melhor_rota:
        return melhor_rota
    return rota

def aplicar_swap_best_first(grafo, rota):
    """BEST-FIRST: para no primeiro swap que melhora (custo global)."""
    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    custo_atual = custo_rota(rota, grafo)

    for i in range(n - 1):
        for j in range(i + 1, n):
            nova_rota = swap_simples(grafo, rota, i, j)
            if not nova_rota:
                continue

            custo_novo = custo_rota(nova_rota, grafo)
            if custo_novo < custo_atual:
                ganho = custo_atual - custo_novo
                return nova_rota
    return rota

###############################################################
# SIMULATED ANNEALING
###############################################################

def aplicar_simulated_annealing(grafo, rota):
    """
    Simulated Annealing para melhorar uma rota de coleta.
    Usa swap_simples e custo total da rota.
    """
    T_inicial = Config.SA_TEMPERATURA_INICIAL
    T_min = Config.SA_TEMPERATURA_MIN
    alfa = Config.SA_ALPHA
    iter_por_temp = Config.SA_ITERACOES_POR_TEMPERATURA

    rota_atual = rota
    custo_atual = custo_rota(rota_atual, grafo)

    melhor_rota = rota_atual
    melhor_custo = custo_atual

    T = T_inicial

    coletas = [(u, v) for (u, v, coleta) in rota if coleta]
    n = len(coletas)
    
    if n < 2:
        return rota
    
    while T > T_min:
        for _ in range(iter_por_temp):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                continue

            nova_rota = swap_simples(grafo, rota_atual, i, j)
            if not nova_rota:
                continue

            custo_novo = custo_rota(nova_rota, grafo)
            delta = custo_novo - custo_atual

            if delta < 0:
                aceita = True
            else:
                prob = math.exp(-delta / T)
                aceita = random.random() < prob

            if aceita:
                rota_atual = nova_rota
                custo_atual = custo_novo
                if custo_novo < melhor_custo:
                    melhor_custo = custo_novo
                    melhor_rota = nova_rota

        T *= alfa
    return melhor_rota

###############################################################
# GRASP + DIJKSTRA
###############################################################

def construir_solucao_grasp(grafo, demandas, alfa, capacidade):
    arcos_restantes = set(k for k, q in demandas.items() if q > 0)
    rotas = []

    while arcos_restantes:
        rota = []  # lista de triplas (u, v, coleta: bool)
        carga = 0.0

        # candidatos iniciais que cabem na capacidade
        candidatos = []
        for arco in arcos_restantes:
            if demandas[arco] > capacidade:
                continue
            u, v = arco
            custo = custo_aresta(grafo, u, v)
            candidatos.append((custo, arco))

        if not candidatos:
            break

        candidatos.sort()
        limite = max(1, int(len(candidatos) * alfa))
        _, arco_escolhido = random.choice(candidatos[:limite])
        u, v = arco_escolhido

        rota.append((u, v, True))
        carga += demandas[arco_escolhido]
        arcos_restantes.remove(arco_escolhido)
        atual = v

        while True:
            vizinhos = []
            for viz, custo in grafo[atual]:
                arco = (atual, viz)
                if arco in arcos_restantes and (carga + demandas[arco] <= capacidade):
                    vizinhos.append((custo, arco))

            if vizinhos:
                vizinhos.sort()
                limite = max(1, int(len(vizinhos) * alfa))
                _, proximo_arco = random.choice(vizinhos[:limite])
                u2, v2 = proximo_arco
                rota.append((u2, v2, True))
                carga += demandas[proximo_arco]
                arcos_restantes.remove(proximo_arco)
                atual = v2
                continue

            distancias, prev = dijkstra(grafo, atual)

            menor_distancia = float("inf")
            melhor_candidato = None

            for arco in arcos_restantes:
                if carga + demandas[arco] > capacidade:
                    continue
                u3, v3 = arco
                d = distancias.get(u3, float("inf"))
                if d < menor_distancia:
                    menor_distancia = d
                    melhor_candidato = (d, arco, u3, v3)

            if melhor_candidato and menor_distancia < float("inf"):
                _, arco_dij, u_dij, v_dij = melhor_candidato
                caminho_nos = reconstruir_caminho(prev, atual, u_dij)

                for i in range(len(caminho_nos) - 1):
                    a = caminho_nos[i]
                    b = caminho_nos[i + 1]
                    rota.append((a, b, False))

                rota.append((u_dij, v_dij, True))
                carga += demandas[arco_dij]
                arcos_restantes.remove(arco_dij)
                atual = v_dij
            else:
                break

        ############################################################
        # MELHORIA LOCAL VIA SWAP AO FINAL DA ROTA
        ############################################################
        custo_antigo = custo_rota(rota, grafo)

        if Config.METODO_MELHORIA == "best":
            rota_melhorada = aplicar_swap_best(grafo, rota)
        elif Config.METODO_MELHORIA == "first_rand":
            rota_melhorada = aplicar_swap_first_random(grafo, rota)
        elif Config.METODO_MELHORIA == "best_first":
            rota_melhorada = aplicar_swap_best_first(grafo, rota)
        elif Config.METODO_MELHORIA == "annealing":
            rota_melhorada = aplicar_simulated_annealing(grafo, rota)
        else:
            rota_melhorada = rota
        custo_novo = custo_rota(rota_melhorada, grafo)

        if custo_novo < custo_antigo:
            rota = rota_melhorada

        rotas.append((rota, carga))

    return rotas

# ==============================================================
# SALVA ROTAS EM TXT
# ==============================================================

def salvar_rotas_txt(rotas, grafo, gp, tempo_total=0):
    """
    Mantém o conteúdo/estrutura do TXT, mas salva em rotas/rotas_grasp_{gp}.txt
    """
    Path("rotas").mkdir(parents=True, exist_ok=True)
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
        
# ==============================================================
# XLSX
# ==============================================================

def salvar_xls(rotas, grafo, gp, tempo_s):
    Path("resultados_xls").mkdir(parents=True, exist_ok=True)
    dados = []
    for i, (rota, carga) in enumerate(rotas, 1):
        dados.append({
            "instancia": gp,
            "rota": i,
            "carga": carga,
            "distancia_m": sum(custo_aresta(grafo, u, v) for (u, v, _) in rota),
            "tempo_s": tempo_s,
        })
    tag = getattr(Config, "TAG", "c0_default")
    pd.DataFrame(dados).to_excel(f"resultados_xls/{tag}_resultado_grasp_{gp}.xlsx", index=False)
# ==============================================================
# MAPAS
# ==============================================================

def gerar_mapas(df_arestas, melhor_solucao, gp):
    Path("mapas").mkdir(parents=True, exist_ok=True)
    prefixo = Path("mapas") / f"{Config.TAG}_mapa_grasp_{gp}"

    coord_por_aresta = {
        frozenset((str(row["id1"]), str(row["id2"]))):
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

# ==============================================================
# LOGS AUXILIARES
# ==============================================================

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

# ==============================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================

def executar_grasp():
    inicio = time.time()
    gp = extrair_gp(Config.CAMINHO_ARQUIVO)

    df = ler_arestas(Config.CAMINHO_ARQUIVO)
    grafo, demandas = construir_grafo(df)

    melhor_solucao = None
    melhor_custo = float("inf")

    for _ in range(Config.NUM_ITERACOES_GRASP):
        rotas = construir_solucao_grasp(
            grafo, demandas.copy(), Config.ALFA, Config.CAPACIDADE_CAMINHAO
        )
        custo = sum(custo_aresta(grafo, u, v) for rota, _ in rotas for (u, v, _) in rota)

        if custo < melhor_custo:
            melhor_solucao, melhor_custo = rotas, custo

    tempo_total = time.time() - inicio

    exibir_cargas_rotas(melhor_solucao, Config.CAPACIDADE_CAMINHAO)
    exibir_distancias_rotas(melhor_solucao, grafo)

    if Config.GERAR_HTML:
        gerar_mapas(df, melhor_solucao, gp)
    if Config.GERAR_TXT:
        salvar_rotas_txt(melhor_solucao, grafo, gp, tempo_total)

    salvar_xls(melhor_solucao, grafo, gp, tempo_total)

    return melhor_solucao

# ==============================================================
# EXECUÇÃO PARA TODAS AS INSTÂNCIAS (mantida, simplificada)
# ==============================================================

def executar_grasp_para_todos():
    for i in range(1, 8):
        Config.CAMINHO_ARQUIVO = f"aresta_residuo/arestas_residuo_gp{i}.txt"
        executar_grasp()
# ==============================================================
# CONFIGURAÇÕES
# ==============================================================
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
            "GERAR_HTML": True,
            "GERAR_TXT": True,
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
            "GERAR_HTML": True,
            "GERAR_TXT": True,
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
            "GERAR_HTML": True,
            "GERAR_TXT": True,
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