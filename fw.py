import numpy as np
import math
import pandas as pd
import re
from pathlib import Path
import numpy as np
import pandas as pd

# ===============================================================
# LEITURA DO ARQUIVO DE INSTÂNCIA
# ===============================================================
def ler_instancia_fw(caminho):
    """
    Lê arquivo de arestas no formato:
    id1 lat1 lon1 id2 lat2 lon2 comprimento_m residuo

    Retorna:
        grafo: dict {u: [(v, dist), ...]} 
    """

    df = pd.read_csv(
        caminho,
        sep="\t",
        header=None,
        names=["id1", "lat1", "lon1", "id2", "lat2", "lon2", "dist", "residuo"],
        dtype={"id1": str, "id2": str}
    )

    grafo = {}

    for _, row in df.iterrows():
        u = row["id1"]
        v = row["id2"]
        d = float(row["dist"])

        if u not in grafo:
            grafo[u] = []
        if v not in grafo:
            grafo[v] = []

        grafo[u].append((v, d))
        grafo[v].append((u, d))

    return grafo

# ===============================================================
# FLOYD–WARSHALL
# ===============================================================
def floyd_warshall(grafo):
    """
    grafo: dict {u: [(v, dist), ...]}

    Retorna:
        nodes: lista ordenada de nós
        idx: {node → index}
        dist: matriz NxN com float32
        nxt: matriz NxN com int32 (próximo nó)
    """

    nodes = sorted(grafo.keys())
    n = len(nodes)

    idx = {node: i for i, node in enumerate(nodes)}

    dist = np.full((n, n), np.inf, dtype=np.float32)
    nxt  = np.full((n, n), -1, dtype=np.int32)

    for u in nodes:
        i = idx[u]
        dist[i, i] = 0.0
        nxt[i, i] = i

        for (v, w) in grafo[u]:
            j = idx[v]
            if w < dist[i, j]:
                dist[i, j] = w
                nxt[i, j] = j
                
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, j] > dist[i, k] + dist[k, j]:
                    dist[i, j] = dist[i, k] + dist[k, j]
                    nxt[i, j] = nxt[i, k]

    return nodes, idx, dist, nxt

# ===============================================================
# SALVAMENTO EM BINÁRIO
# ===============================================================
def salvar_fw_binario(nodes, dist, nxt, prefix="fw"):
    # salvar lista de nós
    with open(f"{prefix}_nodes.txt", "w", encoding="utf-8") as f:
        for i, node in enumerate(nodes):
            f.write(f"{i}\t{node}\n")

    # salvar distâncias e next-hop
    dist.astype(np.float32).tofile(f"{prefix}_dist.bin")
    nxt.astype(np.int32).tofile(f"{prefix}_next.bin")

    print("\nArquivos gerados:")
    print(f" - {prefix}_nodes.txt")
    print(f" - {prefix}_dist.bin")
    print(f" - {prefix}_next.bin")
# ===============================================================
# PREFIXO INSTANCIA
# ===============================================================
def _prefix_from_instancia(arquivo_instancia: str) -> str:
    stem = Path(arquivo_instancia).stem
    m = re.search(r'(gp\d+)', stem, re.IGNORECASE)
    if m:
        return f"fw_{m.group(1).lower()}"
    return f"fw_{stem}"

# ===============================================================
# FUNÇÃO PRINCIPAL
# ===============================================================
def gerar_fw_da_instancia(arquivo_instancia: str, prefix: str | None = None):
    if prefix is None:
        prefix = _prefix_from_instancia(arquivo_instancia)

    print("📥 Lendo instância...")
    grafo = ler_instancia_fw(arquivo_instancia)
    print(f"   - Nós (distintos): {len(grafo)}")

    print("🔄 Executando Floyd–Warshall...")
    nodes, idx, dist, nxt = floyd_warshall(grafo)

    print(f"💾 Salvando arquivos binários com prefixo '{prefix}'...")
    salvar_fw_binario(nodes, dist, nxt, prefix)

    print("\n✅ Concluído!")
    return nodes, idx, dist, nxt

if __name__ == "__main__":
    gerar_fw_da_instancia("aresta_residuo/arestas_residuo_gp7.txt")