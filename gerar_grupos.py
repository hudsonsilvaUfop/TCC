import os
import pandas as pd

def get_bairro_group(bairro):
    gp1 = {'Ponte Funda', 'Nova Cachoeirinha', 'Recanto Paraíso', 'Vale da Serra',
           'Rosário', 'Vale do Sol', 'Mangabeiras', 'José Elói', 'São João',
           'São Benedito', 'Castelo', 'Ipiranga', 'Industrial', 'Campo Alegre',
           'Boa Vista', 'Cidade Nova','Jk'}

    gp2 = {'Nova Aclimação', 'Paineiras', 'Sion', 'Campos Elísios',
           'Terminal Rodoviário', 'Poso Marfim', 'Posto 5 Estrelas',
           'Chácaras Burian', 'Tanquinho', 'Tanquinho II', 'Centro Industrial',
           'Tiete', 'Pedreira', 'Santa Cruz', 'Amazonas', 'Jacui', 'Serra do Egito'}

    gp3 = {'Novo Horizonte', 'Aclimação', 'Republica', 'Lourdes',
           'Alvorada', 'Nossa Senhora da Conceicao', 'Nova Esperança'}

    gp4 = {'Lucília', 'São José', 'Promorar', 'São Geraldo', 'Satélite',
           'José de Alencar', 'Loanda', 'Miramar', 'Ernestina Graciana', 'Alto do Loanda','Serra'}

    gp5 = {'Santo Hipolito', 'Teresópolis', 'Cruzeiro Celeste', 'Petrópolis',
           'Nova Monlevade', 'Novo Cruzeiro', 'Santa Cecilia', 'Estrela Dalva',
           'Primeiro de Maio', 'Monte Sagrado', 'Corumbiara de Vanessa', 'ABM',
           'Palmares','Jardim Vitória','Vera Cruz','Feixos'}

    gp6 = {'João Cândido Dias', 'Laranjeiras', 'Metalúrgico', 'Belmonte',
           'Bau', 'Vila Tanque', 'Areia Preta'}

    gp7 = {'Carneirinhos', 'Avenida Getúlio Vargas', 'Avenida Wilson Alvarenga',
           'Pinheiros', 'Santa Barbara', 'São Jorge', 'Monte Santo', 'Entorno da Prefeitura'}

    if bairro in gp1:
        return 'gp1'
    elif bairro in gp2:
        return 'gp2'
    elif bairro in gp3:
        return 'gp3'
    elif bairro in gp4:
        return 'gp4'
    elif bairro in gp5:
        return 'gp5'
    elif bairro in gp6:
        return 'gp6'
    elif bairro in gp7:
        return 'gp7'
    else:
        return 'unknown'


def gerar_arquivos_por_grupo(arquivo_entrada="grafo-viario-completo.txt"):
    with open(arquivo_entrada, 'r', encoding='utf-8') as f:
        linhas = f.readlines()

    # separar vértices e arestas
    idx_arestas = next(i for i, l in enumerate(linhas) if l.strip().startswith("ARESTAS"))
    linhas_vertices = linhas[1:idx_arestas]  # pula a primeira linha "VERTICES"
    linhas_arestas = linhas[idx_arestas + 1:]

    # processar vértices
    vertices_por_grupo = {}
    vertex_info = {}

    for linha in linhas_vertices:
        partes = linha.strip().split()
        if len(partes) < 4:
            continue
        id_, lat, lon = partes[:3]
        bairro = ' '.join(partes[3:])
        grupo = get_bairro_group(bairro)
        vertex_info[id_] = (lat, lon, bairro, grupo)
        if grupo not in vertices_por_grupo:
            vertices_por_grupo[grupo] = {}
        vertices_por_grupo[grupo][id_] = linha.strip()

    # processar arestas
    arestas_por_grupo = {g: [] for g in vertices_por_grupo.keys()}

    for linha in linhas_arestas:
        partes = linha.strip().split()
        if len(partes) != 7:
            continue
        id1, lat1, lon1, id2, lat2, lon2, dist = partes
        grupo1 = vertex_info.get(id1, (None, None, None, ''))[3]
        grupo2 = vertex_info.get(id2, (None, None, None, ''))[3]
        if grupo1 == grupo2 and grupo1 in arestas_por_grupo:
            arestas_por_grupo[grupo1].append(linha.strip())

    # gerar arquivos
    for grupo, vertices in vertices_por_grupo.items():
        nome_saida = f"grafo_{grupo}.txt"
        with open(nome_saida, "w", encoding="utf-8") as out:
            out.write("VERTICES (ID, Latitude, Longitude, Bairro):\n")
            for v in vertices.values():
                out.write(f"{v}\n")
            out.write("\nARESTAS (origem -> destino):\n")
            for a in arestas_por_grupo.get(grupo, []):
                out.write(f"{a}\n")
        print(f"✅ Arquivo gerado: {nome_saida} ({len(vertices)} vértices, {len(arestas_por_grupo.get(grupo, []))} arestas)")

def gerar_txt_com_residuo(
    caminho_arquivo: str,
    habitantes: int,
    taxa_per_capita: float = 0.8,
    saida_txt: str = "arestas_residuo.txt"
):
    """
    Lê o grafo viário no formato do 'grafo_gp.txt', calcula o resíduo (kg/dia)
    proporcional ao comprimento das arestas e grava um TXT tabulado com a coluna 'residuo',
    SEM cabeçalhos.

    Parâmetros:
      - caminho_arquivo: caminho do grafo_gp.txt
      - habitantes: população total (int)
      - taxa_per_capita: kg/hab/dia (padrão 0.8)
      - saida_txt: nome do arquivo de saída .txt
    """
    # --- Ler arquivo inteiro ---
    with open(caminho_arquivo, "r", encoding="utf-8") as f:
        linhas = f.readlines()

    # --- Encontrar início da seção ARESTAS ---
    idx_arestas = None
    for i, ln in enumerate(linhas):
        if ln.strip().startswith("ARESTAS"):
            idx_arestas = i
            break
    if idx_arestas is None:
        raise ValueError("Seção 'ARESTAS' não encontrada no arquivo.")

    # --- Parse das arestas ---
    arestas = []
    for ln in linhas[idx_arestas + 1:]:
        parts = ln.strip().split()
        if len(parts) != 7:
            continue
        id1, lat1, lon1, id2, lat2, lon2, length = parts
        arestas.append({
            "id1": int(id1),
            "lat1": float(lat1),
            "lon1": float(lon1),
            "id2": int(id2),
            "lat2": float(lat2),
            "lon2": float(lon2),
            "comprimento_m": float(length),
        })

    if not arestas:
        raise ValueError("Nenhuma aresta válida encontrada após 'ARESTAS'.")

    df = pd.DataFrame(arestas)

    # --- Cálculo do resíduo ---
    residuo_total = habitantes * taxa_per_capita
    comprimento_total = df["comprimento_m"].sum()
    if comprimento_total <= 0:
        raise ValueError("Comprimento total das arestas é zero ou negativo.")
    densidade_linear = residuo_total / comprimento_total
    df["residuo"] = df["comprimento_m"] * densidade_linear

    # --- Escrita sem cabeçalho ---
    with open(saida_txt, "w", encoding="utf-8") as out:
        for _, row in df.iterrows():
            linha = f"{int(row['id1'])}\t{row['lat1']:.6f}\t{row['lon1']:.6f}\t"
            linha += f"{int(row['id2'])}\t{row['lat2']:.6f}\t{row['lon2']:.6f}\t"
            linha += f"{row['comprimento_m']:.6f}\t{row['residuo']:.6f}\n"
            out.write(linha)

    return {
        "saida_txt": saida_txt,
        "residuo_total_kg_dia": residuo_total,
        "densidade_linear_kg_por_m_dia": densidade_linear,
        "num_arestas": len(df),
    }
# Executar
if __name__ == "__main__":
  gerar_arquivos_por_grupo()

  for i in range(1, 8):
    info = gerar_txt_com_residuo(
        caminho_arquivo=f"grafo_gp{i}.txt",
        habitantes=83360,
        taxa_per_capita=0.8,
        saida_txt=f"arestas_residuo_gp{i}.txt"
      )
    print(info)
