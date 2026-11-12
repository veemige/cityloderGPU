# CityLODer GPU

Relatório técnico: [otimizações, uso de CUDA e limites das medições](docs/RELATORIO_OTIMIZACOES.md).

Reconstrução de cidades a partir de nuvens LAS/LAZ classificadas e contornos de
edifícios em Shapefile. O projeto exporta CityJSON, malhas OBJ separadas de
edifícios e terreno e uma malha completa que reúne as duas partes.

O caminho Python/GEOS permite gerar malhas com materiais sem depender do
Triangle. O gerador C++/Triangle continua disponível como alternativa.

## Requisitos

- Python 3.10 ou superior; os testes locais foram executados com Python 3.14.
- RAM e espaço em disco proporcionais é área processada. A leitura ocorre em
  blocos/tiles, mas a interpolação e a conexão final mantém estruturas globais
  em memória.
- Shapely >= 2.1 com GEOS >= 3.10, além das dependências de
  `core/requirements.txt`.
- Opcional: GPU NVIDIA e driver compatível com CUDA 13 para usar
  `requirements-gpu.txt`. Sem GPU, utilize `--device cpu`.
- Somente para o caminho C++: Git, CMake >= 3.16 e compilador C++17; no Windows,
  Visual Studio/Build Tools com desenvolvimento desktop em C++.

Execute os comandos a partir da raiz do repositório. Os caminhos dos dados nos
exemplos devem ser substituídos pelos seus.

## Instalação no Windows

```powershell
git clone https://github.com/veemige/cityloderGPU.git
cd cityloderGPU
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r core/requirements.txt
```

Os comandos usam o Python do ambiente diretamente; não é necessário executar
`Activate.ps1` nem alterar a política de execução do PowerShell.

Para habilitar CUDA, instale também:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-gpu.txt
.\.venv\Scripts\python.exe -c "import cupy; cupy.show_config()"
```

O segundo comando deve listar a GPU e as versões CUDA disponíveis.
`requirements-gpu.txt` usa CuPy para CUDA 13; não instale variantes diferentes
como `cupy` e `cupy-cuda13x` simultaneamente no mesmo ambiente.

### CUDA no PATH

Se o CUDA Toolkit estiver instalado, confirme seu caminho real. Por exemplo,
para uma instalação em `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2`,
configure o terminal atual com:

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"
$env:Path = "$env:CUDA_PATH\bin;$env:Path"
nvcc --version
nvidia-smi
```

Para persistir, adicione o diretório `bin` do Toolkit ao `Path` nas Variáveis de
Ambiente do Windows e defina `CUDA_PATH` para a raiz da instalação. Reinicie o
terminal/VS Code. Ter `nvcc` no PATH, isoladamente, não comprova que o CuPy
funciona: confirme também a GPU com `cupy.show_config()`.

## Preparação dos dados

- Informe uma pasta de LAS/LAZ; todas as subpastas são pesquisadas, inclusive
  pastas EPT e arquivos com extensão em maiúsculas.
- Os pontos devem estar classificados: terreno `2` e edifícios `6` por padrão.
  Não há classificação automática de uma nuvem não classificada.
- Informe os footprints em `.shp` e mantenha seus arquivos auxiliares
  (`.shx`, `.dbf`, `.prj`, quando disponíveis) junto dele.
- Nuvens e footprints devem usar o mesmo CRS projetado, preferencialmente em
  metros. O projeto não reprojeta automaticamente os footprints.
- Ruas GeoJSON são opcionais; não é necessário fornecer um arquivo vazio.

## Processar toda a área e gerar as três malhas

Este é o caminho recomendado para as malhas coloridas, interpolação contínua
do terreno e conexão com as bases dos edifícios:

```powershell
.\.venv\Scripts\python.exe .\cityloder_runner.py `
  --las "D:\dados\nuvens" `
  --footprints "D:\dados\edificios.shp" `
  --output .\output `
  --name cidade `
  --mesh-backend cityjson `
  --save-parts `
  --device cuda
```

Para executar sem GPU, substitua `--device cuda` por `--device cpu`.

Uma pasta de entrada seleciona automaticamente o modo por tiles. O tamanho
padrão é 1.000 unidades do CRS, com leitura de LAS/LAZ comuns em blocos de um
milhão de pontos e consulta espacial para COPC. A descompressão do próximo
bloco pode ocorrer enquanto o bloco atual é processado. Edifícios nas bordas
recebem uma margem de carregamento.

Arquivos diferentes que cobrem a mesma região são incluídos por padrão.
Isso **não elimina pontos XYZ repetidos entre arquivos**. `--dedupe` seleciona
um produto por extensão espacial idêntica; não é uma deduplicação dos pontos.

### Saídas

```text
output/
  cidade.city.json
  cidade_tmp/
    building1/ ...
    ground_polygon.off
    tiles/
  cidade/
    buildings_mesh.obj
    ground_mesh.obj
    mesh.obj
    city_materials.mtl
    mesh.report.json
```

- `buildings_mesh.obj`: edifícios, com fachadas bege e telhados terracota.
- `ground_mesh.obj`: terreno verde.
- `mesh.obj`: os mesmos triângulos das duas partes corrigidas, reunidos com
  reutilização de vértices coincidentes; não reconstrói uma terceira geometria.
- `city_materials.mtl`: materiais compartilhados pelos OBJs. Mantenha este
  arquivo na mesma pasta e habilite os materiais no visualizador.
- `mesh.report.json`: contagens de reparos, superfícies degeneradas, bases
  reconciliadas, arestas subdivididas e triângulos exportados.

A representação de terreno usada pelo conversor interpola as amostras
linearmente, ancora suas bordas nas bases dos edifícios e subdivide arestas
para tratar junções em T entre terreno, fachadas e tiles. Os dados de origem
podem conter superfícies degeneradas ou inconsistências; não há garantia de
que toda entrada produza um sólido fechado. OBJ não preserva todos os
atributos e a semântica do CityJSON.

O CityJSON gerado atualmente contém edifícios e, quando fornecidas, ruas.
**Ele não contém o terreno.** Preserve `cidade_tmp`, especialmente
`ground_polygon.off`, para reconstruir a malha completa sem reler LAS/LAZ.

## Refazer apenas as malhas com dados já existentes

Não executa a leitura nem a reconstrução das nuvens de entrada:

```powershell
.\.venv\Scripts\python.exe .\cityjson_to_mesh.py `
  --cityjson .\output\cidade.city.json `
  --ground-off .\output\cidade_tmp\ground_polygon.off `
  --output .\output\cidade\mesh.obj `
  --save-parts `
  --device cuda
```

Este comando substitui as malhas de destino ao concluir a exportação e mantém
o CityJSON de origem. Sem `--ground-off`, exporta somente os objetos presentes
no CityJSON. Sem `--save-parts`, exporta somente o OBJ completo e seus materiais.

## Parâmetros úteis

| Opção do runner | Uso |
| --- | --- |
| `--device auto` | Usa CUDA quando disponível e informa fallbacks. é o padrão. |
| `--device cuda` | Exige CUDA nas etapas aceleradas; falhas nessas etapas interrompem a execução. |
| `--device cpu` | Executa o caminho numérico em CPU. |
| `--gpu-memory-fraction 0.8` | Limita o pool CuPy a 80% da VRAM total; não reserva essa memória. |
| `--tile-size 1000` | Tamanho do tile em unidades do CRS. |
| `--ground-sample-step 25` | Espaçamento da amostragem do terreno; valores menores preservam mais detalhe e aumentam o custo. |
| `--overhang 50` | Margem adicional em torno dos footprints. |
| `--building-class 6 --ground-class 2` | Classes LAS usadas na reconstrução. |
| `--streets "D:\dados\ruas.geojson"` | Inclui ruas; `--graph-target-crs EPSG:31982` informa o CRS de destino do grafo. |
| `--no-resume` | Descarta os resultados temporários e refaz os tiles. |
| `--bbox "minx,miny,maxx,maxy"` | Restringe os tiles a uma região. |
| `--mesh-backend cityjson` | Usa o conversor Python/GEOS, com materiais e conexão das bases. |
| `--mesh-backend auto` | Tenta C++ e recupera via CityJSON se o gerador falhar ou estiver ausente. é o padrão. |
| `--mesh-backend triangle` | Exige o gerador C++/Triangle. |

`--only-tiles-with-buildings` e `--tile-filter` servem para prévias ou recortes;
não os use quando quiser a área inteira. Alterações nas fontes e nos parâmetros
de processamento invalidam o cache dos tiles. Alterar apenas o conversor e
executar `cityjson_to_mesh.py` dispensa reprocessar os tiles.

Consulte as opções de cada entrada:

```powershell
.\.venv\Scripts\python.exe cityloder_runner.py --help
.\.venv\Scripts\python.exe cityjson_to_mesh.py --help
```

### O que usa CUDA

O processamento de nuvens pode manter pontos de cada tile na VRAM e executar
filtros de classe/limites, seleção espacial, pontos dentro de polígonos,
estatísticas dos telhados, ajuste em lote de planos e buscas maiores de
vizinhos na GPU. No conversor de malhas, a avaliação em lote da interpolação
do terreno pode usar CUDA.

Leitura/descompressão, operações Shapely/GEOS, construção do índice de
triangulação e conexão/exportação das malhas usam CPU. Consultas pequenas
podem continuar na CPU para evitar o custo de preparação. Não há promessa de
100% de ocupação da GPU nem de aceleração em toda entrada. Compare tempos
usando `--device cpu`; para comparar o pipeline completo, use `--no-resume`.

## Gerador C++ opcional

O caminho `--mesh-backend cityjson` não precisa desta compilação. Para usar
Triangle, inicialize o submódulo e compile:

```powershell
git submodule update --init --recursive
cmake -S mesh_gen -B build/mesh -DCMAKE_BUILD_TYPE=Release
cmake --build build/mesh --config Release --parallel
```

No Windows/MSVC, informe o executável com
`--mesh-exe .\build\mesh\Release\city_iconic_mesh.exe`. Em geradores de
configuração única, ele pode estar diretamente em `build/mesh`. O CMake pode
baixar a dependência Triangle durante a configuração; é necessária conexão
é internet na primeira compilação.

Se o Triangle emitir `Topological inconsistency after splitting a segment`,
use a recuperação a partir do CityJSON e do terreno preservado. O caminho C++
não oferece todos os materiais e reparos do conversor Python.

## Linux / WSL

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r core/requirements.txt
# Opcional para CUDA 13:
.venv/bin/python -m pip install -r requirements-gpu.txt

.venv/bin/python cityloder_runner.py \
  --las /dados/nuvens --footprints /dados/edificios.shp \
  --output output --name cidade --mesh-backend cityjson --save-parts --device cpu
```

Para refazer somente as malhas, use `cityjson_to_mesh.py` com os mesmos argumentos
do exemplo Windows e caminhos Linux. Os wrappers legados permanecem disponíveis;
os comandos Python acima dão acesso ao fluxo atual por tiles e aos materiais.

## Testes

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Os testes usam dados sintéticos para validar seleção de pontos, equivalência
CPU/CUDA, exportação, materiais, interpolação e conexões de borda. Testes que
exigem GPU são ignorados quando CUDA não está disponível. Não processam seus
dados de cidade.

## Empacotar no Windows (opcional)

```powershell
powershell -ExecutionPolicy Bypass -File .\build_windows_exe.ps1
```

O script instala dependências de empacotamento, compila o gerador C++ e cria
`dist/cityloder.exe` com PyInstaller. Exige CMake, compilador e CUDA compatível
para o caminho GPU. O executável deve ser reempacotado após mudanças no código.
O conversor independente continua acessível por `cityjson_to_mesh.py`.
A validação principal deste repositório usa as entradas Python, não o pacote.

## Arquivos versionados

São versionados os fontes Python/C++, testes, dependências, scripts de build
e documentação. `.gitignore` exclui ambientes virtuais, caches, dados LAS/LAZ,
malhas geradas, materiais gerados, saídas de build, executáveis e configurações
locais. Baixe/prepare os dados e instale as dependências em cada máquina.

## Licença e referência

Código sob [GPL-3.0](LICENSE). O submódulo cinolib e demais dependências têm suas
próprias licenças. O projeto deriva de CityLODer, descrito em *City models from
airborne point clouds*, Tommaso Sorgente et al., Graphical Models (2026),
[DOI: 10.1016/j.gmod.2026.101337](https://doi.org/10.1016/j.gmod.2026.101337).
