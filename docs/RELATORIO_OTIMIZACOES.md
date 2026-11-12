# Relatório técnico das otimizações do CityLODer GPU

Data: 12/11/2026. Código de referência: commit `ff668c2`, seguido da remoção do
fluxo Docker em `e7b7695`.

## 1. Objetivo e critérios

As alterações procuram reduzir varreduras repetidas, transferências entre RAM
e VRAM, pressão de memória e reprocessamento. A aceleração utiliza CUDA via
CuPy onde há implementação disponível; índices espaciais, cache e mudanças de
algoritmo também contribuem independentemente da GPU.

Este relatório distingue três efeitos:

- **Tempo:** redução do trabalho ou execução paralela de operações numéricas.
- **Memória e escala:** possibilidade de processar uma área maior sem carregar
  toda a nuvem simultaneamente.
- **Qualidade e recuperação:** prevenção de falhas e de retrabalho. Algumas
  dessas mudanças aumentam o tempo de uma execução bem-sucedida.

Não foi realizado um benchmark completo, controlado, antes/depois. Portanto,
os mecanismos de otimização estão comprovados pela implementação e pelos
testes funcionais, mas não existe um percentual validado de aceleração global.

## 2. Paralelização em GPU com CUDA

Implementação principal: [cityloder_gpu.py](../core/cityloder_gpu.py).

| Operação | Implementação atual | Benefício pretendido e limite |
| --- | --- | --- |
| Filtragem de classes e limites XY | Máscaras e seleção em CuPy para blocos com pelo menos 65.536 pontos | Processar muitos pontos em paralelo. Há transferência dos dados descomprimidos para a GPU e retorno dos pontos selecionados. |
| Índice espacial do tile | Pontos residentes na VRAM, ordenados por X com `argsort` | Reutilizar os pontos em muitas consultas sem reenviar a nuvem inteira a cada edifício. |
| Seleção por caixa | `searchsorted` delimita a faixa X; uma máscara verifica Y | Evitar testar todos os pontos do tile em cada footprint. O custo depende dos candidatos na faixa X. |
| Preparação das buscas | Limites X de todos os edifícios do lote calculados em conjunto | Reduzir sincronizações e pequenas transferências CPU/GPU por consulta. |
| Pontos dentro de polígonos | Kernel CUDA de ray casting, com blocos de 256 threads e uma thread por ponto | Paralelizar os testes sobre os pontos candidatos; o percurso das arestas do polígono continua dentro de cada thread. |
| Estatísticas dos telhados | Quantis, máximos, médias, produtos matriciais e análise de posto no dispositivo | Evitar trazer a nuvem selecionada inteira de volta para executar essas operações em NumPy. Pequenos resultados ainda retornam à CPU. |
| Ajuste dos planos | Sistemas de três parâmetros resolvidos em lote com `cp.linalg.solve` | Agrupar soluções em uma operação batelada por chamada de processamento dos edifícios, normalmente por tile. |
| Vizinhos mais próximos | `cupyx.scipy.spatial.KDTree`, com consultas divididas em lotes de até 65.536 | Executar consultas maiores na GPU sem criar matrizes densas de todas as distâncias. O índice é reconstruído a cada chamada de `nearest_xy`. |

Há dois caminhos para pontos dentro de polígonos. Quando os pontos já estão
residentes na GPU, o kernel opera diretamente sobre eles. No caminho que recebe
arrays NumPy, o limiar padrão para transferir e testar na GPU é 4.096 pontos.

As consultas de vizinhos usam CUDA quando existem pelo menos 4.096 pontos de
referência e 128 consultas. Consultas menores usam `cKDTree` na CPU, com
`workers=-1`, evitando o custo de montar e transferir um índice GPU pequeno.
Os limiares são escolhas de implementação; não foram calibrados por um
benchmark abrangente de todas as densidades e equipamentos.

### Interpolação do terreno na GPU

Em [TerrainHeightField](../core/cityloder_mesh_repair.py), lotes com pelo menos
4.096 consultas podem avaliar as alturas por coordenadas baricêntricas no CUDA.
As transformações dos triângulos e suas elevações são enviadas uma vez e
reutilizadas no objeto de interpolação.

A construção da triangulação e a identificação do triângulo de cada consulta
continuam na CPU. Portanto, essa etapa é híbrida: somente a avaliação numérica
em lote foi transferida para a GPU.

### Memória e diagnóstico

O pool CuPy tem limite padrão de 80% da VRAM total. O limite facilita o controle
do consumo, mas não representa reserva de memória nem meta de utilização.
O reaproveitamento de alocações pelo pool reduz a necessidade de novas
alocações no driver.

`--device cuda` exige CUDA nas etapas aceleradas e torna suas falhas explícitas;
`auto` permite fallback com aviso; `cpu` desativa o caminho CUDA. O resumo do
pipeline registra contadores de operações, fallbacks e memória do pool.
Esses contadores não equivalem a uma medição de ocupação ou tempo de kernel.

**Limite importante:** não foi implementado processamento simultâneo de vários
tiles nem agendamento paralelo de todos os edifícios. Os laços de orquestração
continuam sequenciais; o paralelismo ocorre dentro das operações sobre arrays,
dos kernels e dos sistemas resolvidos em lote.

## 3. Leitura e processamento espacial por tiles

Implementação: [cityloder_tiled.py](../core/cityloder_tiled.py) e
[cityloder_processPC.py](../core/cityloder_processPC.py).

### 3.1 Divisão espacial

Pastas LAS/LAZ passam pelo processamento por tiles, com tamanho padrão de 1.000
unidades do CRS. Os arquivos são indexados pela extensão real dos cabeçalhos,
inclusive quando um arquivo cobre vários tiles. Os edifícios são atribuídos
por seus centros de caixa e recebem margem para recuperar pontos nas bordas.

Isso reduz o conjunto de pontos necessário em cada etapa de reconstrução e a
pressão sobre RAM/VRAM. Não elimina toda a memória global: footprints,
amostras do terreno e estruturas da montagem final ainda podem ser grandes.

### 3.2 Leitura em blocos

LAS/LAZ comuns são lidos em blocos de um milhão de pontos. As classes e a região
de interesse são filtradas antes da acumulação dos resultados. A maior alocação
de entrada deixa de depender diretamente do tamanho inteiro de cada arquivo.
Os pontos selecionados para o tile, entretanto, continuam sendo acumulados.

Arquivos COPC identificados pelo nome usam consulta espacial com `CopcReader`.
Isso permite solicitar a região de interesse, em vez de percorrer o arquivo
todo. LAS/LAZ comuns ainda podem ser relidos para tiles diferentes; o código
não implementa um índice espacial persistente desses arquivos.

### 3.3 Sobreposição de leitura e cálculo

`prefetch_chunks` usa uma thread auxiliar para antecipar a leitura/descompressão
do próximo bloco enquanto o bloco atual é processado. Há no máximo o bloco
atual e um próximo bloco antecipado no mecanismo de leitura.

É uma sobreposição de I/O/descompressão e processamento, não uma distribuição
da reconstrução entre muitas threads. O ganho depende do disco, da
descompressão e do tempo das operações numéricas.

### 3.4 Evitar leituras repetidas dentro do tile

O conjunto `loaded_paths` impede que um arquivo seja carregado várias vezes
no mesmo tile quando aparece em células vizinhas do índice. Isso não evita
releituras entre tiles diferentes e não remove pontos repetidos entre arquivos.

### 3.5 Amostragem do terreno

O espaçamento padrão de 25 unidades seleciona pontos reais próximos dos nós
da grade; amostras exatas repetidas são removidas. O limite global usa os
contornos locais dos tiles em vez de manter todos os pontos brutos para esse
fim. O grafo de ruas pode ser elevado usando as amostras acumuladas.

Menos amostras significam menos memória e menos entrada para a triangulação.
Existe um compromisso com o detalhe do relevo: reduzir o espaçamento aumenta
a resolução e também o custo.

## 4. Pré-processamento dos footprints e buscas na CPU

Implementações: [cityloder_preprocess_footprint.py](../core/cityloder_preprocess_footprint.py)
e [cityloder_compute_buildings.py](../core/cityloder_compute_buildings.py).

### 4.1 Remoção da matriz densa de adjacência

O procedimento anterior construía uma matriz inteira `N × N` para representar
interseções entre edifícios, além de trabalhar com pontos amostrados dos
contornos. A versão atual usa:

1. `STRtree` para localizar pares de polígonos que se intersectam.
2. Comparação da área de interseção dos candidatos para distinguir sobreposição
   de contato por borda.
3. Conjuntos para armazenar adjacências efetivas.
4. Union-find, com compressão de caminhos e união por rank, para agrupar os
   footprints sobrepostos.

A remoção da matriz é uma redução estrutural importante de memória. Como
exemplo teórico, 60 mil edifícios exigiriam 3,6 bilhões de entradas: cerca de
28,8 GB apenas para essa matriz se cada inteiro ocupar oito bytes. Isso não é
uma medição de consumo da execução atual.

O índice espacial reduz as comparações nas distribuições usuais. O pior caso
continua podendo conter quantidade quadrática de pares se todos os polígonos
se intersectarem.

### 4.2 Reutilização de índices e mapeamentos

No caminho CPU, `cKDTree` restringe a seleção dos pontos a candidatos próximos
do footprint. Na associação de vizinhos, `id_to_final` permite acesso por ID,
enquanto `ground_by_index` reaproveita bases de edifícios já calculadas.
Isso evita buscas globais repetidas para cada prédio.

O recorte por limites da nuvem no pré-processamento é usado pelo caminho
clássico quando esses limites são informados. No modo tiled, o pré-processamento
dos footprints ainda ocorre globalmente antes da seleção dos tiles.

## 5. Cache e eliminação de reprocessamento

Cada tile concluído grava amostras, contorno, contagem de edifícios e tempo em
um cache NPZ comprimido. Os arquivos OFF dos edifícios permanecem no diretório
temporário. Uma nova execução compatível reutiliza os tiles concluídos.

A assinatura do cache inclui caminhos, tamanhos, datas de modificação das
fontes e parâmetros selecionados. É uma assinatura desses metadados, não um
hash completo do conteúdo dos LAS/LAZ. Não há invalidação automática para toda
possível alteração de algoritmo: para garantir reconstrução após mudanças
desse tipo, use `--no-resume`.

Falhas de leitura ou de processamento impedem a gravação do cache de conclusão
daquele tile. A reutilização não significa que todas as etapas desaparecem:
pré-processamento, CityJSON e exportação final ainda podem ser refeitos.

O maior atalho para ajustes apenas de malha é executar `cityjson_to_mesh.py`
com o CityJSON e `ground_polygon.off` já existentes. Esse caminho dispensa
leitura dos LAS/LAZ, seleção dos pontos e ajuste dos telhados. O CityJSON atual
não armazena o terreno, por isso o OFF precisa ser preservado.

## 6. Montagem e exportação das malhas

### 6.1 Otimizações no gerador C++

Em [ground.h](../mesh_gen/src/ground.h) e [city.h](../mesh_gen/src/city.h),
a inserção dos footprints e a acumulação dos edifícios foram modificadas para
evitar reconstruir um octree a cada união. A implementação reutiliza mapas
de coordenadas, adiciona vértices e remapeia os índices das faces.

As alturas do terreno podem ser recuperadas por mapa XY, evitando uma busca
completa de correspondências para cada vértice. A união final acrescenta as
faces das duas partes e reutiliza os vértices compatíveis; não reconstrói os
edifícios do zero. O custo exato ainda depende das estruturas da biblioteca
de malhas e não foi medido isoladamente.

### 6.2 Conversor CityJSON/GEOS

Em [cityjson_to_mesh.py](../cityjson_to_mesh.py), cada superfície é triangulada
em seu plano, respeitando os anéis e pátios. O terreno é recortado e triangulado
por tiles, usando `STRtree` para consultar apenas os footprints relevantes.
As consultas de altura são agrupadas por coordenadas XY únicas antes da
avaliação, reduzindo cálculos repetidos.

O conversor seleciona um LOD por objeto, evitando exportar representações
superpostas. Remove vértices consecutivos repetidos, tenta reparar superfícies
inválidas e registra degenerescências. Isso melhora a taxa de conclusão,
mas não demonstra que GEOS seja sempre mais rápido que Triangle.

O conjunto de vértices é compartilhado entre edifícios e terreno. As partes
são corrigidas uma vez e os mesmos triângulos são exportados nos OBJs
separados e no OBJ completo. Há escrita adicional ao pedir as três saídas,
mas não uma terceira triangulação independente. O escritor usa buffer de
1 MiB e publica o OBJ a partir de um arquivo temporário ao finalizar.

## 7. Correções de qualidade que também afetam o tempo

Estas alterações foram necessárias para o resultado, mas não devem ser
apresentadas como acelerações garantidas:

- Interpolação linear contínua do terreno no lugar da atribuição geral de
  alturas por vizinho mais próximo. Ainda há fallback para consultas fora da
  região interpolável.
- Âncoras nas bases dos edifícios, reconciliação de alturas coincidentes e
  interpolação das alturas ao longo das arestas dessas bases.
- Subdivisão de arestas para tratar junções em T entre tiles, terreno e
  fachadas. A etapa usa índices espaciais e consultas em blocos de até 32.768
  arestas, mas acrescenta trabalho e mantém estruturas globais em memória.
- Coordenadas locais e precisão dupla no ajuste dos telhados, reduzindo erros
  numéricos em coordenadas projetadas de grande magnitude.
- Pequenos termos de regularização nos sistemas dos planos para melhorar sua
  estabilidade.
- Uso de todos os pontos disponíveis nas estatísticas dos telhados, removendo
  a redução anterior de dez vezes em edifícios com muitos pontos. Isso aumenta
  o trabalho e prioriza qualidade.
- Materiais para fachadas, telhados e terreno. As cores facilitam a inspeção,
  mas não aceleram os cálculos e adicionam instruções de material na exportação.

## 8. Recuperação de falhas e simplificação operacional

O modo `--mesh-backend auto` permite recuperar a exportação pelo CityJSON
quando o gerador C++ falha ou está ausente. Evita perder os resultados de
reconstrução já calculados. Se o caminho C++ falhar antes da recuperação,
entretanto, o tempo dessa tentativa já terá sido gasto.

Quando o objetivo é usar os reparos e materiais atuais, selecionar diretamente
`--mesh-backend cityjson` evita essa tentativa inicial.

A execução nativa, a remoção de Docker, o README e o empacotamento opcional
simplificam a instalação e a operação. Não há medição que permita atribuir
uma redução do tempo de cálculo à remoção de Docker.

## 9. Evidências disponíveis

- Foram aprovados 21 testes sintéticos na validação anterior do código:
  equivalência numérica CPU/CUDA, leitura LAZ em blocos, seleção espacial,
  interpolação, preservação de pátios, conexões e materiais.
- Um registro histórico de recuperação da cidade documenta 62.568 edifícios,
  4.235.110 triângulos e tempo total de 390,8 segundos, aproximadamente
  6 minutos e 31 segundos, com RTX 4080 SUPER disponível na máquina.
- Esse registro é da versão anterior às correções finais de interpolação,
  conexões e materiais. Não mede o pipeline LAS/LAZ completo, não comprova
  ocupação da GPU e não representa o tempo da versão atual.
- O mesmo registro informa 385 ocorrências de vértices repetidos removidas,
  64 superfícies reparadas e 107 superfícies degeneradas ignoradas. São
  indicadores de recuperação geométrica, não de aceleração.

Não há tempos controlados antes/depois por etapa, medição de pico de RAM/VRAM
ou perfil completo de kernels. Não é possível afirmar um ganho global como
“duas vezes mais rápido” ou “redução de 80%”.

Nenhuma nova geração da cidade nem benchmark foi executado para este relatório.
A análise utilizou o código, os commits e os registros já existentes.

## 10. Gargalos e limites que permanecem

1. LAS/LAZ comuns podem ser lidos novamente para diferentes tiles.
2. Tiles e edifícios continuam com orquestração sequencial.
3. GEOS, a construção/localização do TIN, correções topológicas e escrita OBJ
   permanecem total ou parcialmente na CPU.
4. Montagem, índices de conexão e armazenamento das partes exigem memória
   proporcional à malha completa.
5. Transferências e sincronizações CPU/GPU continuam existindo; tarefas
   pequenas ou a precisão dupla podem limitar o ganho na GPU.
6. Não há deduplicação global dos pontos XYZ das nuvens. A deduplicação de
   amostras e de vértices de malha atende a outros objetivos.
7. A etapa de conexão pode subdividir triângulos e aumentar o tamanho da saída.

## 11. Como medir o ganho em uma próxima avaliação

Uma comparação válida deve usar as mesmas fontes, footprint, recorte, classes,
espaçamento do terreno, backend, saídas e versão do código. Separe os cenários:

- Execução completa em CPU e CUDA, sem reutilizar os caches dos tiles.
- Reconstrução somente das malhas a partir dos mesmos intermediários.
- Retomada com cache, identificada explicitamente como retomada.

Registre o tempo total e por etapa, os pontos e triângulos processados, os
fallbacks, a memória máxima e a configuração da máquina. Separe a primeira
execução, que pode compilar kernels e aquecer caches, das execuções seguintes.
Compare múltiplas repetições e apresente a mediana e a variação.

O fator de aceleração será `tempo_CPU / tempo_CUDA`; a redução percentual,
`100 × (tempo_CPU − tempo_CUDA) / tempo_CPU`. Só devem ser publicados depois
de obtidos tempos comparáveis. Não foram calculados neste relatório.
