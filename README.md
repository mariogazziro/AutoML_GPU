# AutoML_GPU

Automatic phase correction of NMR spectra using brute force GPU method

Despite the need for efficient techniques to determine accurate solutions, the problem of phase adjustment of
NMR spectroscopy signals has not received much consideration recently, especially in the presence of noisy
signals. To address this gap, we present a novel methodology, based on GPU processing, that is able to find
the optimal parameter set for phase adjustment through an exhaustive search of all possible combinations
of the phase space parameters. In our experiments, we were able to reduce the execution time by as much as
a factor of five. In our case study, we also demonstrate the robustness of the proposed method against the
problem of local minima. Finally, a Bland-Altman analysis is performed to validate the entropies calculated
using CPU and GPU processing for a set of 16 experiments from brain and body metabolites using 1H
and 31P probes. The results demonstrate that our algorithm always finds the globally optimal solution,
while previous CPU-based heuristics often converge to only a locally optimal solution, while simultaneously
running faster.

=====================================================================================================================================

Super aceleração de algoritmos usando GPUs de última geração - CEPID-CEMEAI-USP

Curso prático de elaboração de algoritmos acelerados para GPUs com arquitetura Pascal – será utilizada Titan XP com 3500 núcleos, porém a maioria das técnicas abordadas se aplicam a todas as classes de GPU da fabricante NVIDIA. Após introdução sobre análise do paralelismo de algoritmos, serão apresentados estudos de caso aplicados em matrizes esparsas, ressonância magnética e do processamento de grafos em escalas de milhões. Em alguns dos casos apresentados a chega próximo de 30 vezes, e as técnicas apresentadas, como definição e ajuste fino da grade (GRID) de recursos, escolha dos tipos adequados de memória e técnicas para o escalonador de processos (WARP), podem ser facilmente adaptadas a outros problemas, logo os alunos inscritos são encorajados a trazer seus próprios problemas de pesquisa para paralelização em nossa oficina.
