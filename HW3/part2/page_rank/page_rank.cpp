#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double *scoreOld = new double[numNodes];
  double equalProb = 1.0 / numNodes;
  int i = 0;
  int dynamicChunk = numNodes / 100000;
  const int chunkMin = 4, chunkMax = 10000;
  if (dynamicChunk < chunkMin)
  {
    dynamicChunk = chunkMin;
  }
  else if (dynamicChunk > chunkMax)
  {
    dynamicChunk = chunkMax;
  }
  int cur = 0;
  #pragma omp parallel for private(i)
  for (i = 0; i < numNodes; ++i)
  {
    solution[i] = equalProb;
  }
  bool converged = false;
  while (!converged) {
    double golbalDiff = 0.0;
    double noOutgoingSum = 0.0;
    #pragma omp parallel
    {
      #pragma omp for private(i)
      for (i = 0; i < numNodes; ++i)
      {
        scoreOld[i] = solution[i];
        solution[i] = 0.0;
      }
      #pragma omp for private(cur) schedule(dynamic, dynamicChunk)
      for (cur = 0; cur < numNodes; ++cur)
      {
        const Vertex *start = incoming_begin(g, cur);
        size_t sizeIncome = incoming_size(g, cur);
        for (size_t i = 0; i < sizeIncome; ++i)
        {
          Vertex v = start[i];
          solution[cur] += scoreOld[v] / outgoing_size(g, v);
        }
      }
      #pragma omp for private(i)
      for (i = 0; i < numNodes; ++i)
      {
        solution[i] = damping * solution[i] + (1.0 - damping) / numNodes;
      }
      #pragma omp for private(i) reduction(+: noOutgoingSum)
      for (i = 0; i < numNodes; ++i)
      {
        if (outgoing_size(g, i) == 0)
        {
          noOutgoingSum += damping * scoreOld[i] / numNodes;
        }
      }
      #pragma omp for private(i)
      for (i = 0; i < numNodes; ++i)
      {
        solution[i] += noOutgoingSum;
      }
      #pragma omp for private(i) reduction(+: golbalDiff)
      for (i = 0; i < numNodes; ++i)
      {
        golbalDiff += abs(solution[i] - scoreOld[i]);
      }
      #pragma omp single
      {
        if (golbalDiff < convergence)
        {
          converged = true;
        }
      }
    }
  }
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
