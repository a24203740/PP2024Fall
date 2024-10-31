#include "bfs.h"

#include <utility>
#include <cstddef>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1
#define CHUNKSIZE 16384
void vertex_set_clear(vertex_set *list) { list->count = 0; }

void vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step_void(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                   int *distances) {
  int i = 0;
  #pragma omp parallel for shared(frontier, new_frontier, distances) \
    private(i) schedule(static, CHUNKSIZE)
  for (i = 0; i < frontier->count; i++) {

    int node = frontier->vertices[i];

    int start_edge = g->outgoing_starts[node];
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                              : g->outgoing_starts[node + 1];

    // attempt to add all neighbors to the new frontier
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int outgoing = g->outgoing_edges[neighbor];
      if (distances[outgoing] != NOT_VISITED_MARKER)
        continue;

      bool success = __sync_bool_compare_and_swap(
          &distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1);
      if (!success)
        continue;
      // atomic version of index = new_frontier->count++;
      int index = __sync_fetch_and_add(&new_frontier->count, 1);
      new_frontier->vertices[index] = outgoing;
    }
  }
}

std::pair<int,int> top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                   int *distances) {
  int i = 0;
  int newFrontOutSum = 0, newFrontInSum = 0;
  #pragma omp parallel for shared(frontier, new_frontier, distances) \
    private(i) reduction(+:newFrontOutSum, newFrontInSum) schedule(static, CHUNKSIZE)
  for (i = 0; i < frontier->count; i++) {

    int node = frontier->vertices[i];

    int start_edge = g->outgoing_starts[node];
    int end_edge = (node == g->num_nodes - 1) ? g->num_edges
                                              : g->outgoing_starts[node + 1];

    // attempt to add all neighbors to the new frontier
    for (int neighbor = start_edge; neighbor < end_edge; neighbor++) {
      int outgoing = g->outgoing_edges[neighbor];
      if (distances[outgoing] != NOT_VISITED_MARKER)
        continue;

      bool success = __sync_bool_compare_and_swap(
          &distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1);
      if (!success)
        continue;
      // atomic version of index = new_frontier->count++;
      int index = __sync_fetch_and_add(&new_frontier->count, 1);
      new_frontier->vertices[index] = outgoing;
      int outSE = g->outgoing_starts[outgoing], outED = (outgoing == g->num_nodes - 1) ? g->num_edges
                                                                                       : g->outgoing_starts[outgoing + 1];
      int inSE = g->incoming_starts[outgoing], inED = (outgoing == g->num_nodes - 1) ? g->num_edges
                                                                                     : g->incoming_starts[outgoing + 1];
      newFrontOutSum += outED - outSE;
      newFrontInSum += inED - inSE;
    }
  }
  return std::make_pair(newFrontOutSum, newFrontInSum);
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;


  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
  
  while (frontier->count != 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif
    vertex_set_clear(new_frontier);
    top_down_step_void(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

int bottomUpOneIteration(Graph graph, int *distance, int currentDistance) 
{
  int thisIterationVisitedCount = 0;
  #pragma omp parallel for \
    shared(distance) reduction(+:thisIterationVisitedCount) schedule(dynamic, CHUNKSIZE)
  for (int i = 0; i < graph->num_nodes; i++) {
    if(distance[i] != NOT_VISITED_MARKER) {
      continue;
    }
    int start_edge = graph->incoming_starts[i];
    int end_edge = (i == graph->num_nodes - 1) ? graph->num_edges
                                          : graph->incoming_starts[i + 1];
    for (int edge = start_edge; edge < end_edge; edge++) {
      int incomingNeighbor = graph->incoming_edges[edge];
      if (distance[incomingNeighbor] == currentDistance) {
        distance[i] = currentDistance + 1;
        thisIterationVisitedCount++;
        break;
      }
    }
  }
  return thisIterationVisitedCount;
}

void bfs_bottom_up(Graph graph, solution *sol) {
  for (int i = 0; i < graph->num_nodes; i++) {
    sol->distances[i] = NOT_VISITED_MARKER;
  }
  sol->distances[ROOT_NODE_ID] = 0;
  int visitedCount = 1;
  int currentDistance = 0;
  while (visitedCount < graph->num_nodes) {
    int thisIterationVisitedCount = bottomUpOneIteration(graph, sol->distances, currentDistance);
    currentDistance++;
    visitedCount += thisIterationVisitedCount;
    if (thisIterationVisitedCount == 0) {
      break;
    }
  }
  // For PP students:
  //
  // You will need to implement the "bottom up" BFS here as
  // described in the handout.
  //
  // As a result of your code's execution, sol.distances should be
  // correctly populated for all nodes in the graph.
  //
  // As was done in the top-down case, you may wish to organize your
  // code by creating subroutine bottom_up_step() that is called in
  // each step of the BFS process.
}

void bfs_hybrid(Graph graph, solution *sol) {
  // meta data for hybrid
  bool isTopDown = true;
  int outDegSumOfFrontier = 0;
  int inDegSumOfUnvisited = 0;
  int numOfUnvisited = graph->num_nodes;
  
  // heuristic parameters provided by the handout
  int alpha = 14, beta = 24;

  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;


  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;
 
  if (graph->num_nodes == 1) {
    return;
  }
  
  outDegSumOfFrontier = graph->outgoing_starts[1];
  inDegSumOfUnvisited = graph->num_edges - graph->incoming_starts[1];
  int numOfFrontier = 0;
  numOfUnvisited -= 1; 

  int currentDistance = 0;
  
  while (numOfUnvisited > 0) {
    if (isTopDown) {
      vertex_set_clear(new_frontier);
      auto [out, in] = top_down_step(graph, frontier, new_frontier, sol->distances);
      outDegSumOfFrontier = out;
      inDegSumOfUnvisited -= in;
      numOfUnvisited -= new_frontier->count;
      if (new_frontier->count == 0)break;
      // printf("top %d\n", new_frontier->count);
      std::swap(frontier, new_frontier);
    }
    else {
      int frontierCount = bottomUpOneIteration(graph, sol->distances, currentDistance); 
      numOfUnvisited -= frontierCount;
      // printf("bot %d\n", frontierCount);
      if (frontierCount == 0)break;
    }
    currentDistance++;

    if (isTopDown) {
      if (outDegSumOfFrontier > (inDegSumOfUnvisited / alpha) ) {
        isTopDown = false;
      }
    }
  }

  // For PP students:
  //
  // You will need to implement the "hybrid" BFS here as
  // described in the handout.
}
