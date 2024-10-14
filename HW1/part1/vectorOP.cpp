#include "PPintrin.h"

// implementation of absSerial(), but it is vectorized using PP intrinsics
void absVector(float *values, float *output, int N)
{
  __pp_vec_float x;
  __pp_vec_float result;
  __pp_vec_float zero = _pp_vset_float(0.f);
  __pp_mask maskAll, maskIsNegative, maskIsNotNegative;

  //  Note: Take a careful look at this loop indexing.  This example
  //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
  //  Why is that the case?
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {

    // All ones
    maskAll = _pp_init_ones();

    // All zeros
    maskIsNegative = _pp_init_ones(0);

    // Load vector of values from contiguous memory addresses
    _pp_vload_float(x, values + i, maskAll); // x = values[i];

    // Set mask according to predicate
    _pp_vlt_float(maskIsNegative, x, zero, maskAll); // if (x < 0) {

    // Execute instruction using mask ("if" clause)
    _pp_vsub_float(result, zero, x, maskIsNegative); //   output[i] = -x;

    // Inverse maskIsNegative to generate "else" mask
    maskIsNotNegative = _pp_mask_not(maskIsNegative); // } else {

    // Execute instruction ("else" clause)
    _pp_vload_float(result, values + i, maskIsNotNegative); //   output[i] = x; }

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);
  }
}

void clampedExpVector(float *values, int *exponents, float *output, int N)
{
  //
  // PP STUDENTS TODO: Implement your vectorized version of
  // clampedExpSerial() here.
  //
  // Your solution should work for any value of
  // N and VECTOR_WIDTH, not just when VECTOR_WIDTH divides N
  //
  __pp_vec_float base;
  __pp_vec_int exp;
  __pp_vec_float result;

  __pp_vec_int zeroi = _pp_vset_int(0);
  __pp_vec_int onei = _pp_vset_int(1);
  __pp_vec_float ceilf = _pp_vset_float(9.999999f);
  
  __pp_mask maskAll, maskExpGtZero, maskResultGtCeil;
  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    // init mask
    if(i + VECTOR_WIDTH > N)
    {
      // Last iteration, in case N is not a multiple of VECTOR_WIDTH 
      maskAll = _pp_init_ones(N - i);
    }
    else
    {
      maskAll = _pp_init_ones();
    }
    maskExpGtZero = _pp_init_ones(0);
    maskResultGtCeil = _pp_init_ones(0);


    // Load vector of values from contiguous memory addresses
    _pp_vload_float(base, values + i, maskAll);
    _pp_vload_int(exp, exponents + i, maskAll);

    // mask[j] = 1 if exp[j] > 0
    _pp_vgt_int(maskExpGtZero, exp, zeroi, maskAll);
    // init result with all 1
    _pp_vset_float(result, 1.f, maskAll);

    // if there is any exp > 0
    while(_pp_cntbits(maskExpGtZero) > 0)
    {
      // multiply base with result, decrease exp by 1, update mask
      _pp_vmult_float(result, result, base, maskExpGtZero);
      _pp_vsub_int(exp, exp, onei, maskExpGtZero);
      _pp_vgt_int(maskExpGtZero, exp, zeroi, maskAll);

    }

    // mask[j] = 1 if result[j] > ceil, and use it to clamp result[j] to ceilValue
    _pp_vgt_float(maskResultGtCeil, result, ceilf, maskAll);
    _pp_vmove_float(result, ceilf, maskResultGtCeil);

    // Write results back to memory
    _pp_vstore_float(output + i, result, maskAll);


  }
}

// returns the sum of all elements in values
// You can assume N is a multiple of VECTOR_WIDTH
// You can assume VECTOR_WIDTH is a power of 2
float arraySumVector(float *values, int N)
{
  __pp_mask leftMask, maskAll = _pp_init_ones();
  float result = 0;
  __pp_vec_float adding, adding2;
  //
  // PP STUDENTS TODO: Implement your vectorized version of arraySumSerial here
  //

  for (int i = 0; i < N; i += VECTOR_WIDTH)
  {
    int current = VECTOR_WIDTH;
    _pp_vload_float(adding, values + i, maskAll);
    while (current > 1)
    {
      leftMask = _pp_init_ones(current);
      _pp_hadd_float(adding, adding);
      _pp_interleave_float(adding2, adding);
      _pp_vset_float(adding, 0, maskAll);
      _pp_vmove_float(adding, adding2, leftMask);
      current /= 2;
    }
    result += adding.value[0];
  }

  return result;
}