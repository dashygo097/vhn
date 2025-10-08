#pragma once

#ifdef __VITIS_HLS__
#include <ap_fixed.h>
#include <ap_float.h>
#include <ap_int.h>
#endif

#ifdef __VITIS_HLS__
#define UINT(n) typedef ap_uint<n> uint##n;
#define SINT(n) typedef ap_int<n> sint##n;
#define FIXED(w, i) typedef ap_fixed<w, i> fixed_##w##_##i;
#define FLOAT(total_bits, exp_bits, mant_bits)                                 \
  typedef ap_float<total_bits,                                                 \
                   exp_bits> float##total_bits##_e##exp_bits##m##mant_bits;
#endif
