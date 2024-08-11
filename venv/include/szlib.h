/**
 * @file szlib.h
 *
 * @section LICENSE
 * Copyright 2024 Mathis Rosenhauer, Moritz Hanke, Joerg Behrens, Luis Kornblueh
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
 * STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
 * OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * @section DESCRIPTION
 *
 * Adaptive Entropy Coding library
 *
 */

#ifndef SZLIB_H
#define SZLIB_H 1

#include <libaec.h>

#define SZ_ALLOW_K13_OPTION_MASK 1
#define SZ_CHIP_OPTION_MASK 2
#define SZ_EC_OPTION_MASK 4
#define SZ_LSB_OPTION_MASK 8
#define SZ_MSB_OPTION_MASK 16
#define SZ_NN_OPTION_MASK 32
#define SZ_RAW_OPTION_MASK 128

#define SZ_OK AEC_OK
#define SZ_OUTBUFF_FULL 2

#define SZ_NO_ENCODER_ERROR -1
#define SZ_PARAM_ERROR AEC_CONF_ERROR
#define SZ_MEM_ERROR AEC_MEM_ERROR

#define SZ_MAX_PIXELS_PER_BLOCK 32
#define SZ_MAX_BLOCKS_PER_SCANLINE 128
#define SZ_MAX_PIXELS_PER_SCANLINE                              \
    (SZ_MAX_BLOCKS_PER_SCANLINE) * (SZ_MAX_PIXELS_PER_BLOCK)

typedef struct SZ_com_t_s
{
    int options_mask;
    int bits_per_pixel;
    int pixels_per_block;
    int pixels_per_scanline;
} SZ_com_t;

LIBAEC_DLL_EXPORTED int SZ_BufftoBuffCompress(
    void *dest, size_t *destLen,
    const void *source, size_t sourceLen,
    SZ_com_t *param);
LIBAEC_DLL_EXPORTED int SZ_BufftoBuffDecompress(
    void *dest, size_t *destLen,
    const void *source, size_t sourceLen,
    SZ_com_t *param);

LIBAEC_DLL_EXPORTED int SZ_encoder_enabled(void);

#endif /* SZLIB_H */
