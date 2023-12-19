/* MIT License Copyright (c) https://tinygrad.org/

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
 
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#ifndef TINYGRAD_COMMON_H_
#define TINYGRAD_COMMON_H_

namespace tinygrad {

#if defined(_MSC_VER) // Microsoft Visual Studio
#define TINYGRAD_INLINE __forceinline
#elif defined(__GNUC__) // GCC
#define TINYGRAD_INLINE inline __attribute__((always_inline))
#elif defined(__clang__) // Clang
#define TINYGRAD_INLINE inline __attribute__((always_inline))
#elif defined(__INTEL_COMPILER) // Intel compiler
#define TINYGRAD_INLINE __forceinline
#else
#define TINYGRAD_INLINE inline // Other compilers, use standard inline
#endif

} // namespace tinygrad

#endif // TINYGRAD_COMMON_H_
