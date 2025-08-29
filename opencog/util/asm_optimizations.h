/*
 * opencog/util/asm_optimizations.h
 *
 * High-performance assembly optimizations for critical cogutil operations
 *
 * Copyright (C) 2024 OpenCog Foundation
 * All Rights Reserved
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License v3 as
 * published by the Free Software Foundation and including the exceptions
 * at http://opencog.org/wiki/Licenses
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program; if not, write to:
 * Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 */

#ifndef _OPENCOG_ASM_OPTIMIZATIONS_H
#define _OPENCOG_ASM_OPTIMIZATIONS_H

#include <cstdint>
#include <cstddef>

namespace opencog {
namespace asm_opt {

/** \addtogroup grp_cogutil
 *  @{
 */

/**
 * High-performance assembly implementations for numerical operations
 */

// Optimized bit manipulation operations
#ifdef __x86_64__

/// Fast integer log2 using BSR instruction
static inline unsigned int fast_integer_log2(size_t v) {
    if (v == 0) return 0;
    
    unsigned long result;
    __asm__ volatile (
        "bsr %1, %0"
        : "=r" (result)
        : "rm" (v)
        : "cc"
    );
    return (unsigned int)result;
}

/// Fast next power of two using bit scan and shift
static inline size_t fast_next_power_of_two(size_t x) {
    if (x <= 1) return 1;
    
    unsigned long bit_pos;
    __asm__ volatile (
        "bsr %1, %0"
        : "=r" (bit_pos)
        : "rm" (x - 1)
        : "cc"
    );
    return 1UL << (bit_pos + 1);
}

/// Fast 32-bit hash function using optimized multiply and shift
static inline uint32_t fast_hash32(uint32_t key) {
    uint32_t result;
    __asm__ volatile (
        "imul $0x9e3779b9, %1, %0\n\t"  // Multiply by golden ratio constant
        "rol $13, %0\n\t"               // Rotate left by 13 bits
        "xor $0x5bd1e995, %0"          // XOR with magic constant
        : "=r" (result)
        : "r" (key)
        : "cc"
    );
    return result;
}

/// Fast 64-bit hash function with x86_64 optimizations
static inline uint64_t fast_hash64(uint64_t key) {
    // Use a simpler but still optimized approach
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccdULL;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53ULL;
    key ^= key >> 33;
    return key;
}

/// Optimized memory comparison using SIMD-style operations
static inline int fast_memcmp(const void* ptr1, const void* ptr2, size_t len) {
    const char* p1 = static_cast<const char*>(ptr1);
    const char* p2 = static_cast<const char*>(ptr2);
    
    // Handle 8-byte chunks for speed
    while (len >= 8) {
        uint64_t val1, val2;
        __asm__ volatile (
            "movq (%1), %0"
            : "=r" (val1)
            : "r" (p1)
            : "memory"
        );
        __asm__ volatile (
            "movq (%1), %0"
            : "=r" (val2)
            : "r" (p2)
            : "memory"
        );
        
        if (val1 != val2) {
            // Find the first differing byte using a simpler approach
            for (int i = 0; i < 8; i++) {
                uint8_t b1 = (val1 >> (i * 8)) & 0xff;
                uint8_t b2 = (val2 >> (i * 8)) & 0xff;
                if (b1 != b2) return b1 - b2;
            }
        }
        
        p1 += 8;
        p2 += 8;
        len -= 8;
    }
    
    // Handle remaining bytes
    while (len > 0) {
        if (*p1 != *p2) return *p1 - *p2;
        p1++;
        p2++;
        len--;
    }
    
    return 0;
}

/// High-performance atomic increment with memory ordering
static inline uint64_t fast_atomic_increment(volatile uint64_t* ptr) {
    uint64_t result;
    __asm__ volatile (
        "lock incq %0\n\t"
        "mov %0, %1"
        : "+m" (*ptr), "=r" (result)
        :
        : "memory", "cc"
    );
    return result;
}

/// High-performance atomic decrement with memory ordering
static inline uint64_t fast_atomic_decrement(volatile uint64_t* ptr) {
    uint64_t result;
    __asm__ volatile (
        "lock decq %0\n\t"
        "mov %0, %1"
        : "+m" (*ptr), "=r" (result)
        :
        : "memory", "cc"
    );
    return result;
}

/// Fast atomic compare-and-swap
static inline bool fast_atomic_cas(volatile uint64_t* ptr, uint64_t expected, uint64_t desired) {
    uint64_t prev;
    __asm__ volatile (
        "lock cmpxchgq %2, %1"
        : "=a" (prev), "+m" (*ptr)
        : "r" (desired), "0" (expected)
        : "memory", "cc"
    );
    return prev == expected;
}

#else
// Fallback implementations for non-x86_64 architectures
static inline unsigned int fast_integer_log2(size_t v) {
    if (v == 0) return 0;
    return (8*sizeof(size_t) - 1) - __builtin_clzl(v);
}

static inline size_t fast_next_power_of_two(size_t x) {
    if (x <= 1) return 1;
    return 1UL << (8*sizeof(size_t) - __builtin_clzl(x-1));
}

static inline uint32_t fast_hash32(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key;
}

static inline uint64_t fast_hash64(uint64_t key) {
    key ^= key >> 33;
    key *= 0xff51afd7ed558ccd;
    key ^= key >> 33;
    key *= 0xc4ceb9fe1a85ec53;
    key ^= key >> 33;
    return key;
}

static inline int fast_memcmp(const void* ptr1, const void* ptr2, size_t len) {
    return __builtin_memcmp(ptr1, ptr2, len);
}

static inline uint64_t fast_atomic_increment(volatile uint64_t* ptr) {
    return __sync_add_and_fetch(ptr, 1);
}

static inline uint64_t fast_atomic_decrement(volatile uint64_t* ptr) {
    return __sync_sub_and_fetch(ptr, 1);
}

static inline bool fast_atomic_cas(volatile uint64_t* ptr, uint64_t expected, uint64_t desired) {
    return __sync_bool_compare_and_swap(ptr, expected, desired);
}
#endif

/**
 * High-performance string processing operations
 */

/// Fast string hash using assembly-optimized FNV-1a algorithm
static inline uint64_t fast_string_hash(const char* str, size_t len) {
    const uint64_t FNV_OFFSET_BASIS = 14695981039346656037ULL;
    const uint64_t FNV_PRIME = 1099511628211ULL;
    
    uint64_t hash = FNV_OFFSET_BASIS;
    
#ifdef __x86_64__
    // Process 8 bytes at a time when possible
    while (len >= 8) {
        uint64_t chunk;
        __asm__ volatile (
            "movq (%1), %0"
            : "=r" (chunk)
            : "r" (str)
            : "memory"
        );
        
        // Process each byte in the chunk
        for (int i = 0; i < 8; i++) {
            uint8_t byte = (chunk >> (i * 8)) & 0xff;
            hash ^= byte;
            __asm__ volatile (
                "imulq %1, %0"
                : "+r" (hash)
                : "r" (FNV_PRIME)
                : "cc"
            );
        }
        
        str += 8;
        len -= 8;
    }
#endif
    
    // Process remaining bytes
    while (len > 0) {
        hash ^= static_cast<uint8_t>(*str);
        hash *= FNV_PRIME;
        str++;
        len--;
    }
    
    return hash;
}

/// Fast string comparison with early exit optimization
static inline int fast_string_compare(const char* str1, const char* str2, size_t len) {
#ifdef __x86_64__
    // Use 64-bit comparisons when possible
    while (len >= 8) {
        uint64_t val1, val2;
        __asm__ volatile (
            "movq (%1), %0"
            : "=r" (val1)
            : "r" (str1)
            : "memory"
        );
        __asm__ volatile (
            "movq (%1), %0"
            : "=r" (val2)
            : "r" (str2)
            : "memory"
        );
        
        if (val1 != val2) {
            return fast_memcmp(str1, str2, 8);
        }
        
        str1 += 8;
        str2 += 8;
        len -= 8;
    }
#endif
    
    // Process remaining bytes
    while (len > 0 && *str1 == *str2) {
        if (*str1 == '\0') return 0;
        str1++;
        str2++;
        len--;
    }
    
    return len > 0 ? (*str1 - *str2) : 0;
}

/** @}*/

} // namespace asm_opt
} // namespace opencog

#endif // _OPENCOG_ASM_OPTIMIZATIONS_H