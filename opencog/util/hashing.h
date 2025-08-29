/*
 * opencog/util/hashing.h
 *
 * Copyright (C) 2002-2007 Novamente LLC
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

#ifndef _OPENCOG_HASHING_H
#define _OPENCOG_HASHING_H

#include <boost/functional/hash.hpp>

#include <opencog/util/tree.h>

#ifdef __x86_64__
// Assembly-optimized hash functions for x86_64
namespace detail {
    
/// Fast 64-bit hash using optimized assembly
inline std::size_t fast_hash_combine(std::size_t seed, std::size_t value) {
    std::size_t result;
    __asm__ volatile (
        "xor %2, %0\n\t"                    // seed ^= value
        "movabs $0x9e3779b97f4a7c15, %%rax\n\t"  // Load golden ratio constant  
        "add %%rax, %0\n\t"                 // Add golden ratio
        "rol $6, %0\n\t"                    // Rotate left by 6
        "add %1, %0"                        // Add original seed shifted
        : "=r" (result)
        : "r" (seed << 6), "r" (value), "0" (seed)
        : "rax", "cc"
    );
    return result;
}

/// Optimized string hash for pointer-based hashing
inline std::size_t fast_ptr_hash(const void* ptr) {
    std::size_t addr = reinterpret_cast<std::size_t>(ptr);
    std::size_t result;
    __asm__ volatile (
        "mov %1, %0\n\t"                    // Copy address
        "shr $3, %0\n\t"                   // Shift right by 3 (remove alignment bits)
        "movabs $0x9e3779b97f4a7c15, %%rax\n\t"  // Golden ratio
        "imul %%rax, %0\n\t"               // Multiply by golden ratio
        "mov %0, %%rax\n\t"                // Copy to rax
        "shr $32, %%rax\n\t"              // Get high 32 bits
        "xor %%rax, %0"                   // XOR high and low parts
        : "=r" (result)
        : "r" (addr)
        : "rax", "cc"
    );
    return result;
}

} // namespace detail
#endif

namespace opencog
{
/** \addtogroup grp_cogutil
 *  @{
 */

//! Functor returning the address of an object pointed by an iterator.
/**
 * Useful for defining the hash function of an iterator.
 */
template<typename It>
struct obj_ptr_hash {
    size_t operator()(const It& it) const {
#ifdef __x86_64__
        return detail::fast_ptr_hash(&(*it));
#else
        return boost::hash_value(&(*it));
#endif
    }
};

template < typename T,
typename Hash = boost::hash<T> >
struct deref_hash {
    deref_hash(const Hash& h = Hash()) : hash(h) {}
    size_t operator()(const T& t) const {
        return hash(*t);
    }
    Hash hash;
};

template < typename T,
typename Equals = std::equal_to<T> >
struct deref_equals {
    deref_equals(const Equals& e = Equals()) : equals(e) {}
    bool operator()(const T& x, const T& y) const {
        return equals(*x, *y);
    }
    Equals equals;
};

template<typename T>
std::size_t hash_value(const tree<T>& tr)
{
    return boost::hash_range(tr.begin(), tr.end());
}

//! Functor comparing the addresses of objects pointed by
//! tree iterators.
/**
 * Useful for storing iterators in a std::map.
 * (the tree has pointer to node, we use that to identify the
 * tree node uniquely).
 */
template<typename It>
struct obj_ptr_cmp {
    bool operator()(const It& lit, const It& rit) const {
        return ((void *) lit.node) < ((void *) rit.node);
    }
};

/** @}*/
} //~namespace opencog

#endif // _OPENCOG_HASHING_H
