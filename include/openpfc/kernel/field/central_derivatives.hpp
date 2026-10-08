// SPDX-FileCopyrightText: 2026 VTT Technical Research Centre of Finland Ltd
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

/** @file central_derivatives.hpp
 * @brief Accessor-based central 3D gradient and symmetric Hessian.
 */
#include <openpfc/kernel/data/host_device.hpp>
#include <openpfc/kernel/field/fd_stencils.hpp>
#include <type_traits>

namespace pfc::field::fd {
/// Cartesian gradient followed by the six independent Hessian components.
template <class T> struct Derivatives3D {
  T x, y, z, xx, yy, zz, xy, xz, yz;
};
namespace detail {
template <int A, int B = -1, class Sample>
OPENPFC_INLINE_HD auto shifted(const Sample &sample, int a, int b = 0) {
  return sample((A == 0 ? a : 0) + (B == 0 ? b : 0),
                (A == 1 ? a : 0) + (B == 1 ? b : 0),
                (A == 2 ? a : 0) + (B == 2 ? b : 0));
}
template <int Order, int A, bool Second, int K = 1, class Sample, class T>
OPENPFC_INLINE_HD T axis_sum(const Sample &sample, T center) {
  if constexpr (K > Order / 2) {
    return T{};
  } else {
    constexpr auto coefficient =
        Second ? EvenCentralD2<Order>::coeffs[K] : EvenCentralD1<Order>::coeffs[K];
    const T plus = shifted<A>(sample, K), minus = shifted<A>(sample, -K);
    // Difference form annihilates a constant without cancellation of weights.
    const T value = Second ? (plus - center) + (minus - center) : plus - minus;
    return T(coefficient) * value +
           axis_sum<Order, A, Second, K + 1>(sample, center);
  }
}
template <int Order, int A, int B, int K = 1, int L = 1, class Sample, class T>
OPENPFC_INLINE_HD T mixed_sum(const Sample &sample) {
  if constexpr (K > Order / 2) {
    return T{};
  } else if constexpr (L > Order / 2) {
    return mixed_sum<Order, A, B, K + 1, 1, Sample, T>(sample);
  } else {
    constexpr auto ck = EvenCentralD1<Order>::coeffs[K];
    constexpr auto cl = EvenCentralD1<Order>::coeffs[L];
    const T value = (shifted<A, B>(sample, K, L) - shifted<A, B>(sample, K, -L)) -
                    (shifted<A, B>(sample, -K, L) - shifted<A, B>(sample, -K, -L));
    return T(ck * cl) * value + mixed_sum<Order, A, B, K, L + 1, Sample, T>(sample);
  }
}
} // namespace detail

/** Evaluate central derivatives at the accessor's origin.
 *
 * Order is 2, 4 or 6. sample(dx,dy,dz) returns float or double scalar samples
 * with relative integer offsets. Positive finite hx/hy/hz, finite samples and
 * valid accessor loads are caller preconditions; no runtime validation occurs.
 * Mixed entries compose the same first derivative in each of the two axes.
 * The maximum offset in each axis is Order/2, including edge/corner samples.
 * The caller supplies periodic indexing or sufficient initialized ghost data,
 * including a boundary extension and coordinate convention. Formal interior
 * accuracy does not certify an arbitrary boundary extension. No MPI exchange,
 * ownership inference, allocation, synchronization or physics is performed.
 */
template <int Order, class Sample, class T>
OPENPFC_INLINE_HD Derivatives3D<T> evaluate(const Sample &sample, T hx, T hy, T hz) {
  static_assert(Order == 2 || Order == 4 || Order == 6, "Order must be 2, 4 or 6");
  static_assert(std::is_floating_point_v<T>, "Spacing must have floating type");
  const T center = sample(0, 0, 0);
  constexpr T d1 = T(EvenCentralD1<Order>::denom);
  constexpr T d2 = T(EvenCentralD2<Order>::denom);
  return {
      detail::axis_sum<Order, 0, false>(sample, center) / (d1 * hx),
      detail::axis_sum<Order, 1, false>(sample, center) / (d1 * hy),
      detail::axis_sum<Order, 2, false>(sample, center) / (d1 * hz),
      detail::axis_sum<Order, 0, true>(sample, center) / (d2 * hx * hx),
      detail::axis_sum<Order, 1, true>(sample, center) / (d2 * hy * hy),
      detail::axis_sum<Order, 2, true>(sample, center) / (d2 * hz * hz),
      detail::mixed_sum<Order, 0, 1, 1, 1, Sample, T>(sample) / (d1 * d1 * hx * hy),
      detail::mixed_sum<Order, 0, 2, 1, 1, Sample, T>(sample) / (d1 * d1 * hx * hz),
      detail::mixed_sum<Order, 1, 2, 1, 1, Sample, T>(sample) / (d1 * d1 * hy * hz)};
}
} // namespace pfc::field::fd
