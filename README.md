# Multidimensional array for Rust

## Overview

The mdarray crate provides a multidimensional array for Rust. Its main target
is for numeric types, however generic types are supported as well. The purpose
is to provide a generic container type that is simple and flexible to use,
with interworking to other crates for e.g. BLAS/LAPACK functionality.

Here are the main features of mdarray:

- Dense array type, with dynamic or inline allocation.
- Static or dynamic array dimensions, or fully dynamic including rank.
- Standard Rust mechanisms are used for e.g. indexing and iteration.
- Generic expressions for multidimensional iteration.

The design is inspired from other Rust crates (ndarray, nalgebra, bitvec, dfdx
and candle), the proposed C++ mdarray and mdspan types, and multidimensional
arrays in other languages.

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
