# Multidimensional array for Rust

## Overview

The mdarray crate provides a multidimensional array for Rust. Its main target
is for numeric types, however generic types are supported as well. The purpose
is to provide a generic container type that is simple and flexible to use,
with interworking to other crates for e.g. BLAS/LAPACK functionality.

Here are the main features of mdarray:

- Dense array type, where the rank and element order is known at compile time.
- Column-major and row-major element order.
- Subarrays (views) can be created with arbitrary shapes and strides.
- Standard Rust mechanisms are used for e.g. slices, indexing and iteration.

The design is inspired from the Rust ndarray, nalgebra and bitvec crates,
the proposed C++ mdarray and mdspan types, and multidimensional arrays in
Julia and Matlab.

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
