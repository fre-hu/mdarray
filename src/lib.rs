//! # Multidimensional array for Rust
//!
//! ## Overview
//!
//! The mdarray crate provides a multidimensional array for Rust. Its main target
//! is for numeric types, however generic types are supported as well. The purpose
//! is to provide a generic container type that is simple and flexible to use,
//! with interworking to other crates for e.g. BLAS/LAPACK functionality.
//!
//! Here are the main features of mdarray:
//!
//! - Dense array type, where the rank is known at compile time.
//! - Static or dynamic array dimensions, with optional inline storage.
//! - Standard Rust mechanisms are used for e.g. indexing and iteration.
//! - Generic expressions for multidimensional iteration.
//!
//! The design is inspired from other Rust crates (ndarray, nalgebra, bitvec, dfdx
//! and candle), the proposed C++ mdarray and mdspan types, and multidimensional
//! arrays in other languages.
//!
//! ## Array types
//!
//! The basic array type is `Tensor` for a dense array that owns the storage,
//! similar to the Rust `Vec` type. It is parameterized by the element type,
//! the shape (i.e. the size of each dimension) and optionally an allocator.
//!
//! `Array` is a dense array which stores elements inline, similar to the Rust
//! `array` type. The shape must consist of dimensions with constant size.
//!
//! `View` and `ViewMut` are array types that refer to a parent array. They are
//! used for example when creating array views without duplicating elements.
//!
//! `Slice` is a generic array reference, similar to the Rust `slice` type.
//! It consists of a pointer to an internal structure that holds the storage
//! and the layout mapping. All arrays can be dereferenced to an array slice.
//!
//! The following type aliases are provided:
//!
//! - `DTensor<T, const N: usize, ...>` for a dense array with a given rank.
//! - `DSlice<T, const N: usize, ...>` for an array slice with a given rank.
//!
//! The rank can be dynamic using the `DynRank` shape type. This is the default
//! for array types if no shape is specified.
//!
//! The layout mapping describes how elements are stored in memory. The mapping
//! is parameterized by the shape and the layout. It contains the dynamic size
//! and stride per dimension when needed.
//!
//! The layout is `Dense` if elements are stored contiguously without gaps, and
//! it is `Strided` if all dimensions can have arbitrary strides.
//!
//! The array elements are stored in row-major or C order, where the first
//! dimension is the outermost one.
//!
//! ## Indexing and views
//!
//! Scalar indexing is done using the normal square-bracket index operator and
//! an array of `usize` per dimension as index. A scalar `usize` can be used for
//! linear indexing. If the layout is `Dense`, a range can also be used to select
//! a slice.
//!
//! An array view can be created with the `view` and `view_mut` methods, which
//! take indices per dimension as arguments. Each index can be either a range
//! or `usize`. The resulting array layout depends on both the layout inferred
//! from the indices and the input layout.
//!
//! For two-dimensional arrays, a view of one column or row can be created with
//! the `col`, `col_mut`, `row` and `row_mut` methods, and a view of the diagonal
//! with `diag` and `diag_mut`.
//!
//! If the array layout is not known, `remap`, `remap_mut` and `into_mapping` can
//! be used to change layout.
//!
//! ## Iteration
//!
//! An iterator can be created from an array with the `iter`, `iter_mut` and
//! `into_iter` methods like for `Vec` and `slice`.
//!
//! Expressions are similar to iterators, but support multidimensional iteration
//! and have consistency checking of shapes. An expression is created with the
//! `expr`, `expr_mut` and `into_expr` methods. Note that the array types `View`
//! and `ViewMut` are also expressions.
//!
//! There are methods for for evaluating expressions or converting into other
//! expressions, such as `eval`, `for_each` and `map`. Two expressions can be
//! merged to an expression of tuples with the `zip` method or free function.
//!
//! When merging expressions, if the rank differs the expression with the lower
//! rank is broadcast into the larger shape by adding outer dimensions. It is not
//! possible to broadcast mutable arrays or when moving elements out of an array.
//!
//! For multidimensional arrays, iteration over a single dimension can be done
//! with `outer_expr`, `outer_expr_mut`, `axis_expr` and `axis_expr_mut`.
//! The resulting expressions give array views of the remaining dimensions.
//!
//! It is also possible to iterate over all except one dimension with `cols`,
//! `cols_mut`, `lanes`, `lanes_mut`, `rows` and `rows_mut`.
//!
//! ## Operators
//!
//! Arithmetic, logical, negation, comparison and compound assignment operators
//! are supported for arrays and expressions.
//!
//! If at least one of the inputs is an array that is passed by value, the
//! operation is evaluated directly and the input array is reused for the result.
//! Otherwise, if all input parameters are array references or expressions, an
//! expression is returned. In the latter case, the result may have a different
//! element type.
//!
//! For comparison operators, the parameters must always be arrays that are passed
//! by reference. For compound assignment operators, the first parameter is always
//! a mutable reference to an array where the result is stored.
//!
//! Scalar parameters must passed using the `fill` function that wraps a value in
//! an `Fill<T>` expression. If a type does not implement the `Copy` trait, the
//! parameter must be passed by reference.
//!
//! ## Example
//!
//! This example implements matrix multiplication and addition `C = A * B + C`.
//! The matrices use row-major ordering, and the inner loop runs over one row in
//! `B` and `C`. By using iterator-like expressions the array bounds checking is
//! avoided, and the compiler is able to vectorize the inner loop.
//!
//! ```
//! use mdarray::{expr::Expression, tensor, view, DSlice};
//!
//! fn matmul(a: &DSlice<f64, 2>, b: &DSlice<f64, 2>, c: &mut DSlice<f64, 2>) {
//!     for (mut ci, ai) in c.rows_mut().zip(a.rows()) {
//!         for (aik, bk) in ai.expr().zip(b.rows()) {
//!             for (cij, bkj) in ci.expr_mut().zip(bk) {
//!                 *cij = aik.mul_add(*bkj, *cij);
//!             }
//!         }
//!     }
//! }
//!
//! let a = view![[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]];
//! let b = view![[0.0, 1.0], [1.0, 1.0]];
//!
//! let mut c = tensor![[0.0; 2]; 3];
//!
//! matmul(&a, &b, &mut c);
//!
//! assert_eq!(c, view![[4.0, 5.0], [5.0, 7.0], [6.0, 9.0]]);
//! ```

#![allow(clippy::comparison_chain)]
#![allow(clippy::needless_range_loop)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(extern_types))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(impl_trait_in_assoc_type))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(unused_results)]

/// Expression module, for multidimensional iteration.
pub mod expr;

/// Module for array slice and view indexing, and for array axis subarray types.
pub mod index;

mod array;
mod dim;
mod layout;
mod macros;
mod mapping;
mod ops;
mod raw_slice;
mod raw_tensor;
mod shape;
mod slice;
mod tensor;
mod traits;
mod view;

#[cfg(feature = "serde")]
mod serde;

#[cfg(not(feature = "nightly"))]
mod alloc {
    pub trait Allocator {}

    #[derive(Copy, Clone, Default, Debug)]
    pub struct Global;

    impl Allocator for Global {}
}

pub use array::Array;
pub use dim::{Const, Dim, Dyn};
pub use layout::{Dense, Layout, Strided};
pub use mapping::{DenseMapping, Mapping, StridedMapping};
pub use ops::{step, StepRange};
pub use shape::{ConstShape, DynRank, IntoShape, Rank, Shape};
pub use slice::{DSlice, Slice};
pub use tensor::{DTensor, Tensor};
pub use traits::IntoCloned;
pub use view::{DView, DViewMut, View, ViewMut};
