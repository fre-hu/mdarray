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
//! - Subarrays (views) can be created with arbitrary shapes and strides.
//! - Standard Rust mechanisms are used for e.g. slices, indexing and iteration.
//! - Generic expressions for multidimensional iteration.
//!
//! The design is inspired from other Rust crates (ndarray, nalgebra, bitvec
//! and dfdx), the proposed C++ mdarray and mdspan types, and multidimensional
//! arrays in other languages.
//!
//! ## Array types
//!
//! The basic array type is `Grid` for a dense array that owns the storage,
//! similar to the Rust `Vec` type. It is parameterized by the element type,
//! the rank (i.e. the number of dimensions) and optionally an allocator.
//!
//! `Expr` and `ExprMut` are array types that refer to a parent array. They are
//! used for example when creating array views without duplicating elements.
//!
//! `Span` is a generic array reference, similar to the Rust `slice` type.
//! It consists of a pointer to an internal structure that holds the storage
//! and the layout mapping. All arrays can be dereferenced to an array span.
//!
//! The following type aliases are provided:
//!
//! - `DGrid<T, const N: usize, ...>` for a dense array with a given rank.
//! - `DSpan<T, const N: usize, ...>` for an array span with a given rank.
//!
//! The layout mapping describes how elements are stored in memory. The mapping
//! is parameterized by the rank and the layout. It contains the shape (i.e.
//! the size of each dimension), and strides per dimension if needed.
//!
//! The layout is `Dense` if elements are stored contiguously without gaps.
//! The layout is `General` if each dimension can have arbitrary stride except
//! for the innermost one, which has unit stride. It is compatible with the
//! BLAS/LAPACK general matrix storage.
//!
//! The layout is `Flat` if the innermost dimension can have arbitrary stride
//! and the other dimensions must follow in order, allowing for linear indexing.
//! The layout is `Strided` if all dimensions can have arbitrary strides.
//!
//! The array elements are stored in column-major or Fortran order, where the
//! first dimension is the innermost one.
//!
//! ## Indexing and views
//!
//! Scalar indexing is done using the normal square-bracket index operator and
//! an array of `usize` per dimension as index.
//!
//! If the array layout supports linear indexing (i.e. the layout is `Dense` or
//! `Flat`), a scalar `usize` can also be used as index. If the layout is `Dense`,
//! a range can be used to select a slice.
//!
//! If linear or slice indexing is possible but the array layout is not known,
//! `remap`, `remap_mut` and `into_mapping` can be used to change layout.
//! Alternatively, `flatten`, `flatten_mut` and `into_flattened` can be used
//! to change to a one-dimensional array.
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
//! ## Iteration
//!
//! An iterator can be created from an array with the `iter`, `iter_mut` and
//! `into_iter` methods like for `Vec` and `slice`.
//!
//! Expressions are similar to iterators, but support multidimensional iteration
//! and have consistency checking of shapes. An expression is created with the
//! `expr`, `expr_mut` and `into_expr` methods. Note that the array types `Expr`
//! and `ExprMut` are also expressions.
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
//! an `Expression<Fill<T>>` expression. If a type does not implement the `Copy`
//! trait, the parameter must be passed by reference.
//!
//! ## Example
//!
//! This example implements matrix multiplication and addition `C = A * B + C`.
//! The matrices use column-major ordering, and the inner loop runs over one column
//! in `A` and `C`. By using iterator-like expressions the array bounds checking
//! is avoided, and the compiler is able to vectorize the inner loop.
//!
//! ```
//! use mdarray::{expr, grid, DSpan, Expression};
//!
//! fn matmul(a: &DSpan<f64, 2>, b: &DSpan<f64, 2>, c: &mut DSpan<f64, 2>) {
//!     for (mut cj, bj) in c.cols_mut().zip(b.cols()) {
//!         for (ak, bkj) in a.cols().zip(bj) {
//!             for (cij, aik) in cj.expr_mut().zip(ak) {
//!                 *cij = aik.mul_add(*bkj, *cij);
//!             }
//!         }
//!     }
//! }
//!
//! let a = expr![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let b = expr![[0.0, 1.0], [1.0, 1.0]];
//!
//! let mut c = grid![[0.0; 3]; 2];
//!
//! matmul(&a, &b, &mut c);
//!
//! assert_eq!(c, expr![[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]);
//! ```

#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(associated_type_defaults))]
#![cfg_attr(feature = "nightly", feature(extern_types))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(int_roundings))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
#![feature(impl_trait_in_assoc_type)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(unused_results)]

/// Expression module, for multidimensional iteration.
pub mod expr;

/// Module for array span and view indexing, and for array axis subarray types.
pub mod index;

/// Array layout mapping module.
pub mod mapping;

mod dim;
mod expression;
mod grid;
mod iter;
mod layout;
mod macros;
mod ops;
mod raw_grid;
mod raw_span;
mod span;
mod traits;

#[cfg(feature = "serde")]
mod serde;

#[cfg(not(feature = "nightly"))]
mod alloc {
    pub trait Allocator {}

    #[derive(Copy, Clone, Default, Debug)]
    pub struct Global;

    impl Allocator for Global {}
}

pub use dim::{Const, Dim, Shape, Strides};
pub use expression::Expression;
pub use grid::{DGrid, Grid};
pub use iter::Iter;
pub use layout::{Dense, Flat, General, Layout, Strided, Uniform, UnitStrided};
pub use ops::{step, StepRange};
pub use span::{DSpan, Span};
pub use traits::{Apply, IntoCloned, IntoExpression};
