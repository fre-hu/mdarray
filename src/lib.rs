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
//! The design is inspired from the Rust ndarray, nalgebra and bitvec crates,
//! the proposed C++ mdarray and mdspan types, and multidimensional arrays in
//! Julia and Matlab.
//!
//! ## Array types
//!
//! The base type for multidimensional arrays is `Array<B>`, where the generic
//! parameter is a buffer for the array storage. The following variants exist:
//!
//! - `Array<GridBuffer>` is a dense array that owns the storage, similar to
//!   the Rust `Vec` type.
//! - `Array<ViewBuffer>` and `Array<ViewBufferMut>` are arrays that refer to a
//!   parent array. They are used for example when creating a view of a larger
//!   array without duplicating elements.
//! - `Array<SpanBuffer>` is used as a generic array reference, similar to the
//!   Rust `slice` type. It consists of a pointer to an internal structure that
//!   holds the storage and the layout mapping. Arrays and array views can be
//!   dereferenced to an array span.
//!
//! The layout mapping describes how elements are stored in memory. The mapping
//! is parameterized by the rank (i.e. the number of dimensions) and the array
//! layout. It contains the shape (i.e. the size in each dimension), and the
//! strides per dimension if needed.
//!
//! The array layout is `Dense` if elements are stored contiguously without gaps.
//! In this case, the strides are calculated from the shape and not stored as
//! part of the layout. The layout is `General` if each dimension can have
//! arbitrary stride except for the innermost one, which has unit stride. It is
//! compatible with the BLAS/LAPACK general matrix storage.
//!
//! The layout is `Flat` if the innermost dimension can have arbitrary stride
//! and the other dimensions must follow in order, allowing for linear indexing.
//! The layout is `Strided` if all dimensions can have arbitrary strides.
//!
//! The array elements are stored in column-major order, also known as Fortran
//! order where the first dimension is the innermost one.
//!
//! The following type aliases are provided:
//!
//! - `Grid<T, const N: usize, A = Global>` for a dense array
//! - `Span<T, const N: usize, F = Dense>` for an array span
//! - `View<T, const N: usize, F = Dense>` for an array view
//! - `ViewMut<T, const N: usize, F = Dense>` for a mutable array view
//!
//! Prefer using `Span` instead of array views for function parameters, since
//! they can refer to either an owned array or an array view. Array views
//! are useful for example when lifetimes need to be maintained in function
//! return types.
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
//! ## Iteration
//!
//! An iterator can be created from an array with the `iter`, `iter_mut` and
//! `into_iter` methods like for `Vec` and `slice`.
//!
//! Expressions are similar to iterators, but have consistency checking of shapes
//! and support multidimensional iteration more efficiently. An expression is
//! created with the `expr`, `expr_mut` and `into_expr` methods.
//!
//! An expression consists of a base type `Expression<P>`, where the generic
//! parameter is a tree of producer nodes. The base type has several methods
//! for evaluating expressions or converting into other expressions, such as
//! `eval`, `for_each` and `map`.
//!
//! Two expressions can be merged to an expression of tuples with the `zip` method
//! or free function. When merging expressions, if the rank differs the expression
//! with the lower rank is broadcast into the larger shape by adding a number of
//! outer dimensions. It is not possible to broadcast mutable arrays or when
//! moving elements out of an array.
//!
//! For multidimensional arrays, iteration over a single dimension can be done
//! with `outer_expr`, `outer_expr_mut`, `axis_expr` and `axis_expr_mut`.
//! The resulting expressions give array views of the remaining dimensions.
//!
//! ## Operators
//!
//! Arithmetic, logical, negation, comparison and compound assignment operators
//! are supported for arrays and expressions.
//!
//! If at least one of the inputs is an array that is passed by value, the input
//! buffer is reused for the result. Otherwise, if all input parameters are array
//! references or expressions, a new array is created for the result. In the
//! latter case, the result may have a different element type.
//!
//! For comparison operators, the parameters must always be arrays that are passed
//! by reference. For compound assignment operators, the first parameter is always
//! a mutable reference to an array where the result is stored.
//!
//! Scalar parameters must passed using the `fill` function that wraps a value in
//! an `Expression<Fill<T>>` expression. If a type does not implement the `Copy`
//! trait, the parameter must be passed by reference.
//!
//! Note that for complex calculations, it can be more efficient to use expressions
//! and element-wise operations to reduce memory accesses and allocations.
//!
//! ## Example
//!
//! This example implements matrix multiplication and addition `C = A * B + C`.
//! The matrices use column-major ordering, and the inner loop runs over one column
//! in `A` and `C`. By using iterator-like expressions the array bounds checking
//! is avoided, and the compiler is able to vectorize the inner loop.
//!
//! ```
//! use mdarray::{grid, view, Grid, Span, View};
//!
//! fn matmul(a: &Span<f64, 2>, b: &Span<f64, 2>, c: &mut Span<f64, 2>) {
//!     for (mut cj, bj) in c.outer_expr_mut().zip(b.outer_expr()) {
//!         for (ak, bkj) in a.outer_expr().zip(bj) {
//!             for (cij, aik) in cj.expr_mut().zip(ak) {
//!                 *cij = aik.mul_add(*bkj, *cij);
//!             }
//!         }
//!     }
//! }
//!
//! let a = view![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
//! let b = view![[0.0, 1.0], [1.0, 1.0]];
//!
//! let mut c = grid![[0.0; 3]; 2];
//!
//! matmul(&a, &b, &mut c);
//!
//! assert_eq!(c, view![[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]);
//! ```

#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(associated_type_defaults))]
#![cfg_attr(feature = "nightly", feature(extern_types))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(int_roundings))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(unused_results)]

/// Buffer module for array storage.
pub mod buffer;

/// Expression module, for multidimensional iteration.
pub mod expr;

/// Module for array span and view indexing, and for array axis subarray types.
pub mod index;

/// Array layout mapping module.
pub mod mapping;

mod array;
mod dim;
mod expression;
mod grid;
mod iter;
mod layout;
mod macros;
mod ops;
mod raw_span;
mod span;
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

#[cfg(feature = "nightly")]
use std::alloc::Global;

#[cfg(not(feature = "nightly"))]
use alloc::Global;
use array::{GridArray, SpanArray, ViewArray, ViewArrayMut};

pub use array::Array;
pub use dim::{Const, Dim, Shape, Strides};
pub use expression::Expression;
pub use iter::Iter;
pub use layout::{Dense, Flat, General, Layout, Strided, Uniform, UnitStrided};
pub use ops::{step, StepRange};
pub use traits::{Apply, IntoCloned, IntoExpression};

/// Dense multidimensional array.
pub type Grid<T, const N: usize, A = Global> = GridArray<T, Const<N>, A>;

/// Multidimensional array span.
pub type Span<T, const N: usize, F = Dense> = SpanArray<T, Const<N>, F>;

/// Multidimensional array view.
pub type View<'a, T, const N: usize, F = Dense> = ViewArray<'a, T, Const<N>, F>;

/// Mutable multidimensional array view.
pub type ViewMut<'a, T, const N: usize, F = Dense> = ViewArrayMut<'a, T, Const<N>, F>;
