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
//! - Dense array type, where the rank and element order is known at compile time.
//! - Column-major and row-major element order.
//! - Subarrays (views) can be created with arbitrary shapes and strides.
//! - Standard Rust mechanisms are used for e.g. slices, indexing and iteration.
//!
//! The design is inspired from the Rust ndarray, nalgebra and bitvec crates,
//! the proposed C++ mdarray and mdspan types, and multidimensional arrays in
//! Julia and Matlab.
//!
//! ## Array types
//!
//! The base types for multidimensional arrays are `GridBase` and `SpanBase`,
//! similar to the Rust `Vec` and `slice` types.
//!
//! `GridBase` consists of a buffer for element storage and information about
//! the array layout. The buffer can either own the storage like `Vec`, or refer
//! to a parent array. The latter case occurs for example when creating a view
//! of a larger array without duplicating elements.
//!
//! `SpanBase` is used as a generic array reference. It consists of pointers
//! to the buffer and the layout, and is stored internally as a fat pointer.
//! It is useful for function parameters where the same `SpanBase` type can
//! refer to either an owned array or an array view.
//!
//! The array layout describes how elements are stored in memory. The layout
//! is parameterized by the rank (i.e. the number of dimensions), the element
//! order and the storage format. It contains the shape (i.e. the size in each
//! dimension), and the strides per dimension if needed.
//!
//! The storage format is `Dense` if elements are stored contiguously without gaps.
//! In this case, the strides are calculated from the shape and not stored as
//! part of the layout. The format is `General` if each dimension can have
//! arbitrary stride except for the innermost one, which has unit stride. It is
//! compatible with the BLAS/LAPACK general matrix storage.
//!
//! The format is `Flat` if the innermost dimension can have arbitrary stride
//! and the other dimensions must follow in order, allowing for linear indexing.
//! The format is `Strided` if all dimensions can have arbitrary strides.
//!
//! The element order is `ColumnMajor` for Fortran order where the innermost
//! dimension is the innermost one, or `RowMajor` for the opposite C order.
//! Besides indexing for element access, the order affects how iteration is done
//! over multiple dimensions.
//!
//! The following type aliases are provided:
//!
//! | Alias                                  | Description                         |
//! | -------------------------------------- | ----------------------------------- |
//! | `Grid<T, const N: usize, A = Global>`  | Dense array with column-major order |
//! | `CGrid<T, const N: usize, A = Global>` | Dense array with row-major order    |
//! | `Span<T, const N: usize, F = Dense>`   | Array span with column-major order  |
//! | `CSpan<T, const N: usize, F = Dense>`  | Array span with row-major order     |
//!
//! ## Indexing and views
//!
//! Scalar indexing is done using the normal square-bracket index operator and
//! an array of `usize` per dimension as index.
//!
//! If the array layout type supports linear indexing, a scalar `usize` can also
//! be used as index. If the array layout type supports slice indexing, a range
//! can be used as index to select a slice.
//!
//! An array view can be created with the `view` and `view_mut` methods and a
//! tuple of indices per dimension as argument. Each index can be either a range
//! or `usize`. The resulting storage format depends on both the format inferred
//! from the indices and the input format.
//!
//! ## Iteration
//!
//! If the array layout type supports linear indexing, an iterator can be created
//! with the `iter`, `iter_mut` and `into_iter` methods like `Vec` and `slice`.
//!
//! For multidimensional arrays, indexing over a single dimension is done
//! with the `outer_iter`/`outer_iter_mut`, `inner_iter`/`inner_iter_mut` and
//! `axis_iter`/`axis_iter_mut` methods. The iterators give array views of
//! the remaining dimensions.
//!
//! If linear indexing is possible but the array layout type is not known, the
//! `reformat`, `reformat_mut` and `into_format` methods can be used to change
//! format with runtime checking. Alternatively, the `flatten`, `flatten_mut`
//! and `into_flattened` methods can be used to change to a one-dimensional
//! array.
//!
//! ## Operators
//!
//! Arithmetic, logical, negation, comparison and compound assignment operators
//! are supported for arrays. For arithmetic, logical and negation operators,
//! at most one parameter can be passed by value. If all parametes are passed by
//! reference, a new array is allocated for the result. For comparison operators,
//! the parameters are always passed by reference.
//!
//! Scalar parameters must be passed using the `fill` function to wrap the value
//! in a `Fill` struct. If the type does not implement the `Copy` trait, the
//! parameter must be passed by reference.
//!
//! Note that for complex calculations it can be more efficient to use iterators
//! and element-wise operations to reduce memory accesses.
//!
//! ## Example
//!
//! This example implements matrix multiplication and addition `C = A * B + C`.
//! The matrices use column-major ordering, and the inner loop runs over one column
//! in `A` and `C`. By using iterators the array bounds checking is avoided, and
//! the compiler is able to vectorize the inner loop.
//!
//! ```
//! use mdarray::{Grid, Span, SubGrid};
//!
//! fn matmul(a: &Span<f64, 2>, b: &Span<f64, 2>, c: &mut Span<f64, 2>) {
//!     assert!(c.shape() == [a.size(0), b.size(1)] && a.size(1) == b.size(0), "shape mismatch");
//!
//!     for (mut cj, bj) in c.outer_iter_mut().zip(b.outer_iter()) {
//!         for (ak, bkj) in a.outer_iter().zip(bj.iter()) {
//!             for (cij, aik) in cj.iter_mut().zip(ak.iter()) {
//!                 *cij = aik.mul_add(*bkj, *cij);
//!             }
//!         }
//!     }
//! }
//!
//! let a = SubGrid::from(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
//! let b = SubGrid::from(&[[0.0, 1.0], [1.0, 1.0]]);
//!
//! let mut c = Grid::from_elem([3, 2], 0.0);
//!
//! matmul(&a, &b, &mut c);
//!
//! println!("{c:?}");
//! # assert!(c == SubGrid::from(&[[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]));
//! ```
//!
//! This will produce the result `[[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]`.

#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![cfg_attr(feature = "nightly", feature(hasher_prefixfree_extras))]
#![cfg_attr(feature = "nightly", feature(int_roundings))]
#![cfg_attr(feature = "nightly", feature(slice_range))]
#![warn(missing_docs)]

/// Module for array span and view indexing, and for array axis subarray types.
pub mod index {
    mod axis;
    mod span;
    mod view;

    pub use axis::{Axis, Const};
    pub use span::SpanIndex;
    pub use view::{DimIndex, Params, ViewIndex};
}

/// Module for array axis and linear array span iterators.
pub mod iter {
    mod sources;

    pub use sources::{AxisIter, AxisIterMut, LinearIter, LinearIterMut};
}

mod buffer;
mod dim;
mod format;
mod grid;
mod layout;
mod mapping;
mod ops;
mod order;
mod span;

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

pub use dim::{Dim, Rank, Shape, Strides};
pub use format::{Dense, Flat, Format, General, Strided, Uniform, UnitStrided};
pub use grid::{DenseGrid, GridBase, SubGrid, SubGridMut};
pub use layout::Layout;
pub use ops::{fill, step, Fill, StepRange};
pub use order::{ColumnMajor, Order, RowMajor};
pub use span::SpanBase;

/// Dense multidimensional array with column-major element order.
pub type Grid<T, const N: usize, A = Global> = DenseGrid<T, Rank<N, ColumnMajor>, A>;

/// Dense multidimensional array with row-major element order.
pub type CGrid<T, const N: usize, A = Global> = DenseGrid<T, Rank<N, RowMajor>, A>;

/// Multidimensional array span with column-major element order.
pub type Span<T, const N: usize, F = Dense> = SpanBase<T, Layout<Rank<N, ColumnMajor>, F>>;

/// Multidimensional array span with row-major element order.
pub type CSpan<T, const N: usize, F = Dense> = SpanBase<T, Layout<Rank<N, RowMajor>, F>>;
