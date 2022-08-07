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
//! Note that this crate requires nightly Rust toolchain.
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
//! is parameterized by the rank (i.e. the number of dimensions), the storage
//! format and the element order. It contains the shape (i.e. the size in each
//! dimension), and the strides per dimension if needed.
//!
//! The storage format is `Dense` if elements are stored contiguously without gaps.
//! In this case, the strides are calculated from the shape and not stored as
//! part of the layout. The format is `General` if each dimension can have
//! arbitrary stride except for the innermost one, which has unit stride. It is
//! compatible with the BLAS/LAPACK general matrix storage.
//!
//! The format is `Linear` if the innermost dimension can have arbitrary stride
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
//! `flat_iter` and `flat_iter_mut` methods can be used instead of `iter` and
//! `iter_mut`. The methods will check at runtime that the layout is valid.
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
//! let mut c = Grid::from([[0.0; 3]; 2]);
//!
//! matmul(&a, &b, &mut c);
//!
//! println!("{c:?}");
//! # assert!(c == SubGrid::from(&[[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]));
//! ```
//!
//! This will produce the result `[[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]`.

#![feature(allocator_api)]
#![feature(generic_associated_types)]
#![feature(int_roundings)]
#![feature(marker_trait_attr)]
#![feature(slice_range)]
#![warn(missing_docs)]

mod buffer;
mod dim;
mod format;
mod grid;
mod index;
mod iter;
mod layout;
mod mapping;
mod ops;
mod order;
mod span;

#[cfg(feature = "serde")]
mod serde;

use std::alloc::Global;

pub use dim::{Const, Dim, Shape, Strides};
pub use format::{Dense, Format, General, Linear, Strided};
pub use format::{NonUniform, NonUnitStrided, Uniform, UnitStrided};
pub use grid::{DenseGrid, GridBase, SubGrid, SubGridMut};
pub use index::{step, DimIndex, PartialRange, SpanIndex, StepRange};
pub use layout::{HasLinearIndexing, HasSliceIndexing, Layout};
pub use ops::{fill, Fill};
pub use order::{ColumnMajor, Order, RowMajor};
pub use span::SpanBase;

/// Dense multidimensional array with column-major element order.
pub type Grid<T, const N: usize, A = Global> = DenseGrid<T, Const<N>, ColumnMajor, A>;

/// Dense multidimensional array with row-major element order.
pub type CGrid<T, const N: usize, A = Global> = DenseGrid<T, Const<N>, RowMajor, A>;

/// Multidimensional array span with column-major element order.
pub type Span<T, const N: usize, F = Dense> = SpanBase<T, Layout<Const<N>, F, ColumnMajor>>;

/// Multidimensional array span with row-major element order.
pub type CSpan<T, const N: usize, F = Dense> = SpanBase<T, Layout<Const<N>, F, RowMajor>>;
