/*!
# Multidimensional array for Rust

## Overview

The mdarray crate provides a multidimensional array for Rust. Its main target
is for numeric types, however generic types are supported as well. The purpose
is to provide a generic container type that is simple and flexible to use,
with interworking to other crates for e.g. BLAS/LAPACK functionality.

Here are the main features of mdarray:

- Dense array type, where the rank and element order is known at compile time.
- Static array type, where the rank, shape and element order is known at compile time.
- Column-major and row-major element order.
- Subarrays (views) can be created with arbitrary shapes and strides.
- Standard Rust mechanisms are used for e.g. slices, indexing and iteration.

The design is inspired from the Rust ndarray and nalgebra crates, the proposed C++
mdspan/mdarray types and multidimensional arrays in Julia and Matlab.

Note that this crate requires nightly Rust toolchain.
*/

#![allow(incomplete_features)]
#![feature(allocator_api)]
#![feature(const_fn_trait_bound)]
#![feature(custom_inner_attributes)]
#![feature(generic_const_exprs)]
#![feature(ptr_metadata)]
#![feature(slice_ptr_len)]
#![feature(slice_range)]
#![warn(missing_docs)]

mod array;
mod dimension;
mod index;
mod iterator;
mod layout;
mod order;
mod raw_vec;
mod sub_array;
mod view;

pub use array::{ArrayBase, DenseArray, StaticArray};
pub use dimension::{Dim1, Dim2};
pub use order::{ColumnMajor, Order, RowMajor};
pub use view::{DenseView, StridedView, ViewBase};

use std::alloc::Global;

/// Dense multidimensional view with column-major element order.
pub type View<T, const N: usize> = DenseView<T, N, ColumnMajor>;

/// Dense multidimensional view with row-major element order.
pub type CView<T, const N: usize> = DenseView<T, N, RowMajor>;

/// Dense multidimensional array with column-major element order.
pub type Array<T, const N: usize, A = Global> = DenseArray<T, N, ColumnMajor, A>;

/// Dense multidimensional array with row-major element order.
pub type CArray<T, const N: usize, A = Global> = DenseArray<T, N, RowMajor, A>;

/// Static 1-dimensional array with column-major element order.
pub type SArray1<T, const X: usize> = StaticArray<T, Dim1<X>, 1, ColumnMajor>;

/// Static 1-dimensional array with row-major element order.
pub type SCArray1<T, const X: usize> = StaticArray<T, Dim1<X>, 1, RowMajor>;

/// Static 2-dimensional array with column-major element order.
pub type SArray2<T, const X: usize, const Y: usize> = StaticArray<T, Dim2<X, Y>, 2, ColumnMajor>;

/// Static 2-dimensional array with row-major element order.
pub type SCArray2<T, const X: usize, const Y: usize> = StaticArray<T, Dim2<X, Y>, 2, RowMajor>;
