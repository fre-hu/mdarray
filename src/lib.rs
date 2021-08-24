/*!
# Multidimensional array for Rust

## Overview

The mdarray crate provides a multidimensional array for Rust. Its main target
is for numeric types, however generic types are supported as well. The purpose
is to provide a generic container type that is simple and flexible to use,
with interworking to other crates for e.g. BLAS/LAPACK functionality.

Here are the main features of mdarray:

- Dense array type where the rank and element order is known at compile time.
- Static array type where the rank, shape and element order is known at compile time.
- Column-major and row-major element order.
- Standard Rust mechanisms are used for e.g. slices, indexing and iteration.

The design is inspired from the Rust ndarray and nalgebra crates, the proposed C++
mdspan/mdarray types and multidimensional arrays in Julia and Matlab.

Note that this crate requires nightly Rust toolchain.
*/

#![allow(incomplete_features)]
#![allow(type_alias_bounds)]
#![feature(allocator_api)]
#![feature(const_evaluatable_checked)]
#![feature(const_generics)]
#![feature(ptr_metadata)]

mod array;
mod dimension;
mod iterator;
mod layout;
mod order;
mod view;

pub use array::{ArrayBase, DenseArray, StaticArray};
pub use dimension::{Dim1, Dim2};
pub use order::Order;
pub use view::{DenseView, StridedView, ViewBase};

/// Dense multidimensional view with column-major element order.
pub type View<T, const N: usize> = DenseView<T, N, { Order::ColumnMajor }>;

/// Dense multidimensional view with row-major element order.
pub type CView<T, const N: usize> = DenseView<T, N, { Order::RowMajor }>;

/// Dense multidimensional array with column-major element order and using global allocator.
pub type Array<T, const N: usize> = DenseArray<T, std::alloc::Global, N, { Order::ColumnMajor }>;

/// Dense multidimensional array with row-major element order and using global allocator.
pub type CArray<T, const N: usize> = DenseArray<T, std::alloc::Global, N, { Order::RowMajor }>;

/// Static 1-dimensional array with column-major element order.
pub type SArray1<T, const S0: usize> = StaticArray<T, Dim1<S0>, 1, { Order::ColumnMajor }>;

/// Static 1-dimensional array with row-major element order.
pub type SCArray1<T, const S0: usize> = StaticArray<T, Dim1<S0>, 1, { Order::RowMajor }>;

/// Static 2-dimensional array with column-major element order.
pub type SArray2<T, const S0: usize, const S1: usize> =
    StaticArray<T, Dim2<S0, S1>, 2, { Order::ColumnMajor }>;

/// Static 2-dimensional array with row-major element order.
pub type SCArray2<T, const S0: usize, const S1: usize> =
    StaticArray<T, Dim2<S0, S1>, 2, { Order::RowMajor }>;
