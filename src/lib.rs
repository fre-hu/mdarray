/*!
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

Note that this crate requires nightly Rust toolchain.

## Array types

The base types for multidimensional arrays are `GridBase` and `SpanBase`,
similar to the Rust `Vec` and `slice` types.

`GridBase` consists of a buffer for element storage and information about
the array layout. The buffer can either own the storage like `Vec`, or refer
to a parent array. The latter case occurs for example when creating a view
of a larger array without duplicating elements.

`SpanBase` is used as a generic array reference. It consists of pointers
to the buffer and the layout, and is stored internally as a fat pointer.
It is useful for function parameters where the same `SpanBase` type can
refer to either an owned array or an array view.

The array layout describes how elements are accessed in memory. The layout
is parameterized by the rank (i.e. the number of dimensions), the storage
format and the element order. It contains the shape (i.e. the size in each
dimension), and the strides per dimension if needed.

The storage format is `Dense` if elements are stored contiguously without gaps.
In this case, the strides are calculated from the shape and not stored as
part of the layout. The format is `General` if each dimension can have an
arbitrary stride, except for the innermost one which must have unit stride.
It is compatible with the BLAS/LAPACK general matrix storage. The format is
`Strided` if the innermost dimension can also have arbitrary stride.

The element order is `ColumnMajor` for Fortran order where the innermost
dimension is the innermost one, or `RowMajor` for the opposite C order.
Besides indexing for element access, the order affects how iteration is done
over multiple dimensions.

The following type aliases are provided:

| Alias                      | Description                              |
| -------------------------- | ---------------------------------------- |
| `Grid<T, const N: usize>`  | Dense array with column-major order      |
| `CGrid<T, const N: usize>` | Dense array with row-major order         |
| `Span<T, const N: usize>`  | Dense array span with column-major order |
| `CSpan<T, const N: usize>` | Dense array span with row-major order    |

## Indexing and views

Scalar indexing is done using the normal square-bracket index operator and
an array of `usize` per dimension as index.

For one-dimensional arrays, indexing can also be done with a scalar `usize`
as index. If the storage format is `Dense` or `General`, a range can be
used as index to select a one-dimensional array span.

An array view can be created with the `view` and `view_mut` methods and a
tuple of indices per dimension as argument. Each index can be either a range
or `usize`. The resulting storage format depends on both the format inferred
from the indices and the input format.

## Iteration

For one-dimensional arrays, an iterator can be created with the `iter`,
`iter_mut` and `into_iter` methods like `Vec` and `slice`.

For multidimensional arrays, indexing over a single dimension is done
with the `outer_iter`/`outer_iter_mut`, `inner_iter`/`inner_iter_mut` and
`axis_iter`/`axis_iter_mut` methods. The iterators give array views of
the remaining dimensions.

For multidimensional arrays with contiguous array layout, it is possible
to use the `flat_iter` and `flat_iter_mut` to iterate over all dimensions.
The methods will check at runtime and panic if the layout is not contiguous.

## Example

The following example implements simple matrix multiplication `C = A * B + C`.
The matrices use column-major ordering, and the inner loop runs over one column
in `A` and `C`. By using iterators the array bounds checking is avoided, and
the compiler is able to vectorize the inner loop.

```
use mdarray::{Grid, Span};

pub fn matmul(a: &Span<f64, 2>, b: &Span<f64, 2>, c: &mut Span<f64, 2>) {
    assert!(c.shape() == [a.size(0), b.size(1)] && a.size(1) == b.size(0), "shape mismatch");

    for (mut cj, bj) in c.outer_iter_mut().zip(b.outer_iter()) {
        for (ak, bkj) in a.outer_iter().zip(bj.iter()) {
            for (cij, aik) in cj.iter_mut().zip(ak.iter()) {
                *cij += aik * bkj;
            }
        }
    }
}

let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].as_ref();
let b = [[0.0, 1.0], [1.0, 1.0]].as_ref();

let mut c = Grid::from([[0.0; 3]; 2]);

matmul(a, b, &mut c);

println!("{c:?}");
# assert!(&c == AsRef::<Span<f64, 2>>::as_ref(&[[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]));
```

This will produce the result `[[4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]`.
*/

#![feature(allocator_api)]
#![feature(const_generics_defaults)]
#![feature(generic_associated_types)]
#![feature(ptr_metadata)]
#![feature(slice_ptr_len)]
#![feature(slice_range)]
#![warn(missing_docs)]

mod aligned_alloc;
mod buffer;
mod dimension;
mod format;
mod grid;
mod index;
mod iterator;
mod layout;
mod mapping;
mod operator;
mod order;
mod raw_vec;
mod span;

use std::alloc::Global;

pub use aligned_alloc::AlignedAlloc;
pub use dimension::{Const, Dim, Shape, Strides};
pub use format::{Dense, Format, General, Strided};
pub use grid::{DenseGrid, GridBase, SubGrid, SubGridMut};
pub use layout::{DenseLayout, GeneralLayout, Layout, StridedLayout};
pub use order::{ColumnMajor, Order, RowMajor};
pub use span::SpanBase;

/// Dense multidimensional array with column-major element order.
pub type Grid<T, const N: usize, A = Global> = DenseGrid<T, Const<N>, ColumnMajor, A>;

/// Dense multidimensional array with row-major element order.
pub type CGrid<T, const N: usize, A = Global> = DenseGrid<T, Const<N>, RowMajor, A>;

/// Dense multidimensional array span with column-major element order.
pub type Span<T, const N: usize> = SpanBase<T, DenseLayout<Const<N>, ColumnMajor>>;

/// Dense multidimensional array span with row-major element order.
pub type CSpan<T, const N: usize> = SpanBase<T, DenseLayout<Const<N>, RowMajor>>;
