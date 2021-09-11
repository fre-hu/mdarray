#![allow(unused_parens)]
#![rustfmt::skip::macros(impl_view)]

use crate::index::{DimIndex, IndexMap, ViewIndex};
use crate::iterator::{Iter, IterMut};
use crate::layout::{Layout, StridedLayout};
use crate::order::{ColumnMajor, Order, RowMajor};
use crate::sub_array::{SubArray, SubArrayMut};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::{self, NonNull};
use std::slice;

/// Multidimensional view into an array with static rank and element order.
#[repr(transparent)]
pub struct ViewBase<T, L: Layout<N, O>, const N: usize, O: Order> {
    _marker: PhantomData<(T, L, O)>,
    _slice: [()],
}

/// Multidimensional view with static rank and element order, and dynamic shape and strides.
pub type StridedView<T, const N: usize, const M: usize, O> =
    ViewBase<T, StridedLayout<N, M, O>, N, O>;

/// Dense multidimensional view with static rank and element order, and dynamic shape.
pub type DenseView<T, const N: usize, O> = StridedView<T, N, 0, O>;

impl<T, L: Layout<N, O>, const N: usize, O: Order> ViewBase<T, L, N, O> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&self) -> *mut T {
        let (data, _) = (self as *const Self).to_raw_parts();

        data as *mut T
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        let (data, _) = (self as *const Self).to_raw_parts();

        data as *const T
    }

    /// Creates a view from a raw pointer and an array layout.
    pub unsafe fn from_raw_parts(data: *const T, layout: &L) -> &Self {
        &*(ptr::from_raw_parts(data.cast(), layout as *const L as usize) as *const Self)
    }

    /// Creates a mutable view from a raw pointer and an array layout.
    pub unsafe fn from_raw_parts_mut(data: *mut T, layout: &L) -> &mut Self {
        &mut *(ptr::from_raw_parts_mut(data.cast(), layout as *const L as usize) as *mut Self)
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the array layout.
    pub fn layout(&self) -> &L {
        let (_, layout) = (self as *const Self).to_raw_parts();

        unsafe { &*(layout as *const L) }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.layout().shape().iter().product()
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> &[usize; N] {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.layout().shape()[dim]
    }
}

impl<T, const N: usize, const M: usize, O: Order> StridedView<T, N, M, O> {
    /// Returns an iterator over the array.
    pub fn iter(&self) -> Iter<T, N, M, O> {
        Iter::new(self)
    }

    /// Returns a mutable iterator over the array.
    pub fn iter_mut(&mut self) -> IterMut<T, N, M, O> {
        IterMut::new(self)
    }

    /// Returns the distance between elements in the specified dimension.
    pub fn stride(&self, dim: usize) -> isize {
        self.layout().stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> &[isize; M] {
        self.layout().strides()
    }
}

impl<T, const N: usize, O: Order> DenseView<T, N, O> {
    /// Returns a mutable slice of all elements in the array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }

    /// Returns a slice of all elements in the array.
    pub fn as_slice(&self) -> &[T] {
        self
    }
}

macro_rules! impl_view {
    ($name:tt, $type:tt, $as_ptr:tt, $n:tt, $m:tt, ($($v:tt),+), ($($x:tt),+), $d:meta) => {
        impl<T> StridedView<T, $n, $m, ColumnMajor> {
            /// Returns a subarray with the specified view of the array.
            ///
            /// The layout must be concrete (i.e. have no generic parameters) to compute the
            /// resulting layout. Note that variants that differ by return types are hidden.
            #[$d]
            pub fn $name<$($x: DimIndex),+>(
                &self,
                $($v: $x),+
            ) -> $type<
                T,
                { <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK },
                {
                    outer_rank::<$n, $m>(
                        <($($x),+) as IndexMap<$n, ColumnMajor>>::CONT,
                        <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK,
                    )
                },
                ColumnMajor,
            > {
                let mut dims = [0; <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK];
                let mut shape = [0; <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK];
                let mut start = [0; $n];

                <($($x),+) as IndexMap<$n, ColumnMajor>>::view_info(
                    &($($v),+),
                    &mut dims,
                    &mut shape,
                    &mut start,
                    self.shape(),
                    0,
                );

                let mut offset = 0;

                for i in 0..$n {
                    offset += start[i] as isize * self.stride(i);
                }

                let mut strides = [0; outer_rank::<$n, $m>(
                    <($($x),+) as IndexMap<$n, ColumnMajor>>::CONT,
                    <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK,
                )];

                for i in 0..strides.len() {
                    strides[i] = self.stride(dims[i + dims.len() - strides.len()]);
                }

                $type::new(
                    unsafe { NonNull::new_unchecked(self.$as_ptr().offset(offset) as *mut T) },
                    StridedLayout::<
                        { <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK },
                        {
                            outer_rank::<$n, $m>(
                                <($($x),+) as IndexMap<$n, ColumnMajor>>::CONT,
                                <($($x),+) as IndexMap<$n, ColumnMajor>>::RANK,
                            )
                        },
                        ColumnMajor,
                    >::new(shape, strides),
                )
            }
        }

        impl<T> StridedView<T, $n, $m, RowMajor> {
            #[doc(hidden)]
            pub fn $name<$($x: DimIndex),+>(
                &self,
                $($v: $x),+
            ) -> $type<
                T,
                { <($($x),+) as IndexMap<$n, RowMajor>>::RANK },
                {
                    outer_rank::<$n, $m>(
                        <($($x),+) as IndexMap<$n, RowMajor>>::CONT,
                        <($($x),+) as IndexMap<$n, RowMajor>>::RANK,
                    )
                },
                RowMajor,
            > {
                let mut dims = [0; <($($x),+) as IndexMap<$n, RowMajor>>::RANK];
                let mut shape = [0; <($($x),+) as IndexMap<$n, RowMajor>>::RANK];
                let mut start = [0; $n];

                <($($x),+) as IndexMap<$n, RowMajor>>::view_info(
                    &($($v),+),
                    &mut dims,
                    &mut shape,
                    &mut start,
                    self.shape(),
                    0,
                );

                let mut offset = 0;

                for i in 0..$n {
                    offset += start[i] as isize * self.stride(i);
                }

                let mut strides = [0; outer_rank::<$n, $m>(
                    <($($x),+) as IndexMap<$n, RowMajor>>::CONT,
                    <($($x),+) as IndexMap<$n, RowMajor>>::RANK,
                )];

                for i in 0..strides.len() {
                    strides[i] = self.stride(dims[i]);
                }

                $type::new(
                    unsafe { NonNull::new_unchecked(self.$as_ptr().offset(offset) as *mut T) },
                    StridedLayout::<
                        { <($($x),+) as IndexMap<$n, RowMajor>>::RANK },
                        {
                            outer_rank::<$n, $m>(
                                <($($x),+) as IndexMap<$n, RowMajor>>::CONT,
                                <($($x),+) as IndexMap<$n, RowMajor>>::RANK,
                            )
                        },
                        RowMajor,
                    >::new(shape, strides),
                )
            }
        }
    };
}

impl_view!(view, SubArray, as_ptr, 1, 0, (x), (X), doc());
impl_view!(view, SubArray, as_ptr, 1, 1, (x), (X), doc(hidden));

impl_view!(view, SubArray, as_ptr, 2, 0, (x, y), (X, Y), doc());
impl_view!(view, SubArray, as_ptr, 2, 1, (x, y), (X, Y), doc(hidden));
impl_view!(view, SubArray, as_ptr, 2, 2, (x, y), (X, Y), doc(hidden));

impl_view!(view, SubArray, as_ptr, 3, 0, (x, y, z), (X, Y, Z), doc());
impl_view!(view, SubArray, as_ptr, 3, 1, (x, y, z), (X, Y, Z), doc(hidden));
impl_view!(view, SubArray, as_ptr, 3, 2, (x, y, z), (X, Y, Z), doc(hidden));
impl_view!(view, SubArray, as_ptr, 3, 3, (x, y, z), (X, Y, Z), doc(hidden));

impl_view!(view, SubArray, as_ptr, 4, 0, (x, y, z, w), (X, Y, Z, W), doc());
impl_view!(view, SubArray, as_ptr, 4, 1, (x, y, z, w), (X, Y, Z, W), doc(hidden));
impl_view!(view, SubArray, as_ptr, 4, 2, (x, y, z, w), (X, Y, Z, W), doc(hidden));
impl_view!(view, SubArray, as_ptr, 4, 3, (x, y, z, w), (X, Y, Z, W), doc(hidden));
impl_view!(view, SubArray, as_ptr, 4, 4, (x, y, z, w), (X, Y, Z, W), doc(hidden));

impl_view!(view, SubArray, as_ptr, 5, 0, (x, y, z, w, u), (X, Y, Z, W, U), doc());
impl_view!(view, SubArray, as_ptr, 5, 1, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view, SubArray, as_ptr, 5, 2, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view, SubArray, as_ptr, 5, 3, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view, SubArray, as_ptr, 5, 4, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view, SubArray, as_ptr, 5, 5, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));

impl_view!(view, SubArray, as_ptr, 6, 0, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc());
impl_view!(view, SubArray, as_ptr, 6, 1, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view, SubArray, as_ptr, 6, 2, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view, SubArray, as_ptr, 6, 3, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view, SubArray, as_ptr, 6, 4, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view, SubArray, as_ptr, 6, 5, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view, SubArray, as_ptr, 6, 6, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));

impl_view!(view_mut, SubArrayMut, as_mut_ptr, 1, 0, (x), (X), doc());
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 1, 1, (x), (X), doc(hidden));

impl_view!(view_mut, SubArrayMut, as_mut_ptr, 2, 0, (x, y), (X, Y), doc());
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 2, 1, (x, y), (X, Y), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 2, 2, (x, y), (X, Y), doc(hidden));

impl_view!(view_mut, SubArrayMut, as_mut_ptr, 3, 0, (x, y, z), (X, Y, Z), doc());
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 3, 1, (x, y, z), (X, Y, Z), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 3, 2, (x, y, z), (X, Y, Z), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 3, 3, (x, y, z), (X, Y, Z), doc(hidden));

impl_view!(view_mut, SubArrayMut, as_mut_ptr, 4, 0, (x, y, z, w), (X, Y, Z, W), doc());
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 4, 1, (x, y, z, w), (X, Y, Z, W), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 4, 2, (x, y, z, w), (X, Y, Z, W), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 4, 3, (x, y, z, w), (X, Y, Z, W), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 4, 4, (x, y, z, w), (X, Y, Z, W), doc(hidden));

impl_view!(view_mut, SubArrayMut, as_mut_ptr, 5, 0, (x, y, z, w, u), (X, Y, Z, W, U), doc());
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 5, 1, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 5, 2, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 5, 3, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 5, 4, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 5, 5, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));

impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 0, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc());
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 1, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 2, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 3, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 4, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 5, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
impl_view!(view_mut, SubArrayMut, as_mut_ptr, 6, 6, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));

impl<T, const N: usize, O: Order> Deref for DenseView<T, N, O> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

impl<T, const N: usize, O: Order> DerefMut for DenseView<T, N, O> {
    fn deref_mut(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }
}

impl<I: ViewIndex<T, N, M, O>, T, const N: usize, const M: usize, O: Order> Index<I>
    for StridedView<T, N, M, O>
{
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<I: ViewIndex<T, N, M, O>, T, const N: usize, const M: usize, O: Order> IndexMut<I>
    for StridedView<T, N, M, O>
{
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

pub const fn outer_rank<const N: usize, const M: usize>(cont: usize, rank: usize) -> usize {
    if N - M < cont {
        rank - (N - M)
    } else {
        rank - cont
    }
}
