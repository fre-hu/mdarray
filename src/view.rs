#![allow(unused_parens)]

use crate::aligned_alloc::AlignedAlloc;
use crate::buffer::{DenseBuffer, FromIterIn};
use crate::dimension::Dim2;
use crate::grid::{DenseGrid, SubGrid, SubGridMut};
use crate::index::{DimIndex, IndexMap, ViewIndex};
use crate::iterator::{Iter, IterMut};
use crate::layout::{Layout, StaticLayout, StridedLayout};
use crate::order::{ColumnMajor, Order, RowMajor};
use std::alloc::{Allocator, Global};
use std::borrow::ToOwned;
use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, Result};
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};
use std::ptr::{self, NonNull};
use std::{mem, slice};

/// Multidimensional array view with static rank and element order.
#[repr(transparent)]
pub struct ViewBase<T, L: Layout<N, O>, const N: usize, O: Order> {
    _marker: PhantomData<(T, L, O)>,
    _slice: [()],
}

/// Strided multidimensional array view with static rank and element order, and dynamic shape.
pub type StridedView<T, const N: usize, const M: usize, O> =
    ViewBase<T, StridedLayout<N, M, O>, N, O>;

/// Dense multidimensional array view with static rank and element order, and dynamic shape.
pub type DenseView<T, const N: usize, O> = StridedView<T, N, 0, O>;

struct DebugState<'a, T: Debug, const N: usize, const M: usize, O: Order> {
    view: &'a StridedView<T, N, M, O>,
    index: [usize; N],
    dim: usize,
}

impl<T, L: Layout<N, O>, const N: usize, O: Order> ViewBase<T, L, N, O> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        (self as *mut Self).cast()
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        (self as *const Self).cast()
    }

    /// Creates an array view from a raw pointer and an array layout.
    pub unsafe fn from_raw_parts(ptr: *const T, layout: &L) -> &Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        if mem::size_of::<L>() == mem::size_of::<usize>() {
            &*(ptr::from_raw_parts(ptr.cast(), mem::transmute_copy(layout)) as *const Self)
        } else {
            &*(ptr::from_raw_parts(ptr.cast(), layout as *const L as usize) as *const Self)
        }
    }

    /// Creates a mutable array view from a raw pointer and an array layout.
    pub unsafe fn from_raw_parts_mut(ptr: *mut T, layout: &L) -> &mut Self {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        if mem::size_of::<L>() == mem::size_of::<usize>() {
            &mut *(ptr::from_raw_parts_mut(ptr.cast(), mem::transmute_copy(layout)) as *mut Self)
        } else {
            &mut *(ptr::from_raw_parts_mut(ptr.cast(), layout as *const L as usize) as *mut Self)
        }
    }

    /// Returns true if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the array layout.
    pub fn layout(&self) -> L {
        let layout = ptr::metadata(self as *const Self);

        if mem::size_of::<L>() == mem::size_of::<usize>() {
            unsafe { mem::transmute_copy(&layout) }
        } else {
            unsafe { *(layout as *const L) }
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.layout().len()
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> [usize; N] {
        self.layout().shape()
    }

    /// Returns the number of elements in the specified dimension.
    pub fn size(&self, dim: usize) -> usize {
        self.layout().size(dim)
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
    pub fn strides(&self) -> [isize; M] {
        self.layout().strides()
    }
}

impl<T: Clone, const N: usize, const M: usize, O: Order> StridedView<T, N, M, O> {
    /// Copies the array view into a new array.
    pub fn to_grid(&self) -> DenseGrid<T, N, O> {
        self.to_grid_in(AlignedAlloc::new(Global))
    }

    /// Copies the array view into a new array with the specified allocator.
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> DenseGrid<T, N, O, A> {
        let buffer = DenseBuffer::<T, 1, O, A>::from_iter_in(self.iter().cloned(), alloc);
        let (ptr, _, capacity, alloc) = buffer.into_raw_parts_with_alloc();

        unsafe { DenseGrid::from_raw_parts_in(ptr, self.shape(), capacity, alloc) }
    }
}

impl<T, const N: usize, O: Order> DenseView<T, N, O> {
    /// Returns a mutable slice of all elements in the array.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Returns a slice of all elements in the array.
    pub fn as_slice(&self) -> &[T] {
        unsafe { slice::from_raw_parts(self.as_ptr(), self.len()) }
    }
}

// There must be multiple view methods with concrete values/types for N, M and O, as
// otherwise the return type will have unconstrained generic constants when T is generic.

macro_rules! impl_view {
    ($v:tt, $g:tt, $p:tt, {$($u:tt)?}, $n:tt, $m:tt, $o:tt, ($($t:tt),+), ($($x:tt),+), $d:meta) => {
        impl<T> StridedView<T, $n, $m, $o> {
            /// Returns a subarray with the specified array view.
            ///
            /// The layout must be concrete (i.e. have no generic parameters) to compute the
            /// resulting layout. Note that variants that differ by return types are hidden.
            #[$d]
            pub fn $v<$($x: DimIndex),+>(
                &$($u)? self,
                $($t: $x),+
            ) -> $g<
                T,
                { <($($x),+) as IndexMap<$n, $o>>::RANK },
                {
                    outer_rank::<$n, $m>(
                        <($($x),+) as IndexMap<$n, $o>>::CONT,
                        <($($x),+) as IndexMap<$n, $o>>::RANK,
                    )
                },
                $o,
            > {
                let mut dims = [0; <($($x),+) as IndexMap<$n, $o>>::RANK];
                let mut shape = [0; <($($x),+) as IndexMap<$n, $o>>::RANK];
                let mut start = [0; $n];

                <($($x),+) as IndexMap<$n, $o>>::view_info(
                    &($($t),+),
                    &mut dims,
                    &mut shape,
                    &mut start,
                    &self.shape(),
                    0,
                );

                let mut offset = 0;

                for i in 0..$n {
                    offset += start[i] as isize * self.stride(i);
                }

                let mut strides = [0; outer_rank::<$n, $m>(
                    <($($x),+) as IndexMap<$n, $o>>::CONT,
                    <($($x),+) as IndexMap<$n, $o>>::RANK,
                )];

                for i in 0..strides.len() {
                    strides[i] = self.stride(dims[$o::select(i + dims.len() - strides.len(), i)]);
                }

                $g::new(
                    unsafe { NonNull::new_unchecked(self.$p().offset(offset) as *mut T) },
                    StridedLayout::new(shape, strides),
                )
            }
        }
    };
}

#[rustfmt::skip]
macro_rules! impl_views {
    ($v:tt, $g:tt, $p:tt, $u:tt, $o:tt, $d:meta) => {
        impl_view!($v, $g, $p, $u, 1, 0, $o, (x), (X), $d);
        impl_view!($v, $g, $p, $u, 1, 1, $o, (x), (X), doc(hidden));

        impl_view!($v, $g, $p, $u, 2, 0, $o, (x, y), (X, Y), $d);
        impl_view!($v, $g, $p, $u, 2, 1, $o, (x, y), (X, Y), doc(hidden));
        impl_view!($v, $g, $p, $u, 2, 2, $o, (x, y), (X, Y), doc(hidden));

        impl_view!($v, $g, $p, $u, 3, 0, $o, (x, y, z), (X, Y, Z), $d);
        impl_view!($v, $g, $p, $u, 3, 1, $o, (x, y, z), (X, Y, Z), doc(hidden));
        impl_view!($v, $g, $p, $u, 3, 2, $o, (x, y, z), (X, Y, Z), doc(hidden));
        impl_view!($v, $g, $p, $u, 3, 3, $o, (x, y, z), (X, Y, Z), doc(hidden));

        impl_view!($v, $g, $p, $u, 4, 0, $o, (x, y, z, w), (X, Y, Z, W), $d);
        impl_view!($v, $g, $p, $u, 4, 1, $o, (x, y, z, w), (X, Y, Z, W), doc(hidden));
        impl_view!($v, $g, $p, $u, 4, 2, $o, (x, y, z, w), (X, Y, Z, W), doc(hidden));
        impl_view!($v, $g, $p, $u, 4, 3, $o, (x, y, z, w), (X, Y, Z, W), doc(hidden));
        impl_view!($v, $g, $p, $u, 4, 4, $o, (x, y, z, w), (X, Y, Z, W), doc(hidden));

        impl_view!($v, $g, $p, $u, 5, 0, $o, (x, y, z, w, u), (X, Y, Z, W, U), $d);
        impl_view!($v, $g, $p, $u, 5, 1, $o, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
        impl_view!($v, $g, $p, $u, 5, 2, $o, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
        impl_view!($v, $g, $p, $u, 5, 3, $o, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
        impl_view!($v, $g, $p, $u, 5, 4, $o, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));
        impl_view!($v, $g, $p, $u, 5, 5, $o, (x, y, z, w, u), (X, Y, Z, W, U), doc(hidden));

        impl_view!($v, $g, $p, $u, 6, 0, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), $d);
        impl_view!($v, $g, $p, $u, 6, 1, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
        impl_view!($v, $g, $p, $u, 6, 2, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
        impl_view!($v, $g, $p, $u, 6, 3, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
        impl_view!($v, $g, $p, $u, 6, 4, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
        impl_view!($v, $g, $p, $u, 6, 5, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
        impl_view!($v, $g, $p, $u, 6, 6, $o, (x, y, z, w, u, v), (X, Y, Z, W, U, V), doc(hidden));
    };
}

impl_views!(view, SubGrid, as_ptr, {}, ColumnMajor, doc());
impl_views!(view, SubGrid, as_ptr, {}, RowMajor, doc(hidden));

impl_views!(view_mut, SubGridMut, as_mut_ptr, {mut}, ColumnMajor, doc());
impl_views!(view_mut, SubGridMut, as_mut_ptr, {mut}, RowMajor, doc(hidden));

pub const fn outer_rank<const N: usize, const M: usize>(cont: usize, rank: usize) -> usize {
    if N - M < cont {
        rank - (N - M)
    } else {
        rank - cont
    }
}

impl<T, const N: usize, O: Order> AsMut<[T]> for DenseView<T, N, O> {
    fn as_mut(&mut self) -> &mut [T] {
        self.as_mut_slice()
    }
}

impl<T, O: Order> AsMut<DenseView<T, 1, O>> for [T] {
    fn as_mut(&mut self) -> &mut DenseView<T, 1, O> {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        unsafe { &mut *ptr::from_raw_parts_mut(self.as_mut_ptr().cast(), self.len()) }
    }
}

impl<T, const X: usize, O: Order> AsMut<DenseView<T, 1, O>> for [T; X] {
    fn as_mut(&mut self) -> &mut DenseView<T, 1, O> {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        unsafe { &mut *ptr::from_raw_parts_mut(self.as_mut_ptr().cast(), X) }
    }
}

impl<T, const X: usize, const Y: usize, O: Order> AsMut<DenseView<T, 2, O>> for [[T; X]; Y] {
    fn as_mut(&mut self) -> &mut DenseView<T, 2, O> {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        let layout = O::select(
            &<Dim2<X, Y> as StaticLayout<2, O>>::LAYOUT,
            &<Dim2<Y, X> as StaticLayout<2, O>>::LAYOUT,
        );

        unsafe { ViewBase::from_raw_parts_mut(self.as_mut_ptr().cast(), &layout) }
    }
}

impl<T, const N: usize, O: Order> AsRef<[T]> for DenseView<T, N, O> {
    fn as_ref(&self) -> &[T] {
        self.as_slice()
    }
}

impl<T, O: Order> AsRef<DenseView<T, 1, O>> for [T] {
    fn as_ref(&self) -> &DenseView<T, 1, O> {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        unsafe { &*ptr::from_raw_parts(self.as_ptr().cast(), self.len()) }
    }
}

impl<T, const X: usize, O: Order> AsRef<DenseView<T, 1, O>> for [T; X] {
    fn as_ref(&self) -> &DenseView<T, 1, O> {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        unsafe { &*ptr::from_raw_parts(self.as_ptr().cast(), X) }
    }
}

impl<T, const X: usize, const Y: usize, O: Order> AsRef<DenseView<T, 2, O>> for [[T; X]; Y] {
    fn as_ref(&self) -> &DenseView<T, 2, O> {
        assert!(mem::size_of::<T>() != 0); // ZST not allowed

        let layout = O::select(
            &<Dim2<X, Y> as StaticLayout<2, O>>::LAYOUT,
            &<Dim2<Y, X> as StaticLayout<2, O>>::LAYOUT,
        );

        unsafe { ViewBase::from_raw_parts(self.as_ptr().cast(), &layout) }
    }
}

impl<T: Debug, const N: usize, const M: usize, O: Order> Debug for StridedView<T, N, M, O> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        DebugState {
            view: self,
            index: [0; N],
            dim: N - 1,
        }
        .fmt(fmt)
    }
}

impl<'a, T: Debug, const N: usize, const M: usize, O: Order> Debug for DebugState<'a, T, N, M, O> {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result {
        let mut list = fmt.debug_list();
        let mut index = self.index;

        for i in 0..self.view.size(O::select(self.dim, N - 1 - self.dim)) {
            index[O::select(self.dim, N - 1 - self.dim)] = i;

            if self.dim == 0 {
                list.entry(&self.view[index]);
            } else {
                list.entry(&DebugState {
                    view: self.view,
                    index,
                    dim: self.dim - 1,
                });
            }
        }

        list.finish()
    }
}

impl<T: Eq, const N: usize, const M: usize, O: Order> Eq for StridedView<T, N, M, O> {}

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

impl<'a, T, const N: usize, const M: usize, O: Order> IntoIterator for &'a StridedView<T, N, M, O> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N, M, O>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, const N: usize, const M: usize, O: Order> IntoIterator
    for &'a mut StridedView<T, N, M, O>
{
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T, N, M, O>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Ord, const M: usize, O: Order> Ord for StridedView<T, 1, M, O> {
    default fn cmp(&self, other: &Self) -> Ordering {
        self.iter().cmp(other.iter())
    }
}

impl<T: Ord, O: Order> Ord for DenseView<T, 1, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<T: PartialEq<U>, U: PartialEq, const N: usize, const M: usize, const K: usize, O: Order>
    PartialEq<StridedView<U, N, K, O>> for StridedView<T, N, M, O>
{
    default fn eq(&self, other: &StridedView<U, N, K, O>) -> bool {
        self.shape() == other.shape() && self.iter().eq(other.iter())
    }
}

impl<T: PartialEq<U>, U: PartialEq, const N: usize, O: Order> PartialEq<DenseView<U, N, O>>
    for DenseView<T, N, O>
{
    fn eq(&self, other: &DenseView<U, N, O>) -> bool {
        self.shape() == other.shape() && self.as_slice() == other.as_slice()
    }
}

impl<T: PartialOrd<U>, U: PartialOrd, const M: usize, const K: usize, O: Order>
    PartialOrd<StridedView<U, 1, K, O>> for StridedView<T, 1, M, O>
{
    default fn partial_cmp(&self, other: &StridedView<U, 1, K, O>) -> Option<Ordering> {
        self.iter().partial_cmp(other.iter())
    }
}

impl<T: PartialOrd, O: Order> PartialOrd for DenseView<T, 1, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.as_slice().partial_cmp(other.as_slice())
    }
}

impl<T: Clone, const N: usize, O: Order> ToOwned for DenseView<T, N, O> {
    type Owned = DenseGrid<T, N, O>;

    fn to_owned(&self) -> DenseGrid<T, N, O> {
        self.to_grid()
    }
}
