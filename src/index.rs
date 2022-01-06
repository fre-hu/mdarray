use std::ops::{Bound, Index, IndexMut, Range, RangeBounds, RangeFrom};
use std::ops::{RangeFull, RangeInclusive, RangeTo, RangeToInclusive};
use std::slice;

use crate::dimension::{Const, Dim, Shape};
use crate::format::UnitStrided;
use crate::layout::{DenseLayout, Layout, StridedLayout};
use crate::mapping::Mapping;
use crate::order::{ColumnMajor, Order, RowMajor};
use crate::span::SpanBase;

pub trait DimIndex {
    type Layout<L: Layout>: Layout;
    type Outer<L: Layout>: Layout;
    type Root<L: Layout>: Layout;

    fn first_info<L: Layout>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Root<L>);

    fn next_info<L: Layout>(
        self,
        offset: isize,
        inner: L,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Outer<L>);
}

pub trait PartialRange: RangeBounds<usize> {}

pub trait ViewIndex<D: Dim, O: Order, L: Layout<Order = O>> {
    type Inner: Layout;
    type Layout: Layout;
    type Outer: Layout;

    fn view_info(self, layout: L) -> (isize, Self::Layout, Self::Outer);
}

impl DimIndex for usize {
    type Layout<L: Layout> = L;
    type Outer<L: Layout> = L::NonDense;
    type Root<L: Layout> = StridedLayout<L::Dim, L::Order>;

    fn first_info<L: Layout>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Root<L>) {
        if self >= size {
            panic_bounds_check(self, size)
        }

        (stride * self as isize, L::default(), StridedLayout::default())
    }

    fn next_info<L: Layout>(
        self,
        offset: isize,
        inner: L,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Outer<L>) {
        if self >= size {
            panic_bounds_check(self, size)
        }

        (offset + stride * self as isize, inner, inner.to_non_dense())
    }
}

impl DimIndex for RangeFull {
    type Layout<L: Layout> = L::Larger;
    type Outer<L: Layout> = L::Larger;
    type Root<L: Layout> = L::Larger;

    fn first_info<L: Layout>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Root<L>) {
        let layout = L::default().add_dim(size, stride);

        (0, layout, layout)
    }

    fn next_info<L: Layout>(
        self,
        offset: isize,
        inner: L,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Outer<L>) {
        let layout = inner.add_dim(size, stride);

        (offset, layout, layout)
    }
}

impl<R: PartialRange> DimIndex for R {
    type Layout<L: Layout> = L::Larger;
    type Outer<L: Layout> = <L::Larger as Mapping<L::Larger>>::NonDense;
    type Root<L: Layout> = <L::Larger as Mapping<L::Larger>>::NonDense;

    fn first_info<L: Layout>(
        self,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Root<L>) {
        let range = slice::range(self, ..size);
        let layout = L::default().add_dim(range.end - range.start, stride);

        (stride * range.start as isize, layout, layout.to_non_dense())
    }

    fn next_info<L: Layout>(
        self,
        offset: isize,
        inner: L,
        size: usize,
        stride: isize,
    ) -> (isize, Self::Layout<L>, Self::Outer<L>) {
        let range = slice::range(self, ..size);
        let layout = inner.add_dim(range.end - range.start, stride);

        (offset + stride * range.start as isize, layout, layout.to_non_dense())
    }
}

impl PartialRange for (Bound<usize>, Bound<usize>) {}
impl PartialRange for Range<usize> {}
impl PartialRange for RangeFrom<usize> {}
impl PartialRange for RangeInclusive<usize> {}
impl PartialRange for RangeTo<usize> {}
impl PartialRange for RangeToInclusive<usize> {}

impl<O: Order, L: Layout<Order = O>, X: DimIndex> ViewIndex<Const<1>, O, L> for X {
    type Inner = L::Reshaped<[usize; 0]>;
    type Layout = X::Layout<Self::Inner>;
    type Outer = X::Root<Self::Inner>;

    fn view_info(self, layout: L) -> (isize, Self::Layout, Self::Outer) {
        self.first_info(layout.size(layout.dim(0)), layout.stride(layout.dim(0)))
    }
}

macro_rules! impl_view_index_cm {
    ($n:tt, ($($xy:tt),+), $z:tt, ($($idx:tt),+), $last:tt) => {
        #[allow(unused_parens)]
        impl<L: Layout<Order = ColumnMajor>, $($xy: DimIndex),+, $z: DimIndex>
            ViewIndex<Const<$n>, ColumnMajor, L> for ($($xy),+, $z)
        {
            type Inner = <($($xy),+) as ViewIndex<Const<{$n - 1}>, ColumnMajor, L>>::Outer;
            type Layout = $z::Layout<Self::Inner>;
            type Outer = $z::Outer<Self::Inner>;

            fn view_info(self, layout: L) -> (isize, Self::Layout, Self::Outer) {
                let dim = layout.dim($n - 1);
                let (offset, _, inner) =
                    <($($xy),+) as ViewIndex<Const<{$n - 1}>, ColumnMajor, L>>::view_info(
                        ($(self.$idx),+),
                        layout
                    );

                self.$last.next_info(offset, inner, layout.size(dim), layout.stride(dim))
            }
        }
    };
}

impl_view_index_cm!(2, (X), Y, (0), 1);
impl_view_index_cm!(3, (X, Y), Z, (0, 1), 2);
impl_view_index_cm!(4, (X, Y, Z), W, (0, 1, 2), 3);
impl_view_index_cm!(5, (X, Y, Z, W), U, (0, 1, 2, 3), 4);
impl_view_index_cm!(6, (X, Y, Z, W, U), V, (0, 1, 2, 3, 4), 5);

macro_rules! impl_view_index_rm {
    ($n:tt, ($($yz:tt),+), ($($idx:tt),+)) => {
        #[allow(unused_parens)]
        impl<L: Layout<Order = RowMajor>, X: DimIndex, $($yz: DimIndex),+>
            ViewIndex<Const<$n>, RowMajor, L> for (X, $($yz),+)
        {
            type Inner = <($($yz),+) as ViewIndex<Const<{$n - 1}>, RowMajor, L>>::Outer;
            type Layout = X::Layout<Self::Inner>;
            type Outer = X::Outer<Self::Inner>;

            fn view_info(self, layout: L) -> (isize, Self::Layout, Self::Outer) {
                let dim = layout.dim($n - 1);
                let (offset, _, inner) =
                    <($($yz),+) as ViewIndex<Const<{$n - 1}>, RowMajor, L>>::view_info(
                        ($(self.$idx),+),
                        layout
                    );

                self.0.next_info(offset, inner, layout.size(dim), layout.stride(dim))
            }
        }
    };
}

impl_view_index_rm!(2, (Y), (1));
impl_view_index_rm!(3, (Y, Z), (1, 2));
impl_view_index_rm!(4, (Y, Z, W), (1, 2, 3));
impl_view_index_rm!(5, (Y, Z, W, U), (1, 2, 3, 4));
impl_view_index_rm!(6, (Y, Z, W, U, V), (1, 2, 3, 4, 5));

impl<S: Shape, T, D: Dim<Shape = S>, L: Layout<Dim = D>> Index<S> for SpanBase<T, L> {
    type Output = T;

    fn index(&self, index: S) -> &T {
        let layout = self.layout();

        for i in 0..L::Dim::RANK {
            if index.as_ref()[i] >= layout.size(i) {
                panic_bounds_check(index.as_ref()[i], layout.size(i))
            }
        }

        unsafe { &*self.as_ptr().offset(layout.offset(index)) }
    }
}

impl<S: Shape, T, D: Dim<Shape = S>, L: Layout<Dim = D>> IndexMut<S> for SpanBase<T, L> {
    fn index_mut(&mut self, index: S) -> &mut T {
        let layout = self.layout();

        for i in 0..L::Dim::RANK {
            if index.as_ref()[i] >= layout.size(i) {
                panic_bounds_check(index.as_ref()[i], layout.size(i))
            }
        }

        unsafe { &mut *self.as_mut_ptr().offset(layout.offset(index)) }
    }
}

impl<T, L: Layout<Dim = Const<1>>> Index<usize> for SpanBase<T, L> {
    type Output = T;

    fn index(&self, index: usize) -> &T {
        if index >= self.size(0) {
            panic_bounds_check(index, self.size(0))
        }

        unsafe { &*self.as_ptr().offset(self.stride(0) * index as isize) }
    }
}

impl<T, L: Layout<Dim = Const<1>>> IndexMut<usize> for SpanBase<T, L> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index >= self.size(0) {
            panic_bounds_check(index, self.size(0))
        }

        unsafe { &mut *self.as_mut_ptr().offset(self.stride(0) * index as isize) }
    }
}

macro_rules! impl_index_range {
    ($type:ty) => {
        impl<T, F: UnitStrided, O: Order, L: Layout<Dim = Const<1>, Format = F, Order = O>>
            Index<$type> for SpanBase<T, L>
        {
            type Output = SpanBase<T, DenseLayout<Const<1>, O>>;

            fn index(&self, index: $type) -> &Self::Output {
                self.as_slice()[index].as_ref()
            }
        }

        impl<T, F: UnitStrided, L: Layout<Dim = Const<1>, Format = F>> IndexMut<$type>
            for SpanBase<T, L>
        {
            fn index_mut(&mut self, index: $type) -> &mut Self::Output {
                self.as_mut_slice()[index].as_mut()
            }
        }
    };
}

impl_index_range!((Bound<usize>, Bound<usize>));
impl_index_range!(Range<usize>);
impl_index_range!(RangeFrom<usize>);
impl_index_range!(RangeInclusive<usize>);
impl_index_range!(RangeFull);
impl_index_range!(RangeTo<usize>);
impl_index_range!(RangeToInclusive<usize>);

#[cold]
#[inline(never)]
#[track_caller]
fn panic_bounds_check(index: usize, len: usize) -> ! {
    panic!("index out of bounds: the len is {len} but the index is {index}")
}
