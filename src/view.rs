use std::slice;

use crate::array::{GridArray, ViewArray, ViewArrayMut};
use crate::buffer::{ViewBuffer, ViewBufferMut};
use crate::dim::{Const, Dim, Shape};
use crate::expr::{Expr, ExprMut};
use crate::expression::Expression;
use crate::index::{self, Axis, DimIndex, Permutation, ViewIndex};
use crate::iter::Iter;
use crate::layout::{Dense, Layout};
use crate::mapping::{DenseMapping, FlatMapping, Mapping, StridedMapping};
use crate::traits::{Apply, IntoExpression};

macro_rules! impl_view {
    ($name:tt, $buffer:tt, $as_ptr:tt, $from_raw_parts:tt, $raw_mut:tt, {$($mut:tt)?}) => {
        impl<'a, T, D: Dim, L: Layout> $name<'a, T, D, L> {
            /// Converts the array view into a one-dimensional array view.
            ///
            /// # Panics
            ///
            /// Panics if the array layout is not uniformly strided.
            pub fn into_flattened(self) -> $name<'a, T, Const<1>, L::Uniform> {
                let len = self.len();

                self.into_shape([len])
            }

            /// Converts the array view into a remapped array view.
            ///
            /// # Panics
            ///
            /// Panics if the memory layout is not compatible with the new array layout.
            pub fn into_mapping<M: Layout>($($mut)? self) -> $name<'a, T, D, M> {
                let mapping = M::Mapping::remap(self.mapping());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a reshaped array view with similar layout.
            ///
            /// # Panics
            ///
            /// Panics if the array length is changed, or the memory layout is not compatible.
            pub fn into_shape<S: Shape>(
                $($mut)? self,
                shape: S
            ) -> $name<'a, T, S::Dim, <S::Dim as Dim>::Layout<L>> {
                let mapping = Mapping::reshape(self.mapping(), shape);

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Divides the array view into two at an index along the outermost dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension.
            pub fn into_split_at(self, mid: usize) -> ($name<'a, T, D, L>, $name<'a, T, D, L>) {
                assert!(D::RANK > 0, "invalid rank");

                self.into_split_dim_at(D::RANK - 1, mid)
            }

            /// Divides the array view into two at an index along the specified dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension.
            pub fn into_split_axis_at<const DIM: usize>(
                self,
                mid: usize,
            ) -> (
                $name<'a, T, D, <Const<DIM> as Axis<D>>::Split<L>>,
                $name<'a, T, D, <Const<DIM> as Axis<D>>::Split<L>>
            )
            where
                Const<DIM>: Axis<D>
            {
                self.into_mapping().into_split_dim_at(DIM, mid)
            }

            /// Creates an array view from a raw pointer and layout.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and a valid array view for the given layout.
            pub unsafe fn new_unchecked(ptr: *$raw_mut T, mapping: L::Mapping<D>) -> Self {
                Self { buffer: $buffer::new_unchecked(ptr, mapping) }
            }

            fn into_split_dim_at(
                $($mut)? self,
                dim: usize,
                mid: usize
            ) -> ($name<'a, T, D, L>, $name<'a, T, D, L>) {
                if mid > self.size(dim) {
                    index::panic_bounds_check(mid, self.size(dim));
                }

                let left_mapping = self.mapping().resize_dim(dim, mid);
                let right_mapping = self.mapping().resize_dim(dim, self.size(dim) - mid);

                // Calculate offset for the second view if non-empty.
                let count = if mid == self.size(dim) { 0 } else { self.stride(dim) * mid as isize };

                unsafe {
                    let left = $name::new_unchecked(self.$as_ptr(), left_mapping);
                    let right = $name::new_unchecked(self.$as_ptr().offset(count), right_mapping);

                    (left, right)
                }
            }
        }

        impl<'a, T, D: Dim> $name<'a, T, D, Dense> {
            /// Converts the array view into a slice of all elements, where the array view
            /// must have dense layout.
            pub fn into_slice($($mut)? self) -> &'a $($mut)? [T] {
                unsafe { slice::$from_raw_parts(self.$as_ptr(), self.len()) }
            }
        }
    };
}

impl_view!(ViewArray, ViewBuffer, as_ptr, from_raw_parts, const, {});
impl_view!(ViewArrayMut, ViewBufferMut, as_mut_ptr, from_raw_parts_mut, mut, {mut});

macro_rules! impl_into_permuted {
    ($n:tt, ($($xyz:tt),+)) => {
        impl<'a, T, L: Layout> ViewArray<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view with the dimensions permuted.
            pub fn into_permuted<$(const $xyz: usize),+>(
                self
            ) -> ViewArray<'a, T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
            where
                ($(Const<$xyz>,)+): Permutation
            {
                let shape = [$(self.size($xyz)),+];
                let strides = [$(self.stride($xyz)),+];

                let mapping = if $n > 1 {
                    Mapping::remap(StridedMapping::new(shape, strides))
                } else {
                    Mapping::remap(FlatMapping::new(shape, strides[0]))
                };

                unsafe { ViewArray::new_unchecked(self.as_ptr(), mapping) }
            }
        }

        impl<'a, T, L: Layout> ViewArrayMut<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view with the dimensions permuted.
            pub fn into_permuted<$(const $xyz: usize),+>(
                mut self
            ) -> ViewArrayMut<'a, T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
            where
                ($(Const<$xyz>,)+): Permutation
            {
                let shape = [$(self.size($xyz)),+];
                let strides = [$(self.stride($xyz)),+];

                let mapping = if $n > 1 {
                    Mapping::remap(StridedMapping::new(shape, strides))
                } else {
                    Mapping::remap(FlatMapping::new(shape, strides[0]))
                };

                unsafe { ViewArrayMut::new_unchecked(self.as_mut_ptr(), mapping) }
            }
        }
    };
}

impl_into_permuted!(1, (X));
impl_into_permuted!(2, (X, Y));
impl_into_permuted!(3, (X, Y, Z));
impl_into_permuted!(4, (X, Y, Z, W));
impl_into_permuted!(5, (X, Y, Z, W, U));
impl_into_permuted!(6, (X, Y, Z, W, U, V));

macro_rules! impl_into_view {
    ($n:tt, ($($xyz:tt),+), ($($idx:tt),+)) => {
        impl<'a, T, L: Layout> ViewArray<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($xyz: DimIndex),+>(
                self,
                $($idx: $xyz),+
            ) -> ViewArray<
                'a,
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { ViewArray::new_unchecked(self.as_ptr().offset(count), mapping) }
            }
        }

        impl<'a, T, L: Layout> ViewArrayMut<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($xyz: DimIndex),+>(
                mut self,
                $($idx: $xyz),+
            ) -> ViewArrayMut<
                'a,
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { ViewArrayMut::new_unchecked(self.as_mut_ptr().offset(count), mapping) }
            }
        }
    };
}

impl_into_view!(1, (X), (x));
impl_into_view!(2, (X, Y), (x, y));
impl_into_view!(3, (X, Y, Z), (x, y, z));
impl_into_view!(4, (X, Y, Z, W), (x, y, z, w));
impl_into_view!(5, (X, Y, Z, W, U), (x, y, z, w, u));
impl_into_view!(6, (X, Y, Z, W, U, V), (x, y, z, w, u, v));

impl<'a, T, U, D: Dim, L: Layout> Apply<U> for ViewArray<'a, T, D, L> {
    type Output = GridArray<U, D>;
    type ZippedWith<I: IntoExpression> = GridArray<U, D::Max<I::Dim>>;

    fn apply<F: FnMut(&'a T) -> U>(self, f: F) -> Self::Output {
        self.into_expr().map(f).eval()
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, mut f: F) -> Self::ZippedWith<I>
    where
        F: FnMut(&'a T, I::Item) -> U,
    {
        self.into_expr().zip(expr).map(|(x, y)| f(x, y)).eval()
    }
}

impl<'a, T, U, D: Dim, L: Layout> Apply<U> for ViewArrayMut<'a, T, D, L> {
    type Output = GridArray<U, D>;
    type ZippedWith<I: IntoExpression> = GridArray<U, D::Max<I::Dim>>;

    fn apply<F: FnMut(&'a mut T) -> U>(self, f: F) -> Self::Output {
        self.into_expr().map(f).eval()
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, mut f: F) -> Self::ZippedWith<I>
    where
        F: FnMut(&'a mut T, I::Item) -> U,
    {
        self.into_expr().zip(expr).map(|(x, y)| f(x, y)).eval()
    }
}

impl<'a, T> From<&'a [T]> for ViewArray<'a, T, Const<1>, Dense> {
    fn from(slice: &'a [T]) -> Self {
        unsafe { Self::new_unchecked(slice.as_ptr(), DenseMapping::new([slice.len()])) }
    }
}

impl<'a, T> From<&'a mut [T]> for ViewArrayMut<'a, T, Const<1>, Dense> {
    fn from(slice: &'a mut [T]) -> Self {
        unsafe { Self::new_unchecked(slice.as_mut_ptr(), DenseMapping::new([slice.len()])) }
    }
}

macro_rules! impl_from_array_ref {
    ($n:tt, ($($size:tt),+), $array:tt) => {
        impl<'a, T, $(const $size: usize),+> From<&'a $array>
            for ViewArray<'a, T, Const<$n>, Dense>
        {
            fn from(array: &'a $array) -> Self {
                let mapping = if [$($size),+].contains(&0) {
                    DenseMapping::new([0; $n])
                } else {
                    DenseMapping::new([$($size),+])
                };

                unsafe { Self::new_unchecked(array.as_ptr().cast(), mapping) }
            }
        }

        impl<'a, T, $(const $size: usize),+> From<&'a mut $array>
            for ViewArrayMut<'a, T, Const<$n>, Dense>
        {
            fn from(array: &'a mut $array) -> Self {
                let mapping = if [$($size),+].contains(&0) {
                    DenseMapping::new([0; $n])
                } else {
                    DenseMapping::new([$($size),+])
                };

                unsafe { Self::new_unchecked(array.as_mut_ptr().cast(), mapping) }
            }
        }
    };
}

impl_from_array_ref!(1, (X), [T; X]);
impl_from_array_ref!(2, (X, Y), [[T; X]; Y]);
impl_from_array_ref!(3, (X, Y, Z), [[[T; X]; Y]; Z]);
impl_from_array_ref!(4, (X, Y, Z, W), [[[[T; X]; Y]; Z]; W]);
impl_from_array_ref!(5, (X, Y, Z, W, U), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array_ref!(6, (X, Y, Z, W, U, V), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<'a, T> From<&'a T> for ViewArray<'a, T, Const<0>, Dense> {
    fn from(value: &'a T) -> Self {
        unsafe { Self::new_unchecked(value, DenseMapping::default()) }
    }
}

impl<'a, T> From<&'a mut T> for ViewArrayMut<'a, T, Const<0>, Dense> {
    fn from(value: &'a mut T) -> Self {
        unsafe { Self::new_unchecked(value, DenseMapping::default()) }
    }
}

impl<'a, T, D: Dim, L: Layout> IntoExpression for ViewArray<'a, T, D, L> {
    type Item = &'a T;
    type Dim = D;
    type Producer = Expr<'a, T, D, L>;

    fn into_expr(self) -> Expression<Self::Producer> {
        Expression::new(unsafe { Expr::new_unchecked(self.as_ptr(), self.mapping()) })
    }
}

impl<'a, T, D: Dim, L: Layout> IntoExpression for ViewArrayMut<'a, T, D, L> {
    type Item = &'a mut T;
    type Dim = D;
    type Producer = ExprMut<'a, T, D, L>;

    fn into_expr(mut self) -> Expression<Self::Producer> {
        Expression::new(unsafe { ExprMut::new_unchecked(self.as_mut_ptr(), self.mapping()) })
    }
}

impl<'a, T, D: 'a + Dim, L: Layout> IntoIterator for ViewArray<'a, T, D, L> {
    type Item = &'a T;
    type IntoIter = Iter<Expr<'a, T, D, L>>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.into_expr().into_producer())
    }
}

impl<'a, T, D: 'a + Dim, L: Layout> IntoIterator for ViewArrayMut<'a, T, D, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, D, L>>;

    fn into_iter(self) -> Self::IntoIter {
        Iter::new(self.into_expr().into_producer())
    }
}
