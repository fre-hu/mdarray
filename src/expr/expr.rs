use std::borrow::{Borrow, BorrowMut};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice;

use crate::dim::{Const, Dim, Shape};
use crate::expr::{Map, Zip};
use crate::expression::Expression;
use crate::index::{self, Axis, DimIndex, Permutation, SpanIndex, ViewIndex};
use crate::iter::Iter;
use crate::layout::{Dense, Flat, Layout};
use crate::mapping::{DenseMapping, FlatMapping, Mapping, StridedMapping};
use crate::raw_span::RawSpan;
use crate::span::Span;
use crate::traits::{Apply, IntoExpression};

/// Expression that gives references to array elements.
pub struct Expr<'a, T, D: Dim, L: Layout = Dense> {
    span: RawSpan<T, D, L>,
    phantom: PhantomData<&'a T>,
}

/// Expression that gives mutable references to array elements.
pub struct ExprMut<'a, T, D: Dim, L: Layout = Dense> {
    span: RawSpan<T, D, L>,
    phantom: PhantomData<&'a mut T>,
}

macro_rules! impl_expr {
    ($name:tt, $as_ptr:tt, $from_raw_parts:tt, $raw_mut:tt, {$($mut:tt)?}, $repeatable:tt) => {
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
                Self { span: RawSpan::new_unchecked(ptr as *mut T, mapping), phantom: PhantomData }
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

        impl<'a, T, L: Layout> $name<'a, T, Const<2>, L> {
            /// Converts the array view into a new array view for the given diagonal,
            /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
            ///
            /// # Panics
            ///
            /// Panics if the absolute index is larger than the number of columns or rows.
            pub fn into_diag($($mut)? self, index: isize) -> $name<'a, T, Const<1>, Flat> {
                let (offset, len) = if index >= 0 {
                    assert!(index as usize <= self.size(1), "invalid diagonal");

                    (index * self.stride(1), self.size(0).min(self.size(1) - (index as usize)))
                } else {
                    assert!(-index as usize <= self.size(0), "invalid diagonal");

                    (-index * self.stride(0), self.size(1).min(self.size(0) - (-index as usize)))
                };

                let count = if len > 0 { offset } else { 0 }; // Offset pointer if non-empty.
                let mapping = FlatMapping::new([len], self.stride(0) + self.stride(1));

                unsafe { $name::new_unchecked(self.$as_ptr().offset(count), mapping) }
            }
        }

        impl<'a, T, D: Dim> $name<'a, T, D> {
            /// Converts the array view into a slice of all elements, where the array view
            /// must have dense layout.
            pub fn into_slice($($mut)? self) -> &'a $($mut)? [T] {
                unsafe { slice::$from_raw_parts(self.$as_ptr(), self.len()) }
            }
        }

        impl<'a, T, U, D: Dim, L: Layout> Apply<U> for &'a $name<'_, T, D, L> {
            type Output<F: FnMut(&'a T) -> U> = Map<Self::IntoExpr, F>;
            type ZippedWith<I: IntoExpression, F: FnMut((&'a T, I::Item)) -> U> =
                Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

            fn apply<F: FnMut(&'a T) -> U>(self, f: F) -> Self::Output<F> {
                self.expr().map(f)
            }

            fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
            where
                F: FnMut((&'a T, I::Item)) -> U,
            {
                self.expr().zip(expr).map(f)
            }
        }

        impl<T, U: ?Sized, D: Dim, L: Layout> AsRef<U> for $name<'_, T, D, L>
        where
            Span<T, D, L>: AsRef<U>,
        {
            fn as_ref(&self) -> &U {
                (**self).as_ref()
            }
        }

        impl<T, D: Dim, L: Layout> Borrow<Span<T, D, L>> for $name<'_, T, D, L> {
            fn borrow(&self) -> &Span<T, D, L> {
                self
            }
        }

        impl<T: Debug, D: Dim, L: Layout> Debug for $name<'_, T, D, L> {
            fn fmt(&self, f: &mut Formatter) -> Result {
                (**self).fmt(f)
            }
        }

        impl<T, D: Dim, L: Layout> Deref for $name<'_, T, D, L> {
            type Target = Span<T, D, L>;

            fn deref(&self) -> &Self::Target {
                self.span.as_span()
            }
        }

        impl<'a, T, D: Dim, L: Layout> Expression for $name<'a, T, D, L> {
            type Dim = D;

            const IS_REPEATABLE: bool = $repeatable;
            const SPLIT_MASK: usize =
                if L::IS_UNIFORM { (1 << D::RANK) >> 1 } else { (1 << D::RANK) - 1 };

            fn shape(&self) -> D::Shape {
                (**self).shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> &'a $($mut)? T {
                let count = if D::RANK > 0 { self.stride(0) * index as isize } else { 0 };

                &$($mut)? *self.$as_ptr().offset(count)
            }

            unsafe fn reset_dim(&mut self, dim: usize, count: usize) {
                let count = -self.stride(dim) * count as isize;
                let ptr = self.span.as_mut_ptr().offset(count);

                self.span.set_ptr(ptr);
            }

            unsafe fn step_dim(&mut self, dim: usize) {
                let ptr = self.span.as_mut_ptr().offset(self.stride(dim));

                self.span.set_ptr(ptr);
            }
        }

        impl<'a, T> From<&'a $($mut)? [T]> for $name<'a, T, Const<1>> {
            fn from(value: &'a $($mut)? [T]) -> Self {
                unsafe { Self::new_unchecked(value.$as_ptr(), DenseMapping::new([value.len()])) }
            }
        }

        impl<'a, T, D: Dim> From<$name<'a, T, D>> for &'a $($mut)? [T] {
            fn from($($mut)? value: $name<T, D>) -> Self {
                unsafe { slice::$from_raw_parts(value.$as_ptr(), value.len()) }
            }
        }

        impl<T: Hash, D: Dim, L: Layout> Hash for $name<'_, T, D, L> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                (**self).hash(state)
            }
        }

        impl<T, D: Dim, L: Layout, I: SpanIndex<T, D, L>> Index<I> for $name<'_, T, D, L> {
            type Output = I::Output;

            fn index(&self, index: I) -> &I::Output {
                index.index(self)
            }
        }

        impl<'a, T, D: Dim, L: Layout> IntoExpression for &'a $name<'_, T, D, L> {
            type Dim = D;
            type IntoExpr = Expr<'a, T, D, L>;

            fn into_expr(self) -> Self::IntoExpr {
                self.expr()
            }
        }

        impl<'a, T, D: Dim, L: Layout> IntoIterator for &'a $name<'_, T, D, L> {
            type Item = &'a T;
            type IntoIter = Iter<Expr<'a, T, D, L>>;

            fn into_iter(self) -> Self::IntoIter {
                self.expr().into_iter()
            }
        }

        impl<'a, T, D: Dim, L: Layout> IntoIterator for $name<'a, T, D, L> {
            type Item = &'a $($mut)? T;
            type IntoIter = Iter<Self>;

            fn into_iter(self) -> Iter<Self> {
                Iter::new(self)
            }
        }
    };
}

impl_expr!(Expr, as_ptr, from_raw_parts, const, {}, true);
impl_expr!(ExprMut, as_mut_ptr, from_raw_parts_mut, mut, {mut}, false);

macro_rules! impl_into_permuted {
    ($n:tt, ($($xyz:tt),+)) => {
        impl<'a, T, L: Layout> Expr<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view with the dimensions permuted.
            pub fn into_permuted<$(const $xyz: usize),+>(
                self
            ) -> Expr<'a, T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
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

                unsafe { Expr::new_unchecked(self.as_ptr(), mapping) }
            }
        }

        impl<'a, T, L: Layout> ExprMut<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view with the dimensions permuted.
            pub fn into_permuted<$(const $xyz: usize),+>(
                mut self
            ) -> ExprMut<'a, T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
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

                unsafe { ExprMut::new_unchecked(self.as_mut_ptr(), mapping) }
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
        impl<'a, T, L: Layout> Expr<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($xyz: DimIndex),+>(
                self,
                $($idx: $xyz),+
            ) -> Expr<
                'a,
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { Expr::new_unchecked(self.as_ptr().offset(count), mapping) }
            }
        }

        impl<'a, T, L: Layout> ExprMut<'a, T, Const<$n>, L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($xyz: DimIndex),+>(
                mut self,
                $($idx: $xyz),+
            ) -> ExprMut<
                'a,
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { ExprMut::new_unchecked(self.as_mut_ptr().offset(count), mapping) }
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

impl<'a, T, U, D: Dim, L: Layout> Apply<U> for &'a mut ExprMut<'_, T, D, L> {
    type Output<F: FnMut(&'a mut T) -> U> = Map<Self::IntoExpr, F>;
    type ZippedWith<I: IntoExpression, F: FnMut((&'a mut T, I::Item)) -> U> =
        Map<Zip<Self::IntoExpr, I::IntoExpr>, F>;

    fn apply<F: FnMut(&'a mut T) -> U>(self, f: F) -> Self::Output<F> {
        self.expr_mut().map(f)
    }

    fn zip_with<I: IntoExpression, F>(self, expr: I, f: F) -> Self::ZippedWith<I, F>
    where
        F: FnMut((&'a mut T, I::Item)) -> U,
    {
        self.expr_mut().zip(expr).map(f)
    }
}

impl<T, U: ?Sized, D: Dim, L: Layout> AsMut<U> for ExprMut<'_, T, D, L>
where
    Span<T, D, L>: AsMut<U>,
{
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, D: Dim, L: Layout> BorrowMut<Span<T, D, L>> for ExprMut<'_, T, D, L> {
    fn borrow_mut(&mut self) -> &mut Span<T, D, L> {
        self
    }
}

impl<T, D: Dim, L: Layout> Clone for Expr<'_, T, D, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, D: Dim, L: Layout> Copy for Expr<'_, T, D, L> {}

impl<T, D: Dim, L: Layout> DerefMut for ExprMut<'_, T, D, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.span.as_mut_span()
    }
}

macro_rules! impl_from_array_ref {
    ($n:tt, ($($size:tt),+), $array:tt) => {
        impl<'a, T, $(const $size: usize),+> From<&'a $array> for Expr<'a, T, Const<$n>> {
            fn from(array: &'a $array) -> Self {
                let mapping = if [$($size),+].contains(&0) {
                    DenseMapping::new([0; $n])
                } else {
                    DenseMapping::new([$($size),+])
                };

                unsafe { Self::new_unchecked(array.as_ptr().cast(), mapping) }
            }
        }

        impl<'a, T, $(const $size: usize),+> From<&'a mut $array> for ExprMut<'a, T, Const<$n>> {
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

impl<T, D: Dim, L: Layout, I: SpanIndex<T, D, L>> IndexMut<I> for ExprMut<'_, T, D, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, D: Dim, L: Layout> IntoExpression for &'a mut ExprMut<'_, T, D, L> {
    type Dim = D;
    type IntoExpr = ExprMut<'a, T, D, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, T, D: Dim, L: Layout> IntoIterator for &'a mut ExprMut<'_, T, D, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, D, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.expr_mut().into_iter()
    }
}

unsafe impl<'a, T: Sync, D: Dim, L: Layout> Send for Expr<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for Expr<'a, T, D, L> {}

unsafe impl<'a, T: Send, D: Dim, L: Layout> Send for ExprMut<'a, T, D, L> {}
unsafe impl<'a, T: Sync, D: Dim, L: Layout> Sync for ExprMut<'a, T, D, L> {}
