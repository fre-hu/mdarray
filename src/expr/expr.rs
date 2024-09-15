use std::borrow::{Borrow, BorrowMut};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice;

use crate::array::Array;
use crate::dim::{Const, Dim, Dyn};
use crate::expr::adapters::{Map, Zip};
use crate::expression::Expression;
use crate::index::{self, Axis, DimIndex, Inner, Outer, Permutation, SpanIndex, ViewIndex};
use crate::iter::Iter;
use crate::layout::{Dense, Flat, Layout};
use crate::mapping::{DenseMapping, FlatMapping, Mapping, StridedMapping};
use crate::raw_span::RawSpan;
use crate::shape::{IntoShape, Rank, Shape};
use crate::span::Span;
use crate::traits::{Apply, IntoExpression};

/// Expression that gives references to array elements.
pub struct Expr<'a, T, S: Shape, L: Layout = Dense> {
    span: RawSpan<T, S, L>,
    phantom: PhantomData<&'a T>,
}

/// Expression that gives mutable references to array elements.
pub struct ExprMut<'a, T, S: Shape, L: Layout = Dense> {
    span: RawSpan<T, S, L>,
    phantom: PhantomData<&'a mut T>,
}

macro_rules! impl_expr {
    ($name:tt, $as_ptr:tt, $from_raw_parts:tt, $raw_mut:tt, {$($mut:tt)?}, $repeatable:tt) => {
        impl<'a, T, S: Shape, L: Layout> $name<'a, T, S, L> {
            /// Converts the array view into a one-dimensional array view.
            ///
            /// # Panics
            ///
            /// Panics if the array layout is not uniformly strided.
            pub fn into_flattened(self) -> $name<'a, T, Dyn, L::Uniform> {
                let len = self.len();

                self.into_shape([len])
            }

            /// Converts the array view into a remapped array view.
            ///
            /// # Panics
            ///
            /// Panics if the memory layout is not compatible with the new array layout.
            pub fn into_mapping<M: Layout>($($mut)? self) -> $name<'a, T, S, M> {
                let mapping = M::Mapping::remap(self.mapping());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a reshaped array view with similar layout.
            ///
            /// # Panics
            ///
            /// Panics if the array length is changed, or the memory layout is not compatible.
            pub fn into_shape<I: IntoShape>(
                $($mut)? self,
                shape: I
            ) -> $name<'a, T, I::IntoShape, <I::IntoShape as Shape>::Layout<L::Uniform, L>> {
                let mapping = Mapping::reshape(self.mapping(), shape.into_shape());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Divides the array view into two at an index along the outermost dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension,
            /// or if the rank is not at least 1.
            pub fn into_split_at(
                self,
                mid: usize,
            ) -> (
                $name<'a, T, <Outer as Axis>::Replace<Dyn, S>, L>,
                $name<'a, T, <Outer as Axis>::Replace<Dyn, S>, L>,
            ) {
                self.into_split_dim_at::<Outer>(mid)
            }

            /// Divides the array view into two at an index along the specified dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension,
            /// or if the dimension is out of bounds.
            pub fn into_split_axis_at<const N: usize>(
                self,
                mid: usize,
            ) -> (
                $name<'a, T, <Inner<N> as Axis>::Replace<Dyn, S>, <Inner<N> as Axis>::Resize<S, L>>,
                $name<'a, T, <Inner<N> as Axis>::Replace<Dyn, S>, <Inner<N> as Axis>::Resize<S, L>>,
            )
            where
                Inner<N>: Axis,
            {
                self.into_split_dim_at::<Inner<N>>(mid)
            }

            /// Creates an array view from a raw pointer and layout.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and a valid array view for the given layout.
            pub unsafe fn new_unchecked(ptr: *$raw_mut T, mapping: L::Mapping<S>) -> Self {
                Self { span: RawSpan::new_unchecked(ptr as *mut T, mapping), phantom: PhantomData }
            }

            fn into_split_dim_at<A: Axis>(
                $($mut)? self,
                mid: usize
            ) -> (
                $name<'a, T, A::Replace<Dyn, S>, A::Resize<S, L>>,
                $name<'a, T, A::Replace<Dyn, S>, A::Resize<S, L>>,
            ) {
                let index = A::index(S::RANK);
                let size = self.dim(index);

                if mid > size {
                    index::panic_bounds_check(mid, size);
                }

                let left_mapping = A::resize(self.mapping(), mid);
                let right_mapping = A::resize(self.mapping(), size - mid);

                // Calculate offset for the second view if non-empty.
                let offset = self.stride(index) * mid as isize;
                let count = if right_mapping.is_empty() { 0 } else { offset };

                unsafe {
                    let left = $name::new_unchecked(self.$as_ptr(), left_mapping);
                    let right = $name::new_unchecked(self.$as_ptr().offset(count), right_mapping);

                    (left, right)
                }
            }
        }

        impl<'a, T, X: Dim, Y: Dim, L: Layout> $name<'a, T, (X, Y), L> {
            /// Converts the array view into a new array view for the given diagonal,
            /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
            ///
            /// # Panics
            ///
            /// Panics if the absolute index is larger than the number of columns or rows.
            pub fn into_diag($($mut)? self, index: isize) -> $name<'a, T, Dyn, Flat> {
                let (offset, len) = if index >= 0 {
                    assert!(index as usize <= self.dim(1), "invalid diagonal");

                    (index * self.stride(1), self.dim(0).min(self.dim(1) - (index as usize)))
                } else {
                    assert!(-index as usize <= self.dim(0), "invalid diagonal");

                    (-index * self.stride(0), self.dim(1).min(self.dim(0) - (-index as usize)))
                };

                let count = if len > 0 { offset } else { 0 }; // Offset pointer if non-empty.
                let mapping = FlatMapping::new(Dyn(len), self.stride(0) + self.stride(1));

                unsafe { $name::new_unchecked(self.$as_ptr().offset(count), mapping) }
            }
        }

        impl<'a, T, S: Shape> $name<'a, T, S> {
            /// Converts the array view into a slice of all elements, where the array view
            /// must have dense layout.
            pub fn into_slice($($mut)? self) -> &'a $($mut)? [T] {
                unsafe { slice::$from_raw_parts(self.$as_ptr(), self.len()) }
            }
        }

        impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a $name<'_, T, S, L> {
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

        impl<T, U: ?Sized, S: Shape, L: Layout> AsRef<U> for $name<'_, T, S, L>
        where
            Span<T, S, L>: AsRef<U>,
        {
            fn as_ref(&self) -> &U {
                (**self).as_ref()
            }
        }

        impl<T, S: Shape, L: Layout> Borrow<Span<T, S, L>> for $name<'_, T, S, L> {
            fn borrow(&self) -> &Span<T, S, L> {
                self
            }
        }

        impl<T: Debug, S: Shape, L: Layout> Debug for $name<'_, T, S, L> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                (**self).fmt(f)
            }
        }

        impl<T, S: Shape, L: Layout> Deref for $name<'_, T, S, L> {
            type Target = Span<T, S, L>;

            fn deref(&self) -> &Self::Target {
                self.span.as_span()
            }
        }

        impl<'a, T, S: Shape, L: Layout> Expression for $name<'a, T, S, L> {
            type Shape = S;

            const IS_REPEATABLE: bool = $repeatable;
            const SPLIT_MASK: usize =
                if L::IS_UNIFORM { (1 << S::RANK) >> 1 } else { (1 << S::RANK) - 1 };

            fn shape(&self) -> S {
                (**self).shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> &'a $($mut)? T {
                let count = if S::RANK > 0 { self.stride(0) * index as isize } else { 0 };

                &$($mut)? *self.$as_ptr().offset(count)
            }

            unsafe fn reset_dim(&mut self, index: usize, count: usize) {
                let count = -self.stride(index) * count as isize;
                let ptr = self.span.as_mut_ptr().offset(count);

                self.span.set_ptr(ptr);
            }

            unsafe fn step_dim(&mut self, index: usize) {
                let ptr = self.span.as_mut_ptr().offset(self.stride(index));

                self.span.set_ptr(ptr);
            }
        }

        impl<'a, T, S: Shape, L: Layout, I> From<&'a $($mut)? I> for $name<'a, T, S, L>
        where
            &'a $($mut)? I: IntoExpression<IntoExpr = $name<'a, T, S, L>>
        {
            fn from(value: &'a $($mut)? I) -> Self {
                value.into_expr()
            }
        }

        impl<'a, T> From<&'a $($mut)? [T]> for $name<'a, T, Dyn> {
            fn from(value: &'a $($mut)? [T]) -> Self {
                let mapping = DenseMapping::new(Dyn(value.len()));

                unsafe { Self::new_unchecked(value.$as_ptr(), mapping) }
            }
        }

        impl<'a, T, S: Shape> From<$name<'a, T, S>> for &'a $($mut)? [T] {
            fn from($($mut)? value: $name<T, S>) -> Self {
                unsafe { slice::$from_raw_parts(value.$as_ptr(), value.len()) }
            }
        }

        impl<T: Hash, S: Shape, L: Layout> Hash for $name<'_, T, S, L> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                (**self).hash(state)
            }
        }

        impl<T, S: Shape, L: Layout, I: SpanIndex<T, S, L>> Index<I> for $name<'_, T, S, L> {
            type Output = I::Output;

            fn index(&self, index: I) -> &I::Output {
                index.index(self)
            }
        }

        impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a $name<'_, T, S, L> {
            type Shape = S;
            type IntoExpr = Expr<'a, T, S, L>;

            fn into_expr(self) -> Self::IntoExpr {
                self.expr()
            }
        }

        impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a $name<'_, T, S, L> {
            type Item = &'a T;
            type IntoIter = Iter<Expr<'a, T, S, L>>;

            fn into_iter(self) -> Self::IntoIter {
                self.expr().into_iter()
            }
        }

        impl<'a, T, S: Shape, L: Layout> IntoIterator for $name<'a, T, S, L> {
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
    ($n:tt, ($($xyz:tt),+), ($($abc:tt),+)) => {
        #[allow(unused_parens)]
        impl<'a, T, $($xyz: Dim,)+ L: Layout> Expr<'a, T, ($($xyz),+), L> {
            /// Converts the array view into a new array view with the dimensions permuted.
            pub fn into_permuted<$(const $abc: usize),+>(
                self
            ) -> Expr<
                'a,
                T,
                <($(Const<$abc>,)+) as Permutation>::Shape<($($xyz),+)>,
                <($(Const<$abc>,)+) as Permutation>::Layout<L>
            >
            where
                ($(Const<$abc>,)+): Permutation
            {
                let dims = [$(self.dim($abc)),+];
                let strides = [$(self.stride($abc)),+];

                let shape = Shape::from_dims(TryFrom::try_from(&dims[..]).expect("invalid rank"));

                let mapping = if $n > 1 {
                    let strides = TryFrom::try_from(&strides[..]).expect("invalid rank");

                    Mapping::remap(StridedMapping::new(shape, strides))
                } else {
                    Mapping::remap(FlatMapping::new(shape, strides[0]))
                };

                unsafe { Expr::new_unchecked(self.as_ptr(), mapping) }
            }
        }

        #[allow(unused_parens)]
        impl<'a, T, $($xyz: Dim,)+ L: Layout> ExprMut<'a, T, ($($xyz),+), L> {
            /// Converts the array view into a new array view with the dimensions permuted.
            pub fn into_permuted<$(const $abc: usize),+>(
                mut self
            ) -> ExprMut<
                'a,
                T,
                <($(Const<$abc>,)+) as Permutation>::Shape<($($xyz),+)>,
                <($(Const<$abc>,)+) as Permutation>::Layout<L>
            >
            where
                ($(Const<$abc>,)+): Permutation
            {
                let dims = [$(self.dim($abc)),+];
                let strides = [$(self.stride($abc)),+];

                let shape = Shape::from_dims(TryFrom::try_from(&dims[..]).expect("invalid rank"));

                let mapping = if $n > 1 {
                    let strides = TryFrom::try_from(&strides[..]).expect("invalid rank");

                    Mapping::remap(StridedMapping::new(shape, strides))
                } else {
                    Mapping::remap(FlatMapping::new(shape, strides[0]))
                };

                unsafe { ExprMut::new_unchecked(self.as_mut_ptr(), mapping) }
            }
        }
    };
}

impl_into_permuted!(1, (X), (A));
impl_into_permuted!(2, (X, Y), (A, B));
impl_into_permuted!(3, (X, Y, Z), (A, B, C));
impl_into_permuted!(4, (X, Y, Z, W), (A, B, C, D));
impl_into_permuted!(5, (X, Y, Z, W, U), (A, B, C, D, E));
impl_into_permuted!(6, (X, Y, Z, W, U, V), (A, B, C, D, E, F));

macro_rules! impl_into_view {
    ($n:tt, ($($xyz:tt),+), ($($abc:tt),+), ($($idx:tt),+)) => {
        #[allow(unused_parens)]
        impl<'a, T, $($xyz: Dim,)+ L: Layout> Expr<'a, T, ($($xyz),+), L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($abc: DimIndex),+>(
                self,
                $($idx: $abc),+
            ) -> Expr<
                'a,
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz),+)>,
                <($($abc,)+) as ViewIndex>::Layout<($($xyz),+), L>,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { Expr::new_unchecked(self.as_ptr().offset(count), mapping) }
            }
        }

        #[allow(unused_parens)]
        impl<'a, T, $($xyz: Dim,)+ L: Layout> ExprMut<'a, T, ($($xyz),+), L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($abc: DimIndex),+>(
                mut self,
                $($idx: $abc),+
            ) -> ExprMut<
                'a,
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz),+)>,
                <($($abc,)+) as ViewIndex>::Layout<($($xyz),+), L>,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { ExprMut::new_unchecked(self.as_mut_ptr().offset(count), mapping) }
            }
        }
    };
}

impl_into_view!(1, (X), (A), (a));
impl_into_view!(2, (X, Y), (A, B), (a, b));
impl_into_view!(3, (X, Y, Z), (A, B, C), (a, b, c));
impl_into_view!(4, (X, Y, Z, W), (A, B, C, D), (a, b, c, d));
impl_into_view!(5, (X, Y, Z, W, U), (A, B, C, D, E), (a, b, c, d, e));
impl_into_view!(6, (X, Y, Z, W, U, V), (A, B, C, D, E, F), (a, b, c, d, e, f));

impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a mut ExprMut<'_, T, S, L> {
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

impl<T, U: ?Sized, S: Shape, L: Layout> AsMut<U> for ExprMut<'_, T, S, L>
where
    Span<T, S, L>: AsMut<U>,
{
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, S: Shape, L: Layout> BorrowMut<Span<T, S, L>> for ExprMut<'_, T, S, L> {
    fn borrow_mut(&mut self) -> &mut Span<T, S, L> {
        self
    }
}

impl<T, S: Shape, L: Layout> Clone for Expr<'_, T, S, L> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T, S: Shape, L: Layout> Copy for Expr<'_, T, S, L> {}

impl<T, S: Shape, L: Layout> DerefMut for ExprMut<'_, T, S, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.span.as_mut_span()
    }
}

macro_rules! impl_from_array_ref {
    ($n:tt, ($($xyz:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<'a, T, $(const $xyz: usize),+> From<&'a Array<T, ($(Const<$xyz>),+)>>
            for Expr<'a, T, Rank<$n>>
        {
            fn from(value: &'a Array<T, ($(Const<$xyz>),+)>) -> Self {
                Self::from(&value.0)
            }
        }

        impl<'a, T, $(const $xyz: usize),+> From<&'a $array> for Expr<'a, T, Rank<$n>> {
            fn from(value: &'a $array) -> Self {
                let mapping = DenseMapping::new(($(Dyn($xyz)),+));

                _ = mapping.shape().checked_len().expect("invalid length");

                unsafe { Self::new_unchecked(value.as_ptr().cast(), mapping) }
            }
        }

        #[allow(unused_parens)]
        impl<'a, T, $(const $xyz: usize),+> From<&'a mut Array<T, ($(Const<$xyz>),+)>>
            for ExprMut<'a, T, Rank<$n>>
        {
            fn from(value: &'a mut Array<T, ($(Const<$xyz>),+)>) -> Self {
                Self::from(&mut value.0)
            }
        }

        impl<'a, T, $(const $xyz: usize),+> From<&'a mut $array> for ExprMut<'a, T, Rank<$n>> {
            fn from(value: &'a mut $array) -> Self {
                let mapping = DenseMapping::new(($(Dyn($xyz)),+));

                _ = mapping.shape().checked_len().expect("invalid length");

                unsafe { Self::new_unchecked(value.as_mut_ptr().cast(), mapping) }
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

impl<T, S: Shape, L: Layout, I: SpanIndex<T, S, L>> IndexMut<I> for ExprMut<'_, T, S, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a mut ExprMut<'_, T, S, L> {
    type Shape = S;
    type IntoExpr = ExprMut<'a, T, S, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a mut ExprMut<'_, T, S, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, S, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.expr_mut().into_iter()
    }
}

unsafe impl<'a, T: Sync, S: Shape, L: Layout> Send for Expr<'a, T, S, L> {}
unsafe impl<'a, T: Sync, S: Shape, L: Layout> Sync for Expr<'a, T, S, L> {}

unsafe impl<'a, T: Send, S: Shape, L: Layout> Send for ExprMut<'a, T, S, L> {}
unsafe impl<'a, T: Sync, S: Shape, L: Layout> Sync for ExprMut<'a, T, S, L> {}

macro_rules! impl_try_from_array_ref {
    ($n:tt, ($($xyz:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<'a, T, $(const $xyz: usize),+> TryFrom<Expr<'a, T, Rank<$n>>>
            for &'a Array<T, ($(Const<$xyz>),+)>
        {
            type Error = Expr<'a, T, Rank<$n>>;

            fn try_from(value: Expr<'a, T, Rank<$n>>) -> Result<Self, Self::Error> {
                Ok(<&'a $array>::try_from(value)?.as_ref())
            }
        }

        impl<'a, T, $(const $xyz: usize),+> TryFrom<Expr<'a, T, Rank<$n>>> for &'a $array {
            type Error = Expr<'a, T, Rank<$n>>;

            fn try_from(value: Expr<'a, T, Rank<$n>>) -> Result<Self, Self::Error> {
                if value.dims() == [$($xyz),+] {
                    Ok(unsafe { &*value.as_ptr().cast() })
                } else {
                    Err(value)
                }
            }
        }

        #[allow(unused_parens)]
        impl<'a, T, $(const $xyz: usize),+> TryFrom<ExprMut<'a, T, Rank<$n>>>
            for &'a mut Array<T, ($(Const<$xyz>),+)>
        {
            type Error = ExprMut<'a, T, Rank<$n>>;

            fn try_from(value: ExprMut<'a, T, Rank<$n>>) -> Result<Self, Self::Error> {
                Ok(<&'a mut $array>::try_from(value)?.as_mut())
            }
        }

        impl<'a, T, $(const $xyz: usize),+> TryFrom<ExprMut<'a, T, Rank<$n>>> for &'a mut $array {
            type Error = ExprMut<'a, T, Rank<$n>>;

            fn try_from(mut value: ExprMut<'a, T, Rank<$n>>) -> Result<Self, Self::Error> {
                if value.dims() == [$($xyz),+] {
                    Ok(unsafe { &mut *value.as_mut_ptr().cast() })
                } else {
                    Err(value)
                }
            }
        }
    };
}

impl_try_from_array_ref!(1, (X), [T; X]);
impl_try_from_array_ref!(2, (X, Y), [[T; X]; Y]);
impl_try_from_array_ref!(3, (X, Y, Z), [[[T; X]; Y]; Z]);
impl_try_from_array_ref!(4, (X, Y, Z, W), [[[[T; X]; Y]; Z]; W]);
impl_try_from_array_ref!(5, (X, Y, Z, W, U), [[[[[T; X]; Y]; Z]; W]; U]);
impl_try_from_array_ref!(6, (X, Y, Z, W, U, V), [[[[[[T; X]; Y]; Z]; W]; U]; V]);
