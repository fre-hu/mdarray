use std::borrow::{Borrow, BorrowMut};
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice;

use crate::dim::{Const, Dim, Dyn};
use crate::expr::{Apply, Expression, IntoExpression, Iter, Map, Zip};
use crate::index::{self, Axis, DimIndex, Permutation, SliceIndex, Split, ViewIndex};
use crate::layout::{Dense, Layout, Strided};
use crate::mapping::{DenseMapping, Mapping, StridedMapping};
use crate::raw_slice::RawSlice;
use crate::shape::{DynRank, IntoShape, Rank, Shape};
use crate::slice::Slice;

/// Multidimensional array view.
pub struct View<'a, T, S: Shape = DynRank, L: Layout = Dense> {
    slice: RawSlice<T, S, L>,
    phantom: PhantomData<&'a T>,
}

/// Mutable multidimensional array view.
pub struct ViewMut<'a, T, S: Shape = DynRank, L: Layout = Dense> {
    slice: RawSlice<T, S, L>,
    phantom: PhantomData<&'a mut T>,
}

/// Multidimensional array view with dynamically-sized dimensions.
pub type DView<'a, T, const N: usize, L = Dense> = View<'a, T, Rank<N>, L>;

/// Mutable multidimensional array view with dynamically-sized dimensions.
pub type DViewMut<'a, T, const N: usize, L = Dense> = ViewMut<'a, T, Rank<N>, L>;

macro_rules! impl_view {
    ($name:tt, $as_ptr:tt, $from_raw_parts:tt, $raw_mut:tt, {$($mut:tt)?}, $repeatable:tt) => {
        impl<'a, T, S: Shape, L: Layout> $name<'a, T, S, L> {
            /// Converts the array view into a new array view for the given diagonal,
            /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
            ///
            /// # Panics
            ///
            /// Panics if the rank is not equal to 2, or if the absolute index is larger
            /// than the number of columns or rows.
            pub fn into_diag($($mut)? self, index: isize) -> $name<'a, T, (Dyn,), Strided> {
                assert!(self.rank() == 2, "invalid rank");

                let (offset, len) = if index >= 0 {
                    assert!(index as usize <= self.dim(1), "invalid diagonal");

                    (index * self.stride(1), self.dim(0).min(self.dim(1) - (index as usize)))
                } else {
                    assert!(-index as usize <= self.dim(0), "invalid diagonal");

                    (-index * self.stride(0), self.dim(1).min(self.dim(0) - (-index as usize)))
                };

                let count = if len > 0 { offset } else { 0 }; // Offset pointer if non-empty.
                let mapping = StridedMapping::new((len,), &[self.stride(0) + self.stride(1)]);

                unsafe { $name::new_unchecked(self.$as_ptr().offset(count), mapping) }
            }

            /// Converts the array view into an array view with dynamic rank.
            pub fn into_dyn(self) -> $name<'a, T, DynRank, L> {
                self.into_mapping()
            }

            /// Converts the array view into a one-dimensional array view.
            ///
            /// # Panics
            ///
            /// Panics if the array layout is not uniformly strided.
            pub fn into_flat(self) -> $name<'a, T, (Dyn,), L> {
                let len = self.len();

                self.into_shape([len])
            }

            /// Converts the array view into a new array view indexing the specified dimension.
            ///
            /// If the dimension to be indexed is know at compile time, the resulting array shape
            /// will maintain constant-sized dimensions. Furthermore, if it is the first dimension
            /// the resulting array view has the same layout as the input.
            ///
            /// # Panics
            ///
            /// Panics if the dimension or the index is out of bounds.
            pub fn into_index_axis<A: Axis>(
                $($mut)? self,
                axis: A,
                index: usize,
            ) -> $name<'a, T, A::Other<S>, Split<A, S, L>> {
                unsafe { Self::index_axis(self.$as_ptr(), self.mapping(), axis, index) }
            }

            /// Converts the array view into a remapped array view.
            ///
            /// # Panics
            ///
            /// Panics if the shape is not matching static rank or constant-sized dimensions,
            /// or if the memory layout is not compatible with the new array layout.
            pub fn into_mapping<R: Shape, K: Layout>($($mut)? self) -> $name<'a, T, R, K> {
                let mapping = Mapping::remap(self.mapping());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a new array view with the dimensions permuted.
            ///
            /// If the permutation is an identity permutation and known at compile time, the
            /// resulting array view has the same layout as the input.
            ///
            /// # Panics
            ///
            /// Panics if the permutation is not valid.
            pub fn into_permuted<I: IntoShape<IntoShape: Permutation>>(
                $($mut)? self,
                perm: I,
            ) -> $name<
                'a,
                T,
                <I::IntoShape as Permutation>::Shape<S>,
                <I::IntoShape as Permutation>::Layout<L>,
            > {
                let mapping = perm.into_dims(|dims| Mapping::permute(self.mapping(), dims));

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a reordered array view.
            pub fn into_reordered(
                $($mut)? self
            ) -> $name<'a, T, S::Reverse, <S::Tail as Shape>::Layout<L>> {
                let mapping = Mapping::reorder(self.mapping());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Converts the array view into a reshaped array view.
            ///
            /// At most one dimension can have dynamic size `usize::MAX`, and is then inferred
            /// from the other dimensions and the array length.
            ///
            /// # Examples
            ///
            /// ```
            /// use mdarray::view;
            ///
            /// let v = view![[1, 2, 3], [4, 5, 6]];
            ///
            /// assert_eq!(v.into_shape([!0, 2]), view![[1, 2], [3, 4], [5, 6]]);
            /// ```
            ///
            /// # Panics
            ///
            /// Panics if the array length is changed, or if the memory layout is not compatible.
            pub fn into_shape<I: IntoShape>(
                $($mut)? self,
                shape: I
            ) -> $name<'a, T, I::IntoShape, L> {
                let mapping = self.mapping().reshape(shape.into_shape());

                unsafe { $name::new_unchecked(self.$as_ptr(), mapping) }
            }

            /// Divides the array view into two at an index along the first dimension.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension,
            /// or if the rank is not at least 1.
            pub fn into_split_at(
                self,
                mid: usize,
            ) -> (
                $name<'a, T, <Const<0> as Axis>::Replace<Dyn, S>, L>,
                $name<'a, T, <Const<0> as Axis>::Replace<Dyn, S>, L>,
            ) {
                self.into_split_axis_at(Const::<0>, mid)
            }

            /// Divides the array view into two at an index along the specified dimension.
            ///
            /// If the dimension to be divided is know at compile time, the resulting array
            /// shape will maintain constant-sized dimensions. Furthermore, if it is the first
            /// dimension the resulting array views have the same layout as the input.
            ///
            /// # Panics
            ///
            /// Panics if the split point is larger than the number of elements in that dimension,
            /// or if the dimension is out of bounds.
            pub fn into_split_axis_at<A: Axis>(
                $($mut)? self,
                axis: A,
                mid: usize,
            ) -> (
                $name<'a, T, A::Replace<Dyn, S>, Split<A, S, L>>,
                $name<'a, T, A::Replace<Dyn, S>, Split<A, S, L>>,
            ) {
                unsafe { Self::split_axis_at(self.$as_ptr(), self.mapping(), axis, mid) }
            }

            /// Creates an array view from a raw pointer and layout.
            ///
            /// # Safety
            ///
            /// The pointer must be non-null and a valid array view for the given layout.
            pub unsafe fn new_unchecked(ptr: *$raw_mut T, mapping: L::Mapping<S>) -> Self {
                let slice = RawSlice::new_unchecked(ptr as *mut T, mapping);

                Self { slice, phantom: PhantomData }
            }

            pub(crate) unsafe fn index_axis<A: Axis>(
                ptr: *$raw_mut T,
                mapping: &L::Mapping<S>,
                axis: A,
                index: usize,
            ) -> $name<'a, T, A::Other<S>, Split<A, S, L>> {
                let size = mapping.dim(axis.index(mapping.rank()));

                if index >= size {
                    index::panic_bounds_check(index, size);
                }

                let new_mapping = axis.remove(mapping);

                // Calculate offset for the new view if non-empty.
                let offset = mapping.stride(axis.index(mapping.rank())) * index as isize;
                let count = if new_mapping.is_empty() { 0 } else { offset };

                unsafe { $name::new_unchecked(ptr.offset(count), new_mapping) }
            }

            pub(crate) unsafe fn split_axis_at<A: Axis>(
                ptr: *$raw_mut T,
                mapping: &L::Mapping<S>,
                axis: A,
                mid: usize,
            ) -> (
                $name<'a, T, A::Replace<Dyn, S>, Split<A, S, L>>,
                $name<'a, T, A::Replace<Dyn, S>, Split<A, S, L>>,
            ) {
                let index = axis.index(mapping.rank());
                let size = mapping.dim(index);

                if mid > size {
                    index::panic_bounds_check(mid, size);
                }

                let first_mapping = axis.resize(mapping, mid);
                let second_mapping = axis.resize(mapping, size - mid);

                // Calculate offset for the second view if non-empty.
                let offset = mapping.stride(index) * mid as isize;
                let count = if second_mapping.is_empty() { 0 } else { offset };

                unsafe {
                    let first = $name::new_unchecked(ptr, first_mapping);
                    let second = $name::new_unchecked(ptr.offset(count), second_mapping);

                    (first, second)
                }
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
            Slice<T, S, L>: AsRef<U>,
        {
            fn as_ref(&self) -> &U {
                (**self).as_ref()
            }
        }

        impl<T, S: Shape, L: Layout> Borrow<Slice<T, S, L>> for $name<'_, T, S, L> {
            fn borrow(&self) -> &Slice<T, S, L> {
                self
            }
        }

        impl<T: Debug, S: Shape, L: Layout> Debug for $name<'_, T, S, L> {
            fn fmt(&self, f: &mut Formatter) -> fmt::Result {
                (**self).fmt(f)
            }
        }

        impl<T, S: Shape, L: Layout> Deref for $name<'_, T, S, L> {
            type Target = Slice<T, S, L>;

            fn deref(&self) -> &Self::Target {
                self.slice.as_slice()
            }
        }

        impl<'a, T, S: Shape, L: Layout> Expression for $name<'a, T, S, L> {
            type Shape = S;

            const IS_REPEATABLE: bool = $repeatable;

            fn shape(&self) -> &S {
                (**self).shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> &'a $($mut)? T {
                let count = self.slice.mapping().inner_stride() * index as isize;

                &$($mut)? *self.slice.$as_ptr().offset(count)
            }

            fn inner_rank(&self) -> usize {
                if L::IS_DENSE {
                    // For static rank 0, the inner stride is 0 so we allow inner rank >0.
                    if S::RANK == Some(0) { usize::MAX } else { self.rank() }
                } else {
                    // For rank 0, the inner stride is always 0 so we can allow inner rank >0.
                    if self.rank() > 0 { 1 } else { usize::MAX }
                }
            }

            unsafe fn reset_dim(&mut self, index: usize, count: usize) {
                let count = -self.stride(index) * count as isize;
                let ptr = self.slice.as_mut_ptr().offset(count);

                self.slice.set_ptr(ptr);
            }

            unsafe fn step_dim(&mut self, index: usize) {
                let ptr = self.slice.as_mut_ptr().offset(self.stride(index));

                self.slice.set_ptr(ptr);
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

        impl<'a, T> From<&'a $($mut)? [T]> for $name<'a, T, (Dyn,)> {
            fn from(value: &'a $($mut)? [T]) -> Self {
                let mapping = DenseMapping::new((value.len(),));

                unsafe { Self::new_unchecked(value.$as_ptr(), mapping) }
            }
        }

        impl<'a, T, D: Dim> From<$name<'a, T, (D,)>> for &'a $($mut)? [T] {
            fn from($($mut)? value: $name<T, (D,)>) -> Self {
                unsafe { slice::$from_raw_parts(value.$as_ptr(), value.len()) }
            }
        }

        impl<T: Hash, S: Shape, L: Layout> Hash for $name<'_, T, S, L> {
            fn hash<H: Hasher>(&self, state: &mut H) {
                (**self).hash(state)
            }
        }

        impl<T, S: Shape, L: Layout, I: SliceIndex<T, S, L>> Index<I> for $name<'_, T, S, L> {
            type Output = I::Output;

            fn index(&self, index: I) -> &I::Output {
                index.index(self)
            }
        }

        impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a $name<'_, T, S, L> {
            type Shape = S;
            type IntoExpr = View<'a, T, S, L>;

            fn into_expr(self) -> Self::IntoExpr {
                self.expr()
            }
        }

        impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a $name<'_, T, S, L> {
            type Item = &'a T;
            type IntoIter = Iter<View<'a, T, S, L>>;

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

impl_view!(View, as_ptr, from_raw_parts, const, {}, true);
impl_view!(ViewMut, as_mut_ptr, from_raw_parts_mut, mut, {mut}, false);

macro_rules! impl_into_view {
    ($n:tt, ($($xyz:tt),+), ($($abc:tt),+), ($($idx:tt),+)) => {
        impl<'a, T, $($xyz: Dim,)+ L: Layout> View<'a, T, ($($xyz,)+), L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($abc: DimIndex),+>(
                self,
                $($idx: $abc),+
            ) -> View<
                'a,
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz,)+)>,
                <($($abc,)+) as ViewIndex>::Layout<L>,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { View::new_unchecked(self.as_ptr().offset(count), mapping) }
            }
        }

        impl<'a, T, $($xyz: Dim,)+ L: Layout> ViewMut<'a, T, ($($xyz,)+), L> {
            /// Converts the array view into a new array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn into_view<$($abc: DimIndex),+>(
                mut self,
                $($idx: $abc),+
            ) -> ViewMut<
                'a,
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz,)+)>,
                <($($abc,)+) as ViewIndex>::Layout<L>,
            > {
                let (offset, mapping) = ($($idx,)+).view_index(self.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { ViewMut::new_unchecked(self.as_mut_ptr().offset(count), mapping) }
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

impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a mut ViewMut<'_, T, S, L> {
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

impl<T, U: ?Sized, S: Shape, L: Layout> AsMut<U> for ViewMut<'_, T, S, L>
where
    Slice<T, S, L>: AsMut<U>,
{
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, S: Shape, L: Layout> BorrowMut<Slice<T, S, L>> for ViewMut<'_, T, S, L> {
    fn borrow_mut(&mut self) -> &mut Slice<T, S, L> {
        self
    }
}

impl<T, S: Shape, L: Layout> Clone for View<'_, T, S, L> {
    fn clone(&self) -> Self {
        Self { slice: self.slice.clone(), phantom: PhantomData }
    }

    fn clone_from(&mut self, source: &Self) {
        self.slice.clone_from(&source.slice);
    }
}

impl<T, S: Shape, L: Layout<Mapping<S>: Copy>> Copy for View<'_, T, S, L> {}

impl<T, S: Shape, L: Layout> DerefMut for ViewMut<'_, T, S, L> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.as_mut_slice()
    }
}

macro_rules! impl_from_array_ref {
    (($($xyz:tt),+), ($($abc:tt),+), $array:tt) => {
        impl<'a, T $(,$xyz: Dim + From<Const<$abc>>)+ $(,const $abc: usize)+> From<&'a $array>
            for View<'a, T, ($($xyz,)+)>
        {
            fn from(value: &'a $array) -> Self {
                let mapping = DenseMapping::new(($($xyz::from(Const::<$abc>),)+));

                _ = mapping.shape().checked_len().expect("invalid length");

                unsafe { Self::new_unchecked(value.as_ptr().cast(), mapping) }
            }
        }

        impl<'a, T $(,$xyz: Dim + From<Const<$abc>>)+ $(,const $abc: usize)+> From<&'a mut $array>
            for ViewMut<'a, T, ($($xyz,)+)>
        {
            fn from(value: &'a mut $array) -> Self {
                let mapping = DenseMapping::new(($($xyz::from(Const::<$abc>),)+));

                _ = mapping.shape().checked_len().expect("invalid length");

                unsafe { Self::new_unchecked(value.as_mut_ptr().cast(), mapping) }
            }
        }
    };
}

impl_from_array_ref!((X), (A), [T; A]);
impl_from_array_ref!((X, Y), (A, B), [[T; B]; A]);
impl_from_array_ref!((X, Y, Z), (A, B, C), [[[T; C]; B]; A]);
impl_from_array_ref!((X, Y, Z, W), (A, B, C, D), [[[[T; D]; C]; B]; A]);
impl_from_array_ref!((X, Y, Z, W, U), (A, B, C, D, E), [[[[[T; E]; D]; C]; B]; A]);
impl_from_array_ref!((X, Y, Z, W, U, V), (A, B, C, D, E, F), [[[[[[T; F]; E]; D]; C]; B]; A]);

impl<T, S: Shape, L: Layout, I: SliceIndex<T, S, L>> IndexMut<I> for ViewMut<'_, T, S, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a mut ViewMut<'_, T, S, L> {
    type Shape = S;
    type IntoExpr = ViewMut<'a, T, S, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a mut ViewMut<'_, T, S, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ViewMut<'a, T, S, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.expr_mut().into_iter()
    }
}

unsafe impl<T: Sync, S: Shape, L: Layout> Send for View<'_, T, S, L> {}
unsafe impl<T: Sync, S: Shape, L: Layout> Sync for View<'_, T, S, L> {}

unsafe impl<T: Send, S: Shape, L: Layout> Send for ViewMut<'_, T, S, L> {}
unsafe impl<T: Sync, S: Shape, L: Layout> Sync for ViewMut<'_, T, S, L> {}

macro_rules! impl_try_from_array_ref {
    (($($xyz:tt),+), ($($abc:tt),+), $array:tt) => {
        impl<'a, T $(,$xyz: Dim)+ $(,const $abc: usize)+> TryFrom<View<'a, T, ($($xyz,)+)>>
            for &'a $array
        {
            type Error = View<'a, T, ($($xyz,)+)>;

            fn try_from(value: View<'a, T, ($($xyz,)+)>) -> Result<Self, Self::Error> {
                if value.shape().with_dims(|dims| dims == &[$($abc),+]) {
                    Ok(unsafe { &*value.as_ptr().cast() })
                } else {
                    Err(value)
                }
            }
        }

        impl<'a, T $(,$xyz: Dim)+ $(,const $abc: usize)+> TryFrom<ViewMut<'a, T, ($($xyz,)+)>>
            for &'a mut $array
        {
            type Error = ViewMut<'a, T, ($($xyz,)+)>;

            fn try_from(mut value: ViewMut<'a, T, ($($xyz,)+)>) -> Result<Self, Self::Error> {
                if value.shape().with_dims(|dims| dims == &[$($abc),+]) {
                    Ok(unsafe { &mut *value.as_mut_ptr().cast() })
                } else {
                    Err(value)
                }
            }
        }
    };
}

impl_try_from_array_ref!((X), (A), [T; A]);
impl_try_from_array_ref!((X, Y), (A, B), [[T; B]; A]);
impl_try_from_array_ref!((X, Y, Z), (A, B, C), [[[T; C]; B]; A]);
impl_try_from_array_ref!((X, Y, Z, W), (A, B, C, D), [[[[T; D]; C]; B]; A]);
impl_try_from_array_ref!((X, Y, Z, W, U), (A, B, C, D, E), [[[[[T; E]; D]; C]; B]; A]);
impl_try_from_array_ref!((X, Y, Z, W, U, V), (A, B, C, D, E, F), [[[[[[T; F]; E]; D]; C]; B]; A]);
