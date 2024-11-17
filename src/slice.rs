#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::marker::PhantomData;
use std::mem;
use std::ops::{Index, IndexMut};
use std::ptr::NonNull;

use crate::dim::{Dim, Dyn};
#[cfg(not(feature = "nightly"))]
use crate::expr::{Apply, Expression, FromExpression, IntoExpression};
#[cfg(feature = "nightly")]
use crate::expr::{Apply, Expression, IntoExpression};
use crate::expr::{AxisExpr, AxisExprMut, Iter, Lanes, LanesMut, Map, Zip};
use crate::index::{Axis, DimIndex, Nth, Permutation, Resize, SliceIndex, Split, ViewIndex};
use crate::layout::{Dense, Layout, Strided};
use crate::mapping::Mapping;
use crate::raw_slice::RawSlice;
use crate::shape::{DynRank, IntoShape, Rank, Shape};
use crate::tensor::Tensor;
use crate::traits::IntoCloned;
use crate::view::{View, ViewMut};

/// Multidimensional array slice.
pub struct Slice<T, S: Shape = DynRank, L: Layout = Dense> {
    phantom: PhantomData<(T, S, L)>,
}

/// Multidimensional array slice with dynamically-sized dimensions.
pub type DSlice<T, const N: usize, L = Dense> = Slice<T, Rank<N>, L>;

impl<T, S: Shape, L: Layout> Slice<T, S, L> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if mem::size_of::<S>() > 0 {
            RawSlice::from_mut_slice(self).as_mut_ptr()
        } else {
            self as *mut Self as *mut T
        }
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        if mem::size_of::<S>() > 0 {
            RawSlice::from_slice(self).as_ptr()
        } else {
            self as *const Self as *const T
        }
    }

    /// Assigns an expression to the array slice with broadcasting, cloning elements if needed.
    ///
    /// # Panics
    ///
    /// Panics if the expression cannot be broadcast to the shape of the array slice.
    pub fn assign<I: IntoExpression<Item: IntoCloned<T>>>(&mut self, expr: I) {
        self.expr_mut().zip(expr).for_each(|(x, y)| y.clone_to(x));
    }

    /// Returns an expression that gives array views iterating over the specified dimension.
    ///
    /// When iterating over the first dimension, the resulting array views have the same
    /// layout as the input. Otherwise, the resulting array views have strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn axis_expr<const N: usize>(&self) -> AxisExpr<T, S, L, Nth<N>>
    where
        Nth<N>: Axis,
    {
        AxisExpr::new(self)
    }

    /// Returns a mutable expression that gives array views iterating over the specified dimension.
    ///
    /// When iterating over the first dimension, the resulting array views have the same
    /// layout as the input. Otherwise, the resulting array views have strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn axis_expr_mut<const N: usize>(&mut self) -> AxisExprMut<T, S, L, Nth<N>>
    where
        Nth<N>: Axis,
    {
        AxisExprMut::new(self)
    }

    /// Returns an array view for the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2, or if the index is out of bounds.
    pub fn col(&self, index: usize) -> View<T, (S::Head,), Strided> {
        let shape = self.shape().with_dims(<(S::Head, <S::Tail as Shape>::Head)>::from_dims);

        self.reshape(shape).into_view(.., index)
    }

    /// Returns a mutable array view for the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2, or if the index is out of bounds.
    pub fn col_mut(&mut self, index: usize) -> ViewMut<T, (S::Head,), Strided> {
        let shape = self.shape().with_dims(<(S::Head, <S::Tail as Shape>::Head)>::from_dims);

        self.reshape_mut(shape).into_view(.., index)
    }

    /// Returns an expression that gives column views iterating over the other dimension.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2.
    pub fn cols(&self) -> Lanes<T, S, L, Nth<0>> {
        assert!(self.rank() == 2, "invalid rank");

        Lanes::new(self)
    }

    /// Returns a mutable expression that gives column views iterating over the other dimension.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2.
    pub fn cols_mut(&mut self) -> LanesMut<T, S, L, Nth<0>> {
        assert!(self.rank() == 2, "invalid rank");

        LanesMut::new(self)
    }

    /// Returns `true` if the array slice contains an element with the given value.
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        contains(self, x)
    }

    /// Returns an array view for the given diagonal of the array slice,
    /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2, or if the absolute index is larger
    /// than the number of columns or rows.
    pub fn diag(&self, index: isize) -> View<T, (Dyn,), Strided> {
        let shape = self.shape().with_dims(<(S::Head, <S::Tail as Shape>::Head)>::from_dims);

        self.reshape(shape).into_diag(index)
    }

    /// Returns a mutable array view for the given diagonal of the array slice,
    /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2, or if the absolute index is larger
    /// than the number of columns or rows.
    pub fn diag_mut(&mut self, index: isize) -> ViewMut<T, (Dyn,), Strided> {
        let shape = self.shape().with_dims(<(S::Head, <S::Tail as Shape>::Head)>::from_dims);

        self.reshape_mut(shape).into_diag(index)
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn dim(&self, index: usize) -> usize {
        self.mapping().dim(index)
    }

    /// Returns an expression over the array slice.
    pub fn expr(&self) -> View<T, S, L> {
        unsafe { View::new_unchecked(self.as_ptr(), self.mapping().clone()) }
    }

    /// Returns a mutable expression over the array slice.
    pub fn expr_mut(&mut self) -> ViewMut<T, S, L> {
        unsafe { ViewMut::new_unchecked(self.as_mut_ptr(), self.mapping().clone()) }
    }

    /// Fills the array slice with elements by cloning `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.expr_mut().for_each(|x| x.clone_from(&value));
    }

    /// Fills the array slice with elements returned by calling a closure repeatedly.
    pub fn fill_with<F: FnMut() -> T>(&mut self, mut f: F) {
        self.expr_mut().for_each(|x| *x = f());
    }

    /// Returns a one-dimensional array view of the array slice.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten(&self) -> View<T, (Dyn,), L> {
        self.reshape([self.len()])
    }

    /// Returns a mutable one-dimensional array view over the array slice.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten_mut(&mut self) -> ViewMut<T, (Dyn,), L> {
        self.reshape_mut([self.len()])
    }

    /// Returns a reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array slice.
    pub unsafe fn get_unchecked<I: SliceIndex<T, S, L>>(&self, index: I) -> &I::Output {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array slice.
    pub unsafe fn get_unchecked_mut<I: SliceIndex<T, S, L>>(&mut self, index: I) -> &mut I::Output {
        index.get_unchecked_mut(self)
    }

    /// Returns `true` if the array strides are consistent with contiguous memory layout.
    pub fn is_contiguous(&self) -> bool {
        self.mapping().is_contiguous()
    }

    /// Returns `true` if the array contains no elements.
    pub fn is_empty(&self) -> bool {
        self.mapping().is_empty()
    }

    /// Returns an iterator over the array slice.
    pub fn iter(&self) -> Iter<View<'_, T, S, L>> {
        self.expr().into_iter()
    }

    /// Returns a mutable iterator over the array slice.
    pub fn iter_mut(&mut self) -> Iter<ViewMut<'_, T, S, L>> {
        self.expr_mut().into_iter()
    }

    /// Returns an expression that gives array views over the specified dimension,
    /// iterating over the other dimensions.
    ///
    /// If the last dimension is specified, the resulting array views have the same layout
    /// as the input. For other dimensions, the resulting array views have strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn lanes<const N: usize>(&self) -> Lanes<T, S, L, Nth<N>>
    where
        Nth<N>: Axis,
    {
        Lanes::new(self)
    }

    /// Returns a mutable expression that gives array views over the specified dimension,
    /// iterating over the other dimensions.
    ///
    /// If the last dimension is specified, the resulting array views have the same layout
    /// as the input. For other dimensions, the resulting array views have strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn lanes_mut<const N: usize>(&mut self) -> LanesMut<T, S, L, Nth<N>>
    where
        Nth<N>: Axis,
    {
        LanesMut::new(self)
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.mapping().len()
    }

    /// Returns the array layout mapping.
    pub fn mapping(&self) -> &L::Mapping<S> {
        if mem::size_of::<S>() > 0 {
            RawSlice::from_slice(self).mapping()
        } else {
            unsafe { &*NonNull::dangling().as_ptr() }
        }
    }

    /// Returns an expression that gives array views iterating over the first dimension.
    ///
    /// Iterating over the first dimension results in array views with the same layout
    /// as the input.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_expr(&self) -> AxisExpr<T, S, L, Nth<0>> {
        AxisExpr::new(self)
    }

    /// Returns a mutable expression that gives array views iterating over the first dimension.
    ///
    /// Iterating over the first dimension results in array views with the same layout
    /// as the input.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_expr_mut(&mut self) -> AxisExprMut<T, S, L, Nth<0>> {
        AxisExprMut::new(self)
    }

    /// Returns the array rank, i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        self.mapping().rank()
    }

    /// Returns a remapped array view of the array slice.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap<M: Layout>(&self) -> View<T, S, M> {
        let mapping = M::Mapping::remap(self.mapping());

        unsafe { View::new_unchecked(self.as_ptr(), mapping) }
    }

    /// Returns a mutable remapped array view of the array slice.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap_mut<M: Layout>(&mut self) -> ViewMut<T, S, M> {
        let mapping = M::Mapping::remap(self.mapping());

        unsafe { ViewMut::new_unchecked(self.as_mut_ptr(), mapping) }
    }

    /// Returns a reordered array view of the array slice.
    pub fn reorder(&self) -> View<T, S::Reverse, <S::Tail as Shape>::Layout<L>> {
        let mapping = Mapping::reorder(self.mapping());

        unsafe { View::new_unchecked(self.as_ptr(), mapping) }
    }

    /// Returns a mutable reordered array view of the array slice.
    pub fn reorder_mut(&mut self) -> ViewMut<T, S::Reverse, <S::Tail as Shape>::Layout<L>> {
        let mapping = Mapping::reorder(self.mapping());

        unsafe { ViewMut::new_unchecked(self.as_mut_ptr(), mapping) }
    }

    /// Returns a reshaped array view of the array slice.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or if the memory layout is not compatible.
    pub fn reshape<I: IntoShape>(&self, shape: I) -> View<T, I::IntoShape, L> {
        let mapping = Mapping::reshape(self.mapping(), shape.into_shape());

        unsafe { View::new_unchecked(self.as_ptr(), mapping) }
    }

    /// Returns a mutable reshaped array view of the array slice.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or if the memory layout is not compatible.
    pub fn reshape_mut<I: IntoShape>(&mut self, shape: I) -> ViewMut<T, I::IntoShape, L> {
        let mapping = Mapping::reshape(self.mapping(), shape.into_shape());

        unsafe { ViewMut::new_unchecked(self.as_mut_ptr(), mapping) }
    }

    /// Returns an array view for the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2, or if the index is out of bounds.
    pub fn row(&self, index: usize) -> View<T, (<S::Tail as Shape>::Head,), L> {
        let shape = self.shape().with_dims(<(S::Head, <S::Tail as Shape>::Head)>::from_dims);

        self.reshape(shape).into_view(index, ..)
    }

    /// Returns a mutable array view for the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2, or if the index is out of bounds.
    pub fn row_mut(&mut self, index: usize) -> ViewMut<T, (<S::Tail as Shape>::Head,), L> {
        let shape = self.shape().with_dims(<(S::Head, <S::Tail as Shape>::Head)>::from_dims);

        self.reshape_mut(shape).into_view(index, ..)
    }

    /// Returns an expression that gives row views iterating over the other dimension.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2.
    pub fn rows(&self) -> Lanes<T, S, L, Nth<1>> {
        assert!(self.rank() == 2, "invalid rank");

        Lanes::new(self)
    }

    /// Returns a mutable expression that gives row views iterating over the other dimension.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not equal to 2.
    pub fn rows_mut(&mut self) -> LanesMut<T, S, L, Nth<1>> {
        assert!(self.rank() == 2, "invalid rank");

        LanesMut::new(self)
    }

    /// Returns the array shape.
    pub fn shape(&self) -> &S {
        self.mapping().shape()
    }

    /// Divides an array slice into two at an index along the first dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the rank is not at least 1.
    pub fn split_at(
        &self,
        mid: usize,
    ) -> (View<T, Resize<Nth<0>, S>, L>, View<T, Resize<Nth<0>, S>, L>) {
        self.split_axis_at::<0>(mid)
    }

    /// Divides a mutable array slice into two at an index along the first dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the rank is not at least 1.
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (ViewMut<T, Resize<Nth<0>, S>, L>, ViewMut<T, Resize<Nth<0>, S>, L>) {
        self.split_axis_at_mut::<0>(mid)
    }

    /// Divides an array slice into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the dimension is out of bounds.
    pub fn split_axis_at<const N: usize>(
        &self,
        mid: usize,
    ) -> (
        View<T, Resize<Nth<N>, S>, Split<Nth<N>, S, L>>,
        View<T, Resize<Nth<N>, S>, Split<Nth<N>, S, L>>,
    )
    where
        Nth<N>: Axis,
    {
        unsafe { View::split_axis_at::<Nth<N>>(self.as_ptr(), self.mapping(), mid) }
    }

    /// Divides a mutable array slice into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the dimension is out of bounds.
    pub fn split_axis_at_mut<const N: usize>(
        &mut self,
        mid: usize,
    ) -> (
        ViewMut<T, Resize<Nth<N>, S>, Split<Nth<N>, S, L>>,
        ViewMut<T, Resize<Nth<N>, S>, Split<Nth<N>, S, L>>,
    )
    where
        Nth<N>: Axis,
    {
        unsafe { ViewMut::split_axis_at::<Nth<N>>(self.as_mut_ptr(), self.mapping(), mid) }
    }

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn stride(&self, index: usize) -> isize {
        self.mapping().stride(index)
    }

    /// Copies the array slice into a new array.
    #[cfg(not(feature = "nightly"))]
    pub fn to_tensor(&self) -> Tensor<T, S>
    where
        T: Clone,
    {
        Tensor::from_expr(self.expr().cloned())
    }

    /// Copies the array slice into a new array.
    #[cfg(feature = "nightly")]
    pub fn to_tensor(&self) -> Tensor<T, S>
    where
        T: Clone,
    {
        self.to_tensor_in(Global)
    }

    /// Copies the array slice into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_tensor_in<A: Allocator>(&self, alloc: A) -> Tensor<T, S, A>
    where
        T: Clone,
    {
        Tensor::from_expr_in(self.expr().cloned(), alloc)
    }

    /// Copies the array slice into a new vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_tensor().into_vec()
    }

    /// Copies the array slice into a new vector with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        self.to_tensor_in(alloc).into_vec()
    }
}

impl<T, L: Layout> Slice<T, DynRank, L> {
    /// Returns the number of elements in each dimension.
    pub fn dims(&self) -> &[usize] {
        self.mapping().dims()
    }
}

impl<T, S: Shape> Slice<T, S, Strided> {
    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> &[isize] {
        self.mapping().strides()
    }
}

macro_rules! impl_permute {
    (($($xyz:tt),+), ($($abc:tt),*)) => {
        impl<T, $($xyz: Dim,)+ L: Layout> Slice<T, ($($xyz,)+), L> {
            /// Returns an array view with the dimensions permuted.
            pub fn permute<$(const $abc: usize),+>(
                &self
            ) -> View<
                T,
                <($(Nth<$abc>,)+) as Permutation>::Shape<($($xyz,)+)>,
                <($(Nth<$abc>,)+) as Permutation>::Layout<L>,
            >
            where
                ($(Nth<$abc>,)+): Permutation
            {
                self.expr().into_permuted()
            }

            /// Returns a mutable array view with the dimensions permuted.
            pub fn permute_mut<$(const $abc: usize),+>(
                &mut self
            ) -> ViewMut<
                T,
                <($(Nth<$abc>,)+) as Permutation>::Shape<($($xyz,)+)>,
                <($(Nth<$abc>,)+) as Permutation>::Layout<L>,
            >
            where
                ($(Nth<$abc>,)+): Permutation
            {
                self.expr_mut().into_permuted()
            }
        }
    };
}

impl_permute!((X), (A));
impl_permute!((X, Y), (A, B));
impl_permute!((X, Y, Z), (A, B, C));
impl_permute!((X, Y, Z, W), (A, B, C, D));
impl_permute!((X, Y, Z, W, U), (A, B, C, D, E));
impl_permute!((X, Y, Z, W, U, V), (A, B, C, D, E, F));

macro_rules! impl_view {
    (($($xyz:tt),+), ($($abc:tt),+), ($($idx:tt),+)) => {
        impl<T, $($xyz: Dim,)+ L: Layout> Slice<T, ($($xyz,)+), L> {
            /// Copies the specified subarray into a new array.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn tensor<$($abc: DimIndex),+>(
                &self,
                $($idx: $abc),+
            ) -> Tensor<T, <($($abc,)+) as ViewIndex>::Shape<($($xyz,)+)>>
            where
                T: Clone,
            {
                self.view($($idx),+).to_tensor()
            }

            /// Returns an array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn view<$($abc: DimIndex),+>(
                &self,
                $($idx: $abc),+
            ) -> View<
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz,)+)>,
                <($($abc,)+) as ViewIndex>::Layout<L>,
            > {
                self.expr().into_view($($idx),+)
            }

            /// Returns a mutable array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn view_mut<$($abc: DimIndex),+>(
                &mut self,
                $($idx: $abc),+,
            ) -> ViewMut<
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz,)+)>,
                <($($abc,)+) as ViewIndex>::Layout<L>,
            > {
                self.expr_mut().into_view($($idx),+)
            }
        }
    };
}

impl_view!((X), (A), (a));
impl_view!((X, Y), (A, B), (a, b));
impl_view!((X, Y, Z), (A, B, C), (a, b, c));
impl_view!((X, Y, Z, W), (A, B, C, D), (a, b, c, d));
impl_view!((X, Y, Z, W, U), (A, B, C, D, E), (a, b, c, d, e));
impl_view!((X, Y, Z, W, U, V), (A, B, C, D, E, F), (a, b, c, d, e, f));

impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a Slice<T, S, L> {
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

impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a mut Slice<T, S, L> {
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

impl<T, S: Shape, L: Layout> AsMut<Slice<T, S, L>> for Slice<T, S, L> {
    fn as_mut(&mut self) -> &mut Slice<T, S, L> {
        self
    }
}

impl<T, S: Shape> AsMut<[T]> for Slice<T, S> {
    fn as_mut(&mut self) -> &mut [T] {
        self.expr_mut().into()
    }
}

impl<T, S: Shape, L: Layout> AsRef<Slice<T, S, L>> for Slice<T, S, L> {
    fn as_ref(&self) -> &Slice<T, S, L> {
        self
    }
}

impl<T, S: Shape> AsRef<[T]> for Slice<T, S> {
    fn as_ref(&self) -> &[T] {
        self.expr().into()
    }
}

impl<T: Debug, S: Shape, L: Layout> Debug for Slice<T, S, L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if self.rank() == 0 {
            self[[]].fmt(f)
        } else {
            f.debug_list().entries(self.outer_expr()).finish()
        }
    }
}

impl<T: Hash, S: Shape, L: Layout> Hash for Slice<T, S, L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..self.rank() {
            #[cfg(not(feature = "nightly"))]
            state.write_usize(self.dim(i));
            #[cfg(feature = "nightly")]
            state.write_length_prefix(self.dim(i));
        }

        self.expr().for_each(|x| x.hash(state));
    }
}

impl<T, S: Shape, L: Layout, I: SliceIndex<T, S, L>> Index<I> for Slice<T, S, L> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: Shape, L: Layout, I: SliceIndex<T, S, L>> IndexMut<I> for Slice<T, S, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a Slice<T, S, L> {
    type Shape = S;
    type IntoExpr = View<'a, T, S, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a mut Slice<T, S, L> {
    type Shape = S;
    type IntoExpr = ViewMut<'a, T, S, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a Slice<T, S, L> {
    type Item = &'a T;
    type IntoIter = Iter<View<'a, T, S, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a mut Slice<T, S, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ViewMut<'a, T, S, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Clone, S: Shape> ToOwned for Slice<T, S> {
    type Owned = Tensor<T, S>;

    fn to_owned(&self) -> Self::Owned {
        self.to_tensor()
    }

    fn clone_into(&self, target: &mut Self::Owned) {
        target.clone_from_slice(self);
    }
}

fn contains<T: PartialEq, S: Shape, L: Layout>(this: &Slice<T, S, L>, value: &T) -> bool {
    if L::IS_DENSE {
        this.remap()[..].contains(value)
    } else if this.rank() < 2 {
        this.iter().any(|x| x == value)
    } else {
        this.outer_expr().into_iter().any(|x| x.contains(value))
    }
}
