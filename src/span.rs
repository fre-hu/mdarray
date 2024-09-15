#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
#[cfg(feature = "nightly")]
use std::marker::PhantomData;
#[cfg(not(feature = "nightly"))]
use std::marker::{PhantomData, PhantomPinned};
use std::mem;
use std::ops::{Index, IndexMut};

use crate::dim::{Const, Dim, Dyn};
use crate::expr::{AxisExpr, AxisExprMut, Expr, ExprMut, Lanes, LanesMut, Map, Zip};
use crate::expression::Expression;
use crate::grid::Grid;
use crate::index::{Axis, DimIndex, Inner, Outer, Permutation, SpanIndex, ViewIndex};
use crate::iter::Iter;
use crate::layout::{Dense, Flat, Layout};
use crate::mapping::Mapping;
use crate::raw_span::RawSpan;
use crate::shape::{IntoShape, Rank, Shape};
#[cfg(not(feature = "nightly"))]
use crate::traits::{Apply, FromExpression, IntoCloned, IntoExpression};
#[cfg(feature = "nightly")]
use crate::traits::{Apply, IntoCloned, IntoExpression};

/// Multidimensional array span.
pub struct Span<T, S: Shape, L: Layout = Dense> {
    phantom: PhantomData<(T, S, L)>,
    #[cfg(not(feature = "nightly"))]
    _pinned: PhantomPinned,
    #[cfg(feature = "nightly")]
    _opaque: Opaque,
}

/// Multidimensional array span with dynamically-sized dimensions.
pub type DSpan<T, const N: usize, L = Dense> = Span<T, Rank<N>, L>;

#[cfg(feature = "nightly")]
extern "C" {
    type Opaque;
}

impl<T, S: Shape, L: Layout> Span<T, S, L> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if mem::size_of::<S>() > 0 {
            RawSpan::from_mut_span(self).as_mut_ptr()
        } else {
            self as *mut Self as *mut T
        }
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        if mem::size_of::<S>() > 0 {
            RawSpan::from_span(self).as_ptr()
        } else {
            self as *const Self as *const T
        }
    }

    /// Assigns an expression to the array span with broadcasting, cloning elements if needed.
    ///
    /// # Panics
    ///
    /// Panics if the expression cannot be broadcast to the shape of the array span.
    pub fn assign<I: IntoExpression<Item: IntoCloned<T>>>(&mut self, expr: I) {
        self.expr_mut().zip(expr).for_each(|(x, y)| y.clone_to(x));
    }

    /// Returns an expression that gives array views iterating over the specified dimension.
    ///
    /// When iterating over the outermost dimension, both the unit inner stride and the
    /// uniform stride properties are maintained, and the resulting array views have
    /// the same layout.
    ///
    /// When iterating over the innermost dimension, the uniform stride property is
    /// maintained but not unit inner stride, and the resulting array views have
    /// flat or strided layout.
    ///
    /// When iterating over the other dimensions, the unit inner stride propery is
    /// maintained but not uniform stride, and the resulting array views have general
    /// or strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn axis_expr<const N: usize>(&self) -> AxisExpr<T, S, L, Inner<N>>
    where
        Inner<N>: Axis,
    {
        AxisExpr::new(self)
    }

    /// Returns a mutable expression that gives array views iterating over the specified dimension.
    ///
    /// When iterating over the outermost dimension, both the unit inner stride and the
    /// uniform stride properties are maintained, and the resulting array views have
    /// the same layout.
    ///
    /// When iterating over the innermost dimension, the uniform stride property is
    /// maintained but not unit inner stride, and the resulting array views have
    /// flat or strided layout.
    ///
    /// When iterating over the other dimensions, the unit inner stride propery is
    /// maintained but not uniform stride, and the resulting array views have general
    /// or strided layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn axis_expr_mut<const N: usize>(&mut self) -> AxisExprMut<T, S, L, Inner<N>>
    where
        Inner<N>: Axis,
    {
        AxisExprMut::new(self)
    }

    /// Returns an expression that gives column views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn cols(&self) -> Lanes<T, S, L, Inner<0>> {
        Lanes::new(self)
    }

    /// Returns a mutable expression that gives column views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn cols_mut(&mut self) -> LanesMut<T, S, L, Inner<0>> {
        LanesMut::new(self)
    }

    /// Returns `true` if the array span contains an element with the given value.
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        contains(self, x)
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn dim(&self, index: usize) -> usize {
        self.mapping().dim(index)
    }

    /// Returns the number of elements in each dimension.
    pub fn dims(&self) -> S::Dims {
        self.mapping().dims()
    }

    /// Returns an expression over the array span.
    pub fn expr(&self) -> Expr<T, S, L> {
        unsafe { Expr::new_unchecked(self.as_ptr(), self.mapping()) }
    }

    /// Returns a mutable expression over the array span.
    pub fn expr_mut(&mut self) -> ExprMut<T, S, L> {
        unsafe { ExprMut::new_unchecked(self.as_mut_ptr(), self.mapping()) }
    }

    /// Fills the array span with elements by cloning `value`.
    pub fn fill(&mut self, value: T)
    where
        T: Clone,
    {
        self.expr_mut().for_each(|x| x.clone_from(&value));
    }

    /// Fills the array span with elements returned by calling a closure repeatedly.
    pub fn fill_with<F: FnMut() -> T>(&mut self, mut f: F) {
        self.expr_mut().for_each(|x| *x = f());
    }

    /// Returns a one-dimensional array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten(&self) -> Expr<T, Dyn, L::Uniform> {
        self.expr().into_flattened()
    }

    /// Returns a mutable one-dimensional array view over the array span.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten_mut(&mut self) -> ExprMut<T, Dyn, L::Uniform> {
        self.expr_mut().into_flattened()
    }

    /// Returns a reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked<I: SpanIndex<T, S, L>>(&self, index: I) -> &I::Output {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked_mut<I: SpanIndex<T, S, L>>(&mut self, index: I) -> &mut I::Output {
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

    /// Returns `true` if the array strides are consistent with uniformly strided memory layout.
    pub fn is_uniformly_strided(&self) -> bool {
        self.mapping().is_uniformly_strided()
    }

    /// Returns an iterator over the array span.
    pub fn iter(&self) -> Iter<Expr<'_, T, S, L>> {
        self.expr().into_iter()
    }

    /// Returns a mutable iterator over the array span.
    pub fn iter_mut(&mut self) -> Iter<ExprMut<'_, T, S, L>> {
        self.expr_mut().into_iter()
    }

    /// Returns an expression that gives array views over the specified dimension,
    /// iterating over the other dimensions.
    ///
    /// If the innermost dimension is specified, the resulting array views have dense or
    /// flat layout. For other dimensions, the resulting array views have flat layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn lanes<const N: usize>(&self) -> Lanes<T, S, L, Inner<N>>
    where
        Inner<N>: Axis,
    {
        Lanes::new(self)
    }

    /// Returns a mutable expression that gives array views over the specified dimension,
    /// iterating over the other dimensions.
    ///
    /// If the innermost dimension is specified, the resulting array views have dense or
    /// flat layout. For other dimensions, the resulting array views have flat layout.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn lanes_mut<const N: usize>(&mut self) -> LanesMut<T, S, L, Inner<N>>
    where
        Inner<N>: Axis,
    {
        LanesMut::new(self)
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.mapping().len()
    }

    /// Returns the array layout mapping.
    pub fn mapping(&self) -> L::Mapping<S> {
        if mem::size_of::<S>() > 0 {
            RawSpan::from_span(self).mapping()
        } else {
            L::Mapping::default()
        }
    }

    /// Returns an expression that gives array views iterating over the outermost dimension.
    ///
    /// Iterating over the outermost dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_expr(&self) -> AxisExpr<T, S, L, Outer> {
        AxisExpr::new(self)
    }

    /// Returns a mutable expression that gives array views iterating over the outermost dimension.
    ///
    /// Iterating over the outermost dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_expr_mut(&mut self) -> AxisExprMut<T, S, L, Outer> {
        AxisExprMut::new(self)
    }

    /// Returns the array rank, i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        S::RANK
    }

    /// Returns a remapped array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap<M: Layout>(&self) -> Expr<T, S, M> {
        self.expr().into_mapping()
    }

    /// Returns a mutable remapped array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap_mut<M: Layout>(&mut self) -> ExprMut<T, S, M> {
        self.expr_mut().into_mapping()
    }

    /// Returns a reshaped array view of the array span, with similar layout.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape<I: IntoShape>(
        &self,
        shape: I,
    ) -> Expr<T, I::IntoShape, <I::IntoShape as Shape>::Layout<L::Uniform, L>> {
        self.expr().into_shape(shape)
    }

    /// Returns a mutable reshaped array view of the array span, with similar layout.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape_mut<I: IntoShape>(
        &mut self,
        shape: I,
    ) -> ExprMut<T, I::IntoShape, <I::IntoShape as Shape>::Layout<L::Uniform, L>> {
        self.expr_mut().into_shape(shape)
    }

    /// Returns an expression that gives row views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 2.
    pub fn rows(&self) -> Lanes<T, S, L, Inner<1>> {
        Lanes::new(self)
    }

    /// Returns a mutable expression that gives row views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 2.
    pub fn rows_mut(&mut self) -> LanesMut<T, S, L, Inner<1>> {
        LanesMut::new(self)
    }

    /// Returns the array shape.
    pub fn shape(&self) -> S {
        self.mapping().shape()
    }

    /// Divides an array span into two at an index along the outermost dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the rank is not at least 1.
    pub fn split_at(
        &self,
        mid: usize,
    ) -> (Expr<T, <Outer as Axis>::Replace<Dyn, S>, L>, Expr<T, <Outer as Axis>::Replace<Dyn, S>, L>)
    {
        self.expr().into_split_at(mid)
    }

    /// Divides a mutable array span into two at an index along the outermost dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the rank is not at least 1.
    pub fn split_at_mut(
        &mut self,
        mid: usize,
    ) -> (
        ExprMut<T, <Outer as Axis>::Replace<Dyn, S>, L>,
        ExprMut<T, <Outer as Axis>::Replace<Dyn, S>, L>,
    ) {
        self.expr_mut().into_split_at(mid)
    }

    /// Divides an array span into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the dimension is out of bounds.
    pub fn split_axis_at<const N: usize>(
        &self,
        mid: usize,
    ) -> (
        Expr<T, <Inner<N> as Axis>::Replace<Dyn, S>, <Inner<N> as Axis>::Resize<S, L>>,
        Expr<T, <Inner<N> as Axis>::Replace<Dyn, S>, <Inner<N> as Axis>::Resize<S, L>>,
    )
    where
        Inner<N>: Axis,
    {
        self.expr().into_split_axis_at(mid)
    }

    /// Divides a mutable array span into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension,
    /// or if the dimension is out of bounds.
    pub fn split_axis_at_mut<const N: usize>(
        &mut self,
        mid: usize,
    ) -> (
        ExprMut<T, <Inner<N> as Axis>::Replace<Dyn, S>, <Inner<N> as Axis>::Resize<S, L>>,
        ExprMut<T, <Inner<N> as Axis>::Replace<Dyn, S>, <Inner<N> as Axis>::Resize<S, L>>,
    )
    where
        Inner<N>: Axis,
    {
        self.expr_mut().into_split_axis_at(mid)
    }

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn stride(&self, index: usize) -> isize {
        self.mapping().stride(index)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> S::Strides {
        self.mapping().strides()
    }

    /// Copies the array span into a new array.
    #[cfg(not(feature = "nightly"))]
    pub fn to_grid(&self) -> Grid<T, S>
    where
        T: Clone,
    {
        Grid::from_expr(self.expr().cloned())
    }

    /// Copies the array span into a new array.
    #[cfg(feature = "nightly")]
    pub fn to_grid(&self) -> Grid<T, S>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> Grid<T, S, A>
    where
        T: Clone,
    {
        self.expr().cloned().eval_in(alloc)
    }

    /// Copies the array span into a new vector.
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.to_grid().into_vec()
    }

    /// Copies the array span into a new vector with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_vec_in<A: Allocator>(&self, alloc: A) -> Vec<T, A>
    where
        T: Clone,
    {
        self.to_grid_in(alloc).into_vec()
    }
}

impl<T, S: Shape> Span<T, S> {
    /// Returns a mutable slice of all elements in the array, which must have dense layout.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.expr_mut().into_slice()
    }

    /// Returns a slice of all elements in the array, which must have dense layout.
    pub fn as_slice(&self) -> &[T] {
        self.expr().into_slice()
    }
}

impl<T, X: Dim, Y: Dim, L: Layout> Span<T, (X, Y), L> {
    /// Returns an array view for the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn col(&self, index: usize) -> Expr<T, X, L::Uniform> {
        self.view(.., index)
    }

    /// Returns a mutable array view for the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn col_mut(&mut self, index: usize) -> ExprMut<T, X, L::Uniform> {
        self.view_mut(.., index)
    }

    /// Returns an array view for the given diagonal of the array span,
    /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the absolute index is larger than the number of columns or rows.
    pub fn diag(&self, index: isize) -> Expr<T, Dyn, Flat> {
        self.expr().into_diag(index)
    }

    /// Returns a mutable array view for the given diagonal of the array span,
    /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the absolute index is larger than the number of columns or rows.
    pub fn diag_mut(&mut self, index: isize) -> ExprMut<T, Dyn, Flat> {
        self.expr_mut().into_diag(index)
    }

    /// Returns an array view for the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn row(&self, index: usize) -> Expr<T, Y, Flat> {
        self.view(index, ..)
    }

    /// Returns a mutable array view for the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn row_mut(&mut self, index: usize) -> ExprMut<T, Y, Flat> {
        self.view_mut(index, ..)
    }
}

macro_rules! impl_permute {
    (($($xyz:tt),+), ($($abc:tt),*)) => {
        #[allow(unused_parens)]
        impl<T, $($xyz: Dim,)+ L: Layout> Span<T, ($($xyz),+), L> {
            /// Returns an array view with the dimensions permuted.
            pub fn permute<$(const $abc: usize),+>(
                &self
            ) -> Expr<
                T,
                <($(Const<$abc>,)+) as Permutation>::Shape<($($xyz),+)>,
                <($(Const<$abc>,)+) as Permutation>::Layout<L>,
            >
            where
                ($(Const<$abc>,)+): Permutation
            {
                self.expr().into_permuted()
            }

            /// Returns a mutable array view with the dimensions permuted.
            pub fn permute_mut<$(const $abc: usize),+>(
                &mut self
            ) -> ExprMut<
                T,
                <($(Const<$abc>,)+) as Permutation>::Shape<($($xyz),+)>,
                <($(Const<$abc>,)+) as Permutation>::Layout<L>,
            >
            where
                ($(Const<$abc>,)+): Permutation
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
        #[allow(unused_parens)]
        impl<T, $($xyz: Dim,)+ L: Layout> Span<T, ($($xyz),+), L> {
            /// Copies the specified subarray into a new array.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn grid<$($abc: DimIndex),+>(
                &self,
                $($idx: $abc),+
            ) -> Grid<T, <($($abc,)+) as ViewIndex>::Shape<($($xyz),+)>>
            where
                T: Clone,
            {
                self.view($($idx),+).to_grid()
            }

            /// Returns an array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn view<$($abc: DimIndex),+>(
                &self,
                $($idx: $abc),+
            ) -> Expr<
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz),+)>,
                <($($abc,)+) as ViewIndex>::Layout<($($xyz),+), L>,
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
            ) -> ExprMut<
                T,
                <($($abc,)+) as ViewIndex>::Shape<($($xyz),+)>,
                <($($abc,)+) as ViewIndex>::Layout<($($xyz),+), L>,
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

impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a Span<T, S, L> {
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

impl<'a, T, U, S: Shape, L: Layout> Apply<U> for &'a mut Span<T, S, L> {
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

impl<T, S: Shape, L: Layout> AsMut<Span<T, S, L>> for Span<T, S, L> {
    fn as_mut(&mut self) -> &mut Span<T, S, L> {
        self
    }
}

impl<T, S: Shape> AsMut<[T]> for Span<T, S> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

impl<T, S: Shape, L: Layout> AsRef<Span<T, S, L>> for Span<T, S, L> {
    fn as_ref(&self) -> &Span<T, S, L> {
        self
    }
}

impl<T, S: Shape> AsRef<[T]> for Span<T, S> {
    fn as_ref(&self) -> &[T] {
        &self[..]
    }
}

impl<T: Debug, S: Shape, L: Layout> Debug for Span<T, S, L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if S::RANK == 0 {
            self[S::Dims::default()].fmt(f)
        } else {
            let mut list = f.debug_list();

            // Empty arrays should give an empty list.
            if !self.is_empty() {
                _ = list.entries(self.outer_expr());
            }

            list.finish()
        }
    }
}

impl<T: Hash, S: Shape, L: Layout> Hash for Span<T, S, L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..S::RANK {
            #[cfg(not(feature = "nightly"))]
            state.write_usize(self.dim(i));
            #[cfg(feature = "nightly")]
            state.write_length_prefix(self.dim(i));
        }

        self.expr().for_each(|x| x.hash(state));
    }
}

impl<T, S: Shape, L: Layout, I: SpanIndex<T, S, L>> Index<I> for Span<T, S, L> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: Shape, L: Layout, I: SpanIndex<T, S, L>> IndexMut<I> for Span<T, S, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a Span<T, S, L> {
    type Shape = S;
    type IntoExpr = Expr<'a, T, S, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoExpression for &'a mut Span<T, S, L> {
    type Shape = S;
    type IntoExpr = ExprMut<'a, T, S, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a Span<T, S, L> {
    type Item = &'a T;
    type IntoIter = Iter<Expr<'a, T, S, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: Shape, L: Layout> IntoIterator for &'a mut Span<T, S, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, S, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

#[cfg(feature = "nightly")]
unsafe impl<T: Send, S: Shape, L: Layout> Send for Span<T, S, L> {}
#[cfg(feature = "nightly")]
unsafe impl<T: Sync, S: Shape, L: Layout> Sync for Span<T, S, L> {}

impl<T: Clone, S: Shape> ToOwned for Span<T, S> {
    type Owned = Grid<T, S>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }

    fn clone_into(&self, target: &mut Self::Owned) {
        target.clone_from_span(self);
    }
}

fn contains<T: PartialEq, S: Shape, L: Layout>(this: &Span<T, S, L>, value: &T) -> bool {
    if L::IS_UNIFORM {
        if L::IS_UNIT_STRIDED {
            this.remap()[..].contains(value)
        } else {
            this.iter().any(|x| x == value)
        }
    } else {
        this.outer_expr().into_iter().any(|x| x.contains(value))
    }
}
