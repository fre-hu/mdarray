#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::fmt::{Debug, Formatter, Result};
use std::hash::{Hash, Hasher};
#[cfg(feature = "nightly")]
use std::marker::PhantomData;
#[cfg(not(feature = "nightly"))]
use std::marker::{PhantomData, PhantomPinned};
use std::ops::{Index, IndexMut};

use crate::dim::{Const, Dim, Shape};
use crate::expr::{AxisExpr, AxisExprMut, Expr, ExprMut, Lanes, LanesMut, Map, Zip};
use crate::expression::Expression;
use crate::grid::Grid;
use crate::index::{Axis, DimIndex, Permutation, SpanIndex, ViewIndex};
use crate::iter::Iter;
use crate::layout::{Dense, Flat, Layout, Strided};
use crate::mapping::Mapping;
use crate::raw_span::RawSpan;
use crate::traits::{Apply, IntoCloned, IntoExpression};

/// Multidimensional array span.
pub struct Span<T, D: Dim, L: Layout = Dense> {
    phantom: PhantomData<(T, D, L)>,
    #[cfg(not(feature = "nightly"))]
    _pinned: PhantomPinned,
    #[cfg(feature = "nightly")]
    _opaque: Opaque,
}

/// Multidimensional array span with the specified rank.
pub type DSpan<T, const N: usize, L = Dense> = Span<T, Const<N>, L>;

#[cfg(feature = "nightly")]
extern "C" {
    type Opaque;
}

type ValidMapping<D, L> = <<D as Dim>::Layout<L> as Layout>::Mapping<D>;

impl<T, D: Dim, L: Layout> Span<T, D, L> {
    /// Returns a mutable pointer to the array buffer.
    pub fn as_mut_ptr(&mut self) -> *mut T {
        if D::RANK > 0 {
            RawSpan::from_mut_span(self).as_mut_ptr()
        } else {
            self as *mut Self as *mut T
        }
    }

    /// Returns a raw pointer to the array buffer.
    pub fn as_ptr(&self) -> *const T {
        if D::RANK > 0 {
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
    pub fn axis_expr<const DIM: usize>(&self) -> AxisExpr<T, D, <Const<DIM> as Axis<D>>::Remove<L>>
    where
        Const<DIM>: Axis<D>,
    {
        unsafe {
            AxisExpr::new_unchecked(
                self.as_ptr(),
                Mapping::remove_dim(self.mapping(), DIM),
                self.size(DIM),
                self.stride(DIM),
            )
        }
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
    pub fn axis_expr_mut<const DIM: usize>(
        &mut self,
    ) -> AxisExprMut<T, D, <Const<DIM> as Axis<D>>::Remove<L>>
    where
        Const<DIM>: Axis<D>,
    {
        unsafe {
            AxisExprMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::remove_dim(self.mapping(), DIM),
                self.size(DIM),
                self.stride(DIM),
            )
        }
    }

    /// Returns an expression that gives column views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn cols(&self) -> Lanes<T, D, L::Uniform> {
        assert!(D::RANK > 0, "invalid rank");

        let mapping = ValidMapping::<D::Lower, Strided>::remove_dim(self.mapping(), 0);

        unsafe {
            Lanes::new_unchecked(
                self.as_ptr(),
                Mapping::keep_dim(self.mapping(), 0),
                mapping.shape(),
                mapping.strides(),
            )
        }
    }

    /// Returns a mutable expression that gives column views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn cols_mut(&mut self) -> LanesMut<T, D, L::Uniform> {
        assert!(D::RANK > 0, "invalid rank");

        let mapping = ValidMapping::<D::Lower, Strided>::remove_dim(self.mapping(), 0);

        unsafe {
            LanesMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::keep_dim(self.mapping(), 0),
                mapping.shape(),
                mapping.strides(),
            )
        }
    }

    /// Returns `true` if the array span contains an element with the given value.
    pub fn contains(&self, x: &T) -> bool
    where
        T: PartialEq,
    {
        contains(self, x)
    }

    /// Returns an expression over the array span.
    pub fn expr(&self) -> Expr<T, D, L> {
        unsafe { Expr::new_unchecked(self.as_ptr(), self.mapping()) }
    }

    /// Returns a mutable expression over the array span.
    pub fn expr_mut(&mut self) -> ExprMut<T, D, L> {
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
    pub fn flatten(&self) -> Expr<T, Const<1>, L::Uniform> {
        self.expr().into_flattened()
    }

    /// Returns a mutable one-dimensional array view over the array span.
    ///
    /// # Panics
    ///
    /// Panics if the array layout is not uniformly strided.
    pub fn flatten_mut(&mut self) -> ExprMut<T, Const<1>, L::Uniform> {
        self.expr_mut().into_flattened()
    }

    /// Returns a reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked<I: SpanIndex<T, D, L>>(&self, index: I) -> &I::Output {
        index.get_unchecked(self)
    }

    /// Returns a mutable reference to an element or a subslice, without doing bounds checking.
    ///
    /// # Safety
    ///
    /// The index must be within bounds of the array span.
    pub unsafe fn get_unchecked_mut<I: SpanIndex<T, D, L>>(&mut self, index: I) -> &mut I::Output {
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
    pub fn iter(&self) -> Iter<Expr<'_, T, D, L>> {
        self.expr().into_iter()
    }

    /// Returns a mutable iterator over the array span.
    pub fn iter_mut(&mut self) -> Iter<ExprMut<'_, T, D, L>> {
        self.expr_mut().into_iter()
    }

    /// Returns an expression that gives array views over the specified dimension,
    /// iterating over the other dimensions.
    ///
    /// If the innermost dimension is specified, the resulting array views have dense or
    /// flat layout. For other dimensions, the resulting array views have flat layout.
    pub fn lanes<const DIM: usize>(&self) -> Lanes<T, D, <Const<DIM> as Axis<D>>::Keep<L>>
    where
        Const<DIM>: Axis<D>,
    {
        let mapping = ValidMapping::<D::Lower, Strided>::remove_dim(self.mapping(), DIM);

        unsafe {
            Lanes::new_unchecked(
                self.as_ptr(),
                Mapping::keep_dim(self.mapping(), DIM),
                mapping.shape(),
                mapping.strides(),
            )
        }
    }

    /// Returns a mutable expression that gives array views over the specified dimension,
    /// iterating over the other dimensions.
    ///
    /// If the innermost dimension is specified, the resulting array views have dense or
    /// flat layout. For other dimensions, the resulting array views have flat layout.
    pub fn lanes_mut<const DIM: usize>(
        &mut self,
    ) -> LanesMut<T, D, <Const<DIM> as Axis<D>>::Keep<L>>
    where
        Const<DIM>: Axis<D>,
    {
        let mapping = ValidMapping::<D::Lower, Strided>::remove_dim(self.mapping(), DIM);

        unsafe {
            LanesMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::keep_dim(self.mapping(), DIM),
                mapping.shape(),
                mapping.strides(),
            )
        }
    }

    /// Returns the number of elements in the array.
    pub fn len(&self) -> usize {
        self.mapping().len()
    }

    /// Returns the array layout mapping.
    pub fn mapping(&self) -> L::Mapping<D> {
        if D::RANK > 0 {
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
    pub fn outer_expr(&self) -> AxisExpr<T, D, <D::Lower as Dim>::Layout<L>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisExpr::new_unchecked(
                self.as_ptr(),
                Mapping::remove_dim(self.mapping(), D::RANK - 1),
                self.size(D::RANK - 1),
                self.stride(D::RANK - 1),
            )
        }
    }

    /// Returns a mutable expression that gives array views iterating over the outermost dimension.
    ///
    /// Iterating over the outermost dimension maintains both the unit inner stride and the
    /// uniform stride properties, and the resulting array views have the same layout.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1.
    pub fn outer_expr_mut(&mut self) -> AxisExprMut<T, D, <D::Lower as Dim>::Layout<L>> {
        assert!(D::RANK > 0, "invalid rank");

        unsafe {
            AxisExprMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::remove_dim(self.mapping(), D::RANK - 1),
                self.size(D::RANK - 1),
                self.stride(D::RANK - 1),
            )
        }
    }

    /// Returns the array rank, i.e. the number of dimensions.
    pub fn rank(&self) -> usize {
        D::RANK
    }

    /// Returns a remapped array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap<M: Layout>(&self) -> Expr<T, D, M> {
        self.expr().into_mapping()
    }

    /// Returns a mutable remapped array view of the array span.
    ///
    /// # Panics
    ///
    /// Panics if the memory layout is not compatible with the new array layout.
    pub fn remap_mut<M: Layout>(&mut self) -> ExprMut<T, D, M> {
        self.expr_mut().into_mapping()
    }

    /// Returns a reshaped array view of the array span, with similar layout.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape<S: Shape>(&self, shape: S) -> Expr<T, S::Dim, <S::Dim as Dim>::Layout<L>> {
        self.expr().into_shape(shape)
    }

    /// Returns a mutable reshaped array view of the array span, with similar layout.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed, or the memory layout is not compatible.
    pub fn reshape_mut<S: Shape>(
        &mut self,
        shape: S,
    ) -> ExprMut<T, S::Dim, <S::Dim as Dim>::Layout<L>> {
        self.expr_mut().into_shape(shape)
    }

    /// Returns an expression that gives row views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 2.
    pub fn rows(&self) -> Lanes<T, D, Flat> {
        assert!(D::RANK > 1, "invalid rank");

        let mapping = ValidMapping::<D::Lower, Strided>::remove_dim(self.mapping(), 1);

        unsafe {
            Lanes::new_unchecked(
                self.as_ptr(),
                Mapping::keep_dim(self.mapping(), 1),
                mapping.shape(),
                mapping.strides(),
            )
        }
    }

    /// Returns a mutable expression that gives row views iterating over the other dimensions.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 2.
    pub fn rows_mut(&mut self) -> LanesMut<T, D, Flat> {
        assert!(D::RANK > 1, "invalid rank");

        let mapping = ValidMapping::<D::Lower, Strided>::remove_dim(self.mapping(), 1);

        unsafe {
            LanesMut::new_unchecked(
                self.as_mut_ptr(),
                Mapping::keep_dim(self.mapping(), 1),
                mapping.shape(),
                mapping.strides(),
            )
        }
    }

    /// Returns the shape of the array.
    pub fn shape(&self) -> D::Shape {
        self.mapping().shape()
    }

    /// Returns the number of elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn size(&self, dim: usize) -> usize {
        self.mapping().size(dim)
    }

    /// Divides an array span into two at an index along the outermost dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at(&self, mid: usize) -> (Expr<T, D, L>, Expr<T, D, L>) {
        self.expr().into_split_at(mid)
    }

    /// Divides a mutable array span into two at an index along the outermost dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_at_mut(&mut self, mid: usize) -> (ExprMut<T, D, L>, ExprMut<T, D, L>) {
        self.expr_mut().into_split_at(mid)
    }

    /// Divides an array span into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_axis_at<const DIM: usize>(
        &self,
        mid: usize,
    ) -> (
        Expr<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
        Expr<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.expr().into_split_axis_at(mid)
    }

    /// Divides a mutable array span into two at an index along the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the split point is larger than the number of elements in that dimension.
    pub fn split_axis_at_mut<const DIM: usize>(
        &mut self,
        mid: usize,
    ) -> (
        ExprMut<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
        ExprMut<T, D, <Const<DIM> as Axis<D>>::Split<L>>,
    )
    where
        Const<DIM>: Axis<D>,
    {
        self.expr_mut().into_split_axis_at(mid)
    }

    /// Returns the distance between elements in the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if the dimension is out of bounds.
    pub fn stride(&self, dim: usize) -> isize {
        self.mapping().stride(dim)
    }

    /// Returns the distance between elements in each dimension.
    pub fn strides(&self) -> D::Strides {
        self.mapping().strides()
    }

    /// Copies the array span into a new array.
    #[cfg(not(feature = "nightly"))]
    pub fn to_grid(&self) -> Grid<T, D>
    where
        T: Clone,
    {
        self.expr().cloned().eval()
    }

    /// Copies the array span into a new array.
    #[cfg(feature = "nightly")]
    pub fn to_grid(&self) -> Grid<T, D>
    where
        T: Clone,
    {
        self.to_grid_in(Global)
    }

    /// Copies the array span into a new array with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn to_grid_in<A: Allocator>(&self, alloc: A) -> Grid<T, D, A>
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

impl<T, D: Dim> Span<T, D> {
    /// Returns a mutable slice of all elements in the array, which must have dense layout.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.expr_mut().into_slice()
    }

    /// Returns a slice of all elements in the array, which must have dense layout.
    pub fn as_slice(&self) -> &[T] {
        self.expr().into_slice()
    }
}

impl<T, L: Layout> Span<T, Const<2>, L> {
    /// Returns an array view for the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn col(&self, index: usize) -> Expr<T, Const<1>, L::Uniform> {
        self.view(.., index)
    }

    /// Returns a mutable array view for the specified column.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn col_mut(&mut self, index: usize) -> ExprMut<T, Const<1>, L::Uniform> {
        self.view_mut(.., index)
    }

    /// Returns an array view for the given diagonal of the array span,
    /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the absolute index is larger than the number of columns or rows.
    pub fn diag(&self, index: isize) -> Expr<T, Const<1>, Flat> {
        self.expr().into_diag(index)
    }

    /// Returns a mutable array view for the given diagonal of the array span,
    /// where `index` > 0 is above and `index` < 0 is below the main diagonal.
    ///
    /// # Panics
    ///
    /// Panics if the absolute index is larger than the number of columns or rows.
    pub fn diag_mut(&mut self, index: isize) -> ExprMut<T, Const<1>, Flat> {
        self.expr_mut().into_diag(index)
    }

    /// Returns an array view for the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn row(&self, index: usize) -> Expr<T, Const<1>, Flat> {
        self.view(index, ..).into_mapping()
    }

    /// Returns a mutable array view for the specified row.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn row_mut(&mut self, index: usize) -> ExprMut<T, Const<1>, Flat> {
        self.view_mut(index, ..).into_mapping()
    }
}

macro_rules! impl_permute {
    ($n:tt, ($($xyz:tt),+)) => {
        impl<T, L: Layout> Span<T, Const<$n>, L> {
            /// Returns an array view with the dimensions permuted.
            pub fn permute<$(const $xyz: usize),+>(
                &self
            ) -> Expr<T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
            where
                ($(Const<$xyz>,)+): Permutation
            {
                self.expr().into_permuted()
            }

            /// Returns a mutable array view with the dimensions permuted.
            pub fn permute_mut<$(const $xyz: usize),+>(
                &mut self
            ) -> ExprMut<T, Const<$n>, <($(Const<$xyz>,)+) as Permutation>::Layout<L>>
            where
                ($(Const<$xyz>,)+): Permutation
            {
                self.expr_mut().into_permuted()
            }
        }
    };
}

impl_permute!(1, (X));
impl_permute!(2, (X, Y));
impl_permute!(3, (X, Y, Z));
impl_permute!(4, (X, Y, Z, W));
impl_permute!(5, (X, Y, Z, W, U));
impl_permute!(6, (X, Y, Z, W, U, V));

macro_rules! impl_view {
    ($n:tt, ($($xyz:tt),+), ($($idx:tt),+)) => {
        #[allow(unused_parens)]
        impl<T, L: Layout> Span<T, Const<$n>, L> {
            /// Copies the specified subarray into a new array.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn grid<$($xyz: DimIndex),+>(
                &self,
                $($idx: $xyz),+
            ) -> Grid<T, <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim>
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
            pub fn view<$($xyz: DimIndex),+>(
                &self,
                $($idx: $xyz),+
            ) -> Expr<
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                self.expr().into_view($($idx),+)
            }

            /// Returns a mutable array view for the specified subarray.
            ///
            /// # Panics
            ///
            /// Panics if the subarray is out of bounds.
            pub fn view_mut<$($xyz: DimIndex),+>(
                &mut self,
                $($idx: $xyz),+,
            ) -> ExprMut<
                T,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Dim,
                <($($xyz,)+) as ViewIndex<Const<$n>, L>>::Layout,
            > {
                self.expr_mut().into_view($($idx),+)
            }
        }
    };
}

impl_view!(1, (X), (x));
impl_view!(2, (X, Y), (x, y));
impl_view!(3, (X, Y, Z), (x, y, z));
impl_view!(4, (X, Y, Z, W), (x, y, z, w));
impl_view!(5, (X, Y, Z, W, U), (x, y, z, w, u));
impl_view!(6, (X, Y, Z, W, U, V), (x, y, z, w, u, v));

impl<'a, T, U, D: Dim, L: Layout> Apply<U> for &'a Span<T, D, L> {
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

impl<'a, T, U, D: Dim, L: Layout> Apply<U> for &'a mut Span<T, D, L> {
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

impl<T, D: Dim, L: Layout> AsMut<Span<T, D, L>> for Span<T, D, L> {
    fn as_mut(&mut self) -> &mut Span<T, D, L> {
        self
    }
}

impl<T, D: Dim> AsMut<[T]> for Span<T, D> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self[..]
    }
}

impl<T, D: Dim, L: Layout> AsRef<Span<T, D, L>> for Span<T, D, L> {
    fn as_ref(&self) -> &Span<T, D, L> {
        self
    }
}

impl<T, D: Dim> AsRef<[T]> for Span<T, D> {
    fn as_ref(&self) -> &[T] {
        &self[..]
    }
}

impl<T: Debug, D: Dim, L: Layout> Debug for Span<T, D, L> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        if D::RANK == 0 {
            self[D::Shape::default()].fmt(f)
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

impl<T: Hash, D: Dim, L: Layout> Hash for Span<T, D, L> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for i in 0..D::RANK {
            #[cfg(not(feature = "nightly"))]
            state.write_usize(self.size(i));
            #[cfg(feature = "nightly")]
            state.write_length_prefix(self.size(i));
        }

        self.expr().for_each(|x| x.hash(state));
    }
}

impl<T, D: Dim, L: Layout, I: SpanIndex<T, D, L>> Index<I> for Span<T, D, L> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, D: Dim, L: Layout, I: SpanIndex<T, D, L>> IndexMut<I> for Span<T, D, L> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, D: Dim, L: Layout> IntoExpression for &'a Span<T, D, L> {
    type Dim = D;
    type IntoExpr = Expr<'a, T, D, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, D: Dim, L: Layout> IntoExpression for &'a mut Span<T, D, L> {
    type Dim = D;
    type IntoExpr = ExprMut<'a, T, D, L>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<'a, T, D: Dim, L: Layout> IntoIterator for &'a Span<T, D, L> {
    type Item = &'a T;
    type IntoIter = Iter<Expr<'a, T, D, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, D: Dim, L: Layout> IntoIterator for &'a mut Span<T, D, L> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, D, L>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T: Clone, D: Dim> ToOwned for Span<T, D> {
    type Owned = Grid<T, D>;

    fn to_owned(&self) -> Self::Owned {
        self.to_grid()
    }

    fn clone_into(&self, target: &mut Self::Owned) {
        target.clone_from_span(self);
    }
}

fn contains<T: PartialEq, D: Dim, L: Layout>(this: &Span<T, D, L>, value: &T) -> bool {
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
