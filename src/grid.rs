#[cfg(feature = "nightly")]
use std::alloc::{Allocator, Global};
use std::borrow::{Borrow, BorrowMut};
use std::collections::TryReserveError;
use std::fmt::{self, Debug, Formatter};
use std::hash::{Hash, Hasher};
use std::mem::{self, ManuallyDrop, MaybeUninit};
use std::ops::RangeBounds;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::slice;

#[cfg(not(feature = "nightly"))]
use crate::alloc::{Allocator, Global};
use crate::array::Array;
use crate::buffer::{Buffer, Drain};
use crate::dim::{Const, Dim, Dyn};
use crate::expr::{self, Expr, ExprMut, IntoExpr, Map, Zip};
use crate::expression::Expression;
use crate::index::{Axis, Outer, SpanIndex};
use crate::iter::Iter;
use crate::layout::{Dense, Layout};
use crate::mapping::{DenseMapping, Mapping};
use crate::raw_grid::RawGrid;
use crate::shape::{ConstShape, IntoShape, Rank, Shape};
use crate::span::Span;
use crate::traits::{Apply, FromExpression, IntoCloned, IntoExpression};

#[cfg(not(feature = "nightly"))]
macro_rules! vec_t {
    ($type:ty, $alloc:ty) => {
        Vec<$type>
    };
}

#[cfg(feature = "nightly")]
macro_rules! vec_t {
    ($type:ty, $alloc:ty) => {
        Vec<$type, $alloc>
    };
}

/// Dense multidimensional array.
pub struct Grid<T, S: Shape, A: Allocator = Global> {
    grid: RawGrid<T, S, A>,
}

/// Multidimensional array with dynamically-sized dimensions and dense layout.
pub type DGrid<T, const N: usize, A = Global> = Grid<T, Rank<N>, A>;

impl<T, S: Shape, A: Allocator> Grid<T, S, A> {
    /// Returns a reference to the underlying allocator.
    #[cfg(feature = "nightly")]
    pub fn allocator(&self) -> &A {
        self.grid.allocator()
    }

    /// Moves all elements from another array into the array along the outermost dimension.
    ///
    /// If the array is empty, it is reshaped to match the shape of the other array.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, if the rank is not at least 1,
    /// or if the outermost dimension is not dynamically-sized.
    pub fn append(&mut self, other: &mut Self) {
        self.expand(other.drain(..));
    }

    /// Returns the number of elements the array can hold without reallocating.
    pub fn capacity(&self) -> usize {
        self.grid.capacity()
    }

    /// Clears the array, removing all values.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    pub fn clear(&mut self) {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe {
            self.grid.with_mut_vec(|vec| vec.clear());
            self.grid.set_mapping(DenseMapping::default());
        }
    }

    /// Removes the specified range from the array along the outermost dimension,
    /// and returns the removed range as an expression.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1, or if the outermost dimension
    /// is not dynamically-sized.
    pub fn drain<R: RangeBounds<usize>>(&mut self, range: R) -> IntoExpr<Drain<T, S, A>> {
        assert!(S::RANK > 0, "invalid rank");
        assert!(<Outer as Axis>::Dim::<S>::SIZE.is_none(), "dimension not dynamically-sized");

        #[cfg(not(feature = "nightly"))]
        let range = crate::index::range(range, ..self.dim(S::RANK - 1));
        #[cfg(feature = "nightly")]
        let range = slice::range(range, ..self.dim(S::RANK - 1));

        IntoExpr::new(Drain::new(self, range.start, range.end))
    }

    /// Appends an expression to the array along the outermost dimension with broadcasting,
    /// cloning elements if needed.
    ///
    /// If the rank of the expression equals one less than the rank of the array,
    /// the expression is assumed to have outermost dimension of size 1.
    ///
    /// If the array is empty, it is reshaped to match the shape of the expression.
    ///
    /// # Panics
    ///
    /// Panics if the inner dimensions do not match, if the rank of the expression
    /// is not valid, or if the outermost dimension is not dynamically-sized.
    pub fn expand<I: IntoExpression<Item: IntoCloned<T>>>(&mut self, expr: I) {
        assert!(S::RANK > 0, "invalid rank");
        assert!(I::Shape::RANK + 1 == S::RANK || I::Shape::RANK == S::RANK, "invalid rank");
        assert!(<Outer as Axis>::Dim::<S>::SIZE.is_none(), "dimension not dynamically-sized");

        let expr = expr.into_expr();
        let len = expr.len();

        if len > 0 {
            let inner_dims = &expr.dims()[..S::RANK - 1];
            let mut dims = self.dims();

            if self.is_empty() {
                dims[..S::RANK - 1].copy_from_slice(inner_dims);
                dims[S::RANK - 1] = 0;
            } else {
                assert!(inner_dims == &dims[..S::RANK - 1], "inner dimensions mismatch");
            }

            dims[S::RANK - 1] += if I::Shape::RANK < S::RANK { 1 } else { expr.dim(S::RANK - 1) };

            let shape = Shape::from_dims(dims);

            unsafe {
                self.grid.with_mut_vec(|vec| {
                    vec.reserve(len);
                    expr.clone_into_vec(vec);
                });

                self.set_mapping(DenseMapping::new(shape));
            }
        }
    }

    /// Creates an array from the given element with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn from_elem_in<I: IntoShape<IntoShape = S>>(shape: I, elem: T, alloc: A) -> Self
    where
        T: Clone,
    {
        Self::from_expr_in(expr::from_elem(shape, elem), alloc)
    }

    /// Creates an array with the results from the given function with the specified allocator.
    #[cfg(feature = "nightly")]
    pub fn from_fn_in<I: IntoShape<IntoShape = S>, F>(shape: I, f: F, alloc: A) -> Self
    where
        F: FnMut(S::Dims) -> T,
    {
        Self::from_expr_in(expr::from_fn(shape, f), alloc)
    }

    /// Creates an array from raw components of another array with the specified allocator.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the mapping, capacity and allocator.
    #[cfg(feature = "nightly")]
    pub unsafe fn from_raw_parts_in(
        ptr: *mut T,
        mapping: DenseMapping<S>,
        capacity: usize,
        alloc: A,
    ) -> Self {
        Self::from_parts(Vec::from_raw_parts_in(ptr, mapping.len(), capacity, alloc), mapping)
    }

    /// Converts the array into a one-dimensional array.
    pub fn into_flattened(self) -> Grid<T, Dyn, A> {
        self.into_vec().into()
    }

    /// Decomposes an array into its raw components including the allocator.
    #[cfg(feature = "nightly")]
    pub fn into_raw_parts_with_alloc(self) -> (*mut T, DenseMapping<S>, usize, A) {
        let (vec, mapping) = self.grid.into_parts();
        let (ptr, _, capacity, alloc) = vec.into_raw_parts_with_alloc();

        (ptr, mapping, capacity, alloc)
    }

    /// Converts an array with a single element into the contained value.
    ///
    /// # Panics
    ///
    /// Panics if the array length is not equal to one.
    pub fn into_scalar(self) -> T {
        assert!(self.len() == 1, "invalid length");

        self.into_vec().pop().unwrap()
    }

    /// Converts the array into a reshaped array, which must have the same length.
    ///
    /// # Panics
    ///
    /// Panics if the array length is changed.
    pub fn into_shape<I: IntoShape>(self, shape: I) -> Grid<T, I::IntoShape, A> {
        let (vec, mapping) = self.grid.into_parts();

        unsafe { Grid::from_parts(vec, Mapping::reshape(mapping, shape.into_shape())) }
    }

    /// Converts the array into a vector.
    pub fn into_vec(self) -> vec_t!(T, A) {
        let (vec, _) = self.grid.into_parts();

        vec
    }

    /// Returns the array with the given closure applied to each element.
    pub fn map<F: FnMut(T) -> T>(self, f: F) -> Self
    where
        T: Default,
    {
        self.apply(f)
    }

    /// Creates a new, empty array with the specified allocator.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[cfg(feature = "nightly")]
    pub fn new_in(alloc: A) -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::new_in(alloc), DenseMapping::default()) }
    }

    /// Reserves capacity for at least the additional number of elements in the array.
    pub fn reserve(&mut self, additional: usize) {
        unsafe {
            self.grid.with_mut_vec(|vec| vec.reserve(additional));
        }
    }

    /// Reserves the minimum capacity for the additional number of elements in the array.
    pub fn reserve_exact(&mut self, additional: usize) {
        unsafe {
            self.grid.with_mut_vec(|vec| vec.reserve_exact(additional));
        }
    }

    /// Resizes the array to the new shape, creating new elements with the given value.
    pub fn resize<I: IntoShape<IntoShape = S>>(&mut self, new_shape: I, value: T)
    where
        T: Clone,
        A: Clone,
    {
        self.grid.resize_with(new_shape.into_shape(), || value.clone());
    }

    /// Resizes the array to the new shape, creating new elements from the given closure.
    pub fn resize_with<I: IntoShape<IntoShape = S>, F>(&mut self, new_shape: I, f: F)
    where
        A: Clone,
        F: FnMut() -> T,
    {
        self.grid.resize_with(new_shape.into_shape(), f);
    }

    /// Forces the array layout mapping to the new mapping.
    ///
    /// # Safety
    ///
    /// All elements within the array length must be initialized.
    pub unsafe fn set_mapping(&mut self, new_mapping: DenseMapping<S>) {
        self.grid.set_mapping(new_mapping);
    }

    /// Shrinks the capacity of the array with a lower bound.
    pub fn shrink_to(&mut self, min_capacity: usize) {
        unsafe {
            self.grid.with_mut_vec(|vec| vec.shrink_to(min_capacity));
        }
    }

    /// Shrinks the capacity of the array as much as possible.
    pub fn shrink_to_fit(&mut self) {
        unsafe {
            self.grid.with_mut_vec(|vec| vec.shrink_to_fit());
        }
    }

    /// Returns the remaining spare capacity of the array as a slice of `MaybeUninit<T>`.
    ///
    /// The returned slice can be used to fill the array with data, before marking
    /// the data as initialized using the `set_shape` method.
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<T>] {
        let ptr = self.as_mut_ptr();
        let len = self.capacity() - self.len();

        unsafe { slice::from_raw_parts_mut(ptr.add(self.len()).cast(), len) }
    }

    /// Shortens the array along the outermost dimension, keeping the first `size` indices.
    ///
    /// If `size` is greater or equal to the current dimension size, this has no effect.
    ///
    /// Note that this method has no effect on the allocated capacity of the array.
    ///
    /// # Panics
    ///
    /// Panics if the rank is not at least 1, or if the outermost dimension
    /// is not dynamically-sized.
    pub fn truncate(&mut self, size: usize) {
        assert!(S::RANK > 0, "invalid rank");
        assert!(<Outer as Axis>::Dim::<S>::SIZE.is_none(), "dimension not dynamically-sized");

        if size < self.dim(S::RANK - 1) {
            let new_mapping = DenseMapping::resize_dim(self.mapping(), S::RANK - 1, size);

            unsafe {
                self.grid.with_mut_vec(|vec| vec.truncate(new_mapping.len()));
                self.grid.set_mapping(new_mapping);
            }
        }
    }

    /// Tries to reserve capacity for at least the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.grid.with_mut_vec(|vec| vec.try_reserve(additional)) }
    }

    /// Tries to reserve the minimum capacity for the additional number of elements in the array.
    ///
    /// # Errors
    ///
    /// If the capacity overflows, or the allocator reports a failure, then an error is returned.
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        unsafe { self.grid.with_mut_vec(|vec| vec.try_reserve_exact(additional)) }
    }

    /// Creates a new, empty array with the specified capacity and allocator.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    #[cfg(feature = "nightly")]
    pub fn with_capacity_in(capacity: usize, alloc: A) -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::with_capacity_in(capacity, alloc), DenseMapping::default()) }
    }

    #[cfg(not(feature = "nightly"))]
    fn from_expr<E: Expression<Item = T, Shape = S>>(expr: E) -> Self {
        let shape = expr.shape();
        let mut vec = Vec::with_capacity(shape.len());

        expr.clone_into_vec(&mut vec);

        unsafe { Grid::from_parts(vec, DenseMapping::new(shape)) }
    }

    #[cfg(feature = "nightly")]
    pub(crate) fn from_expr_in<E>(expr: E, alloc: A) -> Self
    where
        E: Expression<Item = T, Shape = S>,
    {
        let shape = expr.shape();
        let mut vec = Vec::with_capacity_in(shape.len(), alloc);

        expr.clone_into_vec(&mut vec);

        unsafe { Grid::from_parts(vec, DenseMapping::new(shape)) }
    }

    pub(crate) unsafe fn from_parts(vec: vec_t!(T, A), mapping: DenseMapping<S>) -> Self {
        Self { grid: RawGrid::from_parts(vec, mapping) }
    }
}

#[cfg(not(feature = "nightly"))]
impl<T, S: Shape> Grid<T, S> {
    /// Creates an array from the given element.
    pub fn from_elem<I: IntoShape<IntoShape = S>>(shape: I, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_expr(expr::from_elem(shape, elem))
    }

    /// Creates an array with the results from the given function.
    pub fn from_fn<I: IntoShape<IntoShape = S>, F>(shape: I, f: F) -> Self
    where
        F: FnMut(S::Dims) -> T,
    {
        Self::from_expr(expr::from_fn(shape, f))
    }

    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    pub unsafe fn from_raw_parts(ptr: *mut T, mapping: DenseMapping<S>, capacity: usize) -> Self {
        Self::from_parts(Vec::from_raw_parts(ptr, mapping.len(), capacity), mapping)
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, DenseMapping<S>, usize) {
        let (vec, mapping) = self.grid.into_parts();
        let mut vec = mem::ManuallyDrop::new(vec);

        (vec.as_mut_ptr(), mapping, vec.capacity())
    }

    /// Creates a new, empty array.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    pub fn new() -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::new(), DenseMapping::default()) }
    }

    /// Creates a new, empty array with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    pub fn with_capacity(capacity: usize) -> Self {
        assert!(S::default().checked_len() == Some(0), "default length not zero");

        unsafe { Self::from_parts(Vec::with_capacity(capacity), DenseMapping::default()) }
    }
}

#[cfg(feature = "nightly")]
impl<T, S: Shape> Grid<T, S> {
    /// Creates an array from the given element.
    pub fn from_elem<I: IntoShape<IntoShape = S>>(shape: I, elem: T) -> Self
    where
        T: Clone,
    {
        Self::from_elem_in(shape, elem, Global)
    }

    /// Creates an array with the results from the given function.
    pub fn from_fn<I: IntoShape<IntoShape = S>, F>(shape: I, f: F) -> Self
    where
        F: FnMut(S::Dims) -> T,
    {
        Self::from_fn_in(shape, f, Global)
    }

    /// Creates an array from raw components of another array.
    ///
    /// # Safety
    ///
    /// The pointer must be a valid allocation given the shape and capacity.
    pub unsafe fn from_raw_parts(ptr: *mut T, mapping: DenseMapping<S>, capacity: usize) -> Self {
        Self::from_raw_parts_in(ptr, mapping, capacity, Global)
    }

    /// Decomposes an array into its raw components.
    pub fn into_raw_parts(self) -> (*mut T, DenseMapping<S>, usize) {
        let (ptr, mapping, capacity, _) = self.into_raw_parts_with_alloc();

        (ptr, mapping, capacity)
    }

    /// Creates a new, empty array.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    pub fn new() -> Self {
        Self::new_in(Global)
    }

    /// Creates a new, empty array with the specified capacity.
    ///
    /// # Panics
    ///
    /// Panics if the default array length for the layout mapping is not zero.
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }
}

impl<T: Clone, S: Shape> Grid<T, S> {
    pub(crate) fn clone_from_span(&mut self, span: &Span<T, S>) {
        unsafe {
            self.grid.with_mut_vec(|vec| span[..].clone_into(vec));
            self.grid.set_mapping(span.mapping());
        }
    }
}

impl<'a, T, U, S: Shape, A: Allocator> Apply<U> for &'a Grid<T, S, A> {
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

impl<'a, T, U, S: Shape, A: Allocator> Apply<U> for &'a mut Grid<T, S, A> {
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

impl<T: Default, S: Shape, A: Allocator> Apply<T> for Grid<T, S, A> {
    type Output<F: FnMut(T) -> T> = Self;
    type ZippedWith<I: IntoExpression, F: FnMut((T, I::Item)) -> T> = Self;

    fn apply<F: FnMut(T) -> T>(mut self, mut f: F) -> Self {
        self.expr_mut().for_each(|x| *x = f(mem::take(x)));
        self
    }

    fn zip_with<I: IntoExpression, F>(mut self, expr: I, mut f: F) -> Self
    where
        F: FnMut((T, I::Item)) -> T,
    {
        self.expr_mut().zip(expr).for_each(|(x, y)| *x = f((mem::take(x), y)));
        self
    }
}

impl<T, U: ?Sized, S: Shape, A: Allocator> AsMut<U> for Grid<T, S, A>
where
    Span<T, S>: AsMut<U>,
{
    fn as_mut(&mut self) -> &mut U {
        (**self).as_mut()
    }
}

impl<T, U: ?Sized, S: Shape, A: Allocator> AsRef<U> for Grid<T, S, A>
where
    Span<T, S>: AsRef<U>,
{
    fn as_ref(&self) -> &U {
        (**self).as_ref()
    }
}

impl<T, S: Shape, A: Allocator> Borrow<Span<T, S>> for Grid<T, S, A> {
    fn borrow(&self) -> &Span<T, S> {
        self
    }
}

impl<T, S: Shape, A: Allocator> BorrowMut<Span<T, S>> for Grid<T, S, A> {
    fn borrow_mut(&mut self) -> &mut Span<T, S> {
        self
    }
}

impl<T: Clone, S: Shape, A: Allocator + Clone> Clone for Grid<T, S, A> {
    fn clone(&self) -> Self {
        Self { grid: self.grid.clone() }
    }

    fn clone_from(&mut self, source: &Self) {
        self.grid.clone_from(&source.grid);
    }
}

impl<T: Debug, S: Shape, A: Allocator> Debug for Grid<T, S, A> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl<T, S: Shape> Default for Grid<T, S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, S: Shape, A: Allocator> Deref for Grid<T, S, A> {
    type Target = Span<T, S>;

    fn deref(&self) -> &Self::Target {
        self.grid.as_span()
    }
}

impl<T, S: Shape, A: Allocator> DerefMut for Grid<T, S, A> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.grid.as_mut_span()
    }
}

impl<'a, T: Copy, A: Allocator> Extend<&'a T> for Grid<T, Dyn, A> {
    fn extend<I: IntoIterator<Item = &'a T>>(&mut self, iter: I) {
        self.extend(iter.into_iter().copied());
    }
}

impl<T, A: Allocator> Extend<T> for Grid<T, Dyn, A> {
    fn extend<I: IntoIterator<Item = T>>(&mut self, iter: I) {
        unsafe {
            let len = self.grid.with_mut_vec(|vec| {
                vec.extend(iter);
                vec.len()
            });

            self.set_mapping(DenseMapping::new(Dyn(len)));
        }
    }
}

impl<T: Clone> From<&[T]> for Grid<T, Dyn> {
    fn from(value: &[T]) -> Self {
        Self::from(value.to_vec())
    }
}

impl<T, S: ConstShape> From<Array<T, S>> for Grid<T, S> {
    fn from(value: Array<T, S>) -> Self {
        Self::from_expr(value.into_expr())
    }
}

impl<T, S: Shape, A: Allocator> From<Grid<T, S, A>> for vec_t!(T, A) {
    fn from(value: Grid<T, S, A>) -> Self {
        value.into_vec()
    }
}

impl<'a, T: 'a + Clone, S: Shape, L: Layout, I: IntoExpression<IntoExpr = Expr<'a, T, S, L>>>
    From<I> for Grid<T, S>
{
    fn from(value: I) -> Self {
        Self::from_expr(value.into_expr().cloned())
    }
}

impl<B: Buffer> From<IntoExpr<B>> for Grid<B::Item, B::Shape> {
    fn from(value: IntoExpr<B>) -> Self {
        Self::from_expr(value)
    }
}

impl<T, A: Allocator> From<vec_t!(T, A)> for Grid<T, Dyn, A> {
    fn from(value: vec_t!(T, A)) -> Self {
        let mapping = DenseMapping::new(Dyn(value.len()));

        unsafe { Self::from_parts(value, mapping) }
    }
}

macro_rules! impl_from_array {
    ($n:tt, ($($xyz:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<T: Clone, $(const $xyz: usize),+> From<&Array<T, ($(Const<$xyz>),+)>>
            for Grid<T, Rank<$n>>
        {
            fn from(value: &Array<T, ($(Const<$xyz>),+)>) -> Self {
                Self::from(&value.0)
            }
        }

        impl<T: Clone, $(const $xyz: usize),+> From<&$array> for Grid<T, Rank<$n>> {
            fn from(value: &$array) -> Self {
                Self::from_expr(Expr::from(value).cloned())
            }
        }

        #[allow(unused_parens)]
        impl<T, $(const $xyz: usize),+> From<Array<T, ($(Const<$xyz>),+)>> for Grid<T, Rank<$n>> {
            fn from(value: Array<T, ($(Const<$xyz>),+)>) -> Self {
                Grid::from(value.0)
            }
        }

        impl<T, $(const $xyz: usize),+> From<$array> for Grid<T, Rank<$n>> {
            #[cfg(not(feature = "nightly"))]
            fn from(value: $array) -> Self {
                let mapping = DenseMapping::new(($(Dyn($xyz)),+));
                let capacity = mapping.shape().checked_len().expect("invalid length");

                let ptr = Box::into_raw(Box::new(value));

                unsafe { Self::from_raw_parts(ptr.cast(), mapping, capacity) }
            }

            #[cfg(feature = "nightly")]
            fn from(value: $array) -> Self {
                let mapping = DenseMapping::new(($(Dyn($xyz)),+));
                let capacity = mapping.shape().checked_len().expect("invalid length");

                let (ptr, alloc) = Box::into_raw_with_allocator(Box::new(value));

                unsafe { Self::from_raw_parts_in(ptr.cast(), mapping, capacity, alloc) }
            }
        }
    };
}

impl_from_array!(1, (X), [T; X]);
impl_from_array!(2, (X, Y), [[T; X]; Y]);
impl_from_array!(3, (X, Y, Z), [[[T; X]; Y]; Z]);
impl_from_array!(4, (X, Y, Z, W), [[[[T; X]; Y]; Z]; W]);
impl_from_array!(5, (X, Y, Z, W, U), [[[[[T; X]; Y]; Z]; W]; U]);
impl_from_array!(6, (X, Y, Z, W, U, V), [[[[[[T; X]; Y]; Z]; W]; U]; V]);

impl<T, S: Shape> FromExpression<T, S> for Grid<T, S> {
    #[cfg(not(feature = "nightly"))]
    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::from_expr(expr.into_expr())
    }

    #[cfg(feature = "nightly")]
    fn from_expr<I: IntoExpression<Item = T, Shape = S>>(expr: I) -> Self {
        Self::from_expr_in(expr.into_expr(), Global)
    }
}

impl<T> FromIterator<T> for Grid<T, Dyn> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self::from(Vec::from_iter(iter))
    }
}

impl<T: Hash, S: Shape, A: Allocator> Hash for Grid<T, S, A> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        (**self).hash(state)
    }
}

impl<T, S: Shape, A: Allocator, I: SpanIndex<T, S, Dense>> Index<I> for Grid<T, S, A> {
    type Output = I::Output;

    fn index(&self, index: I) -> &I::Output {
        index.index(self)
    }
}

impl<T, S: Shape, A: Allocator, I: SpanIndex<T, S, Dense>> IndexMut<I> for Grid<T, S, A> {
    fn index_mut(&mut self, index: I) -> &mut I::Output {
        index.index_mut(self)
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoExpression for &'a Grid<T, S, A> {
    type Shape = S;
    type IntoExpr = Expr<'a, T, S>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr()
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoExpression for &'a mut Grid<T, S, A> {
    type Shape = S;
    type IntoExpr = ExprMut<'a, T, S>;

    fn into_expr(self) -> Self::IntoExpr {
        self.expr_mut()
    }
}

impl<T, S: Shape, A: Allocator> IntoExpression for Grid<T, S, A> {
    type Shape = S;
    type IntoExpr = IntoExpr<Grid<ManuallyDrop<T>, S, A>>;

    #[cfg(not(feature = "nightly"))]
    fn into_expr(self) -> Self::IntoExpr {
        let (vec, mapping) = self.grid.into_parts();

        let mut vec = mem::ManuallyDrop::new(vec);
        let (ptr, len, capacity) = (vec.as_mut_ptr(), vec.len(), vec.capacity());

        let grid =
            unsafe { Grid::from_parts(Vec::from_raw_parts(ptr.cast(), len, capacity), mapping) };

        IntoExpr::new(grid)
    }

    #[cfg(feature = "nightly")]
    fn into_expr(self) -> Self::IntoExpr {
        let (ptr, mapping, capacity, alloc) = self.into_raw_parts_with_alloc();

        let grid = unsafe { Grid::from_raw_parts_in(ptr.cast(), mapping, capacity, alloc) };

        IntoExpr::new(grid)
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoIterator for &'a Grid<T, S, A> {
    type Item = &'a T;
    type IntoIter = Iter<Expr<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, T, S: Shape, A: Allocator> IntoIterator for &'a mut Grid<T, S, A> {
    type Item = &'a mut T;
    type IntoIter = Iter<ExprMut<'a, T, S>>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter_mut()
    }
}

impl<T, S: Shape, A: Allocator> IntoIterator for Grid<T, S, A> {
    type Item = T;
    type IntoIter = Iter<IntoExpr<Grid<ManuallyDrop<T>, S, A>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_expr().into_iter()
    }
}

macro_rules! impl_try_from_array {
    ($n:tt, ($($xyz:tt),+), $array:tt) => {
        #[allow(unused_parens)]
        impl<T: Clone, $(const $xyz: usize),+> TryFrom<Grid<T, Rank<$n>>>
            for Array<T, ($(Const<$xyz>),+)>
        {
            type Error = Grid<T, Rank<$n>>;

            fn try_from(value: Grid<T, Rank<$n>>) -> Result<Self, Self::Error> {
                Ok(Array(TryFrom::try_from(value)?))
            }
        }

        impl<T: Clone, $(const $xyz: usize),+> TryFrom<Grid<T, Rank<$n>>> for $array {
            type Error = Grid<T, Rank<$n>>;

            fn try_from(value: Grid<T, Rank<$n>>) -> Result<Self, Self::Error> {
                if value.dims() == [$($xyz),+] {
                    let mut vec = value.into_vec();

                    unsafe {
                        vec.set_len(0);

                        Ok((vec.as_ptr() as *const $array).read())
                    }
                } else {
                    Err(value)
                }
            }
        }
    };
}

impl_try_from_array!(1, (X), [T; X]);
impl_try_from_array!(2, (X, Y), [[T; X]; Y]);
impl_try_from_array!(3, (X, Y, Z), [[[T; X]; Y]; Z]);
impl_try_from_array!(4, (X, Y, Z, W), [[[[T; X]; Y]; Z]; W]);
impl_try_from_array!(5, (X, Y, Z, W, U), [[[[[T; X]; Y]; Z]; W]; U]);
impl_try_from_array!(6, (X, Y, Z, W, U, V), [[[[[[T; X]; Y]; Z]; W]; U]; V]);
