use std::fmt::{Debug, Formatter, Result};

use crate::expr::expression::Expression;
use crate::expr::iter::Iter;
use crate::index::{Axis, Keep, Split};
use crate::layout::Layout;
use crate::mapping::Mapping;
use crate::shape::{IntoShape, Shape};
use crate::slice::Slice;
use crate::view::{View, ViewMut};

/// Array axis expression.
pub struct AxisExpr<'a, T, S: Shape, L: Layout, A: Axis> {
    slice: &'a Slice<T, S, L>,
    axis: A,
    mapping: <Keep<A, S, L> as Layout>::Mapping<(A::Dim<S>,)>,
    offset: isize,
}

/// Mutable array axis expression.
pub struct AxisExprMut<'a, T, S: Shape, L: Layout, A: Axis> {
    slice: &'a mut Slice<T, S, L>,
    axis: A,
    mapping: <Keep<A, S, L> as Layout>::Mapping<(A::Dim<S>,)>,
    offset: isize,
}

/// Expression that repeats an element by cloning.
#[derive(Clone)]
pub struct Fill<T> {
    value: T,
}

/// Expression that gives elements by calling a closure repeatedly.
#[derive(Clone)]
pub struct FillWith<F> {
    f: F,
}

/// Expression with a defined shape that repeats an element by cloning.
#[derive(Clone)]
pub struct FromElem<T, S> {
    shape: S,
    elem: T,
}

/// Expression with a defined shape and elements from the given function.
#[derive(Clone)]
pub struct FromFn<S: Shape, F> {
    shape: S,
    f: F,
    index: S::Dims<usize>,
}

/// Array lanes expression.
pub struct Lanes<'a, T, S: Shape, L: Layout, A: Axis> {
    slice: &'a Slice<T, S, L>,
    axis: A,
    mapping: <Split<A, S, L> as Layout>::Mapping<A::Remove<S>>,
    offset: isize,
}

/// Mutable array lanes expression.
pub struct LanesMut<'a, T, S: Shape, L: Layout, A: Axis> {
    slice: &'a mut Slice<T, S, L>,
    axis: A,
    mapping: <Split<A, S, L> as Layout>::Mapping<A::Remove<S>>,
    offset: isize,
}

/// Creates an expression with elements by cloning `value`.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, tensor, view};
///
/// let mut t = tensor![0; 3];
///
/// t.assign(expr::fill(1));
///
/// assert_eq!(t, view![1; 3]);
/// ```
pub fn fill<T: Clone>(value: T) -> Fill<T> {
    Fill::new(value)
}

/// Creates an expression with elements returned by calling a closure repeatedly.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, tensor, view};
///
/// let mut t = tensor![0; 3];
///
/// t.assign(expr::fill_with(|| 1));
///
/// assert_eq!(t, view![1; 3]);
/// ```
pub fn fill_with<T, F: FnMut() -> T>(f: F) -> FillWith<F> {
    FillWith::new(f)
}

/// Creates an expression with the given shape and elements by cloning `value`.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, expr::FromExpression, view, Tensor};
///
/// let t = Tensor::from_expr(expr::from_elem([2, 3], 1));
///
/// assert_eq!(t, view![[1; 3]; 2]);
/// ```
pub fn from_elem<T: Clone, I: IntoShape>(shape: I, elem: T) -> FromElem<T, I::IntoShape> {
    FromElem::new(shape.into_shape(), elem)
}

/// Creates an expression with the given shape and elements from the given function.
///
/// # Examples
///
/// ```
/// use mdarray::{Tensor, expr, expr::FromExpression, view};
///
/// let t = Tensor::from_expr(expr::from_fn([2, 3], |i| 3 * i[0] + i[1] + 1));
///
/// assert_eq!(t, view![[1, 2, 3], [4, 5, 6]]);
/// ```
pub fn from_fn<T, I: IntoShape, F>(shape: I, f: F) -> FromFn<I::IntoShape, F>
where
    F: FnMut(&[usize]) -> T,
{
    FromFn::new(shape.into_shape(), f)
}

macro_rules! impl_axis_expr {
    ($name:tt, $expr:tt, $as_ptr:tt, {$($mut:tt)?}, $repeatable:tt) => {
        impl<'a, T, S: Shape, L: Layout, A: Axis> $name<'a, T, S, L, A> {
            pub(crate) fn new(
                slice: &'a $($mut)? Slice<T, S, L>,
                axis: A,
            ) -> Self {
                let mapping = axis.get(slice.mapping());

                Self { slice, axis, mapping, offset: 0 }
            }
        }

        impl<'a, T: Debug, S: Shape, L: Layout, A: Axis> Debug for $name<'a, T, S, L, A> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let index = self.axis.index(self.slice.rank());

                f.debug_tuple(stringify!($name)).field(&index).field(&self.slice).finish()
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> Expression for $name<'a, T, S, L, A> {
            type Shape = (A::Dim<S>,);

            const IS_REPEATABLE: bool = $repeatable;

            fn shape(&self) -> &Self::Shape {
                self.mapping.shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
                let offset = self.offset + self.mapping.inner_stride() * index as isize;

                let mapping = self.axis.remove(self.slice.mapping());
                let len = mapping.shape().checked_len().expect("invalid length");

                // If the view is empty, we must not offset the pointer.
                let count = if len == 0 { 0 } else { offset };

                unsafe { $expr::new_unchecked(self.slice.$as_ptr().offset(count), mapping) }
            }

            fn inner_rank(&self) -> usize {
                1
            }

            unsafe fn reset_dim(&mut self, _: usize, _: usize) {
                self.offset = 0;
            }

            unsafe fn step_dim(&mut self, _: usize) {
                self.offset += self.mapping.inner_stride();
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> IntoIterator for $name<'a, T, S, L, A> {
            type Item = $expr<'a, T, A::Remove<S>, Split<A, S, L>>;
            type IntoIter = Iter<Self>;

            fn into_iter(self) -> Iter<Self> {
                Iter::new(self)
            }
        }
    };
}

impl_axis_expr!(AxisExpr, View, as_ptr, {}, true);
impl_axis_expr!(AxisExprMut, ViewMut, as_mut_ptr, {mut}, false);

impl<T, S: Shape, L: Layout, A: Axis> Clone for AxisExpr<'_, T, S, L, A> {
    fn clone(&self) -> Self {
        Self {
            slice: self.slice,
            axis: self.axis,
            mapping: self.mapping.clone(),
            offset: self.offset,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.slice = source.slice;
        self.axis = source.axis;
        self.mapping.clone_from(&source.mapping);
        self.offset = source.offset;
    }
}

impl<T> Fill<T> {
    pub(crate) fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Debug> Debug for Fill<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("Fill").field(&self.value).finish()
    }
}

impl<T: Clone> Expression for Fill<T> {
    type Shape = ();

    const IS_REPEATABLE: bool = true;

    fn shape(&self) -> &() {
        &()
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        self.value.clone()
    }

    fn inner_rank(&self) -> usize {
        usize::MAX
    }

    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<T: Clone> IntoIterator for Fill<T> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}

impl<F> FillWith<F> {
    pub(crate) fn new(f: F) -> Self {
        Self { f }
    }
}

impl<T: Debug, F: FnMut() -> T> Debug for FillWith<F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FillWith").finish()
    }
}

impl<T, F: FnMut() -> T> Expression for FillWith<F> {
    type Shape = ();

    const IS_REPEATABLE: bool = true;

    fn shape(&self) -> &() {
        &()
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        (self.f)()
    }

    fn inner_rank(&self) -> usize {
        usize::MAX
    }

    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<T, F: FnMut() -> T> IntoIterator for FillWith<F> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}

impl<T, S: Shape> FromElem<T, S> {
    pub(crate) fn new(shape: S, elem: T) -> Self {
        _ = shape.checked_len().expect("invalid length");

        Self { shape, elem }
    }
}

impl<T: Debug, S: Shape> Debug for FromElem<T, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FromElem").field(&self.shape).field(&self.elem).finish()
    }
}

impl<T: Clone, S: Shape> Expression for FromElem<T, S> {
    type Shape = S;

    const IS_REPEATABLE: bool = true;

    fn shape(&self) -> &S {
        &self.shape
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        self.elem.clone()
    }

    fn inner_rank(&self) -> usize {
        usize::MAX
    }

    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<T: Clone, S: Shape> IntoIterator for FromElem<T, S> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}

impl<S: Shape, F> FromFn<S, F> {
    pub(crate) fn new(shape: S, f: F) -> Self {
        _ = shape.checked_len().expect("invalid length");

        Self { shape, f, index: S::Dims::default() }
    }
}

impl<S: Shape, F> Debug for FromFn<S, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FromFn").field(&self.shape).finish()
    }
}

impl<T, S: Shape, F: FnMut(&[usize]) -> T> Expression for FromFn<S, F> {
    type Shape = S;

    const IS_REPEATABLE: bool = true;

    fn shape(&self) -> &S {
        &self.shape
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        let value = (self.f)(self.index.as_ref());

        // Increment the last dimension, which will be reset by reset_dim().
        if self.rank() > 0 {
            self.index.as_mut()[self.shape.rank() - 1] += 1;
        }

        value
    }

    fn inner_rank(&self) -> usize {
        if self.shape.rank() > 0 { 1 } else { usize::MAX }
    }

    unsafe fn reset_dim(&mut self, index: usize, _: usize) {
        self.index.as_mut()[index] = 0;
    }

    unsafe fn step_dim(&mut self, index: usize) {
        // Don't increment the last dimension, since it is done in get_unchecked().
        if index + 1 < self.rank() {
            self.index.as_mut()[index] += 1;
        }
    }
}

impl<T, S: Shape, F: FnMut(&[usize]) -> T> IntoIterator for FromFn<S, F> {
    type Item = T;
    type IntoIter = Iter<Self>;

    fn into_iter(self) -> Iter<Self> {
        Iter::new(self)
    }
}

macro_rules! impl_lanes {
    ($name:tt, $expr:tt, $as_ptr:tt, {$($mut:tt)?}, $repeatable:tt) => {
        impl<'a, T, S: Shape, L: Layout, A: Axis> $name<'a, T, S, L, A> {
            pub(crate) fn new(
                slice: &'a $($mut)? Slice<T, S, L>,
                axis: A,
            ) -> Self {
                let mapping = axis.remove(slice.mapping());

                // Ensure that the subarray is valid.
                _ = mapping.shape().checked_len().expect("invalid length");

                Self { slice, axis, mapping, offset: 0 }
            }
        }

        impl<'a, T: Debug, S: Shape, L: Layout, A: Axis> Debug for $name<'a, T, S, L, A> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let index = self.axis.index(self.slice.rank());

                f.debug_tuple(stringify!($name)).field(&index).field(&self.slice).finish()
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> Expression for $name<'a, T, S, L, A> {
            type Shape = A::Remove<S>;

            const IS_REPEATABLE: bool = $repeatable;

            fn shape(&self) -> &Self::Shape {
                self.mapping.shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
                let offset = self.mapping.inner_stride() * index as isize;
                let mapping = self.axis.get(self.slice.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                unsafe { $expr::new_unchecked(self.slice.$as_ptr().offset(count), mapping) }
            }

            fn inner_rank(&self) -> usize {
                if Split::<A, S, L>::IS_DENSE {
                    // For static rank 0, the inner stride is 0 so we allow inner rank >0.
                    if A::Remove::<S>::RANK == Some(0) { usize::MAX } else { self.mapping.rank() }
                } else {
                    // For rank 0, the inner stride is always 0 so we can allow inner rank >0.
                    if self.mapping.rank() > 0 { 1 } else { usize::MAX }
                }
            }

            unsafe fn reset_dim(&mut self, index: usize, count: usize) {
                self.offset -= self.mapping.stride(index) * count as isize;
            }

            unsafe fn step_dim(&mut self, index: usize) {
                self.offset += self.mapping.stride(index);
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> IntoIterator for $name<'a, T, S, L, A> {
            type Item = $expr<'a, T, (A::Dim<S>,), Keep<A, S, L>>;
            type IntoIter = Iter<Self>;

            fn into_iter(self) -> Iter<Self> {
                Iter::new(self)
            }
        }
    };
}

impl_lanes!(Lanes, View, as_ptr, {}, true);
impl_lanes!(LanesMut, ViewMut, as_mut_ptr, {mut}, false);

impl<T, S: Shape, L: Layout, A: Axis> Clone for Lanes<'_, T, S, L, A> {
    fn clone(&self) -> Self {
        Self {
            slice: self.slice,
            axis: self.axis,
            mapping: self.mapping.clone(),
            offset: self.offset,
        }
    }

    fn clone_from(&mut self, source: &Self) {
        self.slice = source.slice;
        self.axis = source.axis;
        self.mapping.clone_from(&source.mapping);
        self.offset = source.offset;
    }
}
