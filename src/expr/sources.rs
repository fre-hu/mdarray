use std::fmt::{Debug, Formatter, Result};
use std::marker::PhantomData;

use crate::expr::expr::{Expr, ExprMut};
use crate::expression::Expression;
use crate::index::Axis;
use crate::iter::Iter;
use crate::layout::Layout;
use crate::mapping::Mapping;
use crate::shape::{IntoShape, Shape};
use crate::span::Span;

/// Array axis expression.
pub struct AxisExpr<'a, T, S: Shape, L: Layout, A: Axis> {
    span: &'a Span<T, S, L>,
    offset: isize,
    phantom: PhantomData<A>,
}

/// Mutable array axis expression.
pub struct AxisExprMut<'a, T, S: Shape, L: Layout, A: Axis> {
    span: &'a mut Span<T, S, L>,
    offset: isize,
    phantom: PhantomData<A>,
}

/// Expression that repeats an element by cloning.
#[derive(Clone, Copy)]
pub struct Fill<T> {
    value: T,
}

/// Expression that gives elements by calling a closure repeatedly.
#[derive(Clone, Copy)]
pub struct FillWith<F> {
    f: F,
}

/// Expression with a defined shape that repeats an element by cloning.
#[derive(Clone, Copy)]
pub struct FromElem<S, T> {
    shape: S,
    elem: T,
}

/// Expression with a defined shape and elements from the given function.
#[derive(Clone, Copy)]
pub struct FromFn<S: Shape, F> {
    shape: S,
    f: F,
    index: S::Dims,
}

/// Array lanes expression.
pub struct Lanes<'a, T, S: Shape, L: Layout, A: Axis> {
    span: &'a Span<T, S, L>,
    offset: isize,
    phantom: PhantomData<A>,
}

/// Mutable array lanes expression.
pub struct LanesMut<'a, T, S: Shape, L: Layout, A: Axis> {
    span: &'a mut Span<T, S, L>,
    offset: isize,
    phantom: PhantomData<A>,
}

/// Creates an expression with elements by cloning `value`.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid};
///
/// let mut g = grid![0; 3];
///
/// g.assign(expr::fill(1));
///
/// assert_eq!(g, expr![1; 3]);
/// ```
pub fn fill<T: Clone>(value: T) -> Fill<T> {
    Fill::new(value)
}

/// Creates an expression with elements returned by calling a closure repeatedly.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, grid};
///
/// let mut g = grid![0; 3];
///
/// g.assign(expr::fill_with(|| 1));
///
/// assert_eq!(g, expr![1; 3]);
/// ```
pub fn fill_with<T, F: FnMut() -> T>(f: F) -> FillWith<F> {
    FillWith::new(f)
}

/// Creates an expression with the given shape and elements by cloning `value`.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression};
///
/// assert_eq!(expr::from_elem([2, 3], 1).eval(), expr![[1; 2]; 3]);
/// ```
pub fn from_elem<T: Clone, I: IntoShape>(shape: I, elem: T) -> FromElem<I::IntoShape, T> {
    FromElem::new(shape.into_shape(), elem)
}

/// Creates an expression with the given shape and elements from the given function.
///
/// # Examples
///
/// ```
/// use mdarray::{expr, Expression};
///
/// assert_eq!(expr::from_fn([2, 3], |[i, j]| 2 * j + i).eval(), expr![[0, 1], [2, 3], [4, 5]]);
/// ```
pub fn from_fn<T, I: IntoShape, F>(shape: I, f: F) -> FromFn<I::IntoShape, F>
where
    F: FnMut(<I::IntoShape as Shape>::Dims) -> T,
{
    FromFn::new(shape.into_shape(), f)
}

macro_rules! impl_axis_expr {
    ($name:tt, $expr:tt, $as_ptr:tt, {$($mut:tt)?}, $repeatable:tt) => {
        impl<'a, T, S: Shape, L: Layout, A: Axis> $name<'a, T, S, L, A> {
            pub(crate) fn new(
                span: &'a $($mut)? Span<T, S, L>,
            ) -> Self {
                // Ensure that the dimension is valid.
                _ = A::index(S::RANK);

                // Ensure that the subarray is valid.
                _ = A::remove(span.mapping()).shape().checked_len().expect("invalid length");

                Self { span, offset: 0, phantom: PhantomData }
            }
        }

        impl<'a, T: Debug, S: Shape, L: Layout, A: Axis> Debug for $name<'a, T, S, L, A> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let index = A::index(S::RANK);

                f.debug_tuple(stringify!($name)).field(&index).field(&self.span).finish()
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> Expression for $name<'a, T, S, L, A> {
            type Shape = A::Dim<S>;

            const IS_REPEATABLE: bool = $repeatable;
            const SPLIT_MASK: usize = 1;

            fn shape(&self) -> Self::Shape {
                A::keep(self.span.mapping()).shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
                let stride = A::keep(self.span.mapping()).stride(0);
                let offset = self.offset + stride * index as isize;

                let mapping = A::remove(self.span.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                $expr::new_unchecked(self.span.$as_ptr().offset(count), mapping)
            }

            unsafe fn reset_dim(&mut self, _: usize, _: usize) {
                self.offset = 0;
            }

            unsafe fn step_dim(&mut self, _: usize) {
                self.offset += A::keep(self.span.mapping()).stride(0);
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> IntoIterator for $name<'a, T, S, L, A> {
            type Item = $expr<'a, T, A::Other<S>, A::Remove<S, L>>;
            type IntoIter = Iter<Self>;

            fn into_iter(self) -> Iter<Self> {
                Iter::new(self)
            }
        }
    };
}

impl_axis_expr!(AxisExpr, Expr, as_ptr, {}, true);
impl_axis_expr!(AxisExprMut, ExprMut, as_mut_ptr, {mut}, false);

impl<'a, T, S: Shape, L: Layout, A: Axis> Clone for AxisExpr<'a, T, S, L, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, S: Shape, L: Layout, A: Axis> Copy for AxisExpr<'a, T, S, L, A> {}

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
    const SPLIT_MASK: usize = 0;

    fn shape(&self) {}

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        self.value.clone()
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
    const SPLIT_MASK: usize = 0;

    fn shape(&self) {}

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        (self.f)()
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

impl<S: Shape, T> FromElem<S, T> {
    pub(crate) fn new(shape: S, elem: T) -> Self {
        _ = shape.checked_len().expect("invalid length");

        Self { shape, elem }
    }
}

impl<S: Shape, T: Debug> Debug for FromElem<S, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FromElem").field(&self.shape).field(&self.elem).finish()
    }
}

impl<S: Shape, T: Clone> Expression for FromElem<S, T> {
    type Shape = S;

    const IS_REPEATABLE: bool = true;
    const SPLIT_MASK: usize = 0;

    fn shape(&self) -> S {
        self.shape
    }

    unsafe fn get_unchecked(&mut self, _: usize) -> T {
        self.elem.clone()
    }

    unsafe fn reset_dim(&mut self, _: usize, _: usize) {}
    unsafe fn step_dim(&mut self, _: usize) {}
}

impl<S: Shape, T: Clone> IntoIterator for FromElem<S, T> {
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

impl<S: Shape, T: Debug, F: FnMut(S::Dims) -> T> Debug for FromFn<S, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        f.debug_tuple("FromFn").field(&self.shape).finish()
    }
}

impl<S: Shape, T, F: FnMut(S::Dims) -> T> Expression for FromFn<S, F> {
    type Shape = S;

    const IS_REPEATABLE: bool = true;
    const SPLIT_MASK: usize = (1 << S::RANK) - 1;

    fn shape(&self) -> S {
        self.shape
    }

    unsafe fn get_unchecked(&mut self, index: usize) -> T {
        if S::RANK > 0 {
            self.index[0] = index;
        }

        (self.f)(self.index)
    }

    unsafe fn reset_dim(&mut self, index: usize, _: usize) {
        self.index[index] = 0;
    }

    unsafe fn step_dim(&mut self, index: usize) {
        self.index[index] += 1;
    }
}

impl<S: Shape, T, F: FnMut(S::Dims) -> T> IntoIterator for FromFn<S, F> {
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
                span: &'a $($mut)? Span<T, S, L>,
            ) -> Self {
                // Ensure that the dimension is valid.
                _ = A::index(S::RANK);

                // Ensure that the subarray is valid.
                _ = A::remove(span.mapping()).shape().checked_len().expect("invalid length");

                Self { span, offset: 0, phantom: PhantomData }
            }
        }

        impl<'a, T: Debug, S: Shape, L: Layout, A: Axis> Debug for $name<'a, T, S, L, A> {
            fn fmt(&self, f: &mut Formatter<'_>) -> Result {
                let index = A::index(S::RANK);

                f.debug_tuple(stringify!($name)).field(&index).field(&self.span).finish()
            }
        }

         impl<'a, T, S: Shape, L: Layout, A: Axis> Expression for $name<'a, T, S, L, A> {
            type Shape = A::Other<S>;

            const IS_REPEATABLE: bool = $repeatable;
            const SPLIT_MASK: usize = ((1 << S::RANK) - 1) >> 1;

            fn shape(&self) -> Self::Shape {
                A::remove(self.span.mapping()).shape()
            }

            unsafe fn get_unchecked(&mut self, index: usize) -> Self::Item {
                let stride = if S::RANK > 1 { A::remove(self.span.mapping()).stride(0) } else { 0 };
                let offset = self.offset + stride * index as isize;

                let mapping = A::keep(self.span.mapping());

                // If the view is empty, we must not offset the pointer.
                let count = if mapping.is_empty() { 0 } else { offset };

                $expr::new_unchecked(self.span.$as_ptr().offset(count), mapping)
            }

            unsafe fn reset_dim(&mut self, index: usize, count: usize) {
                self.offset -= A::remove(self.span.mapping()).stride(index) * count as isize;
            }

            unsafe fn step_dim(&mut self, index: usize) {
                self.offset += A::remove(self.span.mapping()).stride(index);
            }
        }

        impl<'a, T, S: Shape, L: Layout, A: Axis> IntoIterator for $name<'a, T, S, L, A> {
            type Item = $expr<'a, T, A::Dim<S>, A::Keep<S, L>>;
            type IntoIter = Iter<Self>;

            fn into_iter(self) -> Iter<Self> {
                Iter::new(self)
            }
        }
    };
}

impl_lanes!(Lanes, Expr, as_ptr, {}, true);
impl_lanes!(LanesMut, ExprMut, as_mut_ptr, {mut}, false);

impl<'a, T, S: Shape, L: Layout, A: Axis> Clone for Lanes<'a, T, S, L, A> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T, S: Shape, L: Layout, A: Axis> Copy for Lanes<'a, T, S, L, A> {}
